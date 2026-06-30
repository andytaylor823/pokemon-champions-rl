"""CVPN — Counterfactual Value + Policy Network.

Phase 1 implementation: shared Transformer backbone + two heads (policy prior +
scalar value).  Phase 4 hooks (vector CFV head, three-tier public seam) are
designed-in but pinned to trivial values.

See docs/plans/cvpn.md for the full design spec.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from action_space import MOVE_PHASE_COUNT, MOVE_PHASE_OFFSET, TEAM_PREVIEW_COUNT, TEAM_PREVIEW_OFFSET, A
from encoder import ENTITY_FEATURE_DIM, FIELD_FEATURE_DIM, SCALAR_FEATURE_DIM, SIDE_FEATURE_DIM
from vocab import ABILITY_VOCAB, ITEM_VOCAB, MOVE_VOCAB, SPECIES_VOCAB

if TYPE_CHECKING:
    from obs_bundle import ObsBundle

# Fixed prefix tokens before the variable-length entity tokens
_N_GLOBAL_TOKENS = 5  # CLS, field, side_p1, side_p2, meta


@dataclass(frozen=True)
class CVPNConfig:
    """Architecture hyperparameters for the CVPN.

    Head widths derive from action_space constants — never hard-coded (D7).
    Embedding table sizes come from vocab.py singletons (D8).
    """

    # Transformer backbone
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    ffn_mult: int = 4
    dropout: float = 0.1

    # Embedding dimensions
    d_species: int = 32
    d_ability: int = 16
    d_item: int = 16
    d_move: int = 32

    # Vocab sizes (from vocab.py — include reserved id 0 = UNK/NONE)
    n_species: int = SPECIES_VOCAB.size
    n_abilities: int = ABILITY_VOCAB.size
    n_items: int = ITEM_VOCAB.size
    n_moves: int = MOVE_VOCAB.size

    # Input feature dimensions (from encoder sentinel-computed constants)
    entity_feature_dim: int = ENTITY_FEATURE_DIM
    field_feature_dim: int = FIELD_FEATURE_DIM
    side_feature_dim: int = SIDE_FEATURE_DIM
    scalar_feature_dim: int = SCALAR_FEATURE_DIM

    # Action space (from action_space — never hard-coded; D7)
    action_dim: int = A
    team_preview_count: int = TEAM_PREVIEW_COUNT
    move_phase_count: int = MOVE_PHASE_COUNT


# ---------------------------------------------------------------------------
# Move attention pool (D3)
# ---------------------------------------------------------------------------


class MoveAttentionPool(nn.Module):
    """Collapse 4 per-entity move embeddings into one summary via learned attention.

    A single learnable query attends to the 4 move embedding vectors, producing
    a fixed-width summary that can capture move synergies (e.g. Protect + Fake Out).
    """

    def __init__(self, d_move: int) -> None:
        super().__init__()
        self.d_move = d_move
        # Learnable query shared across all entities
        self.query = nn.Parameter(torch.randn(d_move))
        self.k_proj = nn.Linear(d_move, d_move, bias=False)
        self.v_proj = nn.Linear(d_move, d_move, bias=False)
        self._scale = 1.0 / math.sqrt(d_move)

    def forward(self, move_embs: torch.Tensor) -> torch.Tensor:
        """Pool 4 move embeddings into one summary per entity.

        Args:
            move_embs: [*, 4, d_move] — embedded move vectors per entity.

        Returns:
            [*, d_move] — one summary vector per entity.
        """
        # Project keys and values from the 4 move embeddings
        k = self.k_proj(move_embs)  # [*, 4, d_move]
        v = self.v_proj(move_embs)  # [*, 4, d_move]
        # Broadcast query to match leading dims: [*, 1, d_move]
        q = self.query.expand(*move_embs.shape[:-2], 1, self.d_move)
        # Scaled dot-product attention: query attends to 4 keys
        scores = torch.matmul(q, k.transpose(-2, -1)) * self._scale  # [*, 1, 4]
        attn = torch.softmax(scores, dim=-1)  # [*, 1, 4]
        out = torch.matmul(attn, v)  # [*, 1, d_move]
        return out.squeeze(-2)  # [*, d_move]


# ---------------------------------------------------------------------------
# CVPN network
# ---------------------------------------------------------------------------


class CVPN(nn.Module):
    """Counterfactual Value + Policy Network (Phase 1).

    Shared Transformer backbone with two heads:
    - Policy: phase-split (team preview + move phase), assembled into [A] logits
    - Value: scalar in [-1, 1] from CLS via tanh

    Input:  ObsBundle (TensorDict from encoder)
    Output: (policy_logits, value)
    """

    def __init__(self, config: CVPNConfig | None = None) -> None:
        super().__init__()
        cfg = config or CVPNConfig()
        self.config = cfg

        # --- Embedding tables (D8) — sized from vocab, padding_idx=0 for UNK ---
        self.species_embed = nn.Embedding(cfg.n_species, cfg.d_species, padding_idx=0)
        self.ability_embed = nn.Embedding(cfg.n_abilities, cfg.d_ability, padding_idx=0)
        self.item_embed = nn.Embedding(cfg.n_items, cfg.d_item, padding_idx=0)
        # Shared move embedding table across all 4 slots
        self.move_embed = nn.Embedding(cfg.n_moves, cfg.d_move, padding_idx=0)

        # --- Move attention pool (D3) ---
        self.move_pool = MoveAttentionPool(cfg.d_move)

        # --- Input projections (D2) ---
        # Entity: concat of continuous features + all embedding outputs → d_model
        entity_in_dim = cfg.entity_feature_dim + cfg.d_species + cfg.d_ability + cfg.d_item + cfg.d_move
        self.entity_proj = nn.Linear(entity_in_dim, cfg.d_model)
        # Global tokens: type-specific projections, each → d_model
        self.field_proj = nn.Linear(cfg.field_feature_dim, cfg.d_model)
        self.side_proj = nn.Linear(cfg.side_feature_dim, cfg.d_model)  # shared for both sides
        self.meta_proj = nn.Linear(cfg.scalar_feature_dim, cfg.d_model)

        # CLS token — learnable nn.Parameter, randomly initialized, trained
        self.cls_token = nn.Parameter(torch.randn(cfg.d_model))

        # --- Transformer backbone (D1) — pre-norm, no positional encoding ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ffn_mult * cfg.d_model,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # Final LayerNorm after the last block (standard pre-norm convention)
        self.backbone = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.n_layers,
            norm=nn.LayerNorm(cfg.d_model),
            enable_nested_tensor=False,  # incompatible with norm_first=True
        )

        # --- Policy head (D4) — phase-split, unified [A] output ---
        self.tp_head = nn.Linear(cfg.d_model, cfg.team_preview_count)
        self.mp_head = nn.Linear(cfg.d_model, cfg.move_phase_count)

        # --- Value head (D5) — scalar from CLS ---
        self.value_head = nn.Linear(cfg.d_model, 1)

    def forward(self, obs: ObsBundle) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the full CVPN forward pass.

        Args:
            obs: ObsBundle TensorDict, unbatched (batch_size=[]) or
                 batched (batch_size=[B]) from collate_obs_bundles.

        Returns:
            (policy_logits, value) where:
            - policy_logits: [A] (unbatched) or [B, A] — illegal entries at -inf
            - value: scalar [] (unbatched) or [B] — in [-1, 1]
        """
        # Handle both batched and unbatched inputs uniformly
        unbatched = obs.batch_size == torch.Size([])
        if unbatched:
            obs = obs.unsqueeze(0)

        batch_size = obs.batch_size[0]

        # --- 1. Embed categorical IDs (D8) ---
        species_emb = self.species_embed(obs["ids", "species"])  # [B, N, d_species]
        ability_emb = self.ability_embed(obs["ids", "ability"])  # [B, N, d_ability]
        item_emb = self.item_embed(obs["ids", "item"])  # [B, N, d_item]
        move_emb_raw = self.move_embed(obs["ids", "moves"])  # [B, N, 4, d_move]

        # --- 2. Move attention pool (D3) ---
        # MoveAttentionPool is rank-agnostic ([*, 4, d_move]), so feed the 4-D
        # tensor directly — matmul broadcasting handles [B, N, ...] natively.
        move_summary = self.move_pool(move_emb_raw)  # [B, N, d_move]

        # --- 3. Entity input projection ---
        entities = obs["entities"]  # [B, N, entity_feature_dim]
        entity_input = torch.cat([entities, species_emb, ability_emb, item_emb, move_summary], dim=-1)
        entity_tokens = self.entity_proj(entity_input)  # [B, N, d_model]

        # --- 4. Global token projections ---
        field_tok = self.field_proj(obs["field"]).unsqueeze(1)  # [B, 1, d_model]
        side_p1 = self.side_proj(obs["sides"][:, 0]).unsqueeze(1)  # [B, 1, d_model]
        side_p2 = self.side_proj(obs["sides"][:, 1]).unsqueeze(1)  # [B, 1, d_model]
        meta_tok = self.meta_proj(obs["scalars"]).unsqueeze(1)  # [B, 1, d_model]
        # Expand CLS for the batch
        cls = self.cls_token.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)  # [B, 1, d_model]

        # --- 5. Assemble token sequence [CLS, field, side_p1, side_p2, meta, entities...] ---
        seq = torch.cat([cls, field_tok, side_p1, side_p2, meta_tok, entity_tokens], dim=1)

        # --- 6. Key-padding mask ---
        # ObsBundle: True = real token; torch Transformer: True = IGNORE
        entity_ignore = ~obs["padding_mask"]  # [B, N]
        # The 5 global prefix tokens are always real → never ignored
        global_attend = torch.zeros(batch_size, _N_GLOBAL_TOKENS, dtype=torch.bool, device=seq.device)
        key_padding_mask = torch.cat([global_attend, entity_ignore], dim=1)  # [B, 5+N]

        # --- 7. Transformer backbone ---
        h = self.backbone(seq, src_key_padding_mask=key_padding_mask)  # [B, 5+N, d_model]
        cls_out = h[:, 0]  # [B, d_model] — read the evolved CLS token

        # --- 8. Policy head (D4) ---
        policy_logits = self._assemble_policy(cls_out, obs["action_mask"])

        # --- 9. Value head (D5) ---
        value = torch.tanh(self.value_head(cls_out)).squeeze(-1)  # [B]

        # Strip the batch dim we added for unbatched input
        if unbatched:
            policy_logits = policy_logits.squeeze(0)  # [A]
            value = value.squeeze(0)  # scalar

        return policy_logits, value

    def _assemble_policy(self, cls_out: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        """Build the [B, A] policy logit vector from the two phase heads, then mask to legal.

        Each head produces logits for its own region only; the action_mask
        (phase-exclusive by construction in action_space.legal_mask) zeroes the
        inactive phase's region, so no explicit phase routing is needed.
        """
        cfg = self.config
        # Initialize all logits to -inf, then write both heads into their regions
        logits = torch.full((cls_out.shape[0], cfg.action_dim), float("-inf"), device=cls_out.device, dtype=cls_out.dtype)
        logits[:, TEAM_PREVIEW_OFFSET : TEAM_PREVIEW_OFFSET + cfg.team_preview_count] = self.tp_head(cls_out)
        logits[:, MOVE_PHASE_OFFSET : MOVE_PHASE_OFFSET + cfg.move_phase_count] = self.mp_head(cls_out)
        return logits.masked_fill(~action_mask, float("-inf"))
