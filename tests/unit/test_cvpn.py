"""Unit tests for the CVPN module.

Covers forward-pass shapes, mask correctness, phase-split routing, value range,
gradient flow, and batch/padding invariance. Uses the same real_snapshot.json
fixture pattern as the encoder tests.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import action_space
from action_space import A, MOVE_PHASE_COUNT, MOVE_PHASE_OFFSET, TEAM_PREVIEW_COUNT, TEAM_PREVIEW_OFFSET
from cvpn import CVPN, CVPNConfig, _N_GLOBAL_TOKENS
from encoder import ENTITY_FEATURE_DIM, FIELD_FEATURE_DIM, SCALAR_FEATURE_DIM, SIDE_FEATURE_DIM, encode
from obs_bundle import ObsBundle, collate_obs_bundles, make_obs_bundle
from state_types import BattleSnapshot, FieldSnapshot, PokemonSnapshot, SideSnapshot, StateView

# ---------------------------------------------------------------------------
# Fixtures from real snapshot (same pattern as test_encoder.py)
# ---------------------------------------------------------------------------

_FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "real_snapshot.json"
with open(_FIXTURE_PATH, encoding="utf-8") as _f:
    _REAL = json.load(_f)

_REAL_MON_DICT = _REAL["team_preview"]["snapshot"]["sides"][0]["pokemon"][0]


def _minimal_mon(**overrides) -> PokemonSnapshot:
    data = {**_REAL_MON_DICT, **overrides}
    return PokemonSnapshot.model_validate(data)


def _minimal_field(**overrides) -> FieldSnapshot:
    base = {"weather": None, "weatherDuration": None, "terrain": None, "terrainDuration": None, "pseudoWeather": {}}
    base.update(overrides)
    return FieldSnapshot.model_validate(base)


def _minimal_side(pokemon=None, side_conditions=None, side_id="p1") -> SideSnapshot:
    if pokemon is None:
        pokemon = [_minimal_mon() for _ in range(6)]
    conds = side_conditions or {}
    return SideSnapshot(id=side_id, sideConditions=conds, pokemon=pokemon)


def _team_preview_view() -> StateView:
    """Build a team-preview StateView (12 tokens, 360 legal actions)."""
    return StateView(
        phase="teamPreview",
        terminal=False,
        to_move=["p1", "p2"],
        snapshot=BattleSnapshot(
            turn=0,
            sides=[
                _minimal_side(side_id="p1"),
                _minimal_side(
                    [_minimal_mon(species="venusaur", ability="chlorophyll") for _ in range(6)],
                    side_id="p2",
                ),
            ],
            field=_minimal_field(),
        ),
        legal={
            "p1": {"teamSize": 6},
            "p2": {"teamSize": 6},
        },
        utility=None,
    )


def _move_phase_view() -> StateView:
    """Build a move-phase StateView (12 tokens, legal mask in move region)."""
    # Construct active mons with moves so the mask has legal move-phase actions
    active_mon = _minimal_mon(active=True, position=0)
    active_mon2 = _minimal_mon(species="venusaur", ability="chlorophyll", active=True, position=1)
    bench = [_minimal_mon(active=False, position=i) for i in range(2, 6)]
    my_mons = [active_mon, active_mon2] + bench[:4]

    opp_active = _minimal_mon(species="garchomp", ability="roughskin", active=True, position=0)
    opp_active2 = _minimal_mon(species="whimsicott", ability="prankster", active=True, position=1)
    opp_bench = [_minimal_mon(species="pelipper", ability="drizzle", active=False, position=i) for i in range(2, 6)]
    opp_mons = [opp_active, opp_active2] + opp_bench[:4]

    # Build a request with two active slots that each have moves
    move_data = [
        {"moves": [{"move": "heatwave", "id": "heatwave", "pp": 10, "maxpp": 10, "target": "allAdjacentFoes", "disabled": False}], "canMegaEvo": False},
        {"moves": [{"move": "protect", "id": "protect", "pp": 10, "maxpp": 10, "target": "self", "disabled": False}], "canMegaEvo": False},
    ]
    side_pokemon = [
        {"condition": "200/200"},  # active slot 0
        {"condition": "180/180"},  # active slot 1
        {"condition": "150/150"},  # bench slot 0
        {"condition": "160/160"},  # bench slot 1
    ]
    legal_request = {"active": move_data, "side": {"pokemon": side_pokemon}}

    return StateView(
        phase="move",
        terminal=False,
        to_move=["p1"],
        snapshot=BattleSnapshot(
            turn=1,
            sides=[
                _minimal_side(my_mons, side_id="p1"),
                _minimal_side(opp_mons, side_id="p2"),
            ],
            field=_minimal_field(),
        ),
        legal={"p1": legal_request, "p2": None},
        utility=None,
    )


def _make_dummy_bundle(n_tokens: int, phase: str = "move") -> ObsBundle:
    """Build a synthetic ObsBundle with correct shapes for testing."""
    entities = torch.randn(n_tokens, ENTITY_FEATURE_DIM)
    species_ids = torch.randint(1, 50, (n_tokens,))
    ability_ids = torch.randint(1, 30, (n_tokens,))
    item_ids = torch.randint(1, 20, (n_tokens,))
    move_ids = torch.randint(1, 100, (n_tokens, 4))
    belief_weight = torch.ones(n_tokens)
    slot_id = torch.zeros(n_tokens, dtype=torch.long)
    field = torch.randn(FIELD_FEATURE_DIM)
    sides = torch.randn(2, SIDE_FEATURE_DIM)

    # Build scalars with phase one-hot
    scalars = torch.zeros(SCALAR_FEATURE_DIM)
    scalars[0] = 0.05  # turn
    if phase == "teamPreview":
        scalars[1] = 1.0
    elif phase == "move":
        scalars[2] = 1.0
    elif phase == "forceSwitch":
        scalars[3] = 1.0

    # Build a mask with at least some legal actions in the active region
    action_mask = torch.zeros(A, dtype=torch.bool)
    if phase == "teamPreview":
        action_mask[TEAM_PREVIEW_OFFSET:TEAM_PREVIEW_OFFSET + 10] = True
    else:
        action_mask[MOVE_PHASE_OFFSET:MOVE_PHASE_OFFSET + 10] = True

    padding_mask = torch.ones(n_tokens, dtype=torch.bool)

    return make_obs_bundle(
        entities=entities,
        species_ids=species_ids,
        ability_ids=ability_ids,
        item_ids=item_ids,
        move_ids=move_ids,
        belief_weight=belief_weight,
        slot_id=slot_id,
        field=field,
        sides=sides,
        scalars=scalars,
        action_mask=action_mask,
        padding_mask=padding_mask,
    )


@pytest.fixture
def default_cvpn():
    """A CVPN with default config, in eval mode for deterministic output."""
    model = CVPN()
    model.eval()
    return model


@pytest.fixture
def tp_obs():
    """ObsBundle from a real-fixture team-preview view."""
    return encode(_team_preview_view(), perspective="p1")


@pytest.fixture
def move_obs():
    """ObsBundle from a real-fixture move-phase view."""
    return encode(_move_phase_view(), perspective="p1")


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestCVPNConfig:
    """Config derives from action_space/vocab constants, not hard-coded."""

    def test_head_widths_match_action_space(self):
        cfg = CVPNConfig()
        assert cfg.team_preview_count == TEAM_PREVIEW_COUNT
        assert cfg.move_phase_count == MOVE_PHASE_COUNT
        assert cfg.action_dim == A
        assert cfg.action_dim == cfg.team_preview_count + cfg.move_phase_count

    def test_vocab_sizes_match(self):
        from vocab import ABILITY_VOCAB, ITEM_VOCAB, MOVE_VOCAB, SPECIES_VOCAB

        cfg = CVPNConfig()
        assert cfg.n_species == SPECIES_VOCAB.size
        assert cfg.n_abilities == ABILITY_VOCAB.size
        assert cfg.n_items == ITEM_VOCAB.size
        assert cfg.n_moves == MOVE_VOCAB.size

    def test_feature_dims_match_encoder(self):
        cfg = CVPNConfig()
        assert cfg.entity_feature_dim == ENTITY_FEATURE_DIM
        assert cfg.field_feature_dim == FIELD_FEATURE_DIM
        assert cfg.side_feature_dim == SIDE_FEATURE_DIM
        assert cfg.scalar_feature_dim == SCALAR_FEATURE_DIM


# ---------------------------------------------------------------------------
# Forward pass shape tests
# ---------------------------------------------------------------------------


class TestForwardShapes:
    """Forward pass produces correctly shaped outputs."""

    def test_unbatched_team_preview(self, default_cvpn, tp_obs):
        policy, value = default_cvpn(tp_obs)
        assert policy.shape == (A,)
        assert value.shape == ()  # scalar

    def test_unbatched_move_phase(self, default_cvpn, move_obs):
        policy, value = default_cvpn(move_obs)
        assert policy.shape == (A,)
        assert value.shape == ()

    def test_batched_same_n(self, default_cvpn):
        obs1 = _make_dummy_bundle(12, phase="teamPreview")
        obs2 = _make_dummy_bundle(12, phase="teamPreview")
        batched = collate_obs_bundles([obs1, obs2])
        policy, value = default_cvpn(batched)
        assert policy.shape == (2, A)
        assert value.shape == (2,)

    def test_batched_variable_n(self, default_cvpn):
        obs_12 = _make_dummy_bundle(12, phase="move")
        obs_8 = _make_dummy_bundle(8, phase="move")
        batched = collate_obs_bundles([obs_12, obs_8])
        policy, value = default_cvpn(batched)
        # After collation N is padded to max(12, 8) = 12
        assert policy.shape == (2, A)
        assert value.shape == (2,)

    def test_dummy_various_n(self, default_cvpn):
        """Forward works with different token counts."""
        for n in (4, 8, 12, 16):
            obs = _make_dummy_bundle(n, phase="move")
            policy, value = default_cvpn(obs)
            assert policy.shape == (A,)


# ---------------------------------------------------------------------------
# Mask correctness
# ---------------------------------------------------------------------------


class TestMaskCorrectness:
    """Illegal logits are -inf; legal probs sum to ~1."""

    def test_illegal_are_neg_inf_team_preview(self, default_cvpn, tp_obs):
        policy, _ = default_cvpn(tp_obs)
        mask = tp_obs["action_mask"]
        illegal = ~mask
        assert (policy[illegal] == float("-inf")).all()

    def test_illegal_are_neg_inf_move_phase(self, default_cvpn, move_obs):
        policy, _ = default_cvpn(move_obs)
        mask = move_obs["action_mask"]
        illegal = ~mask
        assert (policy[illegal] == float("-inf")).all()

    def test_legal_probs_sum_to_one_tp(self, default_cvpn, tp_obs):
        policy, _ = default_cvpn(tp_obs)
        probs = torch.softmax(policy, dim=-1)
        mask = tp_obs["action_mask"]
        legal_prob_sum = probs[mask].sum()
        assert legal_prob_sum.item() == pytest.approx(1.0, abs=1e-5)

    def test_legal_probs_sum_to_one_move(self, default_cvpn, move_obs):
        policy, _ = default_cvpn(move_obs)
        probs = torch.softmax(policy, dim=-1)
        mask = move_obs["action_mask"]
        legal_prob_sum = probs[mask].sum()
        assert legal_prob_sum.item() == pytest.approx(1.0, abs=1e-5)

    def test_illegal_probs_are_zero(self, default_cvpn, tp_obs):
        policy, _ = default_cvpn(tp_obs)
        probs = torch.softmax(policy, dim=-1)
        mask = tp_obs["action_mask"]
        illegal_prob_sum = probs[~mask].sum()
        assert illegal_prob_sum.item() == pytest.approx(0.0, abs=1e-7)

    def test_sampled_index_maps_to_valid_choice(self, default_cvpn, tp_obs):
        """An index sampled from the legal region produces a valid choice string."""
        policy, _ = default_cvpn(tp_obs)
        probs = torch.softmax(policy, dim=-1)
        idx = torch.multinomial(probs, 1).item()
        choice = action_space.index_to_choice_string(idx)
        assert choice.startswith("team ")

    def test_sampled_move_index_maps_to_valid_choice(self, default_cvpn, move_obs):
        policy, _ = default_cvpn(move_obs)
        probs = torch.softmax(policy, dim=-1)
        idx = torch.multinomial(probs, 1).item()
        choice = action_space.index_to_choice_string(idx)
        # Move phase choices contain "move" or "switch"
        assert "move" in choice or "switch" in choice


# ---------------------------------------------------------------------------
# Phase-split routing
# ---------------------------------------------------------------------------


class TestPhaseSplit:
    """Team-preview lights only [0, 360); move lights only [360, 1089)."""

    def test_team_preview_region_only(self, default_cvpn, tp_obs):
        policy, _ = default_cvpn(tp_obs)
        # Move-phase region must be all -inf
        move_region = policy[MOVE_PHASE_OFFSET:]
        assert (move_region == float("-inf")).all()
        # Team-preview region should have at least some finite logits
        tp_region = policy[TEAM_PREVIEW_OFFSET:TEAM_PREVIEW_OFFSET + TEAM_PREVIEW_COUNT]
        assert (tp_region > float("-inf")).any()

    def test_move_phase_region_only(self, default_cvpn, move_obs):
        policy, _ = default_cvpn(move_obs)
        # Team-preview region must be all -inf
        tp_region = policy[:TEAM_PREVIEW_COUNT]
        assert (tp_region == float("-inf")).all()
        # Move-phase region should have at least some finite logits
        move_region = policy[MOVE_PHASE_OFFSET:]
        assert (move_region > float("-inf")).any()

    def test_force_switch_uses_move_region(self):
        """forceSwitch phase should use the move-phase head."""
        obs = _make_dummy_bundle(8, phase="forceSwitch")
        model = CVPN()
        model.eval()
        policy, _ = model(obs)
        tp_region = policy[:TEAM_PREVIEW_COUNT]
        assert (tp_region == float("-inf")).all()
        move_region = policy[MOVE_PHASE_OFFSET:]
        assert (move_region > float("-inf")).any()


# ---------------------------------------------------------------------------
# Value range
# ---------------------------------------------------------------------------


class TestValueRange:
    """Value output is bounded in [-1, 1] by tanh."""

    def test_value_in_range_tp(self, default_cvpn, tp_obs):
        _, value = default_cvpn(tp_obs)
        assert -1.0 <= value.item() <= 1.0

    def test_value_in_range_move(self, default_cvpn, move_obs):
        _, value = default_cvpn(move_obs)
        assert -1.0 <= value.item() <= 1.0

    def test_value_in_range_many_random(self, default_cvpn):
        """Run 20 random bundles — all values must be in [-1, 1]."""
        for _ in range(20):
            obs = _make_dummy_bundle(torch.randint(4, 16, (1,)).item())
            _, value = default_cvpn(obs)
            assert -1.0 <= value.item() <= 1.0


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


class TestGradientFlow:
    """loss.backward() populates grads on backbone + both heads + embeddings."""

    def test_all_params_have_grad(self, tp_obs, move_obs):
        """Exercise BOTH phases so every head and projection gets gradient.

        Team-preview alone won't touch mp_head; move-phase alone won't touch
        tp_head.  Combining both ensures full coverage.  We also use non-zero
        field/side inputs so the input projections' *weights* (not just biases)
        get gradient.
        """
        model = CVPN()
        model.train()

        # Forward both phases and accumulate a combined loss
        total_loss = torch.tensor(0.0)
        for obs in (tp_obs, move_obs):
            policy, value = model(obs)
            # Stable cross-entropy-style loss from legal logits
            mask = obs["action_mask"]
            legal_logits = policy[mask]
            log_probs = torch.log_softmax(legal_logits, dim=-1)
            policy_loss = -log_probs.mean()
            value_loss = (value - 0.5) ** 2
            total_loss = total_loss + policy_loss + value_loss

        total_loss.backward()

        # Every parameter should have a non-None gradient
        no_grad_params = []
        for name, param in model.named_parameters():
            if param.grad is None:
                no_grad_params.append(name)

        assert no_grad_params == [], f"Parameters with no gradient: {no_grad_params}"

    def test_embedding_tables_receive_grad(self, tp_obs):
        model = CVPN()
        model.train()
        policy, value = model(tp_obs)

        loss = policy.sum() + value
        loss.backward()

        # Check each embedding table specifically
        for name in ("species_embed", "ability_embed", "item_embed", "move_embed"):
            embed = getattr(model, name)
            assert embed.weight.grad is not None, f"{name}.weight.grad is None"

    def test_cls_token_receives_grad(self, tp_obs):
        model = CVPN()
        model.train()
        policy, value = model(tp_obs)
        (policy.sum() + value).backward()
        assert model.cls_token.grad is not None
        assert model.cls_token.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Batch / padding invariance
# ---------------------------------------------------------------------------


class TestBatchInvariance:
    """Collated batched output matches unbatched per-sample output."""

    def test_batched_matches_unbatched(self):
        """Two bundles collated should produce the same per-sample results
        as running each unbatched (modulo floating-point tolerance)."""
        torch.manual_seed(42)
        model = CVPN()
        model.eval()

        obs1 = _make_dummy_bundle(12, phase="teamPreview")
        obs2 = _make_dummy_bundle(12, phase="teamPreview")

        # Unbatched forward
        with torch.no_grad():
            p1, v1 = model(obs1)
            p2, v2 = model(obs2)

        # Batched forward
        batched = collate_obs_bundles([obs1, obs2])
        with torch.no_grad():
            p_batch, v_batch = model(batched)

        # Policy logits should match per sample
        torch.testing.assert_close(p_batch[0], p1, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(p_batch[1], p2, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(v_batch[0], v1, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(v_batch[1], v2, atol=1e-5, rtol=1e-5)

    def test_padding_does_not_leak(self):
        """Padding tokens should not change the result for the non-padded sample.

        Compare a bundle with N=8 run alone vs. collated with a longer N=12
        bundle (which forces padding on the N=8 sample).
        """
        torch.manual_seed(42)
        model = CVPN()
        model.eval()

        obs_short = _make_dummy_bundle(8, phase="move")
        obs_long = _make_dummy_bundle(12, phase="move")

        # Run the short bundle alone
        with torch.no_grad():
            p_alone, v_alone = model(obs_short)

        # Collate with the longer bundle (short gets padded to N=12)
        batched = collate_obs_bundles([obs_short, obs_long])
        with torch.no_grad():
            p_batch, v_batch = model(batched)

        # The short bundle's results should be the same
        torch.testing.assert_close(p_batch[0], p_alone, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(v_batch[0], v_alone, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Move attention pool
# ---------------------------------------------------------------------------


class TestMoveAttentionPool:
    """The move pool sub-module produces correct shapes."""

    def test_pool_shape(self):
        from cvpn import MoveAttentionPool
        pool = MoveAttentionPool(d_move=32)
        x = torch.randn(10, 4, 32)  # 10 entities, 4 moves, d_move=32
        out = pool(x)
        assert out.shape == (10, 32)

    def test_pool_batched(self):
        from cvpn import MoveAttentionPool
        pool = MoveAttentionPool(d_move=16)
        x = torch.randn(5, 4, 16)  # 5 entities
        out = pool(x)
        assert out.shape == (5, 16)

    def test_zero_moves_produce_finite_output(self):
        """All-zero move IDs (padding) should still produce finite output."""
        from cvpn import MoveAttentionPool
        pool = MoveAttentionPool(d_move=32)
        x = torch.zeros(3, 4, 32)  # zero embeddings (from padding_idx=0)
        out = pool(x)
        assert out.shape == (3, 32)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Custom config
# ---------------------------------------------------------------------------


class TestCustomConfig:
    """CVPN works with non-default hyperparameters."""

    def test_small_model(self):
        cfg = CVPNConfig(d_model=32, n_heads=2, n_layers=1, d_species=8, d_ability=8, d_item=8, d_move=8)
        model = CVPN(cfg)
        model.eval()
        obs = _make_dummy_bundle(8)
        policy, value = model(obs)
        assert policy.shape == (A,)
        assert value.shape == ()

    def test_large_model(self):
        cfg = CVPNConfig(d_model=256, n_heads=8, n_layers=6, d_species=64, d_ability=32, d_item=32, d_move=64)
        model = CVPN(cfg)
        model.eval()
        obs = _make_dummy_bundle(12, phase="teamPreview")
        policy, value = model(obs)
        assert policy.shape == (A,)
        assert value.shape == ()


# ---------------------------------------------------------------------------
# Real-fixture end-to-end smoke test
# ---------------------------------------------------------------------------


class TestRealFixtureSmoke:
    """Run the full pipeline: real snapshot → encoder → CVPN."""

    def test_team_preview_through_cvpn(self, default_cvpn, tp_obs):
        policy, value = default_cvpn(tp_obs)
        assert policy.shape == (A,)
        assert -1.0 <= value.item() <= 1.0
        # At least some legal actions
        assert (policy > float("-inf")).any()

    def test_move_phase_through_cvpn(self, default_cvpn, move_obs):
        policy, value = default_cvpn(move_obs)
        assert policy.shape == (A,)
        assert -1.0 <= value.item() <= 1.0
        assert (policy > float("-inf")).any()


# ---------------------------------------------------------------------------
# _assemble_policy edge cases
# ---------------------------------------------------------------------------


class TestAssemblePolicyEdgeCases:
    """Direct and indirect tests for _assemble_policy boundary behaviour."""

    def test_all_illegal_mask_produces_all_neg_inf(self):
        """When action_mask is all-False, every logit must be -inf (no NaN)."""
        model = CVPN()
        model.eval()
        obs = _make_dummy_bundle(8, phase="move")
        # Zero out the entire action mask
        obs["action_mask"] = torch.zeros(A, dtype=torch.bool)
        with torch.no_grad():
            policy, value = model(obs)
        assert (policy == float("-inf")).all(), "All logits should be -inf when no legal actions"
        assert torch.isfinite(value), "Value should remain finite even with no legal actions"

    def test_region_boundary_team_preview(self):
        """Team-preview head writes exactly to [TEAM_PREVIEW_OFFSET, TEAM_PREVIEW_OFFSET + TEAM_PREVIEW_COUNT)."""
        model = CVPN()
        model.eval()
        # Build a mask that is True only in the TP region
        action_mask = torch.zeros(A, dtype=torch.bool)
        action_mask[TEAM_PREVIEW_OFFSET:TEAM_PREVIEW_OFFSET + TEAM_PREVIEW_COUNT] = True
        obs = _make_dummy_bundle(12, phase="teamPreview")
        obs["action_mask"] = action_mask
        with torch.no_grad():
            policy, _ = model(obs)
        # TP region should all be finite (head wrote real logits, mask kept them)
        tp_slice = policy[TEAM_PREVIEW_OFFSET:TEAM_PREVIEW_OFFSET + TEAM_PREVIEW_COUNT]
        assert torch.isfinite(tp_slice).all(), "All TP logits should be finite"
        # Move region should be all -inf (head wrote logits, but mask zeroed them)
        move_slice = policy[MOVE_PHASE_OFFSET:MOVE_PHASE_OFFSET + MOVE_PHASE_COUNT]
        assert (move_slice == float("-inf")).all(), "Move region should be all -inf in TP phase"

    def test_region_boundary_move_phase(self):
        """Move-phase head writes exactly to [MOVE_PHASE_OFFSET, MOVE_PHASE_OFFSET + MOVE_PHASE_COUNT)."""
        model = CVPN()
        model.eval()
        # Build a mask that is True only in the move region
        action_mask = torch.zeros(A, dtype=torch.bool)
        action_mask[MOVE_PHASE_OFFSET:MOVE_PHASE_OFFSET + MOVE_PHASE_COUNT] = True
        obs = _make_dummy_bundle(12, phase="move")
        obs["action_mask"] = action_mask
        with torch.no_grad():
            policy, _ = model(obs)
        # Move region should all be finite
        move_slice = policy[MOVE_PHASE_OFFSET:MOVE_PHASE_OFFSET + MOVE_PHASE_COUNT]
        assert torch.isfinite(move_slice).all(), "All move logits should be finite"
        # TP region should be all -inf
        tp_slice = policy[TEAM_PREVIEW_OFFSET:TEAM_PREVIEW_OFFSET + TEAM_PREVIEW_COUNT]
        assert (tp_slice == float("-inf")).all(), "TP region should be all -inf in move phase"

    def test_no_gap_between_regions(self):
        """The two regions together span exactly [0, A) with no unaddressed indices."""
        assert TEAM_PREVIEW_OFFSET == 0, "TP region should start at 0"
        assert MOVE_PHASE_OFFSET == TEAM_PREVIEW_COUNT, "Move region starts right after TP"
        assert MOVE_PHASE_OFFSET + MOVE_PHASE_COUNT == A, "Move region ends at A"

    def test_softmax_on_all_neg_inf_produces_nan_not_crash(self):
        """Softmax on all-inf logits produces NaN (expected) — confirm no crash."""
        model = CVPN()
        model.eval()
        obs = _make_dummy_bundle(8, phase="move")
        obs["action_mask"] = torch.zeros(A, dtype=torch.bool)
        with torch.no_grad():
            policy, _ = model(obs)
        # Softmax on all -inf is degenerate but should not raise
        probs = torch.softmax(policy, dim=-1)
        assert probs.shape == (A,)


# ---------------------------------------------------------------------------
# MoveAttentionPool extended tests
# ---------------------------------------------------------------------------


class TestMoveAttentionPoolExtended:
    """Additional coverage for attention pool: 4D input, attention weights, gradients."""

    def test_pool_4d_input(self):
        """The actual CVPN feeds [B, N, 4, d_move] — verify shape with leading batch dim."""
        from cvpn import MoveAttentionPool

        pool = MoveAttentionPool(d_move=32)
        # Simulate batched entity moves: 2 samples, 6 entities each, 4 moves
        x = torch.randn(2, 6, 4, 32)
        out = pool(x)
        assert out.shape == (2, 6, 32), "Should collapse the 4-move dim, preserving [B, N]"

    def test_attention_weights_sum_to_one(self):
        """Internal softmax attention weights must sum to 1 along the key dim."""
        from cvpn import MoveAttentionPool

        pool = MoveAttentionPool(d_move=16)
        pool.eval()
        x = torch.randn(5, 4, 16)

        # Manually compute attention weights to verify
        k = pool.k_proj(x)
        q = pool.query.expand(5, 1, 16)
        scores = torch.matmul(q, k.transpose(-2, -1)) * pool._scale
        attn = torch.softmax(scores, dim=-1)  # [5, 1, 4]
        # Each entity's attention weights across 4 moves should sum to 1
        attn_sums = attn.squeeze(-2).sum(dim=-1)  # [5]
        assert torch.allclose(attn_sums, torch.ones(5), atol=1e-6)

    def test_attention_weights_with_extreme_inputs(self):
        """Verify softmax normalisation holds with large/small magnitude inputs."""
        from cvpn import MoveAttentionPool

        pool = MoveAttentionPool(d_move=16)
        pool.eval()

        # Large magnitude inputs
        x_large = torch.randn(3, 4, 16) * 1000.0
        out_large = pool(x_large)
        assert torch.isfinite(out_large).all(), "Output should be finite with large inputs"

        # Very small inputs
        x_small = torch.randn(3, 4, 16) * 1e-8
        out_small = pool(x_small)
        assert torch.isfinite(out_small).all(), "Output should be finite with small inputs"

    def test_pool_gradient_flow_isolation(self):
        """query, k_proj.weight, and v_proj.weight all receive gradients."""
        from cvpn import MoveAttentionPool

        pool = MoveAttentionPool(d_move=16)
        x = torch.randn(4, 4, 16, requires_grad=True)
        out = pool(x)
        loss = out.sum()
        loss.backward()

        assert pool.query.grad is not None, "query should receive gradient"
        assert pool.query.grad.abs().sum() > 0, "query gradient should be non-zero"
        assert pool.k_proj.weight.grad is not None, "k_proj should receive gradient"
        assert pool.v_proj.weight.grad is not None, "v_proj should receive gradient"


# ---------------------------------------------------------------------------
# Numerical stability
# ---------------------------------------------------------------------------


class TestNumericalStability:
    """CVPN outputs remain finite under extreme input conditions."""

    def test_large_entity_features(self):
        """Very large entity feature values should not produce NaN/inf."""
        model = CVPN()
        model.eval()
        obs = _make_dummy_bundle(8, phase="move")
        # Amplify entity features to extreme magnitudes
        obs["entities"] = obs["entities"] * 1e4
        with torch.no_grad():
            policy, value = model(obs)
        mask = obs["action_mask"]
        assert torch.isfinite(policy[mask]).all(), "Legal logits should be finite with large inputs"
        assert torch.isfinite(value), "Value should be finite with large inputs"

    def test_near_zero_entity_features(self):
        """Near-zero entity features should produce finite outputs."""
        model = CVPN()
        model.eval()
        obs = _make_dummy_bundle(8, phase="move")
        obs["entities"] = obs["entities"] * 1e-8
        with torch.no_grad():
            policy, value = model(obs)
        mask = obs["action_mask"]
        assert torch.isfinite(policy[mask]).all(), "Legal logits should be finite with tiny inputs"
        assert torch.isfinite(value), "Value should be finite with tiny inputs"


# ---------------------------------------------------------------------------
# Global token count invariant
# ---------------------------------------------------------------------------


class TestGlobalTokenCount:
    """_N_GLOBAL_TOKENS matches the actual assembled sequence."""

    def test_n_global_tokens_value(self):
        """The constant should be 5: CLS, field, side_p1, side_p2, meta."""
        assert _N_GLOBAL_TOKENS == 5

    def test_sequence_length_matches_n_plus_entities(self):
        """Assembled sequence should have exactly _N_GLOBAL_TOKENS + N entity tokens."""
        model = CVPN()
        model.eval()
        n_entities = 10
        obs = _make_dummy_bundle(n_entities, phase="move")
        # Unsqueeze to add batch dim (mimicking the forward's internal logic)
        obs_batched = obs.unsqueeze(0)
        # Reproduce the forward's token assembly to check seq length
        species_emb = model.species_embed(obs_batched["ids", "species"])
        ability_emb = model.ability_embed(obs_batched["ids", "ability"])
        item_emb = model.item_embed(obs_batched["ids", "item"])
        move_emb_raw = model.move_embed(obs_batched["ids", "moves"])
        move_summary = model.move_pool(move_emb_raw)
        entities = obs_batched["entities"]
        entity_input = torch.cat([entities, species_emb, ability_emb, item_emb, move_summary], dim=-1)
        entity_tokens = model.entity_proj(entity_input)

        field_tok = model.field_proj(obs_batched["field"]).unsqueeze(1)
        side_p1 = model.side_proj(obs_batched["sides"][:, 0]).unsqueeze(1)
        side_p2 = model.side_proj(obs_batched["sides"][:, 1]).unsqueeze(1)
        meta_tok = model.meta_proj(obs_batched["scalars"]).unsqueeze(1)
        cls = model.cls_token.unsqueeze(0).unsqueeze(0).expand(1, 1, -1)
        seq = torch.cat([cls, field_tok, side_p1, side_p2, meta_tok, entity_tokens], dim=1)

        expected_seq_len = _N_GLOBAL_TOKENS + n_entities
        assert seq.shape[1] == expected_seq_len, (
            f"Sequence length {seq.shape[1]} != {_N_GLOBAL_TOKENS} + {n_entities}"
        )


# ---------------------------------------------------------------------------
# Dropout train/eval behaviour
# ---------------------------------------------------------------------------


class TestDropoutBehaviour:
    """Dropout must be active in train mode and inactive in eval mode."""

    def test_eval_mode_deterministic(self):
        """Two forward passes in eval mode produce identical outputs."""
        torch.manual_seed(99)
        model = CVPN()
        model.eval()
        obs = _make_dummy_bundle(8, phase="move")
        with torch.no_grad():
            p1, v1 = model(obs)
            p2, v2 = model(obs)
        torch.testing.assert_close(p1, p2)
        torch.testing.assert_close(v1, v2)

    def test_train_mode_stochastic(self):
        """Two forward passes in train mode should differ due to dropout.

        With dropout=0.1 and a reasonably sized model, the probability of
        identical outputs is vanishingly small.
        """
        torch.manual_seed(100)
        model = CVPN()
        model.train()
        obs = _make_dummy_bundle(12, phase="teamPreview")
        p1, _v1 = model(obs)
        p2, _v2 = model(obs)
        # At least the policy logits should differ due to stochastic dropout
        assert not torch.allclose(p1, p2, atol=1e-7), (
            "Train-mode outputs should differ due to dropout"
        )


# ---------------------------------------------------------------------------
# State dict save/load round-trip
# ---------------------------------------------------------------------------


class TestStateDictRoundTrip:
    """CVPN checkpoint serialization preserves behaviour."""

    def test_save_load_produces_identical_output(self):
        """Save state_dict, load into fresh model, assert identical forward output."""
        torch.manual_seed(42)
        model = CVPN()
        model.eval()
        obs = _make_dummy_bundle(10, phase="move")

        with torch.no_grad():
            p_orig, v_orig = model(obs)

        # Save and reload
        state = model.state_dict()
        model2 = CVPN()
        model2.load_state_dict(state)
        model2.eval()

        with torch.no_grad():
            p_loaded, v_loaded = model2(obs)

        torch.testing.assert_close(p_orig, p_loaded)
        torch.testing.assert_close(v_orig, v_loaded)

    def test_save_load_custom_config(self):
        """Round-trip also works with non-default config."""
        cfg = CVPNConfig(d_model=64, n_heads=2, n_layers=2, d_species=16, d_ability=8, d_item=8, d_move=16)
        torch.manual_seed(7)
        model = CVPN(cfg)
        model.eval()
        obs = _make_dummy_bundle(8, phase="teamPreview")

        with torch.no_grad():
            p_orig, v_orig = model(obs)

        state = model.state_dict()
        model2 = CVPN(cfg)
        model2.load_state_dict(state)
        model2.eval()

        with torch.no_grad():
            p_loaded, v_loaded = model2(obs)

        torch.testing.assert_close(p_orig, p_loaded)
        torch.testing.assert_close(v_orig, v_loaded)


# ---------------------------------------------------------------------------
# Config propagation to layer dimensions
# ---------------------------------------------------------------------------


class TestConfigPropagation:
    """Custom CVPNConfig values are wired into actual network layer dimensions."""

    def test_custom_d_model_propagates(self):
        cfg = CVPNConfig(d_model=64, n_heads=2, n_layers=2, d_species=16, d_ability=8, d_item=8, d_move=16)
        model = CVPN(cfg)
        # Backbone layers should use d_model=64
        assert model.cls_token.shape == (64,)
        assert model.value_head.in_features == 64
        assert model.value_head.out_features == 1
        assert model.tp_head.in_features == 64
        assert model.mp_head.in_features == 64

    def test_head_out_features_match_action_space(self):
        model = CVPN()
        assert model.tp_head.out_features == TEAM_PREVIEW_COUNT
        assert model.mp_head.out_features == MOVE_PHASE_COUNT

    def test_embedding_table_sizes_match_config(self):
        cfg = CVPNConfig()
        model = CVPN(cfg)
        assert model.species_embed.num_embeddings == cfg.n_species
        assert model.ability_embed.num_embeddings == cfg.n_abilities
        assert model.item_embed.num_embeddings == cfg.n_items
        assert model.move_embed.num_embeddings == cfg.n_moves

    def test_embedding_dims_match_config(self):
        cfg = CVPNConfig(d_species=48, d_ability=24, d_item=24, d_move=48)
        model = CVPN(cfg)
        assert model.species_embed.embedding_dim == 48
        assert model.ability_embed.embedding_dim == 24
        assert model.item_embed.embedding_dim == 24
        assert model.move_embed.embedding_dim == 48

    def test_entity_proj_input_dim(self):
        """Entity projection input = entity_feature_dim + d_species + d_ability + d_item + d_move."""
        cfg = CVPNConfig(d_species=16, d_ability=8, d_item=8, d_move=16)
        model = CVPN(cfg)
        expected_in = cfg.entity_feature_dim + 16 + 8 + 8 + 16
        assert model.entity_proj.in_features == expected_in


# ---------------------------------------------------------------------------
# Adversarial / boundary input testing
# ---------------------------------------------------------------------------


def _make_zero_bundle(n_tokens: int, phase: str = "move") -> ObsBundle:
    """Bundle with all-zero entity features and all-zero IDs (padding_idx=0)."""
    obs = _make_dummy_bundle(n_tokens, phase)
    obs["entities"] = torch.zeros(n_tokens, ENTITY_FEATURE_DIM)
    obs["ids", "species"] = torch.zeros(n_tokens, dtype=torch.long)
    obs["ids", "ability"] = torch.zeros(n_tokens, dtype=torch.long)
    obs["ids", "item"] = torch.zeros(n_tokens, dtype=torch.long)
    obs["ids", "moves"] = torch.zeros(n_tokens, 4, dtype=torch.long)
    return obs


class TestAdversarialInputs:
    """CVPN must produce finite, correctly shaped output under adversarial conditions."""

    def test_all_zero_features_and_ids(self):
        """Maximally uninformative input: zeros everywhere, all embeddings hit padding_idx=0."""
        model = CVPN()
        model.eval()
        obs = _make_zero_bundle(8, phase="move")
        with torch.no_grad():
            policy, value = model(obs)
        mask = obs["action_mask"]
        assert policy.shape == (A,)
        assert torch.isfinite(policy[mask]).all(), "Legal logits must be finite with all-zero input"
        assert torch.isfinite(value), "Value must be finite with all-zero input"
        assert -1.0 <= value.item() <= 1.0

    def test_single_entity_n1(self):
        """N=1 is the minimum entity count — 6-token sequence (5 global + 1 entity)."""
        model = CVPN()
        model.eval()
        obs = _make_dummy_bundle(1, phase="move")
        with torch.no_grad():
            policy, value = model(obs)
        assert policy.shape == (A,)
        assert torch.isfinite(value), "Value must be finite with N=1"
        assert -1.0 <= value.item() <= 1.0

    def test_max_length_n24(self):
        """N=24 is the upper bound (12 per side + Phase 4 candidates)."""
        model = CVPN()
        model.eval()
        obs = _make_dummy_bundle(24, phase="move")
        with torch.no_grad():
            policy, value = model(obs)
        assert policy.shape == (A,)
        assert torch.isfinite(value), "Value must be finite with N=24"
        assert -1.0 <= value.item() <= 1.0

    def test_nearly_all_padded_one_real_token(self):
        """N=12 entities but only 1 is real — attention over 5 global + 1 entity."""
        model = CVPN()
        model.eval()
        obs = _make_dummy_bundle(12, phase="move")
        # Only the first entity token is real; the rest are padding
        pmask = torch.zeros(12, dtype=torch.bool)
        pmask[0] = True
        obs["padding_mask"] = pmask
        with torch.no_grad():
            policy, value = model(obs)
        mask = obs["action_mask"]
        assert policy.shape == (A,)
        assert torch.isfinite(policy[mask]).all(), "Legal logits must be finite with 11/12 padded"
        assert torch.isfinite(value), "Value must be finite with nearly all padding"

    def test_huge_entity_magnitudes_1e6(self):
        """Entity features at 1e6 magnitude — LayerNorm should absorb this."""
        model = CVPN()
        model.eval()
        obs = _make_dummy_bundle(8, phase="move")
        obs["entities"] = obs["entities"] * 1e6
        with torch.no_grad():
            policy, value = model(obs)
        mask = obs["action_mask"]
        assert torch.isfinite(policy[mask]).all(), "Legal logits must be finite with 1e6 inputs"
        assert torch.isfinite(value), "Value must be finite with 1e6 inputs"

    def test_ids_at_vocab_boundary(self):
        """All IDs set to max valid index — catches off-by-one in embedding tables."""
        from vocab import ABILITY_VOCAB, ITEM_VOCAB, MOVE_VOCAB, SPECIES_VOCAB

        model = CVPN()
        model.eval()
        n = 8
        obs = _make_dummy_bundle(n, phase="move")
        # Set every ID to the last valid index (size - 1, since size includes the UNK slot)
        obs["ids", "species"] = torch.full((n,), SPECIES_VOCAB.size - 1, dtype=torch.long)
        obs["ids", "ability"] = torch.full((n,), ABILITY_VOCAB.size - 1, dtype=torch.long)
        obs["ids", "item"] = torch.full((n,), ITEM_VOCAB.size - 1, dtype=torch.long)
        obs["ids", "moves"] = torch.full((n, 4), MOVE_VOCAB.size - 1, dtype=torch.long)
        with torch.no_grad():
            policy, value = model(obs)
        mask = obs["action_mask"]
        assert policy.shape == (A,)
        assert torch.isfinite(policy[mask]).all(), "Legal logits must be finite at vocab boundary"
        assert torch.isfinite(value), "Value must be finite at vocab boundary"

    def test_all_zero_with_team_preview_phase(self):
        """All-zero input in team preview phase (different head exercised)."""
        model = CVPN()
        model.eval()
        obs = _make_zero_bundle(12, phase="teamPreview")
        with torch.no_grad():
            policy, value = model(obs)
        mask = obs["action_mask"]
        assert policy.shape == (A,)
        assert torch.isfinite(policy[mask]).all()
        assert torch.isfinite(value)


# ---------------------------------------------------------------------------
# Direct _assemble_policy unit tests
# ---------------------------------------------------------------------------


class TestAssemblePolicyDirect:
    """Call _assemble_policy directly with known CLS vectors to isolate slicing logic."""

    def test_region_values_match_head_outputs(self):
        """With an all-True mask, each region must contain exactly the head's raw output."""
        model = CVPN()
        model.eval()
        cls_out = torch.ones(1, model.config.d_model)
        all_legal = torch.ones(1, A, dtype=torch.bool)

        with torch.no_grad():
            logits = model._assemble_policy(cls_out, all_legal)
            expected_tp = model.tp_head(cls_out)  # [1, TEAM_PREVIEW_COUNT]
            expected_mp = model.mp_head(cls_out)  # [1, MOVE_PHASE_COUNT]

        tp_slice = logits[:, TEAM_PREVIEW_OFFSET:TEAM_PREVIEW_OFFSET + TEAM_PREVIEW_COUNT]
        mp_slice = logits[:, MOVE_PHASE_OFFSET:MOVE_PHASE_OFFSET + MOVE_PHASE_COUNT]
        torch.testing.assert_close(tp_slice, expected_tp)
        torch.testing.assert_close(mp_slice, expected_mp)

    def test_off_by_one_region_boundaries(self):
        """Boundary indices between regions must land in the correct region."""
        model = CVPN()
        model.eval()
        cls_out = torch.randn(1, model.config.d_model)
        all_legal = torch.ones(1, A, dtype=torch.bool)

        with torch.no_grad():
            logits = model._assemble_policy(cls_out, all_legal)

        # Last TP index and first MP index are both finite (written by their heads)
        last_tp_idx = TEAM_PREVIEW_OFFSET + TEAM_PREVIEW_COUNT - 1
        first_mp_idx = MOVE_PHASE_OFFSET
        assert torch.isfinite(logits[0, last_tp_idx]), "Last TP index must be finite"
        assert torch.isfinite(logits[0, first_mp_idx]), "First MP index must be finite"

        # Last MP index is the end of the action space
        last_mp_idx = MOVE_PHASE_OFFSET + MOVE_PHASE_COUNT - 1
        assert last_mp_idx == A - 1, "Last move-phase index must be A-1"
        assert torch.isfinite(logits[0, last_mp_idx]), "Last MP index must be finite"

    def test_mask_zeroes_precise_indices(self):
        """A mask with exactly one True per region produces exactly 2 finite logits."""
        model = CVPN()
        model.eval()
        cls_out = torch.randn(1, model.config.d_model)

        # Exactly one legal action per region
        mask = torch.zeros(1, A, dtype=torch.bool)
        tp_legal_idx = TEAM_PREVIEW_OFFSET + 5
        mp_legal_idx = MOVE_PHASE_OFFSET + 10
        mask[0, tp_legal_idx] = True
        mask[0, mp_legal_idx] = True

        with torch.no_grad():
            logits = model._assemble_policy(cls_out, mask)

        finite_mask = torch.isfinite(logits[0])
        assert finite_mask.sum() == 2, f"Expected exactly 2 finite logits, got {finite_mask.sum()}"
        assert finite_mask[tp_legal_idx], "The TP legal index must be finite"
        assert finite_mask[mp_legal_idx], "The MP legal index must be finite"

    def test_batched_per_sample_independence(self):
        """Each sample in a batch gets its own mask applied independently."""
        model = CVPN()
        model.eval()
        batch = 3
        cls_out = torch.randn(batch, model.config.d_model)

        # Different masks per sample: sample 0 has TP only, sample 1 has MP only,
        # sample 2 has both
        masks = torch.zeros(batch, A, dtype=torch.bool)
        masks[0, TEAM_PREVIEW_OFFSET:TEAM_PREVIEW_OFFSET + TEAM_PREVIEW_COUNT] = True
        masks[1, MOVE_PHASE_OFFSET:MOVE_PHASE_OFFSET + MOVE_PHASE_COUNT] = True
        masks[2, :] = True  # all legal

        with torch.no_grad():
            logits = model._assemble_policy(cls_out, masks)

        # Sample 0: entire move region must be -inf
        assert (logits[0, MOVE_PHASE_OFFSET:] == float("-inf")).all()
        # Sample 1: entire TP region must be -inf
        assert (logits[1, :TEAM_PREVIEW_COUNT] == float("-inf")).all()
        # Sample 2: no -inf (all-True mask, both heads write finite values)
        assert torch.isfinite(logits[2]).all()

    def test_distinct_cls_vectors_produce_distinct_logits(self):
        """Two different CLS vectors must produce different head outputs."""
        model = CVPN()
        model.eval()
        all_legal = torch.ones(1, A, dtype=torch.bool)

        with torch.no_grad():
            logits_a = model._assemble_policy(torch.zeros(1, model.config.d_model), all_legal)
            logits_b = model._assemble_policy(torch.ones(1, model.config.d_model), all_legal)

        assert not torch.allclose(logits_a, logits_b), "Different CLS inputs must produce different logits"


# ---------------------------------------------------------------------------
# Token ordering convention verification
# ---------------------------------------------------------------------------


class TestTokenOrdering:
    """Verify the token sequence is [CLS, field, side_p1, side_p2, meta, entities...]."""

    def _capture_backbone_input(self, model: CVPN, obs: ObsBundle) -> torch.Tensor:
        """Run a forward pass and capture the seq tensor fed to the backbone."""
        captured = {}

        def hook(module, args, kwargs):
            # nn.TransformerEncoder receives src as the first positional arg
            captured["seq"] = args[0].detach().clone()

        handle = model.backbone.register_forward_pre_hook(hook, with_kwargs=True)
        try:
            with torch.no_grad():
                model(obs)
        finally:
            handle.remove()
        return captured["seq"]

    def _compute_expected_tokens(self, model: CVPN, obs: ObsBundle) -> dict[str, torch.Tensor]:
        """Independently compute what each token position should contain."""
        # Add batch dim for unbatched obs
        unbatched = obs.batch_size == torch.Size([])
        if unbatched:
            obs = obs.unsqueeze(0)

        batch_size = obs.batch_size[0]
        cfg = model.config

        with torch.no_grad():
            # Embeddings
            species_emb = model.species_embed(obs["ids", "species"])
            ability_emb = model.ability_embed(obs["ids", "ability"])
            item_emb = model.item_embed(obs["ids", "item"])
            move_emb_raw = model.move_embed(obs["ids", "moves"])
            move_summary = model.move_pool(move_emb_raw)

            entity_input = torch.cat(
                [obs["entities"], species_emb, ability_emb, item_emb, move_summary], dim=-1
            )
            entity_tokens = model.entity_proj(entity_input)  # [B, N, d_model]

            cls = model.cls_token.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
            field_tok = model.field_proj(obs["field"]).unsqueeze(1)
            side_p1 = model.side_proj(obs["sides"][:, 0]).unsqueeze(1)
            side_p2 = model.side_proj(obs["sides"][:, 1]).unsqueeze(1)
            meta_tok = model.meta_proj(obs["scalars"]).unsqueeze(1)

        return {
            "cls": cls[:, 0],            # [B, d_model]
            "field": field_tok[:, 0],     # [B, d_model]
            "side_p1": side_p1[:, 0],     # [B, d_model]
            "side_p2": side_p2[:, 0],     # [B, d_model]
            "meta": meta_tok[:, 0],       # [B, d_model]
            "entities": entity_tokens,    # [B, N, d_model]
        }

    def test_token_positions_synthetic(self):
        """Synthetic bundle: each position in the backbone input matches its expected token."""
        model = CVPN()
        model.eval()
        n_entities = 8
        obs = _make_dummy_bundle(n_entities, phase="move")

        seq = self._capture_backbone_input(model, obs)  # [1, 5+N, d_model]
        expected = self._compute_expected_tokens(model, obs)

        torch.testing.assert_close(seq[:, 0], expected["cls"], msg="Position 0 must be CLS")
        torch.testing.assert_close(seq[:, 1], expected["field"], msg="Position 1 must be field")
        torch.testing.assert_close(seq[:, 2], expected["side_p1"], msg="Position 2 must be side_p1")
        torch.testing.assert_close(seq[:, 3], expected["side_p2"], msg="Position 3 must be side_p2")
        torch.testing.assert_close(seq[:, 4], expected["meta"], msg="Position 4 must be meta")
        torch.testing.assert_close(
            seq[:, 5:], expected["entities"], msg="Positions 5+ must be entity tokens"
        )

    def test_token_positions_real_fixture(self, tp_obs):
        """Real-fixture team-preview bundle: same ordering convention holds."""
        model = CVPN()
        model.eval()

        seq = self._capture_backbone_input(model, tp_obs)
        expected = self._compute_expected_tokens(model, tp_obs)

        torch.testing.assert_close(seq[:, 0], expected["cls"], msg="Position 0 must be CLS")
        torch.testing.assert_close(seq[:, 1], expected["field"], msg="Position 1 must be field")
        torch.testing.assert_close(seq[:, 2], expected["side_p1"], msg="Position 2 must be side_p1")
        torch.testing.assert_close(seq[:, 3], expected["side_p2"], msg="Position 3 must be side_p2")
        torch.testing.assert_close(seq[:, 4], expected["meta"], msg="Position 4 must be meta")
        torch.testing.assert_close(
            seq[:, 5:], expected["entities"], msg="Positions 5+ must be entity tokens"
        )

    def test_cls_is_learned_parameter_not_input_derived(self):
        """Position 0 must be the raw cls_token parameter, not derived from any input."""
        model = CVPN()
        model.eval()

        # Two different inputs must produce the same CLS value at position 0
        obs_a = _make_dummy_bundle(6, phase="move")
        obs_b = _make_dummy_bundle(10, phase="teamPreview")

        seq_a = self._capture_backbone_input(model, obs_a)
        seq_b = self._capture_backbone_input(model, obs_b)

        torch.testing.assert_close(
            seq_a[:, 0], seq_b[:, 0],
            msg="CLS token at position 0 must be identical regardless of input"
        )
        # And it must equal the raw parameter
        torch.testing.assert_close(
            seq_a[0, 0], model.cls_token,
            msg="CLS position must equal the learned nn.Parameter"
        )

    def test_sides_not_swapped(self):
        """Position 2 is side_p1, position 3 is side_p2 — not the reverse.

        We verify by constructing an obs where sides[0] != sides[1] and checking
        that the projections land in the correct positions.
        """
        model = CVPN()
        model.eval()
        obs = _make_dummy_bundle(8, phase="move")
        # Make the two sides distinguishable
        obs["sides"][0] = torch.ones(SIDE_FEATURE_DIM) * 10.0
        obs["sides"][1] = torch.ones(SIDE_FEATURE_DIM) * -10.0

        seq = self._capture_backbone_input(model, obs)

        with torch.no_grad():
            expected_p1 = model.side_proj(obs["sides"][0])
            expected_p2 = model.side_proj(obs["sides"][1])

        torch.testing.assert_close(seq[0, 2], expected_p1, msg="Position 2 must be side_p1")
        torch.testing.assert_close(seq[0, 3], expected_p2, msg="Position 3 must be side_p2")
        # Confirm they are actually different (the test is meaningful)
        assert not torch.allclose(seq[0, 2], seq[0, 3]), "Sides must be distinguishable"

    def test_key_padding_mask_alignment(self):
        """The 5 global prefix positions are always attended; entity positions mirror obs padding_mask."""
        model = CVPN()
        model.eval()
        n = 10
        obs = _make_dummy_bundle(n, phase="move")
        # Make a non-trivial padding mask: first 6 real, last 4 padded
        pmask = torch.zeros(n, dtype=torch.bool)
        pmask[:6] = True
        obs["padding_mask"] = pmask

        # Capture the key_padding_mask sent to the backbone
        captured = {}

        def hook(module, args, kwargs):
            captured["kpm"] = kwargs.get("src_key_padding_mask")
            if captured["kpm"] is None and len(args) > 1:
                captured["kpm"] = args[1]

        handle = model.backbone.register_forward_pre_hook(hook, with_kwargs=True)
        try:
            with torch.no_grad():
                model(obs)
        finally:
            handle.remove()

        kpm = captured["kpm"]  # [1, 5+N] — True = IGNORE in torch convention
        assert kpm is not None, "Key padding mask must be passed to backbone"

        # Global prefix (first 5): never ignored → all False
        assert not kpm[0, :_N_GLOBAL_TOKENS].any(), "Global prefix tokens must never be ignored"

        # Entity portion: inverted obs padding_mask
        entity_kpm = kpm[0, _N_GLOBAL_TOKENS:]
        expected_entity_ignore = ~pmask
        torch.testing.assert_close(
            entity_kpm, expected_entity_ignore,
            msg="Entity key-padding mask must be the inverse of obs padding_mask"
        )
