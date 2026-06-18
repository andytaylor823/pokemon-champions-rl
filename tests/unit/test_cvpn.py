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
