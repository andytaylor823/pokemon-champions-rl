"""Integration tests for the GT-CFR search module.

Exercises the full search pipeline with a real SimClient subprocess and a CVPN
with random weights. Validates that search produces well-formed results from
both team preview and move-phase states.
"""
from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from action_space import (
    A,
    ACTIONS_PER_SLOT,
    MOVE_PHASE_OFFSET,
    MoveAction,
    _index_to_slot_action,
    action_to_choice_contextual,
    legal_mask,
)
from cvpn import CVPN
from search import SearchConfig, SearchResult, search
from sim_client import SimClient, SimError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rng_seed(rng: random.Random) -> list[int]:
    """Generate a 4-element PRNG seed list."""
    return [rng.randint(0, 0xFFFF) for _ in range(4)]


def _pick_aggressive(mask: np.ndarray, rng: random.Random) -> int:
    """Pick a legal action biased toward foe-targeting attack moves.

    Scores each legal joint action: +2 per slot using a MoveAction, +1 extra
    per slot targeting a foe (target 1 or 2).  Ties broken randomly.
    """
    legal_indices = np.flatnonzero(mask)
    best_score = -1
    best_actions: list[int] = []

    for idx in legal_indices:
        if idx < MOVE_PHASE_OFFSET:
            score = 0
        else:
            joint_idx = idx - MOVE_PHASE_OFFSET
            s1 = joint_idx // ACTIONS_PER_SLOT
            s2 = joint_idx % ACTIONS_PER_SLOT
            score = 0
            for si in (s1, s2):
                action = _index_to_slot_action(si)
                if isinstance(action, MoveAction):
                    score += 2
                    if action.target in (1, 2):
                        score += 1

        if score > best_score:
            best_score = score
            best_actions = [idx]
        elif score == best_score:
            best_actions.append(idx)

    return rng.choice(best_actions)


# Use a small budget for integration tests to keep them fast
_FAST_CONFIG = SearchConfig(
    k_actions=3,
    max_chance_children=2,
    expansion_budget=4,
    cfr_iters_per_expansion=5,
    c_puct=2.0,
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSearchFromTeamPreview:
    """Run search from the team preview phase."""

    @pytest.mark.slow
    def test_produces_valid_result(self, sim_client: SimClient, team_a: list, team_b: list):
        """Search at team preview returns a well-formed SearchResult."""
        # Set up a battle at team preview
        live_handle, view = sim_client.new_battle(team_a, team_b, seed=[1, 2, 3, 4])
        assert view.phase == "teamPreview"

        # Create CVPN with random weights (no training — just testing shape)
        net = CVPN()
        net.eval()

        # Run search
        result = search(view, sim_client, net, from_handle=live_handle, config=_FAST_CONFIG)

        # Validate result shape
        assert isinstance(result, SearchResult)
        assert isinstance(result.value, float)
        assert -1.0 <= result.value <= 1.0

        # Both sides should have a strategy
        for s in ["p1", "p2"]:
            assert s in result.strategy
            assert s in result.policy_target

            # Strategy probabilities should sum to ~1
            strat = result.strategy[s]
            if strat:
                total_prob = sum(strat.values())
                assert abs(total_prob - 1.0) < 1e-6, f"Strategy for {s} sums to {total_prob}"

            # Policy target should be a valid [A] vector
            pt = result.policy_target[s]
            assert pt.shape == (A,)
            assert pt.sum() > 0  # at least some mass
            assert abs(pt.sum() - 1.0) < 1e-5

            # All actions in the strategy should be legal
            mask = legal_mask(view.legal.get(s), view.phase)
            for action_idx in strat:
                assert mask[action_idx], f"Action {action_idx} in strategy for {s} is not legal"


class TestSearchFromMovePhase:
    """Run search from the move phase (after team preview)."""

    @pytest.mark.slow
    def test_produces_valid_result(self, sim_client: SimClient, team_a: list, team_b: list):
        """Search at the move phase returns a well-formed SearchResult."""
        rng = random.Random(42)

        # Create battle and advance past team preview
        live_handle, view = sim_client.new_battle(team_a, team_b, seed=[1, 2, 3, 4])

        # Open a search session to advance through team preview
        session, root, root_view = sim_client.open_search(from_handle=live_handle)
        try:
            # Choose teams
            res = sim_client.step(root, {"p1": "team 1234", "p2": "team 1234"}, seed=_rng_seed(rng))
            move_view = res.view
            move_handle = res.child

            # We should now be in the move phase
            assert move_view.phase == "move", f"Expected move phase, got {move_view.phase}"
        finally:
            sim_client.close_search(session)

        # Now run search from the move-phase view
        net = CVPN()
        net.eval()

        # We need a handle at the move-phase state to search from
        # Re-create the battle and advance to move phase for the live handle
        live2, _ = sim_client.new_battle(team_a, team_b, seed=[1, 2, 3, 4])
        session2, root2, _ = sim_client.open_search(from_handle=live2)
        res2 = sim_client.step(root2, {"p1": "team 1234", "p2": "team 1234"}, seed=_rng_seed(rng))
        move_handle2 = res2.child
        move_view2 = res2.view
        sim_client.close_search(session2)

        # Use the live handle stepped to move phase
        # Actually, step on the live handle directly to get a handle at move phase
        live3, _ = sim_client.new_battle(team_a, team_b, seed=[1, 2, 3, 4])
        rng2 = random.Random(42)
        step_res = sim_client.step(live3, {"p1": "team 1234", "p2": "team 1234"}, seed=_rng_seed(rng2))
        move_handle_live = step_res.child
        move_view_live = step_res.view
        assert move_view_live.phase == "move", f"Expected move phase, got {move_view_live.phase}"

        result = search(move_view_live, sim_client, net, from_handle=move_handle_live, config=_FAST_CONFIG)

        assert isinstance(result, SearchResult)
        assert -1.0 <= result.value <= 1.0

        # Validate strategy for each acting side
        for s in move_view_live.to_move:
            assert s in result.strategy
            strat = result.strategy[s]
            if strat:
                total_prob = sum(strat.values())
                assert abs(total_prob - 1.0) < 1e-6

                # All actions should be legal
                mask = legal_mask(move_view_live.legal.get(s), move_view_live.phase)
                for action_idx in strat:
                    assert mask[action_idx], f"Illegal action {action_idx} in {s}'s strategy"


class TestSearchFromForceSwitch:
    """Run search from the forceSwitch phase (gap #26).

    Seeds were discovered offline with ``scripts/find_force_switch_seeds.py``
    using an aggressive action-selection heuristic.  Both reach forceSwitch
    deterministically at turn 1 with the current teams.
    """

    # Known-good seeds: both hit forceSwitch at turn 1 with aggressive play
    _KNOWN_SEEDS: list[list[int]] = [[0, 1, 2, 3], [5, 6, 7, 8]]

    @pytest.mark.slow
    @pytest.mark.parametrize("battle_seed", _KNOWN_SEEDS)
    def test_produces_valid_result(self, sim_client: SimClient, team_a: list, team_b: list, battle_seed: list):
        """Search at forceSwitch returns a well-formed SearchResult."""
        rng = random.Random(battle_seed[0])
        max_turns = 10

        # Advance past team preview
        live_handle, _ = sim_client.new_battle(team_a, team_b, seed=battle_seed)
        step_res = sim_client.step(live_handle, {"p1": "team 1234", "p2": "team 1234"}, seed=_rng_seed(rng))
        current_handle = step_res.child
        current_view = step_res.view

        # Play aggressively until forceSwitch (expected at turn 1)
        for turn in range(max_turns):
            if current_view.terminal:
                pytest.fail(f"Battle ended at turn {turn} before forceSwitch — seed {battle_seed} may need replacing")

            if current_view.phase == "forceSwitch":
                break

            masks = {s: legal_mask(current_view.legal.get(s), current_view.phase) for s in current_view.to_move}
            stepped = False
            for _attempt in range(10):
                choices = {}
                for s in current_view.to_move:
                    choices[s] = action_to_choice_contextual(_pick_aggressive(masks[s], rng), None)
                try:
                    step_res = sim_client.step(current_handle, choices, seed=_rng_seed(rng))
                    current_handle = step_res.child
                    current_view = step_res.view
                    stepped = True
                    break
                except SimError:
                    continue
            if not stepped:
                pytest.fail(f"No valid choice found at turn {turn} — legal_mask/engine discrepancy")
        else:
            pytest.fail(f"No forceSwitch within {max_turns} turns — seed {battle_seed} may need replacing")

        # We have a forceSwitch view — run search on it
        net = CVPN()
        net.eval()

        result = search(current_view, sim_client, net, from_handle=current_handle, config=_FAST_CONFIG)

        assert isinstance(result, SearchResult)
        assert -1.0 <= result.value <= 1.0

        for s in current_view.to_move:
            assert s in result.strategy
            strat = result.strategy[s]
            if strat:
                total_prob = sum(strat.values())
                assert abs(total_prob - 1.0) < 1e-6

                mask = legal_mask(current_view.legal.get(s), current_view.phase)
                for action_idx in strat:
                    assert mask[action_idx], f"Illegal action {action_idx} in {s}'s forceSwitch strategy"


class TestSearchDeterminism:
    """Verify that search with the same seed and network produces consistent results."""

    @pytest.mark.slow
    def test_same_net_same_view_consistent_value_range(self, sim_client: SimClient, team_a: list, team_b: list):
        """Running search twice with the same network should produce values in [-1, 1]."""
        live_handle, view = sim_client.new_battle(team_a, team_b, seed=[1, 2, 3, 4])

        # Fix the random seed for reproducibility
        torch.manual_seed(0)
        net = CVPN()
        net.eval()

        # Run search twice
        random.seed(123)
        result1 = search(view, sim_client, net, from_handle=live_handle, config=_FAST_CONFIG)
        random.seed(123)
        result2 = search(view, sim_client, net, from_handle=live_handle, config=_FAST_CONFIG)

        # Both should be well-formed (values may differ slightly due to SimClient handle IDs)
        assert -1.0 <= result1.value <= 1.0
        assert -1.0 <= result2.value <= 1.0

        # Strategy structure should be the same (same top-k actions)
        for s in view.to_move:
            assert set(result1.strategy[s].keys()) == set(result2.strategy[s].keys())
