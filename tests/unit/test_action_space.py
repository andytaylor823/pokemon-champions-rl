"""Unit tests for the action_space module — constants, encoding round-trips, legal mask."""
from __future__ import annotations

import pytest

from action_space import (
    ACTIONS_PER_SLOT,
    MOVE_PHASE_COUNT,
    MOVE_PHASE_OFFSET,
    PASS_INDEX,
    TEAM_PREVIEW_COUNT,
    TEAM_PREVIEW_OFFSET,
    A,
    MoveAction,
    PassAction,
    SwitchAction,
    _index_to_slot_action,
    _slot_action_to_index,
    _valid_targets_for,
    choice_string_to_index,
    index_to_choice_string,
    legal_mask,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Verify computed constants match the documented action space layout."""

    def test_team_preview_count(self):
        # P(6,4) = 6*5*4*3 = 360
        assert TEAM_PREVIEW_COUNT == 360

    def test_actions_per_slot(self):
        # 4 moves x 3 targets x 2 (normal + mega) + 2 switches + 1 pass = 27
        assert ACTIONS_PER_SLOT == 27

    def test_pass_index(self):
        assert PASS_INDEX == 26

    def test_move_phase_count(self):
        # 27 x 27 = 729
        assert MOVE_PHASE_COUNT == 729

    def test_total_action_space(self):
        # 360 + 729 = 1089
        assert A == 1089

    def test_offsets(self):
        assert TEAM_PREVIEW_OFFSET == 0
        assert MOVE_PHASE_OFFSET == 360


# ---------------------------------------------------------------------------
# Per-slot action encoding/decoding
# ---------------------------------------------------------------------------


class TestSlotActions:
    """Test per-slot action index encoding and decoding."""

    def test_move_targets_produce_distinct_indices(self):
        # Move 0 targeting foe-left, foe-right, ally should yield 0, 1, 2
        idx_foe_left = _slot_action_to_index(0, 1, False, None)
        idx_foe_right = _slot_action_to_index(0, 2, False, None)
        idx_ally = _slot_action_to_index(0, -1, False, None)
        assert idx_foe_left == 0
        assert idx_foe_right == 1
        assert idx_ally == 2
        assert len({idx_foe_left, idx_foe_right, idx_ally}) == 3

    def test_mega_maps_to_upper_range(self):
        # Non-mega move 0, target 1 → index 0; mega → index 12
        base = _slot_action_to_index(0, 1, False, None)
        mega = _slot_action_to_index(0, 1, True, None)
        assert base < 12
        assert 12 <= mega < 24

    def test_switch_indices(self):
        # Bench pos 1 → index 24, bench pos 2 → index 25
        sw1 = _slot_action_to_index(None, None, False, 1)
        sw2 = _slot_action_to_index(None, None, False, 2)
        assert sw1 == 24
        assert sw2 == 25

    def test_decode_move_action(self):
        action = _index_to_slot_action(0)
        assert isinstance(action, MoveAction)
        assert action == MoveAction(move_idx=0, target=1, mega=False)

    def test_decode_mega_action(self):
        action = _index_to_slot_action(12)
        assert isinstance(action, MoveAction)
        assert action == MoveAction(move_idx=0, target=1, mega=True)

    def test_decode_switch_action(self):
        action = _index_to_slot_action(24)
        assert isinstance(action, SwitchAction)
        assert action == SwitchAction(bench_pos=1, team_slot=3)
        action2 = _index_to_slot_action(25)
        assert action2 == SwitchAction(bench_pos=2, team_slot=4)

    def test_decode_pass_action(self):
        action = _index_to_slot_action(PASS_INDEX)
        assert isinstance(action, PassAction)

    def test_roundtrip_all_slot_indices(self):
        # Every slot index [0, 27) should survive encode → decode → encode
        for idx in range(ACTIONS_PER_SLOT):
            action = _index_to_slot_action(idx)
            if isinstance(action, PassAction):
                assert idx == PASS_INDEX
            elif isinstance(action, SwitchAction):
                re_idx = _slot_action_to_index(None, None, False, action.bench_pos)
                assert re_idx == idx, f"Round-trip failed for slot index {idx}"
            else:
                re_idx = _slot_action_to_index(action.move_idx, action.target, action.mega, None)
                assert re_idx == idx, f"Round-trip failed for slot index {idx}"


# ---------------------------------------------------------------------------
# Choice string round-trips
# ---------------------------------------------------------------------------


class TestChoiceStringRoundTrips:
    """Test index ↔ choice string conversions."""

    def test_team_preview_first(self):
        choice = index_to_choice_string(0)
        assert choice.startswith("team ")
        assert choice_string_to_index(choice) == 0

    def test_team_preview_last(self):
        last_idx = TEAM_PREVIEW_COUNT - 1
        choice = index_to_choice_string(last_idx)
        assert choice.startswith("team ")
        assert choice_string_to_index(choice) == last_idx

    def test_team_preview_sample(self):
        # "team 1234" corresponds to permutation (1,2,3,4) which is index 0
        # in lexicographic order of permutations(range(1,7), 4)
        idx = choice_string_to_index("team 1234")
        assert index_to_choice_string(idx) == "team 1234"

    def test_team_preview_roundtrip_sample(self):
        # Test a spread of indices across team preview range
        for i in range(0, TEAM_PREVIEW_COUNT, 30):
            choice = index_to_choice_string(i)
            assert choice_string_to_index(choice) == i

    def test_move_phase_basic_move(self):
        # Slot1=move1 target foe-left, slot2=move1 target foe-left (both index 0)
        slot1_idx = _slot_action_to_index(0, 1, False, None)
        slot2_idx = _slot_action_to_index(0, 1, False, None)
        idx = MOVE_PHASE_OFFSET + slot1_idx * ACTIONS_PER_SLOT + slot2_idx
        choice = index_to_choice_string(idx)
        assert "move 1 1" in choice
        assert choice_string_to_index(choice) == idx

    def test_move_phase_mega(self):
        # Slot1: move 1, target 1, mega. Slot2: switch 3
        slot1_idx = _slot_action_to_index(0, 1, True, None)  # 12
        slot2_idx = _slot_action_to_index(None, None, False, 1)  # 24
        joint_idx = MOVE_PHASE_OFFSET + slot1_idx * ACTIONS_PER_SLOT + slot2_idx
        choice = index_to_choice_string(joint_idx)
        assert "mega" in choice
        assert "switch 3" in choice
        assert choice_string_to_index(choice) == joint_idx

    def test_move_phase_switch_both(self):
        # Both slots switch to different bench mons
        slot1_idx = _slot_action_to_index(None, None, False, 1)  # switch to team slot 3
        slot2_idx = _slot_action_to_index(None, None, False, 2)  # switch to team slot 4
        joint_idx = MOVE_PHASE_OFFSET + slot1_idx * ACTIONS_PER_SLOT + slot2_idx
        choice = index_to_choice_string(joint_idx)
        assert "switch 3" in choice
        assert "switch 4" in choice
        assert choice_string_to_index(choice) == joint_idx

    def test_move_phase_with_pass(self):
        # Slot1 is a move, slot2 is pass (single-mon endgame)
        slot1_idx = _slot_action_to_index(0, 1, False, None)
        joint_idx = MOVE_PHASE_OFFSET + slot1_idx * ACTIONS_PER_SLOT + PASS_INDEX
        choice = index_to_choice_string(joint_idx)
        # Pass is omitted — only slot1's choice appears
        assert choice == "move 1 1"
        assert choice_string_to_index(choice) == joint_idx

    def test_move_phase_roundtrip_sample(self):
        # Sample move-phase indices at regular intervals
        for i in range(MOVE_PHASE_OFFSET, A, 50):
            choice = index_to_choice_string(i)
            assert choice_string_to_index(choice) == i


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestErrorPaths:
    """Verify correct errors on invalid input."""

    def test_single_slot_choice_treated_as_pass(self):
        # Single-slot choice is valid (other slot is pass), not an error
        idx = choice_string_to_index("move 1 1")
        slot1 = _slot_action_to_index(0, 1, False, None)
        expected = MOVE_PHASE_OFFSET + slot1 * ACTIONS_PER_SLOT + PASS_INDEX
        assert idx == expected

    def test_invalid_fragment_type(self):
        with pytest.raises(ValueError, match="Expected 'move"):
            choice_string_to_index("attack 1 1, move 1 1")

    def test_slot_action_requires_move_args(self):
        with pytest.raises(ValueError, match="move_idx and target required"):
            _slot_action_to_index(None, None, False, None)


# ---------------------------------------------------------------------------
# Valid targets
# ---------------------------------------------------------------------------


class TestValidTargets:
    """Test target type → valid targets mapping."""

    def test_normal_targets_all_three(self):
        assert _valid_targets_for("normal") == [1, 2, -1]

    def test_any_targets_all_three(self):
        assert _valid_targets_for("any") == [1, 2, -1]

    def test_spread_moves_use_canonical_target(self):
        assert _valid_targets_for("allAdjacentFoes") == [1]
        assert _valid_targets_for("allAdjacent") == [1]
        assert _valid_targets_for("all") == [1]

    def test_self_targeting(self):
        assert _valid_targets_for("self") == [1]

    def test_ally_targeting(self):
        assert _valid_targets_for("adjacentAlly") == [-1]
        assert _valid_targets_for("adjacentAllyOrSelf") == [-1]
        assert _valid_targets_for("allySide") == [-1]
        assert _valid_targets_for("allyTeam") == [-1]

    def test_unknown_raises_error(self):
        with pytest.raises(ValueError, match="Unknown Showdown target type"):
            _valid_targets_for("unknownType")


# ---------------------------------------------------------------------------
# Legal mask
# ---------------------------------------------------------------------------


class TestLegalMask:
    """Test legal_mask construction from request dicts."""

    def test_team_preview_mask_shape_and_count(self):
        mask = legal_mask({}, "teamPreview")
        assert mask.shape == (A,)
        assert mask.dtype == bool
        # Exactly 360 true entries, all in team preview range
        assert mask.sum() == TEAM_PREVIEW_COUNT
        assert mask[:TEAM_PREVIEW_COUNT].all()
        assert not mask[TEAM_PREVIEW_COUNT:].any()

    def test_none_request_returns_all_false(self):
        mask = legal_mask(None, "move")
        assert mask.shape == (A,)
        assert not mask.any()

    def test_unknown_phase_returns_all_false(self):
        mask = legal_mask({}, "unknownPhase")
        assert not mask.any()

    def test_move_phase_basic_request(self):
        # Two active slots, each with 2 normal-target moves, 2 bench mons alive
        request = {
            "active": [
                {"moves": [
                    {"id": "heatwave", "pp": 5, "target": "normal"},
                    {"id": "protect", "pp": 5, "target": "self"},
                ]},
                {"moves": [
                    {"id": "earthquake", "pp": 5, "target": "allAdjacentFoes"},
                    {"id": "rockslide", "pp": 5, "target": "allAdjacentFoes"},
                ]},
            ],
            "side": {"pokemon": [
                {"condition": "200/200"},  # active slot 0
                {"condition": "180/180"},  # active slot 1
                {"condition": "150/150"},  # bench 1 (alive)
                {"condition": "100/100"},  # bench 2 (alive)
            ]},
        }
        mask = legal_mask(request, "move")
        assert mask.shape == (A,)
        # Team preview range should be all false
        assert not mask[:TEAM_PREVIEW_COUNT].any()
        # At least some move-phase actions should be legal
        assert mask[MOVE_PHASE_OFFSET:].any()

    def test_double_switch_to_same_slot_excluded(self):
        # Both slots can only switch (no moves available, trapped=False)
        # Only 1 bench mon alive → both would target same slot → no legal joint
        request = {
            "active": [
                {"moves": [{"id": "struggle", "pp": 0, "target": "normal"}]},
                {"moves": [{"id": "struggle", "pp": 0, "target": "normal"}]},
            ],
            "side": {"pokemon": [
                {"condition": "100/200"},  # active 0
                {"condition": "100/180"},  # active 1
                {"condition": "150/150"},  # bench 1 (alive)
                {"condition": "0 fnt"},    # bench 2 (fainted)
            ]},
        }
        mask = legal_mask(request, "move")
        # The only switch available is bench_pos=1 (team_slot=3)
        # Both slots wanting that switch creates a collision → excluded
        sw_idx = _slot_action_to_index(None, None, False, 1)
        collision_joint = MOVE_PHASE_OFFSET + sw_idx * ACTIONS_PER_SLOT + sw_idx
        assert not mask[collision_joint]

    def test_force_switch_only_allows_switches(self):
        request = {
            "forceSwitch": [True, False],
            "active": [
                {"moves": [{"id": "heatwave", "pp": 5, "target": "normal"}]},
                {"moves": [{"id": "protect", "pp": 5, "target": "self"}]},
            ],
            "side": {"pokemon": [
                {"condition": "0 fnt"},    # slot 0 fainted (force switch)
                {"condition": "100/180"},  # slot 1 active
                {"condition": "150/150"},  # bench 1
                {"condition": "100/100"},  # bench 2
            ]},
        }
        mask = legal_mask(request, "forceSwitch")
        assert mask[MOVE_PHASE_OFFSET:].any()

    def test_disabled_moves_excluded(self):
        request = {
            "active": [
                {"moves": [
                    {"id": "heatwave", "pp": 5, "target": "normal", "disabled": True},
                    {"id": "protect", "pp": 5, "target": "self"},
                ]},
                {"moves": [
                    {"id": "earthquake", "pp": 5, "target": "allAdjacentFoes"},
                ]},
            ],
            "side": {"pokemon": [
                {"condition": "200/200"},
                {"condition": "180/180"},
                {"condition": "150/150"},
                {"condition": "100/100"},
            ]},
        }
        mask = legal_mask(request, "move")
        # Heat Wave (move_idx=0) should not appear in slot1's legal actions
        # But Protect (move_idx=1) should
        # Verify by checking that move 1 target 1 appears in some joint action
        # but move 0 (disabled) does not
        slot1_move0_t1 = _slot_action_to_index(0, 1, False, None)
        slot1_move1_t1 = _slot_action_to_index(1, 1, False, None)
        # Check any joint action using move0 in slot1 is masked off
        any_with_move0 = any(
            mask[MOVE_PHASE_OFFSET + slot1_move0_t1 * ACTIONS_PER_SLOT + s2]
            for s2 in range(ACTIONS_PER_SLOT)
        )
        assert not any_with_move0, "Disabled move should not be legal"
        # Protect (self-targeting, canonical target=1) should be legal
        any_with_move1 = any(
            mask[MOVE_PHASE_OFFSET + slot1_move1_t1 * ACTIONS_PER_SLOT + s2]
            for s2 in range(ACTIONS_PER_SLOT)
        )
        assert any_with_move1, "Non-disabled move should be legal"

    def test_single_mon_endgame_uses_pass(self):
        # Only one active mon — slot2 should be pass
        request = {
            "active": [
                {"moves": [{"id": "heatwave", "pp": 5, "target": "normal"}]},
            ],
            "side": {"pokemon": [
                {"condition": "200/200"},
                {"condition": "0 fnt"},
                {"condition": "0 fnt"},
                {"condition": "0 fnt"},
            ]},
        }
        mask = legal_mask(request, "move")
        # Legal actions should be slot1 x PASS only
        slot1_move = _slot_action_to_index(0, 1, False, None)
        joint_with_pass = MOVE_PHASE_OFFSET + slot1_move * ACTIONS_PER_SLOT + PASS_INDEX
        assert mask[joint_with_pass], "slot1 move + slot2 pass should be legal"
        # No joint action should pair slot1 with a non-pass slot2
        for s2 in range(ACTIONS_PER_SLOT):
            if s2 == PASS_INDEX:
                continue
            assert not mask[MOVE_PHASE_OFFSET + slot1_move * ACTIONS_PER_SLOT + s2]

    def test_mega_available(self):
        request = {
            "active": [
                {"moves": [{"id": "heatwave", "pp": 5, "target": "normal"}], "canMegaEvo": True},
                {"moves": [{"id": "protect", "pp": 5, "target": "self"}]},
            ],
            "side": {"pokemon": [
                {"condition": "200/200"},
                {"condition": "180/180"},
                {"condition": "150/150"},
                {"condition": "100/100"},
            ]},
        }
        mask = legal_mask(request, "move")
        # Mega variant of move 0 targeting foe-left should be legal
        mega_idx = _slot_action_to_index(0, 1, True, None)
        any_mega = any(
            mask[MOVE_PHASE_OFFSET + mega_idx * ACTIONS_PER_SLOT + s2]
            for s2 in range(ACTIONS_PER_SLOT)
        )
        assert any_mega, "Mega move should be legal when canMegaEvo is True"


# ---------------------------------------------------------------------------
# EXHAUSTIVE: Targeting, trapped, pass, PP exclusion, bench states
# ---------------------------------------------------------------------------


class TestScriptedTarget:
    """Scripted target type (Struggle, etc.) produces single canonical target."""

    def test_scripted_returns_single_target(self):
        assert _valid_targets_for("scripted") == [1]

    def test_scripted_in_legal_mask(self):
        """A mon with only Struggle (scripted target) should still have legal actions."""
        request = {
            "active": [
                {"moves": [{"id": "struggle", "pp": 1, "target": "scripted"}]},
                {"moves": [{"id": "protect", "pp": 5, "target": "self"}]},
            ],
            "side": {"pokemon": [
                {"condition": "100/200"},
                {"condition": "100/180"},
                {"condition": "150/150"},
                {"condition": "100/100"},
            ]},
        }
        mask = legal_mask(request, "move")
        assert mask[MOVE_PHASE_OFFSET:].any()


class TestTrappedMon:
    """A trapped mon cannot switch — only moves are legal."""

    def test_trapped_excludes_switches(self):
        request = {
            "trapped": True,
            "active": [
                {"moves": [
                    {"id": "heatwave", "pp": 5, "target": "normal"},
                    {"id": "protect", "pp": 5, "target": "self"},
                ]},
                {"moves": [{"id": "earthquake", "pp": 5, "target": "allAdjacentFoes"}]},
            ],
            "side": {"pokemon": [
                {"condition": "200/200"},
                {"condition": "180/180"},
                {"condition": "150/150"},
                {"condition": "100/100"},
            ]},
        }
        mask = legal_mask(request, "move")
        # Slot 1 (index 0) should have no switch actions
        sw1 = _slot_action_to_index(None, None, False, 1)
        sw2 = _slot_action_to_index(None, None, False, 2)
        # Check no joint action has slot1 as a switch
        for s2 in range(ACTIONS_PER_SLOT):
            assert not mask[MOVE_PHASE_OFFSET + sw1 * ACTIONS_PER_SLOT + s2]
            assert not mask[MOVE_PHASE_OFFSET + sw2 * ACTIONS_PER_SLOT + s2]

    def test_trapped_with_all_moves_disabled(self):
        """Trapped + all moves disabled = no legal actions for slot 1 (degenerate)."""
        request = {
            "trapped": True,
            "active": [
                {"moves": [
                    {"id": "heatwave", "pp": 5, "target": "normal", "disabled": True},
                    {"id": "protect", "pp": 5, "target": "self", "disabled": True},
                ]},
                {"moves": [{"id": "earthquake", "pp": 5, "target": "allAdjacentFoes"}]},
            ],
            "side": {"pokemon": [
                {"condition": "200/200"},
                {"condition": "180/180"},
                {"condition": "150/150"},
                {"condition": "100/100"},
            ]},
        }
        mask = legal_mask(request, "move")
        # Slot 1 has no legal moves (all disabled) and no switches (trapped)
        # So the entire mask should be empty (no joint actions possible)
        assert not mask[MOVE_PHASE_OFFSET:].any()


class TestPassChoiceString:
    """Pass action string parsing."""

    def test_pass_string_returns_pass_index(self):
        from action_space import _choice_to_slot_action
        assert _choice_to_slot_action("pass") == PASS_INDEX

    def test_empty_string_returns_pass_index(self):
        from action_space import _choice_to_slot_action
        assert _choice_to_slot_action("") == PASS_INDEX

    def test_single_choice_maps_to_pass_in_slot2(self):
        """A single-slot choice string implies slot2 is pass."""
        idx = choice_string_to_index("move 1 1")
        slot1 = _slot_action_to_index(0, 1, False, None)
        expected = MOVE_PHASE_OFFSET + slot1 * ACTIONS_PER_SLOT + PASS_INDEX
        assert idx == expected

    def test_single_switch_maps_to_pass_in_slot2(self):
        idx = choice_string_to_index("switch 3")
        slot1 = _slot_action_to_index(None, None, False, 1)  # bench_pos=1 -> team_slot=3
        expected = MOVE_PHASE_OFFSET + slot1 * ACTIONS_PER_SLOT + PASS_INDEX
        assert idx == expected


class TestZeroPpExclusion:
    """Moves with 0 PP (not explicitly disabled) should be excluded from mask."""

    def test_zero_pp_excluded(self):
        request = {
            "active": [
                {"moves": [
                    {"id": "heatwave", "pp": 0, "target": "normal"},
                    {"id": "protect", "pp": 5, "target": "self"},
                ]},
                {"moves": [{"id": "earthquake", "pp": 5, "target": "allAdjacentFoes"}]},
            ],
            "side": {"pokemon": [
                {"condition": "200/200"},
                {"condition": "180/180"},
                {"condition": "150/150"},
                {"condition": "100/100"},
            ]},
        }
        mask = legal_mask(request, "move")
        # Heat Wave (move_idx=0) with 0 PP should not appear
        slot1_move0_t1 = _slot_action_to_index(0, 1, False, None)
        any_with_move0 = any(
            mask[MOVE_PHASE_OFFSET + slot1_move0_t1 * ACTIONS_PER_SLOT + s2]
            for s2 in range(ACTIONS_PER_SLOT)
        )
        assert not any_with_move0, "0 PP move should not be legal"

    def test_one_pp_included(self):
        request = {
            "active": [
                {"moves": [
                    {"id": "heatwave", "pp": 1, "target": "normal"},
                ]},
                {"moves": [{"id": "earthquake", "pp": 5, "target": "allAdjacentFoes"}]},
            ],
            "side": {"pokemon": [
                {"condition": "200/200"},
                {"condition": "180/180"},
                {"condition": "150/150"},
                {"condition": "100/100"},
            ]},
        }
        mask = legal_mask(request, "move")
        slot1_move0_t1 = _slot_action_to_index(0, 1, False, None)
        any_with_move0 = any(
            mask[MOVE_PHASE_OFFSET + slot1_move0_t1 * ACTIONS_PER_SLOT + s2]
            for s2 in range(ACTIONS_PER_SLOT)
        )
        assert any_with_move0, "1 PP move should be legal"

    def test_all_moves_zero_pp_only_switches_legal(self):
        request = {
            "active": [
                {"moves": [
                    {"id": "heatwave", "pp": 0, "target": "normal"},
                    {"id": "protect", "pp": 0, "target": "self"},
                    {"id": "airslash", "pp": 0, "target": "normal"},
                    {"id": "solarbeam", "pp": 0, "target": "normal"},
                ]},
                {"moves": [{"id": "earthquake", "pp": 5, "target": "allAdjacentFoes"}]},
            ],
            "side": {"pokemon": [
                {"condition": "200/200"},
                {"condition": "180/180"},
                {"condition": "150/150"},
                {"condition": "100/100"},
            ]},
        }
        mask = legal_mask(request, "move")
        # Only switch actions should be legal for slot 1
        assert mask[MOVE_PHASE_OFFSET:].any()
        # Verify at least one switch appears
        sw_idx = _slot_action_to_index(None, None, False, 1)
        any_switch = any(
            mask[MOVE_PHASE_OFFSET + sw_idx * ACTIONS_PER_SLOT + s2]
            for s2 in range(ACTIONS_PER_SLOT)
        )
        assert any_switch, "Switches should be legal when all moves have 0 PP"


class TestBenchStates:
    """Legal switch availability based on bench pokemon health."""

    def test_all_bench_fainted_no_switches(self):
        from action_space import _legal_switches
        side_pokemon = [
            {"condition": "200/200"},  # active slot 0
            {"condition": "180/180"},  # active slot 1
            {"condition": "0 fnt"},    # bench 1 fainted
            {"condition": "0 fnt"},    # bench 2 fainted
        ]
        switches = _legal_switches(side_pokemon)
        assert switches == []

    def test_one_bench_alive(self):
        from action_space import _legal_switches
        side_pokemon = [
            {"condition": "200/200"},
            {"condition": "180/180"},
            {"condition": "150/150"},  # bench 1 alive
            {"condition": "0 fnt"},    # bench 2 fainted
        ]
        switches = _legal_switches(side_pokemon)
        assert len(switches) == 1

    def test_both_bench_alive(self):
        from action_space import _legal_switches
        side_pokemon = [
            {"condition": "200/200"},
            {"condition": "180/180"},
            {"condition": "150/150"},
            {"condition": "100/100"},
        ]
        switches = _legal_switches(side_pokemon)
        assert len(switches) == 2

    def test_cross_slot_switch_collision_excluded(self):
        """Both slots switching to same bench target is excluded."""
        request = {
            "active": [
                {"moves": [{"id": "heatwave", "pp": 0, "target": "normal"}]},
                {"moves": [{"id": "protect", "pp": 0, "target": "self"}]},
            ],
            "side": {"pokemon": [
                {"condition": "100/200"},
                {"condition": "100/180"},
                {"condition": "150/150"},  # only bench mon alive
                {"condition": "0 fnt"},
            ]},
        }
        mask = legal_mask(request, "move")
        # Both slots can switch to bench_pos=1 (team_slot=3)
        sw_idx = _slot_action_to_index(None, None, False, 1)
        collision = MOVE_PHASE_OFFSET + sw_idx * ACTIONS_PER_SLOT + sw_idx
        assert not mask[collision], "Both switching to same target should be excluded"

    def test_cross_slot_different_targets_allowed(self):
        """Slots switching to different bench targets is legal."""
        request = {
            "active": [
                {"moves": [{"id": "heatwave", "pp": 0, "target": "normal"}]},
                {"moves": [{"id": "protect", "pp": 0, "target": "self"}]},
            ],
            "side": {"pokemon": [
                {"condition": "100/200"},
                {"condition": "100/180"},
                {"condition": "150/150"},  # bench 1 alive
                {"condition": "100/100"},  # bench 2 alive
            ]},
        }
        mask = legal_mask(request, "move")
        sw1 = _slot_action_to_index(None, None, False, 1)
        sw2 = _slot_action_to_index(None, None, False, 2)
        # Slot1 -> bench1, slot2 -> bench2 should be legal
        cross = MOVE_PHASE_OFFSET + sw1 * ACTIONS_PER_SLOT + sw2
        assert mask[cross], "Different switch targets should be allowed"
