"""Action space module — canonical joint action index for VGC doubles.

Defines a fixed-size flat action space covering both team preview and the move
phase. The CVPN policy head outputs logits of shape [A] (one softmax over the
full joint space). At runtime, an `action_mask` bool tensor marks which indices
are legal.

Layout:
  [0, TEAM_PREVIEW_COUNT)           — team preview actions (P(6,4) = 360 orderings)
  [TEAM_PREVIEW_COUNT, A)           — move-phase joint actions (slot1 x slot2)

Per-slot actions during move phase (ACTIONS_PER_SLOT = 26):
  [0, 12)   — move 1-4 x target {1, 2, -1} (foe-left, foe-right, ally)
  [12, 24)  — move 1-4 x target {1, 2, -1} + mega evolution
  [24, 25)  — switch to bench position 1 (team slot 3)
  [25, 26)  — switch to bench position 2 (team slot 4)

Showdown targeting conventions:
  - target  1 = opponent slot 1 (left foe)
  - target  2 = opponent slot 2 (right foe)
  - target -1 = ally
  For spread/self-targeting moves, only one target is legal (mask handles it).
"""

from __future__ import annotations

from itertools import permutations

import numpy as np

# --- Constants ----------------------------------------------------------------

# Move phase: per-slot action decomposition
NUM_MOVES = 4
TARGETS = (1, 2, -1)  # foe-left, foe-right, ally
NUM_TARGETS = len(TARGETS)
NUM_SWITCHES = 2  # bench positions (team slots 3 and 4 in a bring-4 format)

# Per-slot: 4 moves x 3 targets = 12 base + 12 mega + 2 switches = 26
ACTIONS_PER_SLOT = NUM_MOVES * NUM_TARGETS * 2 + NUM_SWITCHES  # 26

# Team preview: P(6,4) = 6*5*4*3 = 360 orderings
TEAM_SIZE = 6
BRING_COUNT = 4
_TEAM_PERMS = list(permutations(range(1, TEAM_SIZE + 1), BRING_COUNT))
TEAM_PREVIEW_COUNT = len(_TEAM_PERMS)  # 360

# Move phase: joint = slot1 x slot2
MOVE_PHASE_COUNT = ACTIONS_PER_SLOT * ACTIONS_PER_SLOT  # 676

# Total canonical action space
TEAM_PREVIEW_OFFSET = 0
MOVE_PHASE_OFFSET = TEAM_PREVIEW_COUNT  # 360
A = TEAM_PREVIEW_COUNT + MOVE_PHASE_COUNT  # 1036


# --- Team preview helpers -----------------------------------------------------


def _team_perm_to_index(perm: tuple[int, ...]) -> int:
    """Map a team ordering tuple (e.g. (1,3,4,2)) to its canonical index."""
    return _TEAM_PERMS.index(perm)


def _index_to_team_perm(idx: int) -> tuple[int, ...]:
    """Map a team preview index to the ordering tuple."""
    return _TEAM_PERMS[idx]


# --- Per-slot action encoding/decoding ----------------------------------------


def _slot_action_to_index(move_idx: int | None, target: int | None, mega: bool, switch_pos: int | None) -> int:
    """Encode a single-slot action into its per-slot index [0, 26)."""
    if switch_pos is not None:
        # switch_pos is 1-indexed bench position (1 or 2)
        return NUM_MOVES * NUM_TARGETS * 2 + (switch_pos - 1)
    # Move action — both args must be provided
    if move_idx is None or target is None:
        raise ValueError("move_idx and target required for move actions")
    target_idx = TARGETS.index(target)
    base = move_idx * NUM_TARGETS + target_idx
    if mega:
        base += NUM_MOVES * NUM_TARGETS  # offset into mega section
    return base


def _index_to_slot_action(idx: int) -> dict:
    """Decode a per-slot index [0, 26) into its components."""
    mega_offset = NUM_MOVES * NUM_TARGETS  # 12
    switch_offset = mega_offset * 2  # 24
    if idx >= switch_offset:
        # Switch action
        bench_pos = idx - switch_offset + 1
        return {"type": "switch", "bench_pos": bench_pos, "team_slot": bench_pos + 2}
    mega = idx >= mega_offset
    if mega:
        idx -= mega_offset
    move_idx = idx // NUM_TARGETS
    target_idx = idx % NUM_TARGETS
    return {"type": "move", "move_idx": move_idx, "target": TARGETS[target_idx], "mega": mega}


# --- Public API ---------------------------------------------------------------


def index_to_choice_string(action_idx: int) -> str:
    """Convert a canonical action index to a Showdown choice string.

    For team preview: returns e.g. "team 1342"
    For move phase: returns e.g. "move 1 1 mega, switch 3"
    """
    if action_idx < TEAM_PREVIEW_OFFSET + TEAM_PREVIEW_COUNT:
        # Team preview action
        perm = _index_to_team_perm(action_idx - TEAM_PREVIEW_OFFSET)
        return f"team {''.join(str(p) for p in perm)}"

    # Move phase joint action
    joint_idx = action_idx - MOVE_PHASE_OFFSET
    slot1_idx = joint_idx // ACTIONS_PER_SLOT
    slot2_idx = joint_idx % ACTIONS_PER_SLOT

    slot1_str = _slot_action_to_choice(slot1_idx)
    slot2_str = _slot_action_to_choice(slot2_idx)
    return f"{slot1_str}, {slot2_str}"


def _slot_action_to_choice(slot_idx: int) -> str:
    """Convert a per-slot action index to its Showdown choice string fragment."""
    action = _index_to_slot_action(slot_idx)
    if action["type"] == "switch":
        return f"switch {action['team_slot']}"
    # Move: "move N T [mega]" where N is 1-indexed move slot, T is target
    move_num = action["move_idx"] + 1  # 1-indexed for Showdown
    target = action["target"]
    mega_str = " mega" if action["mega"] else ""
    return f"move {move_num} {target}{mega_str}"


def choice_string_to_index(choice: str) -> int:
    """Convert a Showdown choice string to its canonical action index.

    Inverse of index_to_choice_string.
    """
    choice = choice.strip()
    if choice.startswith("team "):
        digits = choice[5:]
        perm = tuple(int(d) for d in digits)
        return TEAM_PREVIEW_OFFSET + _team_perm_to_index(perm)

    # Move phase: "slot1_choice, slot2_choice"
    parts = choice.split(", ")
    if len(parts) != 2:
        raise ValueError(f"Expected joint choice 'X, Y', got: {choice!r}")
    slot1_idx = _choice_to_slot_action(parts[0])
    slot2_idx = _choice_to_slot_action(parts[1])
    return MOVE_PHASE_OFFSET + slot1_idx * ACTIONS_PER_SLOT + slot2_idx


def _choice_to_slot_action(fragment: str) -> int:
    """Parse a single slot's choice string fragment into its per-slot index."""
    fragment = fragment.strip()
    if fragment.startswith("switch "):
        team_slot = int(fragment.split()[1])
        bench_pos = team_slot - 2  # team slot 3 -> bench pos 1, slot 4 -> bench pos 2
        return _slot_action_to_index(None, None, False, bench_pos)

    # "move N T [mega]"
    parts = fragment.split()
    if parts[0] != "move":
        raise ValueError(f"Expected 'move ...' or 'switch ...', got: {fragment!r}")
    move_num = int(parts[1])  # 1-indexed
    target = int(parts[2])
    mega = "mega" in parts[3:]
    return _slot_action_to_index(move_num - 1, target, mega, None)


def legal_mask(request: dict, phase: str) -> np.ndarray:
    """Build a bool mask of shape [A] from a Showdown activeRequest object.

    Args:
        request: The raw Showdown request for one side (from view["legal"][side]).
        phase: The battle phase ("teamPreview", "move", "forceSwitch").

    Returns:
        Boolean numpy array of shape [A]. True = legal action at that index.
    """
    mask = np.zeros(A, dtype=bool)

    if phase == "teamPreview":
        # All team orderings are legal during team preview
        mask[TEAM_PREVIEW_OFFSET : TEAM_PREVIEW_OFFSET + TEAM_PREVIEW_COUNT] = True
        return mask

    if phase in ("move", "forceSwitch"):
        _fill_move_phase_mask(mask, request)
        return mask

    return mask


def _fill_move_phase_mask(mask: np.ndarray, request: dict) -> None:
    """Fill the move-phase portion of the mask from a Showdown request."""
    active = request.get("active", [])
    side_pokemon = request.get("side", {}).get("pokemon", [])

    # Determine per-slot legal actions
    slot1_legal = _slot_legal_actions(active, side_pokemon, slot_idx=0, request=request)
    slot2_legal = _slot_legal_actions(active, side_pokemon, slot_idx=1, request=request)

    # Build joint mask (outer product), excluding illegal combos
    for s1 in slot1_legal:
        for s2 in slot2_legal:
            # Can't both switch to the same bench mon
            a1 = _index_to_slot_action(s1)
            a2 = _index_to_slot_action(s2)
            if a1["type"] == "switch" and a2["type"] == "switch" and a1["team_slot"] == a2["team_slot"]:
                continue
            joint_idx = MOVE_PHASE_OFFSET + s1 * ACTIONS_PER_SLOT + s2
            mask[joint_idx] = True


def _slot_legal_actions(active: list, side_pokemon: list, slot_idx: int, request: dict) -> list[int]:
    """Determine legal per-slot action indices for one active slot."""
    legal: list[int] = []

    # Handle force switch: only switches are legal
    force_switch = request.get("forceSwitch", [])
    if force_switch and slot_idx < len(force_switch) and force_switch[slot_idx]:
        return _legal_switches(side_pokemon)

    # If this slot doesn't exist (one mon left), return pass-like empty
    if slot_idx >= len(active):
        # Only one active mon — slot 2 gets a "pass": allow all switches as placeholder
        # In practice this means the joint action degenerates to slot1's choices
        return [NUM_MOVES * NUM_TARGETS * 2]  # first switch as placeholder

    slot_data = active[slot_idx]
    moves = slot_data.get("moves", [])
    can_mega = slot_data.get("canMegaEvo", False)

    # Determine which moves are usable
    for move_idx, move in enumerate(moves):
        if move.get("disabled") or move.get("pp", 0) <= 0:
            continue
        # Determine valid targets for this move
        target_type = move.get("target", "normal")
        valid_targets = _valid_targets_for(target_type)
        for target in valid_targets:
            # Base move (no mega)
            idx = _slot_action_to_index(move_idx, target, mega=False, switch_pos=None)
            legal.append(idx)
            # With mega if available
            if can_mega:
                idx_mega = _slot_action_to_index(move_idx, target, mega=True, switch_pos=None)
                legal.append(idx_mega)

    # Add legal switches
    if not request.get("trapped"):
        legal.extend(_legal_switches(side_pokemon))

    return legal


def _legal_switches(side_pokemon: list) -> list[int]:
    """Determine which switch actions are legal based on bench availability."""
    switches: list[int] = []
    # In a bring-4 format, bench positions are team slots 3 and 4 (0-indexed: 2 and 3)
    for i, mon in enumerate(side_pokemon):
        # Skip active mons (positions 0 and 1) and fainted mons
        if i < 2:
            continue
        if i > 3:
            break  # only 4 brought
        if mon.get("condition", "").endswith(" fnt") or mon.get("fainted"):
            continue
        bench_pos = i - 1  # team slot 3 → bench_pos 1, slot 4 → bench_pos 2
        if bench_pos in (1, 2):
            idx = _slot_action_to_index(None, None, False, bench_pos)
            switches.append(idx)
    return switches


def _valid_targets_for(target_type: str) -> list[int]:
    """Map Showdown target type to valid target integers for our canonical space."""
    # Single-target moves aimed at foes or ally
    if target_type in ("normal", "any"):
        return [1, 2, -1]
    # Spread moves hitting all adjacent foes (target is irrelevant but use 1 as canonical)
    if target_type in ("allAdjacentFoes", "allAdjacent", "all"):
        return [1]
    # Self-targeting (Protect, Swords Dance, etc.)
    if target_type == "self":
        return [1]
    # Ally-only (Helping Hand, Heal Pulse targeting ally)
    if target_type in ("adjacentAlly", "adjacentAllyOrSelf", "allySide", "allyTeam"):
        return [-1]
    # Default: allow both foe targets
    return [1, 2]
