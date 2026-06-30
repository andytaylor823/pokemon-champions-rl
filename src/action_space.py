"""Action space module — canonical joint action index for VGC doubles.

Defines a fixed-size flat action space covering both team preview and the move
phase. The CVPN policy head outputs logits of shape [A] (one softmax over the
full joint space). At runtime, an `action_mask` bool tensor marks which indices
are legal.

Layout:
  [0, TEAM_PREVIEW_COUNT)           — team preview actions (P(6,4) = 360 orderings)
  [TEAM_PREVIEW_COUNT, A)           — move-phase joint actions (slot1 x slot2)

Per-slot actions during move phase (ACTIONS_PER_SLOT = 27):
  [0, 12)   — move 1-4 x target {1, 2, -1} (foe-left, foe-right, ally)
  [12, 24)  — move 1-4 x target {1, 2, -1} + mega evolution
  [24, 25)  — switch to bench position 1 (team slot 3)
  [25, 26)  — switch to bench position 2 (team slot 4)
  [26]      — pass (empty slot, no action required)

Showdown targeting conventions (slot-dependent for allies):
  - target  1 = opponent slot 1 (left foe)  — same from both slots
  - target  2 = opponent slot 2 (right foe) — same from both slots
  - target -1 = ally from SLOT 1's perspective (i.e. slot 0)
  - target -2 = ally from SLOT 0's perspective (i.e. slot 1)

Our canonical per-slot space always uses -1 to mean "ally" regardless of slot.
The slot-specific Showdown target is resolved when emitting a choice string
(see _canonical_to_showdown_target / _showdown_to_canonical_target).

  For spread/self-targeting moves, only one target is legal (mask handles it).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Literal

import numpy as np

# --- Typed slot action models -------------------------------------------------


@dataclass(frozen=True)
class MoveAction:
    """A per-slot move action (use move N targeting T, optionally mega)."""

    type: Literal["move"] = "move"
    move_idx: int = 0
    target: int = 0
    mega: bool = False


@dataclass(frozen=True)
class SwitchAction:
    """A per-slot switch action (swap to bench position)."""

    type: Literal["switch"] = "switch"
    bench_pos: int = 0
    team_slot: int = 0


@dataclass(frozen=True)
class PassAction:
    """A per-slot pass — the slot is empty and requires no action."""

    type: Literal["pass"] = "pass"


SlotAction = MoveAction | SwitchAction | PassAction

# --- Constants ----------------------------------------------------------------

# Move phase: per-slot action decomposition
NUM_MOVES = 4
TARGETS = (1, 2, -1)  # foe-left, foe-right, ally
NUM_TARGETS = len(TARGETS)
NUM_SWITCHES = 2  # bench positions (team slots 3 and 4 in a bring-4 format)

# Per-slot: 4 moves x 3 targets = 12 base + 12 mega + 2 switches + 1 pass = 27
PASS_INDEX = NUM_MOVES * NUM_TARGETS * 2 + NUM_SWITCHES  # 26
ACTIONS_PER_SLOT = PASS_INDEX + 1  # 27

# Team preview: P(6,4) = 6*5*4*3 = 360 orderings
TEAM_SIZE = 6
BRING_COUNT = 4
_TEAM_PERMS = list(permutations(range(1, TEAM_SIZE + 1), BRING_COUNT))
_TEAM_PERM_INDEX: dict[tuple[int, ...], int] = {p: i for i, p in enumerate(_TEAM_PERMS)}
TEAM_PREVIEW_COUNT = len(_TEAM_PERMS)  # 360

# Move phase: joint = slot1 x slot2
MOVE_PHASE_COUNT = ACTIONS_PER_SLOT * ACTIONS_PER_SLOT  # 729

# Total canonical action space
TEAM_PREVIEW_OFFSET = 0
MOVE_PHASE_OFFSET = TEAM_PREVIEW_COUNT  # 360
A = TEAM_PREVIEW_COUNT + MOVE_PHASE_COUNT  # 1089

# Precomputed mapping: switch action index → bench position (for cross-slot constraint)
_SWITCH_INDEX_TO_BENCH: dict[int, int] = {NUM_MOVES * NUM_TARGETS * 2 + (pos - 1): pos for pos in (1, 2)}

# Showdown ally-target numbers differ by active slot position:
#   Slot 0 targets its ally (slot 1) with -2
#   Slot 1 targets its ally (slot 0) with -1
_ALLY_SHOWDOWN_TARGET: dict[int, int] = {0: -2, 1: -1}

# Move target types that do NOT accept an explicit target in the choice string
_NO_TARGET_TYPES = frozenset({"allAdjacentFoes", "self", "allySide", "foeSide", "all", "allAdjacent", "scripted", "randomNormal", "allies", "allyTeam"})


def _canonical_to_showdown_target(canonical_target: int, slot_pos: int) -> int:
    """Translate canonical target (-1 = ally) to Showdown's slot-specific target."""
    if canonical_target == -1:
        return _ALLY_SHOWDOWN_TARGET[slot_pos]
    return canonical_target


def _showdown_to_canonical_target(showdown_target: int, slot_pos: int) -> int:
    """Translate Showdown's slot-specific target back to canonical (-1 = ally)."""
    if showdown_target == _ALLY_SHOWDOWN_TARGET.get(slot_pos):
        return -1
    return showdown_target


# --- Team preview helpers -----------------------------------------------------


def _team_perm_to_index(perm: tuple[int, ...]) -> int:
    """Map a team ordering tuple (e.g. (1,3,4,2)) to its canonical index."""
    return _TEAM_PERM_INDEX[perm]


def _index_to_team_perm(idx: int) -> tuple[int, ...]:
    """Map a team preview index to the ordering tuple."""
    return _TEAM_PERMS[idx]


# --- Per-slot action encoding/decoding ----------------------------------------


def _slot_action_to_index(move_idx: int | None, target: int | None, mega: bool, switch_pos: int | None) -> int:
    """Encode a single-slot action into its per-slot index [0, 27)."""
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


def _index_to_slot_action(idx: int) -> SlotAction:
    """Decode a per-slot index [0, 27) into a typed SlotAction."""
    if idx == PASS_INDEX:
        return PassAction()
    mega_offset = NUM_MOVES * NUM_TARGETS  # 12
    switch_offset = mega_offset * 2  # 24
    if idx >= switch_offset:
        bench_pos = idx - switch_offset + 1
        return SwitchAction(bench_pos=bench_pos, team_slot=bench_pos + 2)
    mega = idx >= mega_offset
    if mega:
        idx -= mega_offset
    move_idx = idx // NUM_TARGETS
    target_idx = idx % NUM_TARGETS
    return MoveAction(move_idx=move_idx, target=TARGETS[target_idx], mega=mega)


# --- Public API ---------------------------------------------------------------


def choice_string_to_index(choice: str) -> int:
    """Convert a Showdown choice string to its canonical action index.

    Inverse of action_to_choice_contextual.
    """
    choice = choice.strip()
    if choice.startswith("team "):
        digits = choice[5:]
        perm = tuple(int(d) for d in digits)
        return TEAM_PREVIEW_OFFSET + _team_perm_to_index(perm)

    # Move phase: "slot1_choice, slot2_choice" or single choice (pass in other slot)
    parts = choice.split(", ")
    if len(parts) == 2:
        slot1_idx = _choice_to_slot_action(parts[0], slot_pos=0)
        slot2_idx = _choice_to_slot_action(parts[1], slot_pos=1)
    elif len(parts) == 1:
        # Single slot choice — other slot is pass
        slot1_idx = _choice_to_slot_action(parts[0], slot_pos=0)
        slot2_idx = PASS_INDEX
    else:
        raise ValueError(f"Expected joint choice 'X, Y' or single choice, got: {choice!r}")
    return MOVE_PHASE_OFFSET + slot1_idx * ACTIONS_PER_SLOT + slot2_idx


def _choice_to_slot_action(fragment: str, slot_pos: int = 1) -> int:
    """Parse a single slot's choice string fragment into its per-slot index.

    Args:
        fragment: Showdown choice string fragment (e.g. "move 1 2", "switch 3").
        slot_pos: Active slot position (0 or 1) — needed to map Showdown's
            slot-specific ally targets back to canonical -1.
    """
    fragment = fragment.strip()
    if not fragment or fragment == "pass":
        return PASS_INDEX
    if fragment.startswith("switch "):
        team_slot = int(fragment.split()[1])
        bench_pos = team_slot - 2  # team slot 3 -> bench pos 1, slot 4 -> bench pos 2
        return _slot_action_to_index(None, None, False, bench_pos)

    # "move N [T] [mega]" — target may be omitted for self/spread moves
    parts = fragment.split()
    if parts[0] != "move":
        raise ValueError(f"Expected 'move ...' or 'switch ...', got: {fragment!r}")
    move_num = int(parts[1])  # 1-indexed
    raw_target = int(parts[2]) if len(parts) > 2 and parts[2] != "mega" else 1
    target = _showdown_to_canonical_target(raw_target, slot_pos)
    mega = "mega" in parts[2:]
    return _slot_action_to_index(move_num - 1, target, mega, None)


def legal_mask(request: dict | None, phase: str) -> np.ndarray:
    """Build a bool mask of shape [A] from a Showdown activeRequest object.

    Args:
        request: The raw Showdown request for one side, or None if the
            perspective has no legal actions (returns all-zeros).
        phase: The battle phase ("teamPreview", "move", "forceSwitch").

    Returns:
        Boolean numpy array of shape [A]. True = legal action at that index.
    """
    mask = np.zeros(A, dtype=bool)

    if request is None:
        return mask

    if phase == "teamPreview":
        mask[TEAM_PREVIEW_OFFSET : TEAM_PREVIEW_OFFSET + TEAM_PREVIEW_COUNT] = True
        return mask

    if phase in ("move", "forceSwitch"):
        _fill_move_phase_mask(mask, request)
        return mask

    return mask


def _fill_move_phase_mask(mask: np.ndarray, request: dict) -> None:
    """Fill the move-phase portion of the mask from a Showdown request."""
    side_pokemon = request.get("side", {}).get("pokemon", [])
    force_switch = request.get("forceSwitch", [])

    # Double forced-switch (both active fainted) is a joint constraint the per-slot
    # outer product cannot express: with exactly one eligible bench mon, both slots
    # enumerate the same lone switch and the cross-slot collision guard erases the
    # only joint, leaving an empty mask. Showdown instead wants the mon brought into
    # one slot with the other slot explicitly passed ("switch N, pass"), so handle
    # the both-forced case directly.
    if len(force_switch) >= 2 and force_switch[0] and force_switch[1]:
        _fill_double_force_switch(mask, side_pokemon)
        return

    active = request.get("active", [])
    slot1_legal = _slot_legal_actions(active, side_pokemon, slot_idx=0, request=request)
    slot2_legal = _slot_legal_actions(active, side_pokemon, slot_idx=1, request=request)

    # Build joint mask (outer product), excluding illegal combos
    for s1 in slot1_legal:
        for s2 in slot2_legal:
            # Can't both switch to the same bench mon (check via precomputed map)
            if s1 in _SWITCH_INDEX_TO_BENCH and _SWITCH_INDEX_TO_BENCH.get(s1) == _SWITCH_INDEX_TO_BENCH.get(s2):
                continue
            joint_idx = MOVE_PHASE_OFFSET + s1 * ACTIONS_PER_SLOT + s2
            mask[joint_idx] = True


def _fill_double_force_switch(mask: np.ndarray, side_pokemon: list) -> None:
    """Set legal joints when BOTH active slots must switch (a double faint).

    Showdown requires bringing in as many bench Pokemon as possible:
      - >= 2 eligible bench mons: both slots switch, to distinct mons;
      - exactly one: it fills either slot while the other passes (both orderings);
      - none: both slots pass (usually a terminal state, included for completeness).
    """
    switches = _legal_switches(side_pokemon)

    def _set(s1: int, s2: int) -> None:
        mask[MOVE_PHASE_OFFSET + s1 * ACTIONS_PER_SLOT + s2] = True

    if len(switches) >= 2:
        for s1 in switches:
            for s2 in switches:
                if _SWITCH_INDEX_TO_BENCH[s1] == _SWITCH_INDEX_TO_BENCH[s2]:
                    continue
                _set(s1, s2)
    elif len(switches) == 1:
        (only_switch,) = switches
        _set(only_switch, PASS_INDEX)
        _set(PASS_INDEX, only_switch)
    else:
        _set(PASS_INDEX, PASS_INDEX)


def _slot_legal_actions(active: list, side_pokemon: list, slot_idx: int, request: dict) -> list[int]:
    """Determine legal per-slot action indices for one active slot."""
    legal: list[int] = []

    # Handle force switch: only switches are legal
    force_switch = request.get("forceSwitch", [])
    if force_switch and slot_idx < len(force_switch) and force_switch[slot_idx]:
        # No bench Pokemon available → pass (slot stays empty; game may be terminal)
        return _legal_switches(side_pokemon) or [PASS_INDEX]

    # If this slot doesn't exist (one mon left), it is a pass
    if slot_idx >= len(active):
        return [PASS_INDEX]

    # Fainted active slot with no bench replacement: Showdown still lists it in
    # `active` with full move data, but rejects any choice for it.  Treat as pass.
    if slot_idx < len(side_pokemon):
        slot_condition = side_pokemon[slot_idx].get("condition", "")
        if slot_condition.endswith(" fnt"):
            return [PASS_INDEX]

    slot_data = active[slot_idx]
    moves = slot_data.get("moves", [])
    can_mega = slot_data.get("canMegaEvo", False)
    # Per-slot trapping (e.g. charging a multi-turn move like Solar Beam)
    slot_trapped = slot_data.get("trapped", False)

    for move_idx, move in enumerate(moves):
        if _is_charging(move):
            idx = _slot_action_to_index(move_idx, 1, mega=False, switch_pos=None)
            legal.append(idx)
            continue

        if move.get("disabled") or move.get("pp", 0) <= 0:
            continue
        target_type = move.get("target", "normal")
        valid_targets = _valid_targets_for(target_type)
        for target in valid_targets:
            idx = _slot_action_to_index(move_idx, target, mega=False, switch_pos=None)
            legal.append(idx)
            if can_mega:
                idx_mega = _slot_action_to_index(move_idx, target, mega=True, switch_pos=None)
                legal.append(idx_mega)

    # Add legal switches (blocked by per-slot OR top-level trapping)
    if not slot_trapped and not request.get("trapped"):
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


_TARGET_TYPE_MAP: dict[str, list[int]] = {
    # Single-target moves aimed at foes or ally
    "normal": [1, 2, -1],
    "any": [1, 2, -1],
    # Single-target but restricted to foes only (no ally target option)
    "adjacentFoe": [1, 2],
    # Spread moves hitting all adjacent foes (target is irrelevant but use 1 as canonical)
    "allAdjacentFoes": [1],
    "allAdjacent": [1],
    "all": [1],
    # Self-targeting (Protect, Swords Dance, etc.)
    "self": [1],
    # Ally-only (Helping Hand, Heal Pulse targeting ally)
    "adjacentAlly": [-1],
    "adjacentAllyOrSelf": [-1],
    "allySide": [-1],
    "allyTeam": [-1],
    # "allies" hits user + ally without a target choice (Life Dew, Howl, etc.)
    "allies": [-1],
    # Engine-chosen target: Struggle, Counter, Mirror Coat, etc.
    "scripted": [1],
    # Multi-turn rampage moves (Outrage, Thrash, Petal Dance) — engine picks random foe
    "randomNormal": [1],
    # Hazards targeting the opposing side (Stealth Rock, Spikes, Sticky Web, Toxic Spikes)
    "foeSide": [1],
}


def _valid_targets_for(target_type: str) -> list[int]:
    """Map Showdown target type to valid target integers for our canonical space."""
    try:
        return _TARGET_TYPE_MAP[target_type]
    except KeyError:
        raise ValueError(f"Unknown Showdown target type: {target_type!r}") from None


def _is_charging(move_data: dict) -> bool:
    """True if this move entry represents a charging/locked multi-turn move.

    Showdown omits both ``pp`` and ``target`` for charging moves (e.g. the
    second turn of Solar Beam). This is the single predicate both legal_mask
    and the choice-string builder should use.
    """
    return "pp" not in move_data


# ---------------------------------------------------------------------------
# Forced-decision predicate
# ---------------------------------------------------------------------------


def forced_actions(legal: dict, to_move: list[str], phase: str) -> dict[str, int] | None:
    """Return {side: sole_legal_idx} if every acting side has exactly one legal action.

    Returns None if to_move is empty or any side has more than one legal action
    (meaning a genuine decision exists).
    """
    if not to_move:
        return None
    out: dict[str, int] = {}
    for s in to_move:
        m = legal_mask(legal.get(s), phase)
        if int(m.sum()) != 1:
            return None
        out[s] = int(np.argmax(m))
    return out


# ---------------------------------------------------------------------------
# Context-aware choice string builder (uses legal request for target stripping)
# ---------------------------------------------------------------------------


def _slot_choice_contextual(action: SlotAction, slot_moves: list[dict] | None, slot_pos: int, force_pass: bool = False) -> str:
    """Build a Showdown choice fragment, stripping target if the move doesn't accept one.

    Combines slot-aware ally targeting with target-type stripping for spread/self moves.
    A passed slot normally emits nothing (the caller collapses it away), but a slot
    under a forceSwitch must emit an explicit "pass" so Showdown receives a choice for
    every forced slot — e.g. "switch 3, pass" in a double faint with one bench mon left,
    which Showdown rejects when written as the bare "switch 3".
    """
    if isinstance(action, PassAction):
        return "pass" if force_pass else ""
    if isinstance(action, SwitchAction):
        return f"switch {action.team_slot}"

    move_num = action.move_idx + 1
    mega_str = " mega" if action.mega else ""

    if slot_moves and action.move_idx < len(slot_moves):
        move_data = slot_moves[action.move_idx]
        if _is_charging(move_data):
            return f"move {move_num}{mega_str}"
        move_target_type = move_data.get("target", "normal")
        if move_target_type in _NO_TARGET_TYPES:
            return f"move {move_num}{mega_str}"

    showdown_target = _canonical_to_showdown_target(action.target, slot_pos)
    return f"move {move_num} {showdown_target}{mega_str}"


def action_to_choice_contextual(action_idx: int, legal_request: dict | None) -> str:
    """Convert a canonical action index to a valid Showdown choice string.

    For team preview: returns e.g. "team 1342".
    For move phase: returns e.g. "move 1 1 mega, switch 3".
    When one slot is a pass, it is omitted (e.g. "move 1 1" with no comma).

    Uses the legal request to strip targets for spread/self/charging moves and
    to resolve slot-specific ally target numbers. Pass legal_request=None for
    context-free conversion (targets are always emitted).
    """
    if action_idx < TEAM_PREVIEW_COUNT:
        perm = _index_to_team_perm(action_idx - TEAM_PREVIEW_OFFSET)
        return f"team {''.join(str(p) for p in perm)}"

    joint_idx = action_idx - MOVE_PHASE_OFFSET
    slot1_idx = joint_idx // ACTIONS_PER_SLOT
    slot2_idx = joint_idx % ACTIONS_PER_SLOT

    slot1_action = _index_to_slot_action(slot1_idx)
    slot2_action = _index_to_slot_action(slot2_idx)

    active = (legal_request or {}).get("active", [])
    force_switch = (legal_request or {}).get("forceSwitch", [])
    slot1_moves = active[0].get("moves", []) if len(active) > 0 else []
    slot2_moves = active[1].get("moves", []) if len(active) > 1 else []

    # A slot under a forceSwitch must emit an explicit "pass" (see _slot_choice_contextual).
    force_pass_0 = len(force_switch) > 0 and bool(force_switch[0])
    force_pass_1 = len(force_switch) > 1 and bool(force_switch[1])

    slot1_str = _slot_choice_contextual(slot1_action, slot1_moves, slot_pos=0, force_pass=force_pass_0)
    slot2_str = _slot_choice_contextual(slot2_action, slot2_moves, slot_pos=1, force_pass=force_pass_1)

    if slot1_str and slot2_str:
        return f"{slot1_str}, {slot2_str}"
    if slot1_str:
        return slot1_str
    if slot2_str:
        return f"pass, {slot2_str}"
    return "pass"
