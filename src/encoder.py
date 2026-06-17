"""Encoder module — translates a SimClient StateView into an ObsBundle.

Phase 1 implementation: omniscient view, all belief_weights = 1.0, exactly 12
entity tokens (6 per side). Phase 4 will add variable-length opponent candidates
with belief_weight < 1.0 from the BeliefModel.

The encoder is the sole consumer of battle state for the NN. It reads typed
Pydantic models (state_types.py) validated by SimClient, never raw dicts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

import action_space
from obs_bundle import ObsBundle, make_obs_bundle
from vocab import ABILITY_VOCAB, ITEM_VOCAB, MOVE_VOCAB, NATURE_VOCAB, SPECIES_VOCAB

if TYPE_CHECKING:
    from state_types import (
        BattleSnapshot,
        FieldSnapshot,
        MoveSnapshot,
        PokemonSnapshot,
        SideSnapshot,
        StateView,
    )

# --- Normalization constants --------------------------------------------------
MAX_STAT = 200
MAX_TURNS = 20

# --- Feature dimension constants (computed from layout below) -----------------
# Status: 7 possible (brn, par, slp, frz, tox, psn, none -> 7 one-hot)
NUM_STATUS = 7
_STATUS_MAP = {"brn": 0, "par": 1, "slp": 2, "frz": 3, "tox": 4, "psn": 5}

NUM_NATURES = 25
NUM_STATS = 6
NUM_BOOSTS = 7  # atk, def, spa, spd, spe, accuracy, evasion
NUM_MOVE_FEATURES = 8  # pp_fraction + disabled flag = 2 per move x 4 moves
NUM_VOLATILE_FEATURES = 3  # substitute_hp, stall_counter, active_turns

# Positional/state flags: is_active, is_bench, is_fainted, item_consumed,
#                         physical_slot (3 one-hot: left/right/bench), side_flag,
#                         belief_weight = 8
NUM_FLAGS = 8

ENTITY_FEATURE_DIM = (
    1  # hp_fraction
    + NUM_STATS  # normalized final stats
    + NUM_BOOSTS  # stat stages / 6
    + NUM_STATUS  # status one-hot
    + NUM_NATURES  # nature one-hot
    + NUM_MOVE_FEATURES  # per-move pp_frac + disabled
    + NUM_VOLATILE_FEATURES  # sub hp, stall counter, active turns
    + NUM_FLAGS  # positional + state flags
)

NUM_WEATHERS = 5
NUM_TERRAINS = 5
FIELD_FEATURE_DIM = NUM_WEATHERS + 1 + NUM_TERRAINS + 1 + 2 + 1  # 15

SIDE_FEATURE_DIM = 11

SCALAR_FEATURE_DIM = 7

# Weather/terrain ID maps — lowercase to match engine status IDs
_WEATHER_MAP = {"raindance": 0, "sunnyday": 1, "sandstorm": 2, "snow": 3, "hail": 3}
_TERRAIN_MAP = {"electricterrain": 0, "grassyterrain": 1, "mistyterrain": 2, "psychicterrain": 3}

_PHASE_MAP = {"teamPreview": 0, "move": 1, "forceSwitch": 2, "terminal": 3}


def encode(
    view: StateView,
    perspective: str,
    belief: dict | None = None,
) -> ObsBundle:
    """Encode a SimClient StateView into an ObsBundle tensor bundle.

    Args:
        view: Typed StateView from SimClient.
        perspective: "p1" or "p2" — whose viewpoint to encode from.
        belief: Optional belief dict for Phase 4 candidates (None in Phase 1).

    Returns:
        ObsBundle TensorDict ready for CVPN consumption.
    """
    snapshot = view.snapshot
    sides = snapshot.sides
    field_data = snapshot.field

    # Reorder sides: perspective player first, opponent second
    if perspective == sides[0].id:
        my_side, opp_side = sides[0], sides[1]
    else:
        my_side, opp_side = sides[1], sides[0]

    # Build entity tokens: my 6 + opponent 6 = 12 in Phase 1
    my_pokemon = my_side.pokemon
    opp_pokemon = opp_side.pokemon

    n_tokens = len(my_pokemon) + len(opp_pokemon)
    entities = torch.zeros(n_tokens, ENTITY_FEATURE_DIM)
    species_ids = torch.zeros(n_tokens, dtype=torch.long)
    ability_ids = torch.zeros(n_tokens, dtype=torch.long)
    item_ids = torch.zeros(n_tokens, dtype=torch.long)
    move_ids = torch.zeros(n_tokens, 4, dtype=torch.long)
    belief_weight = torch.ones(n_tokens)  # 1.0 for Phase 1
    slot_id = torch.zeros(n_tokens, dtype=torch.long)

    # Encode my pokemon (slot_id = 0 for mine)
    for i, mon in enumerate(my_pokemon):
        entities[i] = _encode_pokemon_features(mon)
        species_ids[i] = SPECIES_VOCAB.encode(mon.species or "")
        ability_ids[i] = ABILITY_VOCAB.encode(mon.ability or "")
        item_ids[i] = ITEM_VOCAB.encode(mon.item or "")
        move_ids[i] = _encode_move_ids(mon.moves)
        slot_id[i] = 0

    # Encode opponent pokemon (slot_id = 1 for opponent)
    offset = len(my_pokemon)
    for i, mon in enumerate(opp_pokemon):
        entities[offset + i] = _encode_pokemon_features(mon, is_opponent=True)
        species_ids[offset + i] = SPECIES_VOCAB.encode(mon.species or "")
        ability_ids[offset + i] = ABILITY_VOCAB.encode(mon.ability or "")
        item_ids[offset + i] = ITEM_VOCAB.encode(mon.item or "")
        move_ids[offset + i] = _encode_move_ids(mon.moves)
        slot_id[offset + i] = 1

    # Build field features
    field_tensor = _encode_field(field_data)

    # Build sides features [2, Fs]: my side first, opponent second
    sides_tensor = torch.stack([_encode_side(my_side), _encode_side(opp_side)])

    # Build scalars
    scalars_tensor = _encode_scalars(snapshot, view, perspective)

    # Build action mask
    phase = view.phase
    legal = view.legal
    mask = action_space.legal_mask(legal[perspective], phase) if perspective in legal else np.zeros(action_space.A, dtype=bool)
    action_mask = torch.from_numpy(mask)

    # Padding mask: all true in Phase 1 (no padding within a single observation)
    padding_mask = torch.ones(n_tokens, dtype=torch.bool)

    return make_obs_bundle(
        entities=entities,
        species_ids=species_ids,
        ability_ids=ability_ids,
        item_ids=item_ids,
        move_ids=move_ids,
        belief_weight=belief_weight,
        slot_id=slot_id,
        field=field_tensor,
        sides=sides_tensor,
        scalars=scalars_tensor,
        action_mask=action_mask,
        padding_mask=padding_mask,
    )


# --- Private helpers ----------------------------------------------------------


def _encode_pokemon_features(mon: PokemonSnapshot, is_opponent: bool = False) -> torch.Tensor:
    """Encode a single pokemon's continuous features into a float tensor [F]."""
    feats = torch.zeros(ENTITY_FEATURE_DIM)
    idx = 0

    # HP fraction
    feats[idx] = mon.hp / max(mon.maxhp, 1)
    idx += 1

    # Normalized final stats (hp, atk, def, spa, spd, spe)
    for stat_key in ("hp", "atk", "def", "spa", "spd", "spe"):
        feats[idx] = mon.stats.get(stat_key, 0) / MAX_STAT
        idx += 1

    # Stat stages / 6 (normalized to roughly [-1, 1])
    for boost_key in ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion"):
        feats[idx] = mon.boosts.get(boost_key, 0) / 6.0
        idx += 1

    # Status one-hot (7 slots: brn, par, slp, frz, tox, psn, none)
    if mon.status and mon.status in _STATUS_MAP:
        feats[idx + _STATUS_MAP[mon.status]] = 1.0
    else:
        feats[idx + NUM_STATUS - 1] = 1.0  # "none" slot
    idx += NUM_STATUS

    # Nature one-hot (25 natures)
    nature_id = NATURE_VOCAB.encode(mon.nature or "")
    if 1 <= nature_id <= NUM_NATURES:
        feats[idx + nature_id - 1] = 1.0
    idx += NUM_NATURES

    # Per-move features: pp_fraction + disabled (4 moves x 2 = 8)
    for m_idx in range(4):
        if m_idx < len(mon.moves):
            move = mon.moves[m_idx]
            feats[idx] = move.pp / max(move.maxpp, 1)
            feats[idx + 1] = 1.0 if move.disabled else 0.0
        idx += 2

    # Volatile features: substitute_hp (normalized), stall_counter, active_turns
    sub_data = mon.volatileDetails.get("substitute")
    feats[idx] = (sub_data.hp or 0) / MAX_STAT if sub_data and sub_data.hp is not None else 0.0
    idx += 1
    stall_data = mon.volatileDetails.get("stall")
    feats[idx] = (stall_data.counter or 0) / 6.0 if stall_data and stall_data.counter is not None else 0.0
    idx += 1
    feats[idx] = mon.activeTurns / MAX_TURNS
    idx += 1

    # Positional/state flags
    item_consumed = bool(mon.lastItem) and not mon.item

    feats[idx] = 1.0 if mon.active else 0.0
    idx += 1
    feats[idx] = 1.0 if (not mon.active and not mon.fainted) else 0.0  # is_bench
    idx += 1
    feats[idx] = 1.0 if mon.fainted else 0.0
    idx += 1
    feats[idx] = 1.0 if item_consumed else 0.0
    idx += 1

    # Physical slot one-hot (3: active-left=0, active-right=1, bench=2)
    # Engine positions are 0-indexed: 0 = first active slot, 1 = second active slot
    if mon.active and mon.position == 0:
        feats[idx] = 1.0  # active-left
    elif mon.active and mon.position == 1:
        feats[idx + 1] = 1.0  # active-right
    else:
        feats[idx + 2] = 1.0  # bench
    idx += 3

    # Side flag: 0 for mine, 1 for opponent
    feats[idx] = 1.0 if is_opponent else 0.0
    idx += 1

    return feats


def _encode_move_ids(moves: list[MoveSnapshot]) -> torch.Tensor:
    """Encode move slot IDs into an int64 tensor [4]."""
    ids = torch.zeros(4, dtype=torch.long)
    for i, move in enumerate(moves[:4]):
        ids[i] = MOVE_VOCAB.encode(move.id)
    return ids


def _encode_field(field_data: FieldSnapshot) -> torch.Tensor:
    """Encode field-level features into a float tensor [Ff]."""
    feats = torch.zeros(FIELD_FEATURE_DIM)
    idx = 0

    # Weather one-hot (5 slots) + duration
    if field_data.weather and field_data.weather in _WEATHER_MAP:
        feats[idx + _WEATHER_MAP[field_data.weather]] = 1.0
    idx += NUM_WEATHERS
    feats[idx] = (field_data.weatherDuration or 0) / MAX_TURNS
    idx += 1

    # Terrain one-hot (5 slots) + duration
    if field_data.terrain and field_data.terrain in _TERRAIN_MAP:
        feats[idx + _TERRAIN_MAP[field_data.terrain]] = 1.0
    idx += NUM_TERRAINS
    feats[idx] = (field_data.terrainDuration or 0) / MAX_TURNS
    idx += 1

    # Trick Room (active flag + duration)
    tr = field_data.pseudoWeather.get("trickroom")
    feats[idx] = 1.0 if tr else 0.0
    idx += 1
    feats[idx] = (tr.duration or 0) / MAX_TURNS if tr else 0.0
    idx += 1

    # Gravity
    gravity = field_data.pseudoWeather.get("gravity")
    feats[idx] = 1.0 if gravity else 0.0
    idx += 1

    return feats


def _encode_side(side_data: SideSnapshot) -> torch.Tensor:
    """Encode per-side features into a float tensor [Fs]."""
    feats = torch.zeros(SIDE_FEATURE_DIM)
    conds = side_data.sideConditions
    idx = 0

    # Tailwind: active + duration
    tw = conds.get("tailwind")
    feats[idx] = 1.0 if tw is not None else 0.0
    idx += 1
    feats[idx] = (tw.duration or 0) / MAX_TURNS if tw else 0.0
    idx += 1

    # Reflect: active + duration
    reflect = conds.get("reflect")
    feats[idx] = 1.0 if reflect is not None else 0.0
    idx += 1
    feats[idx] = (reflect.duration or 0) / MAX_TURNS if reflect else 0.0
    idx += 1

    # Light Screen: active + duration
    ls = conds.get("lightscreen")
    feats[idx] = 1.0 if ls is not None else 0.0
    idx += 1
    feats[idx] = (ls.duration or 0) / MAX_TURNS if ls else 0.0
    idx += 1

    # Aurora Veil: active + duration
    av = conds.get("auroraveil")
    feats[idx] = 1.0 if av is not None else 0.0
    idx += 1
    feats[idx] = (av.duration or 0) / MAX_TURNS if av else 0.0
    idx += 1

    # Stealth Rock (binary)
    feats[idx] = 1.0 if "stealthrock" in conds else 0.0
    idx += 1

    # Spikes (layer count / 3) — reads .layers, not .duration
    spikes = conds.get("spikes")
    feats[idx] = (spikes.layers or 0) / 3.0 if spikes else 0.0
    idx += 1

    # Mega used (check if any pokemon on this side has mega evolved)
    # For now encode as 0 — mega detection requires checking pokemon data
    feats[idx] = 0.0
    idx += 1

    return feats


def _encode_scalars(snapshot: BattleSnapshot, view: StateView, perspective: str) -> torch.Tensor:
    """Encode global scalar features into a float tensor [Fg]."""
    feats = torch.zeros(SCALAR_FEATURE_DIM)
    idx = 0

    # Turn number (normalized)
    feats[idx] = snapshot.turn / MAX_TURNS
    idx += 1

    # Phase encoding (one-hot, 4 phases)
    if view.phase in _PHASE_MAP:
        feats[idx + _PHASE_MAP[view.phase]] = 1.0
    idx += 4

    # Whose decision (2 bits: am I acting, is opponent acting)
    feats[idx] = 1.0 if perspective in view.to_move else 0.0
    idx += 1
    opp = "p2" if perspective == "p1" else "p1"
    feats[idx] = 1.0 if opp in view.to_move else 0.0
    idx += 1

    return feats
