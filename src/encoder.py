"""Encoder module — translates a SimClient StateView into an ObsBundle.

Phase 1 implementation: omniscient view, all belief_weights = 1.0, exactly 12
entity tokens (6 per side). Phase 4 will add variable-length opponent candidates
with belief_weight < 1.0 from the BeliefModel.

The encoder is the sole consumer of battle state for the NN. It reads from
SimClient's plain-dict StateView, never from raw Showdown objects.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

import action_space
from obs_bundle import ObsBundle, make_obs_bundle
from vocab import ABILITY_VOCAB, ITEM_VOCAB, MOVE_VOCAB, NATURE_VOCAB, SPECIES_VOCAB

# --- Normalization constants --------------------------------------------------
# Reasonable centering values; some features will exceed 1.0 (acceptable)
MAX_STAT = 200
MAX_TURNS = 20

# --- Feature dimension constants (computed from layout below) -----------------
# Status: 7 possible (brn, par, slp, frz, tox, psn, none → 7 one-hot)
NUM_STATUS = 7
_STATUS_MAP = {"brn": 0, "par": 1, "slp": 2, "frz": 3, "tox": 4, "psn": 5}

# Nature: 25 natures
NUM_NATURES = 25

# Stats: 6 continuous (hp is maxhp, atk, def, spa, spd, spe)
NUM_STATS = 6

# Stat stages: 6 (atk, def, spa, spd, spe, accuracy/evasion not included for simplicity)
NUM_BOOSTS = 7  # atk, def, spa, spd, spe, accuracy, evasion

# Per-move features: pp_fraction + disabled flag = 2 per move x 4 moves = 8
NUM_MOVE_FEATURES = 8

# Volatile counters: substitute_hp, stall_counter, active_turns = 3
NUM_VOLATILE_FEATURES = 3

# Positional/state flags: is_active, is_bench, is_fainted, item_consumed,
#                         physical_slot (3 one-hot: left/right/bench), side_flag,
#                         belief_weight = 8
NUM_FLAGS = 8

# HP%: 1
# Total entity feature dim F
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

# Field features: weather one-hot (5) + duration, terrain one-hot (5) + duration,
#                 trick_room + duration, gravity = 14
NUM_WEATHERS = 5
NUM_TERRAINS = 5
FIELD_FEATURE_DIM = NUM_WEATHERS + 1 + NUM_TERRAINS + 1 + 2 + 1  # 15

# Side features (per side): tailwind(1) + tailwind_dur(1) + reflect(1) + reflect_dur(1) +
#                            light_screen(1) + ls_dur(1) + aurora_veil(1) + av_dur(1) +
#                            stealth_rock(1) + spikes(1) + mega_used(1) = 11
SIDE_FEATURE_DIM = 11

# Scalar features: turn_norm, phase_encoding (4), whose_decision (2) = 7
SCALAR_FEATURE_DIM = 7

# Weather/terrain ID maps for one-hot encoding
_WEATHER_MAP = {"RainDance": 0, "SunnyDay": 1, "Sandstorm": 2, "Snow": 3, "Hail": 3}
_TERRAIN_MAP = {"electricterrain": 0, "grassyterrain": 1, "mistyterrain": 2, "psychicterrain": 3}

# Phase encoding
_PHASE_MAP = {"teamPreview": 0, "move": 1, "forceSwitch": 2, "terminal": 3}


def encode(
    view: dict[str, Any],
    perspective: str,
    belief: dict | None = None,
) -> ObsBundle:
    """Encode a SimClient StateView into an ObsBundle tensor bundle.

    Args:
        view: StateView dict from SimClient (snapshot, legal, phase, etc.)
        perspective: "p1" or "p2" — whose viewpoint to encode from
        belief: Optional belief dict for Phase 4 candidates (None in Phase 1)

    Returns:
        ObsBundle TensorDict ready for CVPN consumption.
    """
    snapshot = view["snapshot"]
    sides = snapshot["sides"]
    field_data = snapshot["field"]

    # Reorder sides: perspective player first, opponent second
    if perspective == sides[0]["id"]:
        my_side, opp_side = sides[0], sides[1]
    else:
        my_side, opp_side = sides[1], sides[0]

    # Build entity tokens: my 6 + opponent 6 = 12 in Phase 1
    my_pokemon = my_side["pokemon"]
    opp_pokemon = opp_side["pokemon"]

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
        species_ids[i] = SPECIES_VOCAB.encode(mon.get("species") or "")
        ability_ids[i] = ABILITY_VOCAB.encode(mon.get("ability") or "")
        item_ids[i] = ITEM_VOCAB.encode(mon.get("item") or "")
        move_ids[i] = _encode_move_ids(mon.get("moves", []))
        slot_id[i] = 0  # my team

    # Encode opponent pokemon (slot_id = 1 for opponent)
    offset = len(my_pokemon)
    for i, mon in enumerate(opp_pokemon):
        entities[offset + i] = _encode_pokemon_features(mon, is_opponent=True)
        species_ids[offset + i] = SPECIES_VOCAB.encode(mon.get("species") or "")
        ability_ids[offset + i] = ABILITY_VOCAB.encode(mon.get("ability") or "")
        item_ids[offset + i] = ITEM_VOCAB.encode(mon.get("item") or "")
        move_ids[offset + i] = _encode_move_ids(mon.get("moves", []))
        slot_id[offset + i] = 1  # opponent team

    # Build field features
    field_tensor = _encode_field(field_data)

    # Build sides features [2, Fs]: my side first, opponent second
    sides_tensor = torch.stack(
        [
            _encode_side(my_side),
            _encode_side(opp_side),
        ]
    )

    # Build scalars
    scalars_tensor = _encode_scalars(snapshot, view, perspective)

    # Build action mask
    phase = view.get("phase", "")
    legal = view.get("legal", {})
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


def _encode_pokemon_features(mon: dict, is_opponent: bool = False) -> torch.Tensor:
    """Encode a single pokemon's continuous features into a float tensor [F]."""
    feats = torch.zeros(ENTITY_FEATURE_DIM)
    idx = 0

    # HP fraction
    maxhp = mon.get("maxhp", 1)
    hp = mon.get("hp", 0)
    feats[idx] = hp / max(maxhp, 1)
    idx += 1

    # Normalized final stats (hp, atk, def, spa, spd, spe)
    stats = mon.get("stats", {})
    for stat_key in ("hp", "atk", "def", "spa", "spd", "spe"):
        feats[idx] = stats.get(stat_key, 0) / MAX_STAT
        idx += 1

    # Stat stages / 6 (normalized to roughly [-1, 1])
    boosts = mon.get("boosts", {})
    for boost_key in ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion"):
        feats[idx] = boosts.get(boost_key, 0) / 6.0
        idx += 1

    # Status one-hot (7 slots: brn, par, slp, frz, tox, psn, none)
    status = mon.get("status")
    if status and status in _STATUS_MAP:
        feats[idx + _STATUS_MAP[status]] = 1.0
    else:
        feats[idx + NUM_STATUS - 1] = 1.0  # "none" slot
    idx += NUM_STATUS

    # Nature one-hot (25 natures)
    nature_id = NATURE_VOCAB.encode(mon.get("nature") or "")
    if 1 <= nature_id <= NUM_NATURES:
        feats[idx + nature_id - 1] = 1.0
    idx += NUM_NATURES

    # Per-move features: pp_fraction + disabled (4 moves x 2 = 8)
    moves = mon.get("moves", [])
    for m_idx in range(4):
        if m_idx < len(moves):
            move = moves[m_idx]
            maxpp = move.get("maxpp", 1)
            feats[idx] = move.get("pp", 0) / max(maxpp, 1)
            feats[idx + 1] = 1.0 if move.get("disabled") else 0.0
        idx += 2

    # Volatile features: substitute_hp (normalized), stall_counter, active_turns
    volatile_details = mon.get("volatileDetails", {})
    sub_data = volatile_details.get("substitute", {})
    feats[idx] = sub_data.get("hp", 0) / MAX_STAT
    idx += 1
    stall_data = volatile_details.get("stall", {})
    feats[idx] = stall_data.get("counter", 0) / 6.0
    idx += 1
    feats[idx] = mon.get("activeTurns", 0) / MAX_TURNS
    idx += 1

    # Positional/state flags
    is_active = mon.get("active", False)
    is_fainted = mon.get("fainted", False)
    item_consumed = bool(mon.get("lastItem")) and not mon.get("item")

    feats[idx] = 1.0 if is_active else 0.0
    idx += 1
    feats[idx] = 1.0 if (not is_active and not is_fainted) else 0.0  # is_bench
    idx += 1
    feats[idx] = 1.0 if is_fainted else 0.0
    idx += 1
    feats[idx] = 1.0 if item_consumed else 0.0
    idx += 1

    # Physical slot one-hot (3: active-left=0, active-right=1, bench=2)
    position = mon.get("position", 0)
    if is_active and position == 1:
        feats[idx] = 1.0  # active-left
    elif is_active and position == 2:
        feats[idx + 1] = 1.0  # active-right
    else:
        feats[idx + 2] = 1.0  # bench
    idx += 3

    # Side flag: 0 for mine, 1 for opponent
    feats[idx] = 1.0 if is_opponent else 0.0
    idx += 1

    return feats


def _encode_move_ids(moves: list[dict]) -> torch.Tensor:
    """Encode move slot IDs into an int64 tensor [4]."""
    ids = torch.zeros(4, dtype=torch.long)
    for i, move in enumerate(moves[:4]):
        ids[i] = MOVE_VOCAB.encode(move.get("id", ""))
    return ids


def _encode_field(field_data: dict) -> torch.Tensor:
    """Encode field-level features into a float tensor [Ff]."""
    feats = torch.zeros(FIELD_FEATURE_DIM)
    idx = 0

    # Weather one-hot (5 slots) + duration
    weather = field_data.get("weather")
    if weather and weather in _WEATHER_MAP:
        feats[idx + _WEATHER_MAP[weather]] = 1.0
    idx += NUM_WEATHERS
    weather_dur = field_data.get("weatherDuration")
    feats[idx] = (weather_dur or 0) / MAX_TURNS
    idx += 1

    # Terrain one-hot (5 slots) + duration
    terrain = field_data.get("terrain")
    if terrain and terrain in _TERRAIN_MAP:
        feats[idx + _TERRAIN_MAP[terrain]] = 1.0
    idx += NUM_TERRAINS
    terrain_dur = field_data.get("terrainDuration")
    feats[idx] = (terrain_dur or 0) / MAX_TURNS
    idx += 1

    # Trick Room (active flag + duration)
    pseudo = field_data.get("pseudoWeather", {})
    tr = pseudo.get("trickroom", {}) if isinstance(pseudo, dict) else {}
    feats[idx] = 1.0 if tr else 0.0
    idx += 1
    feats[idx] = (tr.get("duration") or 0) / MAX_TURNS if tr else 0.0
    idx += 1

    # Gravity
    gravity = pseudo.get("gravity", {}) if isinstance(pseudo, dict) else {}
    feats[idx] = 1.0 if gravity else 0.0
    idx += 1

    return feats


def _encode_side(side_data: dict) -> torch.Tensor:
    """Encode per-side features into a float tensor [Fs]."""
    feats = torch.zeros(SIDE_FEATURE_DIM)
    conds = side_data.get("sideConditions", {})
    idx = 0

    # Tailwind: active + duration
    tw = conds.get("tailwind")
    feats[idx] = 1.0 if tw is not None else 0.0
    idx += 1
    feats[idx] = (tw or 0) / MAX_TURNS
    idx += 1

    # Reflect: active + duration
    reflect = conds.get("reflect")
    feats[idx] = 1.0 if reflect is not None else 0.0
    idx += 1
    feats[idx] = (reflect or 0) / MAX_TURNS
    idx += 1

    # Light Screen: active + duration
    ls = conds.get("lightscreen")
    feats[idx] = 1.0 if ls is not None else 0.0
    idx += 1
    feats[idx] = (ls or 0) / MAX_TURNS
    idx += 1

    # Aurora Veil: active + duration
    av = conds.get("auroraveil")
    feats[idx] = 1.0 if av is not None else 0.0
    idx += 1
    feats[idx] = (av or 0) / MAX_TURNS
    idx += 1

    # Stealth Rock (binary)
    feats[idx] = 1.0 if "stealthrock" in conds else 0.0
    idx += 1

    # Spikes (count / 3)
    spikes = conds.get("spikes")
    feats[idx] = (spikes or 0) / 3.0
    idx += 1

    # Mega used (check if any pokemon on this side has mega evolved)
    # For now encode as 0 — mega detection requires checking pokemon data
    feats[idx] = 0.0
    idx += 1

    return feats


def _encode_scalars(snapshot: dict, view: dict, perspective: str) -> torch.Tensor:
    """Encode global scalar features into a float tensor [Fg]."""
    feats = torch.zeros(SCALAR_FEATURE_DIM)
    idx = 0

    # Turn number (normalized)
    feats[idx] = snapshot.get("turn", 0) / MAX_TURNS
    idx += 1

    # Phase encoding (one-hot, 4 phases)
    phase = view.get("phase", "")
    if phase in _PHASE_MAP:
        feats[idx + _PHASE_MAP[phase]] = 1.0
    idx += 4

    # Whose decision (2 bits: am I acting, is opponent acting)
    to_move = view.get("to_move", [])
    feats[idx] = 1.0 if perspective in to_move else 0.0
    idx += 1
    opp = "p2" if perspective == "p1" else "p1"
    feats[idx] = 1.0 if opp in to_move else 0.0
    idx += 1

    return feats
