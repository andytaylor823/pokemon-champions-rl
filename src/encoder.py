"""Encoder module — translates a SimClient StateView into an ObsBundle.

Phase 1 implementation: omniscient view, all belief_weights = 1.0, exactly 12
entity tokens (6 per side). Phase 4 will add variable-length opponent candidates
with belief_weight < 1.0 from the BeliefModel.

The encoder is the sole consumer of battle state for the NN. It reads typed
Pydantic models (state_types.py) validated by SimClient, never raw dicts.
"""

from __future__ import annotations

import torch

import action_space
from obs_bundle import ObsBundle, make_obs_bundle
from state_types import (
    BattleSnapshot,
    FieldSnapshot,
    MoveSnapshot,
    PokemonSnapshot,
    SideConditionSnapshot,
    SideSnapshot,
    StateView,
)
from vocab import ABILITY_VOCAB, ITEM_VOCAB, MOVE_VOCAB, NATURE_VOCAB, SPECIES_VOCAB

# --- Normalization constants --------------------------------------------------
MAX_STAT = 200
MAX_TURNS = 20

# --- Feature dimension constants (semantic widths for sub-encoders) ----------
# Status: 7 possible (brn, par, slp, frz, tox, psn, none -> 7 one-hot)
NUM_STATUS = 7
_STATUS_MAP = {"brn": 0, "par": 1, "slp": 2, "frz": 3, "tox": 4, "psn": 5}

NUM_NATURES = 25
NUM_MOVE_SLOTS = 4  # moves per Pokémon (hard game constant)
NUM_MOVE_FEATURES = 2 * NUM_MOVE_SLOTS  # pp_fraction + disabled flag per move

NUM_WEATHERS = 4
NUM_TERRAINS = 4

# Weather/terrain ID maps — lowercase to match engine status IDs
_WEATHER_MAP = {"raindance": 0, "sunnyday": 1, "sandstorm": 2, "snow": 3, "hail": 3}
_TERRAIN_MAP = {"electricterrain": 0, "grassyterrain": 1, "mistyterrain": 2, "psychicterrain": 3}

_PHASE_MAP = {"teamPreview": 0, "move": 1, "forceSwitch": 2, "terminal": 3}

# Binary volatile flags — order defines tensor layout. Adding a new volatile is
# a data change: append the engine ID string here (and add a test).
BINARY_VOLATILES: tuple[str, ...] = (
    # Tier 1: action-constraining
    "trapped", "partiallytrapped", "lockedmove", "mustrecharge", "twoturnmove",
    "encore", "taunt", "disable", "torment",
    # Tier 2: mechanic-altering / multi-turn
    "focusenergy", "charge", "throatchop", "confusion", "leechseed",
    "magnetrise", "healblock", "smackdown", "imprison", "saltcure",
    "unburden", "protosynthesis", "quarkdrive", "noretreat",
)
# Offset where binary volatile flags begin in the _volatile_counters tensor.
# Preceding slots: sub_hp, stall, active_turns, yawn, flash_fire, perish_song.
BINARY_VOLATILE_OFFSET = 6


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
    # Dims derived at module load via sentinel encoding — see bottom of file
    entities = torch.zeros(n_tokens, ENTITY_FEATURE_DIM)
    species_ids = torch.zeros(n_tokens, dtype=torch.long)
    ability_ids = torch.zeros(n_tokens, dtype=torch.long)
    item_ids = torch.zeros(n_tokens, dtype=torch.long)
    move_ids = torch.zeros(n_tokens, NUM_MOVE_SLOTS, dtype=torch.long)
    belief_weight = torch.ones(n_tokens)  # 1.0 for Phase 1
    slot_id = torch.zeros(n_tokens, dtype=torch.long)

    # Encode entity tokens: my 6 first, then opponent 6 (perspective-relative order)
    token = 0
    for pokemon, sid, is_opp in ((my_pokemon, 0, False), (opp_pokemon, 1, True)):
        for mon in pokemon:
            entities[token] = _encode_pokemon_features(mon, is_opponent=is_opp)
            species_ids[token] = SPECIES_VOCAB.encode(mon.species or "")
            ability_ids[token] = ABILITY_VOCAB.encode(mon.ability or "")
            item_ids[token] = ITEM_VOCAB.encode(mon.item or "")
            move_ids[token] = _encode_move_ids(mon.moves)
            slot_id[token] = sid
            token += 1

    # Build field features
    field_tensor = _encode_field(field_data)

    # Build sides features [2, Fs]: my side first, opponent second
    sides_tensor = torch.stack([_encode_side(my_side), _encode_side(opp_side)])

    # Build scalars
    scalars_tensor = _encode_scalars(snapshot, view, perspective)

    # Build action mask
    phase = view.phase
    mask = action_space.legal_mask(view.legal.get(perspective), phase)
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


# [ENCODING-CHECKPOINT-4] Sub-encoders — add new features in the appropriate section below
# --- Entity sub-encoders (each returns a fixed-width sub-tensor) -----------


def _hp_fraction(mon: PokemonSnapshot) -> torch.Tensor:
    """HP as a fraction of max HP. [1]"""
    return torch.tensor([mon.hp / max(mon.maxhp, 1)])


def _norm_stats(mon: PokemonSnapshot) -> torch.Tensor:
    """Normalized final stats (hp, atk, def, spa, spd, spe). [6]"""
    return torch.tensor([mon.stats.get(k, 0) / MAX_STAT for k in ("hp", "atk", "def", "spa", "spd", "spe")])


def _boost_stages(mon: PokemonSnapshot) -> torch.Tensor:
    """Stat stages / 6, normalized to roughly [-1, 1]. [7]"""
    return torch.tensor([mon.boosts.get(k, 0) / 6.0 for k in ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")])


def _status_onehot(mon: PokemonSnapshot) -> torch.Tensor:
    """Status condition one-hot (brn, par, slp, frz, tox, psn, none). [7]"""
    vec = torch.zeros(NUM_STATUS)
    if mon.status and mon.status in _STATUS_MAP:
        vec[_STATUS_MAP[mon.status]] = 1.0
    else:
        vec[NUM_STATUS - 1] = 1.0  # "none" slot
    return vec


def _nature_onehot(mon: PokemonSnapshot) -> torch.Tensor:
    """Nature one-hot over 25 natures. [25]"""
    vec = torch.zeros(NUM_NATURES)
    nature_id = NATURE_VOCAB.encode(mon.nature or "")
    if 1 <= nature_id <= NUM_NATURES:
        vec[nature_id - 1] = 1.0
    return vec


def _move_pp_flags(mon: PokemonSnapshot) -> torch.Tensor:
    """Per-move pp_fraction + disabled flag (NUM_MOVE_SLOTS moves x 2). [NUM_MOVE_FEATURES]"""
    vec = torch.zeros(NUM_MOVE_FEATURES)
    for i in range(NUM_MOVE_SLOTS):
        if i < len(mon.moves):
            move = mon.moves[i]
            vec[i * 2] = move.pp / max(move.maxpp, 1)
            vec[i * 2 + 1] = 1.0 if move.disabled else 0.0
    return vec


def _volatile_counters(mon: PokemonSnapshot) -> torch.Tensor:
    """Substitute HP, stall counter, active turns, yawn, flash fire, perish song, + binary volatile flags. [29]"""
    sub_data = mon.volatileDetails.get("substitute")
    sub_hp = (sub_data.hp or 0) / MAX_STAT if sub_data and sub_data.hp is not None else 0.0
    stall_data = mon.volatileDetails.get("stall")
    stall = (stall_data.counter or 0) / 6.0 if stall_data and stall_data.counter is not None else 0.0
    active_turns = mon.activeTurns / MAX_TURNS
    # Yawn sets a 1-turn drowsy countdown; the decision-critical signal is simply "is drowsy"
    yawn = 1.0 if "yawn" in mon.volatiles else 0.0
    # Flash Fire activated: 1.5x boost to Fire moves — binary on/off
    flash_fire = 1.0 if "flashfire" in mon.volatiles else 0.0
    # Perish Song: counter normalized to [0,1] — 3=just applied, 1=faints next turn
    perish_data = mon.volatileDetails.get("perishsong")
    perish_song = (perish_data.duration or 0) / 3.0 if perish_data and perish_data.duration is not None else 0.0

    # Binary volatiles: presence/absence flags. Adding a new volatile is a data
    # change (append to this tuple) — no variable, no return-tensor edit needed.
    binary = [1.0 if name in mon.volatiles else 0.0 for name in BINARY_VOLATILES]

    return torch.tensor([sub_hp, stall, active_turns, yawn, flash_fire, perish_song] + binary)


def _slot_flags(mon: PokemonSnapshot, is_opponent: bool) -> torch.Tensor:
    """Positional + state flags (is_active, is_bench, is_fainted, item_consumed,
    physical_slot 3-way one-hot, side_flag). [8]"""
    item_consumed = bool(mon.lastItem) and not mon.item

    # Engine positions are 0-indexed: 0 = first active slot, 1 = second active slot
    pos_left = 1.0 if mon.active and mon.position == 0 else 0.0
    pos_right = 1.0 if mon.active and mon.position == 1 else 0.0
    pos_bench = 0.0 if mon.active and mon.position in (0, 1) else 1.0

    return torch.tensor(
        [
            float(mon.active),
            float(not mon.active and not mon.fainted),
            float(mon.fainted),
            float(item_consumed),
            pos_left,
            pos_right,
            pos_bench,
            float(is_opponent),
        ]
    )


def _encode_pokemon_features(mon: PokemonSnapshot, is_opponent: bool = False) -> torch.Tensor:
    """Encode a single pokemon's continuous features into a float tensor [F]."""
    return torch.cat(
        [
            _hp_fraction(mon),
            _norm_stats(mon),
            _boost_stages(mon),
            _status_onehot(mon),
            _nature_onehot(mon),
            _move_pp_flags(mon),
            _volatile_counters(mon),
            _slot_flags(mon, is_opponent),
        ]
    )


def _encode_move_ids(moves: list[MoveSnapshot]) -> torch.Tensor:
    """Encode move slot IDs into an int64 tensor [NUM_MOVE_SLOTS]."""
    ids = torch.zeros(NUM_MOVE_SLOTS, dtype=torch.long)
    for i, move in enumerate(moves[:NUM_MOVE_SLOTS]):
        ids[i] = MOVE_VOCAB.encode(move.id)
    return ids


# [ENCODING-CHECKPOINT-4] Field sub-encoders
# --- Field sub-encoders -----------------------------------------------------


def _weather_onehot_dur(field_data: FieldSnapshot) -> torch.Tensor:
    """Weather one-hot (4 slots) + normalized duration. [5]"""
    vec = torch.zeros(NUM_WEATHERS + 1)
    if field_data.weather and field_data.weather in _WEATHER_MAP:
        vec[_WEATHER_MAP[field_data.weather]] = 1.0
    vec[NUM_WEATHERS] = (field_data.weatherDuration or 0) / MAX_TURNS
    return vec


def _terrain_onehot_dur(field_data: FieldSnapshot) -> torch.Tensor:
    """Terrain one-hot (4 slots) + normalized duration. [5]"""
    vec = torch.zeros(NUM_TERRAINS + 1)
    if field_data.terrain and field_data.terrain in _TERRAIN_MAP:
        vec[_TERRAIN_MAP[field_data.terrain]] = 1.0
    vec[NUM_TERRAINS] = (field_data.terrainDuration or 0) / MAX_TURNS
    return vec


def _trick_room(field_data: FieldSnapshot) -> torch.Tensor:
    """Trick Room active flag + normalized duration. [2]"""
    tr = field_data.pseudoWeather.get("trickroom")
    return torch.tensor(
        [
            1.0 if tr else 0.0,
            (tr.duration or 0) / MAX_TURNS if tr else 0.0,
        ]
    )


def _gravity(field_data: FieldSnapshot) -> torch.Tensor:
    """Gravity active flag. [1]"""
    g = field_data.pseudoWeather.get("gravity")
    return torch.tensor([1.0 if g else 0.0])


def _encode_field(field_data: FieldSnapshot) -> torch.Tensor:
    """Encode field-level features into a float tensor [Ff]."""
    return torch.cat(
        [
            _weather_onehot_dur(field_data),
            _terrain_onehot_dur(field_data),
            _trick_room(field_data),
            _gravity(field_data),
        ]
    )


# [ENCODING-CHECKPOINT-4] Side sub-encoders
# --- Side sub-encoders ------------------------------------------------------


def _side_screen(name: str, conds: dict[str, SideConditionSnapshot]) -> torch.Tensor:
    """Encode a single duration-based side condition (active + duration). [2]"""
    entry = conds.get(name)
    return torch.tensor(
        [
            1.0 if entry is not None else 0.0,
            (entry.duration or 0) / MAX_TURNS if entry else 0.0,
        ]
    )


def _side_hazards(conds: dict[str, SideConditionSnapshot]) -> torch.Tensor:
    """Stealth Rock (binary) + Spikes (layers / 3). [2]"""
    spikes = conds.get("spikes")
    return torch.tensor(
        [
            1.0 if "stealthrock" in conds else 0.0,
            (spikes.layers or 0) / 3.0 if spikes else 0.0,
        ]
    )


def _side_mega() -> torch.Tensor:
    """Mega-used flag (stub — always 0 until mega detection is wired). [1]"""
    return torch.tensor([0.0])


def _encode_side(side_data: SideSnapshot) -> torch.Tensor:
    """Encode per-side features into a float tensor [Fs]."""
    conds = side_data.sideConditions
    return torch.cat(
        [
            _side_screen("tailwind", conds),
            _side_screen("reflect", conds),
            _side_screen("lightscreen", conds),
            _side_screen("auroraveil", conds),
            _side_hazards(conds),
            _side_mega(),
        ]
    )


# [ENCODING-CHECKPOINT-4] Scalar sub-encoders
# --- Scalar sub-encoders ----------------------------------------------------


def _turn_norm(snapshot: BattleSnapshot) -> torch.Tensor:
    """Turn number normalized by MAX_TURNS. [1]"""
    return torch.tensor([snapshot.turn / MAX_TURNS])


def _phase_onehot(view: StateView) -> torch.Tensor:
    """Phase one-hot (teamPreview, move, forceSwitch, terminal). [4]"""
    vec = torch.zeros(4)
    if view.phase in _PHASE_MAP:
        vec[_PHASE_MAP[view.phase]] = 1.0
    return vec


def _whose_decision(view: StateView, perspective: str) -> torch.Tensor:
    """Two bits: am I acting, is opponent acting. [2]"""
    opp = "p2" if perspective == "p1" else "p1"
    return torch.tensor(
        [
            1.0 if perspective in view.to_move else 0.0,
            1.0 if opp in view.to_move else 0.0,
        ]
    )


def _encode_scalars(snapshot: BattleSnapshot, view: StateView, perspective: str) -> torch.Tensor:
    """Encode global scalar features into a float tensor [Fg]."""
    return torch.cat(
        [
            _turn_norm(snapshot),
            _phase_onehot(view),
            _whose_decision(view, perspective),
        ]
    )


# --- Derived dimension constants (sentinel calls — no manual bookkeeping) ----

_EMPTY_MON = PokemonSnapshot(
    species="",
    nature="",
    level=50,
    gender="N",
    hp=0,
    maxhp=1,
    fainted=False,
    status=None,
    statusState={"stage": None, "time": None},
    ability="",
    item=None,
    lastItem=None,
    active=False,
    position=0,
    activeTurns=0,
    teraType=None,
    terastallized=None,
    stats={},
    boosts={},
    moves=[],
    volatiles=[],
    volatileDetails={},
)
_EMPTY_FIELD = FieldSnapshot(
    weather=None,
    weatherDuration=None,
    terrain=None,
    terrainDuration=None,
    pseudoWeather={},
)
_EMPTY_SIDE = SideSnapshot(id="p1", sideConditions={}, pokemon=[])
_EMPTY_SNAPSHOT = BattleSnapshot(turn=0, field=_EMPTY_FIELD, sides=[_EMPTY_SIDE, _EMPTY_SIDE])
_EMPTY_VIEW = StateView(
    phase="",
    to_move=[],
    legal={},
    snapshot=_EMPTY_SNAPSHOT,
    terminal=False,
    utility=None,
)

ENTITY_FEATURE_DIM: int = _encode_pokemon_features(_EMPTY_MON).shape[0]
FIELD_FEATURE_DIM: int = _encode_field(_EMPTY_FIELD).shape[0]
SIDE_FEATURE_DIM: int = _encode_side(_EMPTY_SIDE).shape[0]
SCALAR_FEATURE_DIM: int = _encode_scalars(_EMPTY_SNAPSHOT, _EMPTY_VIEW, "p1").shape[0]

# The encoder's schema identity, read live by ReplayBuffer's load guard (replay_buffer.py
# §6). The derived dim constants above auto-catch any *width* change. ENCODER_SCHEMA_VERSION
# is the manual companion: bump it on a semantic-but-same-width change the dims cannot see
# (e.g. reordering or re-meaning features within a fixed F), which invalidates a saved buffer.
ENCODER_SCHEMA_VERSION: int = 1
