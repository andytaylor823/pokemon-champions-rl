"""Streamlit dashboard for interactive Pokemon build clustering.

pylint: disable=invalid-name
  Streamlit scripts use module-level widget values that aren't true constants.
"""
# pylint: disable=invalid-name,wrong-import-position,redefined-outer-name

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from meta_priors.clustering import (
    ArchetypeSummary,
    DistanceWeights,
    auto_cluster,
    cluster_sets,
    compute_distance_matrix,
    summarize_archetypes,
)
from meta_priors.data_loader import PokemonSet, load_all_tournaments

_SPRITES_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "sprites"


def _sprite_path(species_id: str) -> Path | None:
    """Resolve a species ID to its local sprite PNG, with base-form fallback."""
    sid = re.sub(r"[^a-z0-9-]", "", species_id.lower())
    exact = _SPRITES_DIR / f"{sid}.png"
    if exact.exists():
        return exact
    base = re.sub(r"[^a-z0-9]", "", species_id.lower())
    base_path = _SPRITES_DIR / f"{base}.png"
    if base_path.exists():
        return base_path
    return None

st.set_page_config(page_title="Build Clustering Explorer", layout="wide")


def _fmt_pct(value: float) -> str:
    """Format a fraction as a percentage, using one decimal place below 10%."""
    if value < 0.1:
        return f"{value:.1%}"
    return f"{value:.0%}"


@st.cache_data
def get_data() -> tuple[dict[str, list[dict]], list[str]]:
    """Load tournament data once, return serialisable dicts + sorted species."""
    _, species_index = load_all_tournaments()
    serialised = {
        species: [
            {
                "species": p.species,
                "item": p.item,
                "ability": p.ability,
                "moves": sorted(p.moves),
                "teammates": list(p.teammates),
                "player": p.player,
                "placing": p.placing,
                "games_played": p.games_played,
            }
            for p in sets
        ]
        for species, sets in species_index.items()
    }
    ordered = sorted(
        species_index.keys(),
        key=lambda s: -sum(p["games_played"] for p in serialised[s]),
    )
    return serialised, ordered


def _rebuild_sets(raw: list[dict]) -> list[PokemonSet]:
    return [
        PokemonSet(
            species=r["species"],
            item=r["item"],
            ability=r["ability"],
            moves=frozenset(r["moves"]),
            teammates=tuple(r["teammates"]),
            player=r["player"],
            placing=r["placing"],
            games_played=r.get("games_played", 1),
        )
        for r in raw
    ]


species_data, sorted_species = get_data()

# ── Sidebar controls ──────────────────────────────────────────────────────────

st.sidebar.header("Species")
def _species_label(s: str) -> str:
    games = sum(p["games_played"] for p in species_data[s])
    return f"{s} ({games} games / {len(species_data[s])} sets)"

search_query = st.sidebar.text_input(
    "Search Pokemon", placeholder="Type to filter…"
)
if search_query:
    filtered_species = [
        s for s in sorted_species if search_query.lower() in s.lower()
    ]
else:
    filtered_species = sorted_species

if not filtered_species:
    st.sidebar.warning("No Pokemon match your search.")
    st.stop()

species = st.sidebar.selectbox(
    "Select Pokemon",
    filtered_species,
    format_func=_species_label,
)

sprite = _sprite_path(species)
if sprite:
    st.sidebar.image(str(sprite), width=96)

n_sets = len(species_data[species])
n_games = sum(p["games_played"] for p in species_data[species])
st.sidebar.caption(f"{n_games} game-appearances from {n_sets} players")

st.sidebar.divider()
st.sidebar.header("Distance Weights")
w_moves = st.sidebar.slider("Moves (Jaccard)", 0.0, 3.0, value=1.0, step=0.1)
w_item = st.sidebar.slider("Item", 0.0, 3.0, value=1.5, step=0.1)
w_ability = st.sidebar.slider("Ability", 0.0, 3.0, value=0.1, step=0.1)

st.sidebar.divider()
st.sidebar.header("Clustering")

method = st.sidebar.radio("Method", ["agglomerative", "dbscan"], horizontal=True)
linkage = "complete"

if method == "agglomerative":
    linkage = st.sidebar.selectbox("Linkage", ["complete", "average", "single"])
    auto_k = st.sidebar.toggle("Auto-detect k (silhouette)", value=True)
else:
    auto_k = False

# ── Compute clusters ──────────────────────────────────────────────────────────

sets = _rebuild_sets(species_data[species])
weights = DistanceWeights(moves=w_moves, item=w_item, ability=w_ability)

if method == "agglomerative" and auto_k:
    labels, archetypes = auto_cluster(sets, weights, linkage=linkage)
    st.sidebar.caption(f"{len(archetypes)} archetype(s) via hard-partition + auto-k")
elif method == "agglomerative":
    dist_matrix = compute_distance_matrix(sets, weights)
    max_k = min(8, n_sets)
    n_clusters = st.sidebar.slider(
        "Number of clusters", 1, max_k, value=min(3, max_k)
    )
    labels = (
        np.zeros(len(sets), dtype=int)
        if n_clusters == 1
        else cluster_sets(
            dist_matrix,
            method="agglomerative",
            n_clusters=n_clusters,
            linkage=linkage,
        )
    )
    archetypes = summarize_archetypes(sets, labels)
else:
    dist_matrix = compute_distance_matrix(sets, weights)
    eps = st.sidebar.slider("DBSCAN eps", 0.1, 3.0, value=0.8, step=0.1)
    labels = cluster_sets(dist_matrix, method="dbscan", eps=eps)
    archetypes = summarize_archetypes(sets, labels)

# ── Page header with sprite ──────────────────────────────────────────────────

header_cols = st.columns([1, 6])
if sprite:
    with header_cols[0]:
        st.image(str(sprite), width=120)
with header_cols[1]:
    st.title(species)
    st.caption(
        f"{n_games} game-appearances from {n_sets} players  ·  "
        f"{len(archetypes)} archetype{'s' if len(archetypes) != 1 else ''}"
    )


# ── Interactive prior distribution ─────────────────────────────────────────────

all_species_moves = sorted({mv for s in sets for mv in s.moves})
all_species_items = sorted({s.item for s in sets})
all_species_abilities = sorted({s.ability for s in sets})

st.markdown("#### Meta Prior")

col_mv, col_it, col_ab = st.columns(3)

with col_mv:
    pinned_moves: list[str] = st.pills(
        "Known Moves",
        all_species_moves,
        selection_mode="multi",
        key=f"pin_mv_{species}",
    )

with col_it:
    pinned_item: str | None = st.pills(
        "Held Item",
        all_species_items,
        selection_mode="single",
        key=f"pin_it_{species}",
    )

with col_ab:
    pinned_ability: str | None = st.pills(
        "Ability",
        all_species_abilities,
        selection_mode="single",
        key=f"pin_ab_{species}",
    )

filtered_mask = np.ones(len(sets), dtype=bool)
if pinned_item is not None:
    filtered_mask &= np.array([s.item == pinned_item for s in sets])
if pinned_ability is not None:
    filtered_mask &= np.array([s.ability == pinned_ability for s in sets])
for mv in pinned_moves or []:
    filtered_mask &= np.array([mv in s.moves for s in sets])

filtered = [s for s, m in zip(sets, filtered_mask) if m]
has_observation = (
    pinned_item is not None or bool(pinned_moves) or pinned_ability is not None
)

if filtered:
    total_w = sum(p.games_played for p in filtered)
    move_cts: Counter[str] = Counter()
    item_cts: Counter[str] = Counter()
    ability_cts: Counter[str] = Counter()
    for p in filtered:
        w = p.games_played
        for mv in p.moves:
            move_cts[mv] += w
        item_cts[p.item] += w
        ability_cts[p.ability] += w

    with col_mv:
        for mv, c in move_cts.most_common():
            freq = c / total_w
            bar_len = int(freq * 20)
            st.text(
                f"{'█' * bar_len}{'░' * (20 - bar_len)} "
                f"{_fmt_pct(freq):>5}  {mv}"
            )
    with col_it:
        for it_name, c in item_cts.most_common():
            st.text(f"{_fmt_pct(c / total_w):>5}  {it_name}")
    with col_ab:
        for ab_name, c in ability_cts.most_common():
            st.text(f"{_fmt_pct(c / total_w):>5}  {ab_name}")

    if has_observation:
        st.caption(
            f"Conditional on pinned observations: "
            f"{total_w} game-appearances from {len(filtered)} players"
        )
else:
    st.warning("No sets match all pinned observations.")

prior_weights = np.array([a.frequency for a in archetypes])
if has_observation and filtered:
    label_to_arch = {a.label: idx for idx, a in enumerate(archetypes)}
    arch_games = np.zeros(len(archetypes))
    for i in np.where(filtered_mask)[0]:
        lbl = int(labels[i])
        if lbl in label_to_arch:
            arch_games[label_to_arch[lbl]] += sets[i].games_played
    total_arch = arch_games.sum()
    posterior = arch_games / total_arch if total_arch > 0 else prior_weights

    st.markdown("**Archetype Weights**")
    weight_cols = st.columns(len(archetypes))
    for wt_col, arch, prior_w, post_w in zip(
        weight_cols, archetypes, prior_weights, posterior
    ):
        label = (
            arch.partition_key
            if arch.partition_key is not None
            else f"Archetype {arch.label + 1}"
        )
        delta = float(post_w - prior_w)
        delta_str = f"{delta:+.1%}" if abs(delta) >= 0.005 else None
        with wt_col:
            st.metric(label, _fmt_pct(float(post_w)), delta_str)
else:
    posterior = prior_weights

st.divider()

# ── Archetype cards ───────────────────────────────────────────────────────────


def _top_teammates(
    cluster_sets_list: list[PokemonSet], top_n: int = 6
) -> list[tuple[str, float]]:
    """Return the most common teammates weighted by games played."""
    counts: Counter[str] = Counter()
    for member in cluster_sets_list:
        w = member.games_played
        for mate in member.teammates:
            counts[mate] += w
    weighted_total = sum(m.games_played for m in cluster_sets_list)
    return [
        (mate, c / weighted_total) for mate, c in counts.most_common(top_n)
    ] if weighted_total else []


def _render_archetype(  # pylint: disable=too-many-locals
    archetype: ArchetypeSummary,
    cluster_members: list[PokemonSet],
    posterior_weight: float | None = None,
) -> None:
    if archetype.label == -1:
        label_text = "Noise"
    elif archetype.partition_key is not None:
        label_text = archetype.partition_key
    else:
        label_text = f"Archetype {archetype.label + 1}"
    if posterior_weight is not None:
        weight_text = (
            f"{_fmt_pct(archetype.frequency)} \u2192 {_fmt_pct(posterior_weight)}"
        )
    else:
        weight_text = _fmt_pct(archetype.frequency)
    st.subheader(
        f"{label_text}  \u2014  {archetype.count} games / "
        f"{archetype.raw_count} players ({weight_text})"
    )

    col_moves, col_item, col_ability = st.columns(3)

    with col_moves:
        st.markdown("**Moves**")
        for move, freq in archetype.move_frequencies.items():
            bar_len = int(freq * 20)
            pct = _fmt_pct(freq)
            st.text(f"{'█' * bar_len}{'░' * (20 - bar_len)} {pct:>5}  {move}")

    with col_item:
        st.markdown("**Item**")
        for item_name, freq in archetype.item_distribution.items():
            st.text(f"{_fmt_pct(freq):>5}  {item_name}")

    with col_ability:
        st.markdown("**Ability**")
        for ability_name, freq in archetype.ability_distribution.items():
            st.text(f"{_fmt_pct(freq):>5}  {ability_name}")

    teammates = _top_teammates(cluster_members)
    if teammates:
        st.markdown("**Top Teammates**")
        mate_cols = st.columns(len(teammates))
        for col, (mate, freq) in zip(mate_cols, teammates):
            mate_sprite = _sprite_path(mate)
            with col:
                if mate_sprite:
                    st.image(str(mate_sprite), width=48)
                st.caption(f"{mate}\n{_fmt_pct(freq)}")


clusters_by_label: dict[int, list[PokemonSet]] = {}
for pset, label in zip(sets, labels):
    clusters_by_label.setdefault(int(label), []).append(pset)

for _arch_idx, _arch in enumerate(archetypes):
    _pw = float(posterior[_arch_idx]) if has_observation else None
    _render_archetype(
        _arch, clusters_by_label.get(_arch.label, []), posterior_weight=_pw
    )
    st.divider()

# ── Raw sets table ────────────────────────────────────────────────────────────

with st.expander("Raw sets table"):
    rows = []
    for pset, label in zip(sets, labels):
        rows.append(
            {
                "Cluster": int(label) + 1,
                "Player": pset.player,
                "Games": pset.games_played,
                "Item": pset.item,
                "Ability": pset.ability,
                "Moves": ", ".join(sorted(pset.moves)),
            }
        )
    df = pd.DataFrame(rows).sort_values(["Cluster", "Item", "Moves"])
    st.dataframe(df, width="stretch", hide_index=True)

# ── Distance heatmap ──────────────────────────────────────────────────────────

with st.expander("Distance heatmap"):
    heatmap_dm = compute_distance_matrix(sets, weights)
    sort_idx = labels.argsort()
    sorted_dist = heatmap_dm[sort_idx][:, sort_idx]
    fig = px.imshow(
        sorted_dist,
        color_continuous_scale="Viridis",
        labels={"color": "Distance"},
        aspect="equal",
    )
    fig.update_layout(
        xaxis_title="Set index (sorted by cluster)",
        yaxis_title="Set index (sorted by cluster)",
        height=500,
    )
    st.plotly_chart(fig, width="stretch")
