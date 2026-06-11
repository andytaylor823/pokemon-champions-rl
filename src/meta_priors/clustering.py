"""Per-species build clustering with configurable distance weights."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score

from meta_priors.data_loader import PokemonSet
from meta_priors.legality import _load_list, _LEGAL_DIR

SILHOUETTE_THRESHOLD = 0.25
WITHIN_PARTITION_THRESHOLD = 0.50
MIN_GAIN = 0.08
MIN_CLUSTER_FRACTION = 0.03
MINOR_ITEM_DISTANCE = 0.25

MEGA_STONES: frozenset[str] = frozenset(
    item
    for item in _load_list(_LEGAL_DIR / "items.txt")
    if item.endswith("ite") or item.endswith("ite X") or item.endswith("ite Y")
)

SET_DEFINING_ITEMS: frozenset[str] = MEGA_STONES | {"Choice Scarf"}


@dataclass
class DistanceWeights:
    moves: float = 1.0
    item: float = 1.5
    ability: float = 0.1


def _is_mega(pset: PokemonSet) -> bool:
    """True if the set holds a Mega Stone or the species is already a mega form."""
    return pset.item in MEGA_STONES or "-mega" in pset.species


def _item_distance(item_a: str, item_b: str) -> float:
    """Tiered item distance: full weight for set-defining items, reduced otherwise.

    Set-defining items (Choice Scarf, Mega Stones) fundamentally change how a
    Pokemon is played.  Swapping between non-defining items (berries, type
    boosters, Focus Sash, etc.) is a tuning choice within the same archetype.
    """
    if item_a == item_b:
        return 0.0
    if item_a in SET_DEFINING_ITEMS or item_b in SET_DEFINING_ITEMS:
        return 1.0
    return MINOR_ITEM_DISTANCE


def _jaccard_distance(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 0.0
    return 1.0 - len(a & b) / len(a | b)


def compute_distance_matrix(
    sets: list[PokemonSet], weights: DistanceWeights
) -> np.ndarray:
    n = len(sets)
    mega_flags = [_is_mega(s) for s in sets]
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            ability_w = (
                0.0 if mega_flags[i] and mega_flags[j] else weights.ability
            )
            d = (
                weights.moves * _jaccard_distance(sets[i].moves, sets[j].moves)
                + weights.item * _item_distance(sets[i].item, sets[j].item)
                + ability_w
                * (0.0 if sets[i].ability == sets[j].ability else 1.0)
            )
            dist[i, j] = d
            dist[j, i] = d
    return dist


def cluster_sets(
    distance_matrix: np.ndarray,
    method: str = "agglomerative",
    n_clusters: int = 3,
    linkage: str = "complete",
    eps: float = 0.5,
) -> np.ndarray:
    """Return integer cluster labels for each set.

    method: 'agglomerative' or 'dbscan'
    """
    if method == "agglomerative":
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="precomputed",
            linkage=linkage,
        )
    elif method == "dbscan":
        model = DBSCAN(metric="precomputed", eps=eps, min_samples=2)
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    return model.fit_predict(distance_matrix)


def find_best_k(  # pylint: disable=too-many-arguments
    distance_matrix: np.ndarray,
    *,
    linkage: str = "complete",
    max_k: int = 8,
    threshold: float = SILHOUETTE_THRESHOLD,
    min_gain: float = MIN_GAIN,
    min_cluster_fraction: float = MIN_CLUSTER_FRACTION,
) -> tuple[int, dict[int, float]]:
    """Pick the best number of clusters via silhouette with parsimony.

    Walks k=2..max_k.  Starts with the smallest valid k and only moves
    to k+1 if the silhouette score improves by at least *min_gain*.
    This favours the simplest k that captures the major splits.

    A candidate k is rejected if any resulting cluster has fewer than
    *min_cluster_fraction* of the samples (absolute floor of 1).
    Combined with the silhouette threshold this prevents peeling off a
    handful of outlier sets into a phantom archetype.

    If no valid k meets the *threshold*, returns k=1 (one dominant build).

    Returns (best_k, {k: score} for all k evaluated).
    """
    n = distance_matrix.shape[0]
    max_k = min(max_k, n - 1)
    if max_k < 2:
        return 1, {}

    min_size = max(min_cluster_fraction * n, 1)
    scores: dict[int, float] = {}
    for k in range(2, max_k + 1):
        labels = cluster_sets(
            distance_matrix, method="agglomerative", n_clusters=k, linkage=linkage
        )
        if len(set(labels)) < 2:
            continue
        cluster_sizes = Counter(int(lbl) for lbl in labels)
        if any(size < min_size for size in cluster_sizes.values()):
            continue
        scores[k] = float(
            silhouette_score(distance_matrix, labels, metric="precomputed")
        )

    if not scores:
        return 1, scores

    smallest_valid_k = min(scores)
    if scores[smallest_valid_k] < threshold:
        return 1, scores

    best_k = smallest_valid_k
    for k in sorted(scores):
        if k == smallest_valid_k:
            continue
        if scores[k] - scores[best_k] >= min_gain:
            best_k = k

    return best_k, scores


@dataclass
class ArchetypeSummary:  # pylint: disable=too-many-instance-attributes
    label: int
    count: int
    frequency: float
    raw_count: int = 0
    partition_key: str | None = None
    move_frequencies: dict[str, float] = field(default_factory=dict)
    item_distribution: dict[str, float] = field(default_factory=dict)
    ability_distribution: dict[str, float] = field(default_factory=dict)


def summarize_archetypes(  # pylint: disable=too-many-locals
    sets: list[PokemonSet],
    labels: np.ndarray,
    partition_key: str | None = None,
) -> list[ArchetypeSummary]:
    weighted_total = sum(p.games_played for p in sets)
    clusters: dict[int, list[PokemonSet]] = {}
    for pset, label in zip(sets, labels):
        clusters.setdefault(int(label), []).append(pset)

    summaries = []
    for label in sorted(clusters):
        members = clusters[label]
        raw_count = len(members)
        weighted_count = sum(m.games_played for m in members)

        move_counts: Counter[str] = Counter()
        item_counts: Counter[str] = Counter()
        ability_counts: Counter[str] = Counter()

        for m in members:
            w = m.games_played
            for mv in m.moves:
                move_counts[mv] += w
            item_counts[m.item] += w
            ability_counts[m.ability] += w

        summaries.append(
            ArchetypeSummary(
                label=label,
                count=weighted_count,
                raw_count=raw_count,
                frequency=weighted_count / weighted_total if weighted_total else 0.0,
                partition_key=partition_key,
                move_frequencies={
                    mv: c / weighted_count
                    for mv, c in move_counts.most_common()
                },
                item_distribution={
                    it: c / weighted_count
                    for it, c in item_counts.most_common()
                },
                ability_distribution={
                    ab: c / weighted_count
                    for ab, c in ability_counts.most_common()
                },
            )
        )

    return summaries


# ---------------------------------------------------------------------------
# Hard-partition pipeline
# ---------------------------------------------------------------------------


def partition_by_item(
    sets: list[PokemonSet],
) -> dict[str | None, list[PokemonSet]]:
    """Group sets by set-defining item (Mega Stones, Choice Scarf).

    Returns a dict mapping the item name to its sets.  Sets without a
    set-defining item are keyed under ``None``.
    """
    partitions: dict[str | None, list[PokemonSet]] = {}
    for pset in sets:
        key = pset.item if pset.item in SET_DEFINING_ITEMS else None
        partitions.setdefault(key, []).append(pset)
    return partitions


def auto_cluster(  # pylint: disable=too-many-locals
    sets: list[PokemonSet],
    weights: DistanceWeights,
    *,
    linkage: str = "complete",
) -> tuple[np.ndarray, list[ArchetypeSummary]]:
    """Hard-partition by set-defining item, then cluster within partitions.

    Returns (labels, archetypes) where *labels* has one int per set in
    the original *sets* order and *archetypes* has globally-unique label
    ids and informative ``partition_key`` values.
    """
    partitions = partition_by_item(sets)
    set_id_to_label: dict[int, int] = {}
    all_archetypes: list[ArchetypeSummary] = []
    weighted_total = sum(p.games_played for p in sets)
    label_offset = 0

    for part_key in sorted(partitions, key=lambda k: (k is None, k or "")):
        part_sets = partitions[part_key]
        n = len(part_sets)

        if part_key is not None or n < 2:
            # Set-defining item partitions → single archetype each.
            # The hard partition already captured the meaningful split.
            for pset in part_sets:
                set_id_to_label[id(pset)] = label_offset
            for arch in summarize_archetypes(
                part_sets,
                np.zeros(n, dtype=int) + label_offset,
                partition_key=part_key,
            ):
                arch.frequency = (
                    arch.count / weighted_total if weighted_total else 0.0
                )
                all_archetypes.append(arch)
            label_offset += 1
            continue

        # "Other" partition: sub-cluster by moves + minor items.
        # Uses a higher silhouette threshold because the distance range is
        # compressed (no set-defining item distances).
        dm = compute_distance_matrix(part_sets, weights)
        best_k, _ = find_best_k(
            dm, linkage=linkage, threshold=WITHIN_PARTITION_THRESHOLD
        )

        if best_k == 1:
            local_labels = np.zeros(n, dtype=int)
        else:
            local_labels = cluster_sets(
                dm, method="agglomerative", n_clusters=best_k, linkage=linkage
            )

        global_labels = local_labels + label_offset
        for pset, glbl in zip(part_sets, global_labels):
            set_id_to_label[id(pset)] = int(glbl)

        for arch in summarize_archetypes(part_sets, global_labels, partition_key=part_key):
            arch.frequency = arch.count / weighted_total if weighted_total else 0.0
            all_archetypes.append(arch)

        label_offset += int(local_labels.max()) + 1

    combined_labels = np.array([set_id_to_label[id(s)] for s in sets])
    return combined_labels, all_archetypes
