"""GT-CFR inner-loop search for Pokemon VGC (Phase 1, perfect-information regime).

Package re-exports so ``from search import search, SearchConfig, ...`` keeps
working after the single-file → package refactor.
"""

from search.cfr import (
    cfr_update_recursive,
    regret_matching,
)
from search.core import search
from search.expansion import (
    expand_turn_node,
    puct_expand_one,
    puct_scores,
    puct_select_cell,
)
from search.strategy import build_policy_target, extract_average_strategy
from search.types import (
    ChanceNode,
    ChanceOutcome,
    InfoSet,
    SearchConfig,
    SearchResult,
    TurnNode,
)

__all__ = [
    "ChanceNode",
    "ChanceOutcome",
    "InfoSet",
    "SearchConfig",
    "SearchResult",
    "TurnNode",
    "build_policy_target",
    "cfr_update_recursive",
    "expand_turn_node",
    "extract_average_strategy",
    "puct_expand_one",
    "puct_scores",
    "puct_select_cell",
    "regret_matching",
    "search",
]
