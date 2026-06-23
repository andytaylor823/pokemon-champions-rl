"""GT-CFR inner-loop search for Pokemon VGC (Phase 1, perfect-information regime).

Package re-exports so ``from search import search, SearchConfig, ...`` keeps
working after the single-file → package refactor.
"""

from search.cfr import (
    _cfr_update,
    _cfr_update_recursive,
    _expanded_children,
    _regret_matching,
)
from search.core import search
from search.expansion import (
    _puct_scores,
    _puct_scores_with_config,
    _puct_select_cell,
)
from search.strategy import _build_policy_target, _extract_average_strategy
from search.types import (
    ChanceNode,
    NodeKind,
    SearchConfig,
    SearchResult,
    Side,
    TurnNode,
)

__all__ = [
    "ChanceNode",
    "NodeKind",
    "SearchConfig",
    "SearchResult",
    "Side",
    "TurnNode",
    "_build_policy_target",
    "_cfr_update",
    "_cfr_update_recursive",
    "_expanded_children",
    "_extract_average_strategy",
    "_puct_scores",
    "_puct_scores_with_config",
    "_puct_select_cell",
    "_regret_matching",
    "search",
]
