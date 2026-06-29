"""GT-CFR inner-loop search entry point for Pokemon VGC.

Implements the main search() function that orchestrates the GT-CFR loop:
  1. Build an ephemeral tree of TurnNodes (k x k grid) and ChanceNodes
  2. Alternate CFR+ regret-update traversals with PUCT-guided expansion
  3. Output the average strategy sigma-bar and a search-refined value
  4. Discard the tree — rebuilt fresh for every decision

Phase 1 pins: belief = delta (one world), scalar value head, no MCCFR sampling.
The Phase-4 upgrade un-pins these without changing this module's call pattern.

Design ref: docs/plans/search.md, docs/architecture/gt-cfr-theory.md §10.2.
Reference impl: toy_examples/leduc_poker/gt_cfr_search.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from action_space import forced_actions
from search.cfr import cfr_update_recursive, tree_value
from search.expansion import expand_turn_node, puct_expand_one
from search.strategy import build_policy_target, extract_average_strategy
from search.types import SearchConfig, SearchResult, TurnNode

if TYPE_CHECKING:
    from cvpn import CVPN
    from sim_client import SimClient
    from state_types import StateView


def search(
    view: StateView,
    sim: SimClient,
    net: CVPN,
    from_handle: int,
    config: SearchConfig | None = None,
) -> SearchResult:
    """Run GT-CFR search for one decision point.

    Args:
        view: Current StateView at the decision point.
        sim: SimClient instance.
        net: CVPN network for leaf evaluation and policy priors.
        from_handle: The battle handle to clone from for search (live battle or
            an existing clone). Search opens its own session internally.
        config: Search hyperparameters (uses defaults if None).

    Returns:
        SearchResult with the average strategy, value, and policy targets.
    """
    if config is None:
        config = SearchConfig()

    # Empty to_move means no side is acting — not a decision point.
    if not view.to_move:
        raise ValueError(
            "search() called with empty to_move. Callers must skip "
            "non-decision states before invoking search()."
        )

    # Callers must filter forced decisions before invoking search() — forced
    # roots have no strategic content and no meaningful value to return.
    if forced_actions(view.legal, view.to_move, view.phase) is not None:
        raise ValueError(
            "search() called on a forced decision (every acting side has exactly "
            "one legal action). Callers must skip forced states before invoking search()."
        )

    sides = view.to_move

    # --- Open a search session from the provided handle ---
    root_session, root_handle, root_view = sim.open_search(from_handle=from_handle)

    try:
        # Use the cloned root's view (it should match the provided view)
        effective_view = root_view if root_view is not None else view

        root = TurnNode(
            handle=root_handle,
            view=effective_view,
            to_move=sides,
        )

        # --- Expand root ---
        expand_turn_node(root, sim, net, config)

        # --- Main loop: alternate CFR+ updates and PUCT expansion ---
        for expansion_idx in range(config.expansion_budget):
            # Phase A: CFR+ updates over the frozen tree
            for t in range(1, config.cfr_iters_per_expansion + 1):
                iteration_num = expansion_idx * config.cfr_iters_per_expansion + t
                cfr_update_recursive(root, iteration_num)

            # Phase B: PUCT-guided expansion of one node
            puct_expand_one(root, sim, net, config)

        # Read-only value query under the converged strategy (no mutation)
        value_p1 = tree_value(root)

        # --- Extract results ---
        strategy_out: dict[str, dict[int, float]] = {}
        policy_target_out: dict[str, np.ndarray] = {}

        for s in sides:
            strategy_out[s] = extract_average_strategy(root, s)
            policy_target_out[s] = build_policy_target(root, s)

    finally:
        # Clean up the search session (frees all cloned handles)
        sim.close_search(root_session)

    return SearchResult(strategy=strategy_out, value=value_p1, policy_target=policy_target_out)
