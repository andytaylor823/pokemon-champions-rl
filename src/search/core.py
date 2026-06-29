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

import numpy as np

from action_space import A, legal_mask
from search.cfr import cfr_update_recursive, tree_value
from search.expansion import expand_turn_node, puct_expand_one
from search.strategy import build_policy_target, extract_average_strategy
from search.types import SearchConfig, SearchResult, TurnNode

if TYPE_CHECKING:
    from cvpn import CVPN
    from sim_client import SimClient
    from state_types import StateView


def _single_move_result(
    view: StateView,
) -> SearchResult | None:
    """Short-circuit when every acting side has exactly one legal action.

    A fully-forced decision carries no strategic choice — its value is
    determined entirely by its successors. No CVPN forward, no regret tables,
    no training tuple (self-play.md §2.2). Value is returned as 0.0 because
    SelfPlay never calls search() on forced decisions; callers that need a
    value (e.g. evaluation) should skip forced states or evaluate separately.

    Returns a SearchResult with deterministic strategy, or None if any side
    has more than one legal action (meaning real search is needed).
    """
    sides = view.to_move
    forced_actions: dict[str, int] = {}

    for s in sides:
        mask = legal_mask(view.legal.get(s), view.phase)
        legal_count = int(mask.sum())
        if legal_count == 1:
            forced_actions[s] = int(np.argmax(mask))
        else:
            assert legal_count > 0, f"Side {s} in to_move has 0 legal actions"
            return None

    strategy: dict[str, dict[int, float]] = {}
    policy_target: dict[str, np.ndarray] = {}
    for s in sides:
        action_idx = forced_actions[s]
        strategy[s] = {action_idx: 1.0}
        pt = np.zeros(A, dtype=np.float32)
        pt[action_idx] = 1.0
        policy_target[s] = pt

    # No CVPN call — forced decisions have no strategic content (self-play.md §2.2, §11)
    return SearchResult(
        strategy=strategy,
        value=0.0,
        policy_target=policy_target,
    )


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

    skip = _single_move_result(view)
    if skip is not None:
        return skip

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
