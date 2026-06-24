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
import torch

from action_space import A, legal_mask
from encoder import encode
from search.cfr import _cfr_update_recursive
from search.expansion import _expand_turn_node, _puct_expand_one
from search.strategy import _build_policy_target, _extract_average_strategy
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

    # --- Single-legal-move skip ---
    sides = view.to_move
    single_action: dict[str, int | None] = {}
    all_single = True

    for s in sides:
        mask = legal_mask(view.legal.get(s), view.phase)
        legal_count = int(mask.sum())
        if legal_count == 1:
            single_action[s] = int(np.argmax(mask))
        elif legal_count == 0:
            single_action[s] = None
        else:
            all_single = False

    if all_single:
        # Both sides have exactly one legal action — no search needed
        strategy: dict[str, dict[int, float]] = {}
        policy_target: dict[str, np.ndarray] = {}
        for s in sides:
            if single_action[s] is not None:
                strategy[s] = {single_action[s]: 1.0}  # type: ignore[dict-item]
                pt = np.zeros(A, dtype=np.float32)
                pt[single_action[s]] = 1.0  # type: ignore[index]
                policy_target[s] = pt
            else:
                strategy[s] = {}
                policy_target[s] = np.zeros(A, dtype=np.float32)

        # Get a value estimate from CVPN for the training target
        with torch.no_grad():
            obs = encode(view, "p1")
            _, value = net(obs)
            value_p1 = float(value.item())

        return SearchResult(strategy=strategy, value=value_p1, policy_target=policy_target)

    # --- Open a search session from the provided handle ---
    root_session, root_handle, root_view = sim.open_search(from_handle=from_handle)

    try:
        # Use the cloned root's view (it should match the provided view)
        effective_view = root_view if root_view is not None else view

        # No acting sides — nothing to search
        if not sides:
            return SearchResult(strategy={}, value=0.0, policy_target={s: np.zeros(A, dtype=np.float32) for s in sides})

        root = TurnNode(
            handle=root_handle,
            view=effective_view,
            to_move=sides,
        )

        # --- Expand root ---
        _expand_turn_node(root, sim, net, root_session, config)

        # --- Main loop: alternate CFR+ updates and PUCT expansion ---
        for expansion_idx in range(config.expansion_budget):
            # Phase A: CFR+ updates over the frozen tree
            for t in range(1, config.cfr_iters_per_expansion + 1):
                iteration_num = expansion_idx * config.cfr_iters_per_expansion + t
                _cfr_update_recursive(root, iteration_num)

            # Phase B: PUCT-guided expansion of one node
            _puct_expand_one(root, sim, net, root_session, config)

        # --- Final CFR+ pass to compute the deep value ---
        # One more traversal to get the root's CFV under the converged strategy,
        # recursing through all expanded children (not just depth 1).
        final_iteration = config.expansion_budget * config.cfr_iters_per_expansion + 1
        value_p1 = _cfr_update_recursive(root, final_iteration)

        # --- Extract results ---
        strategy_out: dict[str, dict[int, float]] = {}
        policy_target_out: dict[str, np.ndarray] = {}

        for s in sides:
            strategy_out[s] = _extract_average_strategy(root, s)
            policy_target_out[s] = _build_policy_target(root, s)

    finally:
        # Clean up the search session (frees all cloned handles)
        sim.close_search(root_session)

    return SearchResult(strategy=strategy_out, value=value_p1, policy_target=policy_target_out)
