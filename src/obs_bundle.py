"""ObsBundle — TensorDict-based observation bundle for the CVPN.

The ObsBundle is the tensor contract between the Encoder and the neural network.
It wraps a TensorDict with named fields for entity tokens, categorical IDs,
field/side/scalar features, action mask, and belief weights.

Provides factory construction and batched collation with automatic padding.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

# Typing alias for clarity
ObsBundle = TensorDict


def make_obs_bundle(
    entities: torch.Tensor,
    species_ids: torch.Tensor,
    ability_ids: torch.Tensor,
    item_ids: torch.Tensor,
    move_ids: torch.Tensor,
    belief_weight: torch.Tensor,
    slot_id: torch.Tensor,
    field: torch.Tensor,
    sides: torch.Tensor,
    scalars: torch.Tensor,
    action_mask: torch.Tensor,
    padding_mask: torch.Tensor,
) -> ObsBundle:
    """Construct an ObsBundle TensorDict from raw tensors.

    Args:
        entities: float [N, F] — continuous features per entity token
        species_ids: int64 [N] — species vocab index for embedding
        ability_ids: int64 [N] — ability vocab index for embedding
        item_ids: int64 [N] — item vocab index for embedding
        move_ids: int64 [N, 4] — move vocab indices for embedding
        belief_weight: float [N] — per-token belief weight (1.0 in Phase 1)
        slot_id: int [N] — grouping tag for candidates
        field: float [Ff] — weather/terrain/pseudo-weather features
        sides: float [2, Fs] — per-side features (tailwind, screens, hazards, etc.)
        scalars: float [Fg] — global scalar features (turn, phase, etc.)
        action_mask: bool [A] — legal action mask
        padding_mask: bool [N] — true = real token, false = padding
    """
    return TensorDict(
        {
            "entities": entities,
            "ids": TensorDict(
                {
                    "species": species_ids,
                    "ability": ability_ids,
                    "item": item_ids,
                    "moves": move_ids,
                },
                batch_size=[],
            ),
            "belief_weight": belief_weight,
            "slot_id": slot_id,
            "field": field,
            "sides": sides,
            "scalars": scalars,
            "action_mask": action_mask,
            "padding_mask": padding_mask,
        },
        batch_size=[],
    )


def collate_obs_bundles(bundles: list[ObsBundle]) -> ObsBundle:
    """Collate a list of ObsBundles into a batched TensorDict with padding.

    Pads the entity axis (N) to the max N in the batch. Sets padding_mask
    to False for padded positions. All other tensors are stacked directly.

    Returns:
        Batched ObsBundle with batch_size=[B].
    """
    if not bundles:
        raise ValueError("Cannot collate an empty list of ObsBundles")

    batch_size = len(bundles)
    max_n = max(b["entities"].shape[0] for b in bundles)

    # Determine feature dimensions from first bundle
    f_dim = bundles[0]["entities"].shape[1]
    move_slots = bundles[0]["ids", "moves"].shape[1]

    # Pre-allocate padded tensors
    entities = torch.zeros(batch_size, max_n, f_dim)
    species_ids = torch.zeros(batch_size, max_n, dtype=torch.long)
    ability_ids = torch.zeros(batch_size, max_n, dtype=torch.long)
    item_ids = torch.zeros(batch_size, max_n, dtype=torch.long)
    move_ids_t = torch.zeros(batch_size, max_n, move_slots, dtype=torch.long)
    belief_weight = torch.zeros(batch_size, max_n)
    slot_id = torch.zeros(batch_size, max_n, dtype=torch.long)
    padding_mask = torch.zeros(batch_size, max_n, dtype=torch.bool)

    # Stack non-entity tensors directly
    fields = torch.stack([b["field"] for b in bundles])
    sides_t = torch.stack([b["sides"] for b in bundles])
    scalars_t = torch.stack([b["scalars"] for b in bundles])
    action_masks = torch.stack([b["action_mask"] for b in bundles])

    # Fill padded entity tensors
    for i, b in enumerate(bundles):
        n = b["entities"].shape[0]
        entities[i, :n] = b["entities"]
        species_ids[i, :n] = b["ids", "species"]
        ability_ids[i, :n] = b["ids", "ability"]
        item_ids[i, :n] = b["ids", "item"]
        move_ids_t[i, :n] = b["ids", "moves"]
        belief_weight[i, :n] = b["belief_weight"]
        slot_id[i, :n] = b["slot_id"]
        padding_mask[i, :n] = True  # real tokens

    return TensorDict(
        {
            "entities": entities,
            "ids": TensorDict(
                {
                    "species": species_ids,
                    "ability": ability_ids,
                    "item": item_ids,
                    "moves": move_ids_t,
                },
                batch_size=[batch_size],
            ),
            "belief_weight": belief_weight,
            "slot_id": slot_id,
            "field": fields,
            "sides": sides_t,
            "scalars": scalars_t,
            "action_mask": action_masks,
            "padding_mask": padding_mask,
        },
        batch_size=[batch_size],
    )
