"""Unit tests for the obs_bundle module — TensorDict construction and collation."""
from __future__ import annotations

import pytest
import torch

from obs_bundle import collate_obs_bundles, make_obs_bundle


def _make_dummy_bundle(n_tokens: int, f_dim: int = 8, a_dim: int = 1036) -> dict:
    """Create dummy tensors with known shapes for a single ObsBundle."""
    return {
        "entities": torch.randn(n_tokens, f_dim),
        "species_ids": torch.randint(0, 100, (n_tokens,)),
        "ability_ids": torch.randint(0, 50, (n_tokens,)),
        "item_ids": torch.randint(0, 80, (n_tokens,)),
        "move_ids": torch.randint(0, 200, (n_tokens, 4)),
        "belief_weight": torch.ones(n_tokens),
        "slot_id": torch.zeros(n_tokens, dtype=torch.long),
        "field": torch.randn(15),
        "sides": torch.randn(2, 11),
        "scalars": torch.randn(7),
        "action_mask": torch.randint(0, 2, (a_dim,), dtype=torch.bool),
        "padding_mask": torch.ones(n_tokens, dtype=torch.bool),
    }


# ---------------------------------------------------------------------------
# make_obs_bundle
# ---------------------------------------------------------------------------


class TestMakeObsBundle:
    """Test factory construction of ObsBundle TensorDicts."""

    def test_all_keys_present(self):
        args = _make_dummy_bundle(12)
        bundle = make_obs_bundle(**args)
        assert "entities" in bundle.keys()
        assert "ids" in bundle.keys()
        assert "belief_weight" in bundle.keys()
        assert "slot_id" in bundle.keys()
        assert "field" in bundle.keys()
        assert "sides" in bundle.keys()
        assert "scalars" in bundle.keys()
        assert "action_mask" in bundle.keys()
        assert "padding_mask" in bundle.keys()

    def test_nested_ids_present(self):
        args = _make_dummy_bundle(12)
        bundle = make_obs_bundle(**args)
        assert "species" in bundle["ids"].keys()
        assert "ability" in bundle["ids"].keys()
        assert "item" in bundle["ids"].keys()
        assert "moves" in bundle["ids"].keys()

    def test_entity_shape(self):
        args = _make_dummy_bundle(12, f_dim=65)
        bundle = make_obs_bundle(**args)
        assert bundle["entities"].shape == (12, 65)

    def test_categorical_id_shapes(self):
        args = _make_dummy_bundle(12)
        bundle = make_obs_bundle(**args)
        assert bundle["ids", "species"].shape == (12,)
        assert bundle["ids", "moves"].shape == (12, 4)

    def test_action_mask_shape(self):
        args = _make_dummy_bundle(12, a_dim=1036)
        bundle = make_obs_bundle(**args)
        assert bundle["action_mask"].shape == (1036,)

    def test_field_and_scalars_shapes(self):
        args = _make_dummy_bundle(12)
        bundle = make_obs_bundle(**args)
        assert bundle["field"].shape == (15,)
        assert bundle["sides"].shape == (2, 11)
        assert bundle["scalars"].shape == (7,)

    def test_padding_mask_all_true(self):
        args = _make_dummy_bundle(12)
        bundle = make_obs_bundle(**args)
        assert bundle["padding_mask"].all()


# ---------------------------------------------------------------------------
# collate_obs_bundles — uniform N
# ---------------------------------------------------------------------------


class TestCollateUniform:
    """Test collation of bundles with the same token count."""

    def test_batch_size_correct(self):
        bundles = [make_obs_bundle(**_make_dummy_bundle(12)) for _ in range(4)]
        batched = collate_obs_bundles(bundles)
        assert batched.batch_size == torch.Size([4])

    def test_entity_shape_batched(self):
        bundles = [make_obs_bundle(**_make_dummy_bundle(12, f_dim=8)) for _ in range(3)]
        batched = collate_obs_bundles(bundles)
        assert batched["entities"].shape == (3, 12, 8)

    def test_ids_shape_batched(self):
        bundles = [make_obs_bundle(**_make_dummy_bundle(12)) for _ in range(3)]
        batched = collate_obs_bundles(bundles)
        assert batched["ids", "species"].shape == (3, 12)
        assert batched["ids", "moves"].shape == (3, 12, 4)

    def test_action_mask_stacked(self):
        bundles = [make_obs_bundle(**_make_dummy_bundle(12)) for _ in range(3)]
        batched = collate_obs_bundles(bundles)
        assert batched["action_mask"].shape == (3, 1036)

    def test_padding_mask_all_true_uniform(self):
        bundles = [make_obs_bundle(**_make_dummy_bundle(12)) for _ in range(3)]
        batched = collate_obs_bundles(bundles)
        # All tokens are real, no padding needed when uniform
        assert batched["padding_mask"].all()


# ---------------------------------------------------------------------------
# collate_obs_bundles — variable N (padding)
# ---------------------------------------------------------------------------


class TestCollateWithPadding:
    """Test collation with variable token counts, verifying padding behavior."""

    def test_pads_to_max_n(self):
        b1 = make_obs_bundle(**_make_dummy_bundle(8, f_dim=8))
        b2 = make_obs_bundle(**_make_dummy_bundle(12, f_dim=8))
        batched = collate_obs_bundles([b1, b2])
        # Entity axis should be padded to max_n=12
        assert batched["entities"].shape == (2, 12, 8)

    def test_padding_mask_false_for_padded_positions(self):
        b1 = make_obs_bundle(**_make_dummy_bundle(8, f_dim=8))
        b2 = make_obs_bundle(**_make_dummy_bundle(12, f_dim=8))
        batched = collate_obs_bundles([b1, b2])
        # First bundle: tokens 0-7 real, 8-11 padded
        assert batched["padding_mask"][0, :8].all()
        assert not batched["padding_mask"][0, 8:].any()
        # Second bundle: all 12 real
        assert batched["padding_mask"][1, :12].all()

    def test_padded_entities_are_zero(self):
        b1 = make_obs_bundle(**_make_dummy_bundle(6, f_dim=8))
        b2 = make_obs_bundle(**_make_dummy_bundle(10, f_dim=8))
        batched = collate_obs_bundles([b1, b2])
        # Padded entity slots should be zero-filled
        assert (batched["entities"][0, 6:] == 0).all()

    def test_padded_ids_are_zero(self):
        b1 = make_obs_bundle(**_make_dummy_bundle(6, f_dim=8))
        b2 = make_obs_bundle(**_make_dummy_bundle(10, f_dim=8))
        batched = collate_obs_bundles([b1, b2])
        # Padded categorical IDs should be 0 (UNK)
        assert (batched["ids", "species"][0, 6:] == 0).all()
        assert (batched["ids", "moves"][0, 6:] == 0).all()

    def test_categorical_ids_preserved(self):
        # Set specific IDs and verify they survive collation
        args = _make_dummy_bundle(4, f_dim=8)
        args["species_ids"] = torch.tensor([10, 20, 30, 40])
        b1 = make_obs_bundle(**args)
        b2 = make_obs_bundle(**_make_dummy_bundle(6, f_dim=8))
        batched = collate_obs_bundles([b1, b2])
        assert (batched["ids", "species"][0, :4] == torch.tensor([10, 20, 30, 40])).all()


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestCollateErrors:
    """Test error paths in collation."""

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="Cannot collate an empty list"):
            collate_obs_bundles([])
