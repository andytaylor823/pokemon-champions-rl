"""Unit tests for the vocab module — normalization, encode/decode, singletons."""
from __future__ import annotations

from vocab import (
    ABILITY_VOCAB,
    ITEM_VOCAB,
    MOVE_VOCAB,
    NATURE_VOCAB,
    SPECIES_VOCAB,
    Vocab,
    _to_showdown_id,
)


# ---------------------------------------------------------------------------
# Showdown ID normalization
# ---------------------------------------------------------------------------


class TestShowdownIdNormalization:
    """Test _to_showdown_id stripping rules."""

    def test_spaces_removed(self):
        assert _to_showdown_id("Heat Wave") == "heatwave"

    def test_hyphens_removed(self):
        assert _to_showdown_id("Wi-Fi") == "wifi"

    def test_punctuation_removed(self):
        assert _to_showdown_id("Mr. Mime") == "mrmime"

    def test_mixed_case_lowered(self):
        assert _to_showdown_id("Charizardite Y") == "charizarditey"

    def test_already_normalized_unchanged(self):
        assert _to_showdown_id("charizard") == "charizard"

    def test_numbers_preserved(self):
        assert _to_showdown_id("Porygon2") == "porygon2"


# ---------------------------------------------------------------------------
# Vocab class basics
# ---------------------------------------------------------------------------


class TestVocabClass:
    """Test the Vocab class interface with a small synthetic vocabulary."""

    def setup_method(self):
        self.v = Vocab({"alpha": 1, "beta": 2, "gamma": 3}, label="test")

    def test_size_includes_unk(self):
        # 3 entries + 1 UNK slot = 4
        assert self.v.size == 4

    def test_len_matches_size(self):
        assert len(self.v) == 4

    def test_encode_known(self):
        assert self.v.encode("alpha") == 1
        assert self.v.encode("beta") == 2

    def test_encode_unknown_returns_zero(self):
        assert self.v.encode("nonexistent") == 0

    def test_decode_known(self):
        assert self.v.decode(1) == "alpha"
        assert self.v.decode(3) == "gamma"

    def test_decode_zero_returns_unk(self):
        assert self.v.decode(0) == "<UNK>"

    def test_decode_out_of_range_returns_unk(self):
        assert self.v.decode(999) == "<UNK>"

    def test_contains_known(self):
        assert "alpha" in self.v

    def test_contains_unknown(self):
        assert "nonexistent" not in self.v

    def test_repr(self):
        r = repr(self.v)
        assert "test" in r
        assert "4" in r

    def test_to_dict(self):
        d = self.v.to_dict()
        assert d["label"] == "test"
        assert d["size"] == 4
        assert d["entries"]["alpha"] == 1


# ---------------------------------------------------------------------------
# Singleton vocab sanity checks
# ---------------------------------------------------------------------------


class TestSingletonVocabs:
    """Verify the module-level vocabs loaded from data files are reasonable."""

    def test_species_vocab_size(self):
        # Legal species pool + UNK slot; actual count is ~186
        assert SPECIES_VOCAB.size > 150

    def test_item_vocab_size(self):
        # ~120 legal items + UNK
        assert ITEM_VOCAB.size > 100

    def test_move_vocab_size(self):
        # Many moves extracted from learnsets
        assert MOVE_VOCAB.size > 50

    def test_ability_vocab_size(self):
        # abilities.txt was added in this commit (142 entries)
        assert ABILITY_VOCAB.size > 100

    def test_nature_vocab_size(self):
        # 25 natures + UNK = 26
        assert NATURE_VOCAB.size == 26

    def test_species_encode_known(self):
        # Charizard is certainly in the legal species list
        idx = SPECIES_VOCAB.encode("Charizard")
        assert idx > 0

    def test_species_contains_normalization(self):
        # "Charizard" and "charizard" should both resolve
        assert "Charizard" in SPECIES_VOCAB
        assert "charizard" in SPECIES_VOCAB

    def test_item_encode_known(self):
        # Choice Scarf is legal per the rules
        idx = ITEM_VOCAB.encode("Choice Scarf")
        assert idx > 0

    def test_move_encode_known(self):
        # Protect is one of the most common moves
        idx = MOVE_VOCAB.encode("Protect")
        assert idx > 0

    def test_nature_encode_known(self):
        idx = NATURE_VOCAB.encode("Adamant")
        assert idx > 0

    def test_ability_encode_known(self):
        idx = ABILITY_VOCAB.encode("Intimidate")
        assert idx > 0


# ---------------------------------------------------------------------------
# Encode / decode round-trip on real vocabs
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Encode then decode should recover the normalized name."""

    def test_species_roundtrip(self):
        idx = SPECIES_VOCAB.encode("Charizard")
        name = SPECIES_VOCAB.decode(idx)
        assert name == "charizard"

    def test_item_roundtrip(self):
        idx = ITEM_VOCAB.encode("Choice Scarf")
        name = ITEM_VOCAB.decode(idx)
        assert name == "choicescarf"

    def test_move_roundtrip(self):
        idx = MOVE_VOCAB.encode("Heat Wave")
        name = MOVE_VOCAB.decode(idx)
        assert name == "heatwave"

    def test_nature_roundtrip(self):
        idx = NATURE_VOCAB.encode("Timid")
        name = NATURE_VOCAB.decode(idx)
        assert name == "timid"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Vocab IDs should be stable across repeated loads."""

    def test_species_ids_stable(self):
        # Encode a few species, verify they always get the same ID
        id1 = SPECIES_VOCAB.encode("Charizard")
        id2 = SPECIES_VOCAB.encode("Charizard")
        assert id1 == id2

    def test_move_ordering_stable(self):
        # Two different moves should have different, consistent IDs
        a = MOVE_VOCAB.encode("Protect")
        b = MOVE_VOCAB.encode("Earthquake")
        assert a != b
        assert a == MOVE_VOCAB.encode("Protect")
        assert b == MOVE_VOCAB.encode("Earthquake")
