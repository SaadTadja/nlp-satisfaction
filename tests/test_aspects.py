"""Tests for aspect detection and sentence attribution (§11)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.aspects import (  # noqa: E402
    attribute_sentences,
    detect_aspects,
    split_sentences,
    suggest_lexicon_terms,
)


def test_split_sentences_basic():
    text = "Produit correct. Livraison trop lente ! Je recommande ?"
    assert len(split_sentences(text)) == 3


def test_split_sentences_handles_newlines_and_empties():
    assert split_sentences("Bien.\n\nMais cher.") == ["Bien.", "Mais cher."]
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_detect_single_aspect():
    assert detect_aspects("Le colis est arrivé en retard") == {"livraison"}


def test_detect_multiple_aspects():
    found = detect_aspects("Le prix est correct mais la qualité est mauvaise")
    assert found == {"prix", "qualité"}


def test_no_aspect_returns_empty():
    assert detect_aspects("Je ne sais pas trop quoi en penser") == set()


def test_matching_is_whole_word():
    """'cherche' must not match the lexicon term 'cher'."""
    assert "prix" not in detect_aspects("Je cherche un autre modèle")


def test_matching_is_case_insensitive():
    assert detect_aspects("LIVRAISON catastrophique") == {"livraison"}


def test_mixed_review_separates_by_sentence():
    """The case that motivates sentence-level attribution (§11 step 3)."""
    text = "Le produit est super. Par contre la livraison est catastrophique."
    pairs = attribute_sentences(text)
    assert len(pairs) == 2
    assert pairs[0][1] == set()          # no aspect term in the praise
    assert pairs[1][1] == {"livraison"}  # complaint is attributed


def test_attribution_keeps_unattributed_sentences():
    """Unattributed sentences are kept, not dropped -- their share is a
    number worth reporting."""
    pairs = attribute_sentences("Bof. Le prix reste élevé.")
    assert len(pairs) == 2
    assert any(a == set() for _, a in pairs)


def test_custom_lexicon_overrides_default():
    custom = {"batterie": {"batterie", "autonomie"}}
    assert detect_aspects("L'autonomie est faible", custom) == {"batterie"}
    assert detect_aspects("Le prix est élevé", custom) == set()


def test_suggest_lexicon_terms_ranks_by_frequency():
    texts = ["livraison lente", "livraison nulle", "produit correct"]
    top = dict(suggest_lexicon_terms(texts, top_k=10))
    assert top["livraison"] == 2


def test_suggest_lexicon_terms_filters_stopwords_and_short_words():
    terms = dict(suggest_lexicon_terms(["avec le prix très bas"], top_k=20))
    assert "avec" not in terms and "très" not in terms
    assert "prix" in terms
