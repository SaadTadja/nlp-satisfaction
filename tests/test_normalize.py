"""Tests for text normalisation (§4.4).

These exist mainly as a drift alarm: if normalisation changes without
``NORMALIZE_VERSION`` being bumped, a stale ``inference_config.json`` could
pair with new code and the deployed model would silently stop matching the
evaluated one.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.normalize import NORMALIZE_VERSION, is_empty, normalize, normalize_all  # noqa: E402


def test_collapses_whitespace_and_trims():
    assert normalize("  trop   lent \n\n vraiment  ") == "trop lent vraiment"


def test_strips_html_and_urls():
    assert normalize("<b>Nul</b>") == "Nul"
    assert normalize("Voir https://exemple.fr/page pour la suite") == (
        "Voir pour la suite"
    )


def test_preserves_signal_bearing_features():
    """Casing, punctuation, accents and emoji all carry sentiment signal in
    review text, so v1 keeps them."""
    out = normalize("CATASTROPHIQUE !!! 😡 déçu")
    assert "CATASTROPHIQUE" in out and "!!!" in out and "😡" in out and "déçu" in out


def test_unicode_is_canonicalised():
    # 'e' + combining acute must normalise to the same string as precomposed 'é'
    assert normalize("déçu") == normalize("déçu")


def test_non_breaking_spaces_are_folded():
    assert normalize("prix élevé") == "prix élevé"
    assert normalize("prix élevé") == "prix élevé"


def test_handles_non_string_input():
    assert normalize(None) == ""
    assert normalize(123) == ""


def test_is_empty_detects_content_free_input():
    assert is_empty("   ")
    assert is_empty("<br/>")
    assert not is_empty("bien")


def test_is_idempotent():
    """normalize(normalize(x)) == normalize(x) -- required because the Space
    and the eval notebook must be able to apply it without coordinating."""
    for raw in ["  <i>Très</i>  cher  ", "Nul.\n\nVraiment", "ok"]:
        assert normalize(normalize(raw)) == normalize(raw)


def test_normalize_all_matches_elementwise():
    texts = [" a ", "<b>b</b>", None]
    assert normalize_all(texts) == [normalize(t) for t in texts]


def test_version_is_declared():
    assert isinstance(NORMALIZE_VERSION, str) and NORMALIZE_VERSION
