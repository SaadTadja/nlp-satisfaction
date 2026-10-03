"""Text normalisation — the single definition used everywhere.

Spec reference: §4.4 (audit) and §3.1 (one prediction path).

This function is imported by training, evaluation, and ``space/app.py``.
It must never be reimplemented inline. If evaluation normalises differently
from serving, the published numbers stop describing the deployed artifact
and nothing will raise an error.

``NORMALIZE_VERSION`` is recorded in ``inference_config.json``. Bump it on any
behavioural change so a stale config cannot silently pair with new code.
"""

from __future__ import annotations

import re
import unicodedata

NORMALIZE_VERSION = "v1"

_HTML_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"https?://\S+|www\.\S+")
_WHITESPACE = re.compile(r"\s+")

# Decision deferred to the §4.4 audit: measure how much of the corpus is
# lowercase / unpunctuated before choosing. Casing carries real signal in
# reviews ("NUL", "CATASTROPHIQUE"), so v1 preserves it. If the audit shows
# the corpus is already lowercased, set this True and bump NORMALIZE_VERSION.
LOWERCASE = False


def normalize(text: str) -> str:
    """Canonicalise one piece of feedback text.

    Conservative by design: it removes markup and unifies whitespace and
    Unicode form, but preserves casing, punctuation, accents and emoji,
    all of which carry sentiment signal in review text.
    """
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _HTML_TAG.sub(" ", text)
    text = _URL.sub(" ", text)
    # NFKC already folds NBSP/narrow-NBSP to plain space; \s then collapses them.
    text = _WHITESPACE.sub(" ", text).strip()
    if LOWERCASE:
        text = text.lower()
    return text


def normalize_all(texts) -> list[str]:
    """Vectorised convenience wrapper."""
    return [normalize(t) for t in texts]


def is_empty(text: str) -> bool:
    """True when the text has no content left after normalisation.

    Used as the first input guard (§3.5) so the model is never called on
    whitespace or emoji-only input that normalisation reduces to nothing.
    """
    return len(normalize(text)) == 0
