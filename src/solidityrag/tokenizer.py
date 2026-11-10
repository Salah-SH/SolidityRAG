"""Identifier-aware tokenizer used for BM25 (and by the test embedder)."""

from __future__ import annotations

import re

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def tokenize(text: str) -> list[str]:
    """Lowercased identifier tokens.

    Each identifier is emitted whole and also split on camelCase and underscores,
    so ``withdrawAll`` yields ``withdrawall``, ``withdraw`` and ``all``. Keeping the
    whole identifier preserves exact matches on things like ``delegatecall``.
    """
    tokens: list[str] = []
    for ident in _IDENT.findall(text):
        whole = ident.lower()
        tokens.append(whole)
        parts = [p.lower() for p in _CAMEL.sub("_", ident).split("_") if p]
        if len(parts) > 1:
            tokens.extend(parts)
    return tokens