"""Test doubles that avoid downloading models or calling an API."""

from __future__ import annotations

import zlib
from types import SimpleNamespace

import numpy as np

from solidityrag.tokenize import tokenize


class HashEmbedder:
    """Deterministic bag-of-words embedder: same interface as the real one."""

    name = "hash-test"

    def __init__(self, dim: int = 64):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text):
            v[zlib.crc32(tok.encode()) % self.dim] += 1.0
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def encode(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def encode_query(self, text: str) -> np.ndarray:
        return self._vec(text)


class FakeLLMClient:
    """Mimics anthropic.Anthropic().messages.create."""

    def __init__(self, reply: str):
        self.reply = reply
        self.calls: list[dict] = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.reply)])