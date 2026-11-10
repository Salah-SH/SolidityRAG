"""Embedding backends. Anything with this small interface can be plugged in."""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
# bge models recommend this prefix for short search queries (not for documents)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    name: str

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return L2-normalized vectors, shape (len(texts), dim)."""

    def encode_query(self, text: str) -> np.ndarray:
        """Return one L2-normalized vector, shape (dim,)."""


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, query_prefix: str = BGE_QUERY_PREFIX):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "Dense retrieval needs sentence-transformers: pip install 'solidityrag[embed]'"
            ) from e
        self.name = model_name
        self.query_prefix = query_prefix
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        texts = list(texts)
        vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 200,
        )
        return np.asarray(vecs, dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        vec = self.model.encode(self.query_prefix + text, normalize_embeddings=True)
        return np.asarray(vec, dtype=np.float32)