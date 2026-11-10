"""The index: chunks plus a BM25 view and (optionally) a dense embedding matrix.

Embeddings are the slow part, so they are saved to disk. BM25 is cheap and is
rebuilt from the chunks on load instead of being serialized.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from ..ingest.chunker import Chunk
from ..tokenize import tokenize
from .embedder import Embedder


class PositiveIdfBM25(BM25Okapi):
    """BM25Okapi with the Lucene-style IDF, log(1 + (N - n + 0.5) / (n + 0.5)).

    The stock formula is exactly 0 for a term found in one of two documents and
    negative for terms in more than half of them, so on a small corpus (for
    example a single contract with a few functions) matching chunks can score
    zero and nothing is returned. This version is always positive.
    """

    def _calc_idf(self, nd):
        for word, freq in nd.items():
            self.idf[word] = math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))
        self.average_idf = sum(self.idf.values()) / len(self.idf) if self.idf else 0.0


class Index:
    def __init__(
        self,
        chunks: list[Chunk],
        embeddings: np.ndarray | None = None,
        embedder_name: str | None = None,
    ):
        if not chunks:
            raise ValueError("Cannot build an index from zero chunks.")
        if embeddings is not None and len(embeddings) != len(chunks):
            raise ValueError("embeddings and chunks must have the same length")
        self.chunks = chunks
        self.embeddings = embeddings
        self.embedder_name = embedder_name
        self.bm25 = PositiveIdfBM25([tokenize(c.embed_text) for c in chunks])

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder | None = None) -> "Index":
        embeddings = None
        name = None
        if embedder is not None:
            embeddings = np.asarray(embedder.encode([c.embed_text for c in chunks]), dtype=np.float32)
            name = embedder.name
        return cls(chunks, embeddings, name)

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        (d / "chunks.json").write_text(
            json.dumps([c.to_dict() for c in self.chunks]), encoding="utf-8"
        )
        meta = {"n_chunks": len(self.chunks), "embedder": self.embedder_name}
        (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        if self.embeddings is not None:
            np.save(d / "embeddings.npy", self.embeddings)

    @classmethod
    def load(cls, directory: str | Path) -> "Index":
        d = Path(directory)
        chunks = [Chunk.from_dict(x) for x in json.loads((d / "chunks.json").read_text(encoding="utf-8"))]
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        emb_path = d / "embeddings.npy"
        embeddings = np.load(emb_path) if emb_path.exists() else None
        return cls(chunks, embeddings, meta.get("embedder"))