"""Run retrievers over the query set and aggregate the results."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from statistics import mean
from typing import Mapping

import pandas as pd
import yaml

from ..ingest.chunker import Chunk
from ..ingest.smartbugs import Annotations
from ..retrieve.base import Retriever
from .metrics import hit_at_k, is_relevant, precision_at_k


def load_queries(path: str | Path | None = None) -> dict[str, list[str]]:
    if path is None:
        text = resources.files("solidityrag.eval").joinpath("queries.yaml").read_text(encoding="utf-8")
    else:
        text = Path(path).read_text(encoding="utf-8")
    return yaml.safe_load(text)


def evaluate(
    retrievers: Mapping[str, Retriever],
    queries: Mapping[str, list[str]],
    annotations: Annotations,
    chunks: list[Chunk],
    k: int = 5,
) -> pd.DataFrame:
    """One row per (retriever, category). Scores are averaged over a category's paraphrases.

    Categories with no relevant chunk in the corpus are skipped: nothing could
    be retrieved correctly, so they would only add noise.
    """
    p_col, h_col = f"precision@{k}", f"hit@{k}"
    rows = []
    for category, qs in queries.items():
        n_relevant = sum(is_relevant(c, annotations, category) for c in chunks)
        if n_relevant == 0:
            continue
        for name, retriever in retrievers.items():
            precisions, hits = [], []
            for q in qs:
                flags = [is_relevant(r.chunk, annotations, category) for r in retriever.search(q, k)]
                precisions.append(precision_at_k(flags, k))
                hits.append(hit_at_k(flags, k))
            rows.append(
                {
                    "retriever": name,
                    "category": category,
                    p_col: mean(precisions),
                    h_col: mean(hits),
                    "n_queries": len(qs),
                    "n_relevant_chunks": n_relevant,
                }
            )
    return pd.DataFrame(rows)


def overall(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Macro average over categories, so large categories don't dominate."""
    cols = [f"precision@{k}", f"hit@{k}"]
    return df.groupby("retriever", sort=False)[cols].mean().round(3)


def per_category(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    return df.pivot(index="category", columns="retriever", values=metric).round(3)