"""Experiment: Solidity-aware function chunks vs fixed-size line windows.

Every configuration is indexed and scored with the same corpus, embedding
model, queries and annotations, so the chunker is the only thing that changes.

    python scripts/chunking_baseline.py data/smartbugs-curated --out benchmarks/v0.2

Two comparisons are written:

* ``chunking.csv`` / ``chunking_summary.csv``: the usual top-k metrics, plus
  ``lines@k`` (distinct lines retrieved) and ``vlines@k`` (distinct annotated
  vulnerable lines found). Top-k alone is unfair across chunkers: bigger
  chunks cover more lines, so they "hit" more often for free.
* ``chunking_budget.csv``: the fair comparison. Each retriever reads ranked
  chunks until a fixed number of source lines is used up, so every chunker gets
  the same reading budget. Each fixed-size chunker is compared with the function
  chunker query by query (paired), with a bootstrap 95% interval on the
  difference.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from solidityrag.eval import evaluate, is_relevant, load_queries
from solidityrag.eval.metrics import lines_covered, vulnerable_lines
from solidityrag.index.embedder import BGE_QUERY_PREFIX, DEFAULT_MODEL, SentenceTransformerEmbedder
from solidityrag.index.store import Index
from solidityrag.ingest import chunk_source, fixed_chunker, load_chunks
from solidityrag.ingest.smartbugs import Annotations
from solidityrag.retrieve import BM25Retriever, DenseRetriever, HybridRetriever

BUDGETS = (40, 80, 160)  # source lines the LLM would read
POOL = 200  # ranked chunks to draw from when filling a budget
BOOTSTRAP = 5000

# window 8 matches the mean function chunk (8.4 lines); 16 and 32 show the size trade-off
CONFIGS = [("function", chunk_source)] + [
    (f"fixed:w={w},o={o}", fixed_chunker(w, o)) for w in (8, 16, 32) for o in (0, w // 2)
]


def budget_scores(retriever, queries, annotations, chunks, budget: int) -> list[dict]:
    """Per-query hit and vulnerable lines found within a fixed line budget.

    Chunks are taken in rank order and skipped once the next one would exceed
    the budget, so no retriever reads more than ``budget`` distinct lines.
    """
    rows = []
    for category, qs in queries.items():
        if not any(is_relevant(c, annotations, category) for c in chunks):
            continue
        for q in qs:
            seen: set = set()
            found: set = set()
            for r in retriever.search(q, POOL):
                new = lines_covered([r.chunk]) - seen
                if len(seen) + len(new) > budget:
                    break
                seen |= new
                found |= vulnerable_lines(r.chunk, annotations, category)
            rows.append({"category": category, "query": q, "hit": float(bool(found)), "vlines": len(found)})
    return rows


def paired_ci(a: np.ndarray, b: np.ndarray, rng) -> tuple[float, float, float]:
    """Mean of a - b over queries, with a bootstrap 95% interval."""
    diff = a - b
    boots = diff[rng.integers(0, len(diff), (BOOTSTRAP, len(diff)))].mean(axis=1)
    return diff.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("smartbugs", type=Path, help="Clone of smartbugs-curated")
    ap.add_argument("--out", type=Path, default=Path("benchmarks/v0.2"))
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    commit = subprocess.run(
        ["git", "-C", str(args.smartbugs), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    annotations = Annotations.from_json(args.smartbugs / "vulnerabilities.json")
    queries = load_queries()
    embedder = SentenceTransformerEmbedder(DEFAULT_MODEL, BGE_QUERY_PREFIX)

    frames, stats, budget_rows = [], [], []
    for name, chunk_fn in CONFIGS:
        chunks = load_chunks(args.smartbugs / "dataset", chunk_fn).chunks
        sizes = np.array([c.end_line - c.start_line + 1 for c in chunks])
        stats.append({"chunking": name, "n_chunks": len(chunks), "mean_lines": sizes.mean().round(1)})
        print(f"{name}: {len(chunks)} chunks, mean {sizes.mean():.1f} lines", flush=True)

        idx = Index.build(chunks, embedder, name)
        bm25 = BM25Retriever(idx)
        dense = DenseRetriever(idx, embedder)
        retrievers = {"bm25": bm25, "dense": dense, "hybrid": HybridRetriever(idx, [dense, bm25])}
        df = evaluate(retrievers, queries, annotations, chunks, k=args.k)
        df.insert(0, "chunking", name)
        frames.append(df)

        # a wider fusion pool so hybrid can fill the largest budget
        retrievers["hybrid"] = HybridRetriever(idx, [dense, bm25], pool=POOL)
        for rname, retriever in retrievers.items():
            for budget in BUDGETS:
                for row in budget_scores(retriever, queries, annotations, chunks, budget):
                    budget_rows.append({"chunking": name, "retriever": rname, "budget": budget, **row})

    df = pd.concat(frames, ignore_index=True)
    k = args.k
    metrics = [f"precision@{k}", f"hit@{k}", f"vlines@{k}", f"lines@{k}"]
    summary = df.groupby(["chunking", "retriever"], sort=False)[metrics].mean().round(3).reset_index()
    summary = summary.merge(pd.DataFrame(stats), on="chunking")
    # vulnerable lines found per 100 lines retrieved: how much of the reading is on target
    summary["vlines_per_100"] = (100 * summary[f"vlines@{k}"] / summary[f"lines@{k}"]).round(2)

    per_query = pd.DataFrame(budget_rows)
    rng = np.random.default_rng(0)
    budget_summary = []
    for (name, rname, budget), g in per_query.groupby(["chunking", "retriever", "budget"], sort=False):
        ref = per_query[
            (per_query.chunking == "function") & (per_query.retriever == rname) & (per_query.budget == budget)
        ]
        row = {"chunking": name, "retriever": rname, "budget": budget, "n_queries": len(g),
               "hit": g.hit.mean().round(3), "vlines": g.vlines.mean().round(2)}
        if name != "function":
            for m in ("hit", "vlines"):
                mean_d, lo, hi = paired_ci(ref[m].to_numpy(float), g[m].to_numpy(float), rng)
                row[f"function_minus_fixed_{m}"] = round(mean_d, 3)
                row[f"{m}_ci95"] = f"[{lo:+.3f}, {hi:+.3f}]"
        budget_summary.append(row)
    budget_summary = pd.DataFrame(budget_summary)

    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "chunking.csv", index=False)
    summary.to_csv(args.out / "chunking_summary.csv", index=False)
    budget_summary.to_csv(args.out / "chunking_budget.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"\nDataset commit: {commit}  Embedder: {DEFAULT_MODEL}  k={k}")
    print("Macro average over categories:")
    print(summary.to_string(index=False))
    print("\nEqual line budget (paired against the function chunker, 95% bootstrap interval):")
    print(budget_summary.to_string(index=False))
    print(f"\nSaved chunking.csv, chunking_summary.csv and chunking_budget.csv to {args.out}")


if __name__ == "__main__":
    main()
