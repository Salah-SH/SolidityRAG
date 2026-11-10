"""Command line interface: index, ask, eval."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .generate.answer import DEFAULT_LLM
from .index.embedder import DEFAULT_MODEL
from .index.store import Index
from .ingest.chunker import chunk_source
from .ingest.fixed import fixed_chunker
from .ingest.loader import load_chunks
from .retrieve.bm25 import BM25Retriever
from .retrieve.dense import DenseRetriever
from .retrieve.hybrid import HybridRetriever

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="SolidityRAG: retrieval toolkit for smart contract security auditing.",
)

DEFAULT_INDEX_DIR = Path(".solidityrag/index")
DEFAULT_ANNOTATIONS = Path("data/smartbugs-curated/vulnerabilities.json")


def _make_embedder(model: str, query_prefix: bool):
    from .index.embedder import BGE_QUERY_PREFIX, SentenceTransformerEmbedder

    return SentenceTransformerEmbedder(model, BGE_QUERY_PREFIX if query_prefix else "")


def _make_retrievers(index: Index, embedder) -> dict:
    bm25 = BM25Retriever(index)
    retrievers = {"bm25": bm25}
    if index.embeddings is not None and embedder is not None:
        dense = DenseRetriever(index, embedder)
        retrievers["dense"] = dense
        retrievers["hybrid"] = HybridRetriever(index, [dense, bm25])
    return retrievers


@app.command("index")
def index_cmd(
    corpus: Path = typer.Argument(..., help="Directory (or file) of Solidity sources."),
    out: Path = typer.Option(DEFAULT_INDEX_DIR, "--out", help="Where to save the index."),
    model: str = typer.Option(DEFAULT_MODEL, help="Sentence-transformers model name."),
    embed: bool = typer.Option(True, "--embed/--no-embed", help="Skip embeddings for a BM25-only index."),
    chunker: str = typer.Option("function", help="function (Solidity-aware) or fixed (line windows, baseline)."),
    window: int = typer.Option(8, help="Lines per window (--chunker fixed)."),
    overlap: int = typer.Option(0, help="Lines shared by consecutive windows (--chunker fixed)."),
):
    """Chunk a corpus, build the BM25 and dense indexes, and save them."""
    if chunker == "function":
        chunk_fn, chunking = chunk_source, "function"
    elif chunker == "fixed":
        if window < 1 or not 0 <= overlap < window:
            raise typer.BadParameter("Need window >= 1 and 0 <= overlap < window.")
        chunk_fn, chunking = fixed_chunker(window, overlap), f"fixed:w={window},o={overlap}"
    else:
        raise typer.BadParameter(f"Unknown chunker '{chunker}' (use function or fixed).")
    loaded = load_chunks(corpus, chunk_fn)
    typer.echo(f"{loaded.n_files} files -> {len(loaded.chunks)} chunks ({chunking})")
    if loaded.empty_files:
        typer.echo(f"{len(loaded.empty_files)} files produced no chunks, for example:")
        for f in loaded.empty_files[:10]:
            typer.echo(f"  {f}")
    embedder = _make_embedder(model, True) if embed else None
    idx = Index.build(loaded.chunks, embedder, chunking)
    idx.save(out)
    typer.echo(f"Saved index to {out}")


@app.command("ask")
def ask_cmd(
    path: Path = typer.Argument(..., help="Directory or file to audit (ignored when --index is given)."),
    question: str = typer.Argument(..., help="What to look for."),
    index_dir: Optional[Path] = typer.Option(None, "--index", help="Use a saved index instead of indexing PATH."),
    k: int = typer.Option(5, help="Number of chunks to retrieve."),
    retriever: str = typer.Option("hybrid", help="bm25, dense or hybrid."),
    llm: str = typer.Option(DEFAULT_LLM, help="Model used to write the answer."),
    model: str = typer.Option(DEFAULT_MODEL, help="Embedding model (when indexing PATH)."),
    query_prefix: bool = typer.Option(True, "--query-prefix/--no-query-prefix"),
):
    """Retrieve evidence and answer a question about it, with citations."""
    from .generate.answer import ask

    if index_dir is not None:
        idx = Index.load(index_dir)
        embedder = _make_embedder(idx.embedder_name, query_prefix) if idx.embeddings is not None else None
    else:
        loaded = load_chunks(path)
        embedder = _make_embedder(model, query_prefix) if retriever != "bm25" else None
        idx = Index.build(loaded.chunks, embedder)

    retrievers = _make_retrievers(idx, embedder)
    if retriever not in retrievers:
        raise typer.BadParameter(f"'{retriever}' is not available for this index (have: {', '.join(retrievers)}).")

    result = ask(question, retrievers[retriever], k=k, model=llm)
    typer.echo(result.text)
    typer.echo("\nEvidence:")
    for label, chunk in result.label_map.items():
        typer.echo(f"  [{label}] {chunk.file}:{chunk.start_line}-{chunk.end_line}  {chunk.contract}.{chunk.name}")
    validity = result.citations.validity
    if validity is None:
        typer.echo("\nCitations: none in the answer.")
    else:
        typer.echo(f"\nCitations valid: {len(result.citations.valid)}/{len(result.citations.cited)}")
        if result.citations.invalid:
            typer.echo(f"Unknown labels: {', '.join(result.citations.invalid)}")


@app.command("eval")
def eval_cmd(
    index_dir: Path = typer.Argument(DEFAULT_INDEX_DIR, help="Saved index of the SmartBugs corpus."),
    annotations: Path = typer.Option(DEFAULT_ANNOTATIONS, help="SmartBugs vulnerabilities.json."),
    k: int = typer.Option(5, help="Cutoff for precision@k and hit@k."),
    queries: Optional[Path] = typer.Option(None, help="Custom queries YAML (default: bundled)."),
    out: Optional[Path] = typer.Option(None, help="Save the per-category results as CSV."),
    query_prefix: bool = typer.Option(True, "--query-prefix/--no-query-prefix"),
):
    """Compare bm25, dense and hybrid retrieval against the SmartBugs annotations."""
    from .eval.run import evaluate, load_queries, overall, per_category
    from .ingest.smartbugs import Annotations

    idx = Index.load(index_dir)
    embedder = _make_embedder(idx.embedder_name, query_prefix) if idx.embeddings is not None else None
    retrievers = _make_retrievers(idx, embedder)
    df = evaluate(retrievers, load_queries(queries), Annotations.from_json(annotations), idx.chunks, k=k)
    if idx.chunking:
        df.insert(0, "chunking", idx.chunking)
    if df.empty:
        raise typer.BadParameter("No category has relevant chunks. Check the index and annotations paths.")

    typer.echo(f"\nChunking: {idx.chunking or 'unknown (index built before v0.2)'}")
    typer.echo(f"Macro average over categories (k={k}):")
    typer.echo(overall(df, k).to_string())
    typer.echo(f"\nPrecision@{k} by category:")
    typer.echo(per_category(df, f"precision@{k}").to_string())
    sizes = df.drop_duplicates("category").set_index("category")["n_relevant_chunks"]
    typer.echo("\nRelevant chunks per category in this corpus (small counts mean noisy scores):")
    typer.echo(sizes.to_string())
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        typer.echo(f"\nSaved results to {out}")