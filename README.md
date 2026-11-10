# SolidityRAG

> A benchmark-first retrieval toolkit for smart contract security auditing.

SolidityRAG helps an LLM audit Solidity code by retrieving the right evidence first. It splits contracts into function-level chunks that keep their structure, retrieves them with keyword search, dense embeddings, or both fused together, and asks an LLM to write findings that cite the code they rest on. A built-in evaluation harness scores each retrieval method against labeled vulnerabilities, so you can measure whether a change actually helps instead of guessing.

**Status: v0.1, a working prototype.** The pipeline runs end to end, but read [Limitations](#limitations) before trusting any number.

## What makes it different

- **Solidity-aware chunking.** One chunk per function, modifier or constructor, tagged with its contract, name and line range, instead of fixed-size text windows that cut functions in half.
- **Evaluation first.** Every retriever runs against the same labeled data with the same metrics, so "hybrid beats keyword search" is something you can check, not something you assume.
- **Evidence you can verify.** The LLM must cite numbered evidence blocks, and a citation checker flags any reference to evidence that was never provided.

## Quickstart

You need Python 3.10 or newer. An Anthropic API key is only needed for the `ask` command.

```bash
git clone https://github.com/<your-username>/SolidityRAG.git
cd SolidityRAG
python -m venv .venv && source .venv/bin/activate
pip install -e ".[embed,llm]"        # or: uv pip install -e ".[embed,llm]"
```

The `embed` extra installs sentence-transformers (and PyTorch), which is large. Skip it to run BM25 only.

Get the benchmark data. It is not bundled with this repo:

```bash
git clone https://github.com/smartbugs/smartbugs-curated data/smartbugs-curated
git -C data/smartbugs-curated rev-parse HEAD     # record this commit with your results
```

Build the index, ask a question, and run the evaluation:

```bash
# 1. Chunk the corpus and build the BM25 and dense indexes (embeddings are cached on disk)
solidityrag index data/smartbugs-curated/dataset

# 2. Ask a question about a contract or a whole directory
export ANTHROPIC_API_KEY=your_key
solidityrag ask data/smartbugs-curated/dataset/reentrancy \
    "Which function sends ether before updating state, and how could it be exploited?"

# 3. Compare bm25, dense and hybrid retrieval against the annotated vulnerable lines
solidityrag eval --out benchmarks/v0.1/results.csv
```

`solidityrag index --no-embed` builds a BM25-only index with no model download. `ask` takes `--retriever bm25|dense|hybrid` and `--k` for the number of chunks.

## Example

<!-- TODO: paste one real `solidityrag ask` output here: the answer, the evidence list and the citation line. -->

## Results

<!-- TODO: replace with the table printed by `solidityrag eval` on your machine. Do not publish numbers you did not produce. -->

| Retriever | Precision@5 | Hit@5 |
|---|---|---|
| bm25 | TBD | TBD |
| dense | TBD | TBD |
| hybrid | TBD | TBD |

Macro average over the SmartBugs Curated categories, 3 query paraphrases per category. Dataset commit: `TBD`. Embedding model: `BAAI/bge-small-en-v1.5`.

A retrieved chunk counts as relevant to a category if its line range covers an annotated vulnerable line of that category. Precision@5 is the share of the top 5 that is relevant, and hit@5 is whether at least one is. Treat these numbers as a pipeline check, not a claim about detection ability (see below).

## How it works

```
.sol files -> chunker -> index (BM25 + dense) -> retrievers (bm25 | dense | hybrid) -> LLM with [C#] citations
                                                       |
                                             evaluation harness (scores each retriever)
```

| Module | Role |
|---|---|
| `ingest/chunker.py` | one chunk per function, modifier or constructor, with contract, name and line range |
| `ingest/smartbugs.py` | loads the labeled vulnerable lines used as ground truth |
| `index/` | embedder interface, sentence-transformers backend, saved index |
| `retrieve/` | BM25, dense and hybrid retrievers behind one interface |
| `generate/` | prompt with labeled evidence blocks, LLM call, citation check |
| `eval/` | query set, relevance rule, precision@k and hit@k |

Design choices worth knowing:

- **Hybrid retrieval uses reciprocal rank fusion.** BM25 scores and cosine similarities are not comparable, but ranks are, so fusion needs no tuned weights.
- **BM25 uses a positive IDF.** The stock formula scores a term that appears in one of two documents as exactly zero, so on a small corpus (such as a single contract) matching chunks can return nothing. This project uses the always-positive Lucene-style formula.
- **Dataset labels are stripped before indexing.** SmartBugs files contain comments such as `// <yes> <report> REENTRANCY` that name the answer. They are removed from everything that is indexed or shown to the model.
- **Evidence labels are short.** The prompt uses `[C1]`, `[C2]` and so on instead of file paths, because models copy long identifiers unreliably.

You can also use the pieces as a library:

```python
from solidityrag.ingest import load_chunks
from solidityrag.index import Index
from solidityrag.retrieve import BM25Retriever

chunks = load_chunks("data/smartbugs-curated/dataset").chunks
index = Index.build(chunks)                      # BM25 only; pass an embedder for dense
for r in BM25Retriever(index).search("external call before balance update", k=3):
    print(r.rank, r.chunk.id, round(r.score, 2))
```

## Limitations

- **The chunker is not a real parser.** It strips comments and strings and matches braces with a regex. It handles the SmartBugs corpus, but unusual syntax can break it. `index` prints any files that produce no chunks.
- **Data contamination.** SmartBugs contracts are public and old, so they are very likely in LLM training data. Results from `ask`, and any claim about detection ability, should be read with that in mind.
- **Small categories are noisy.** Some categories have very few annotated chunks (`short_addresses` has one). `eval` prints the number of relevant chunks per category so you can see which scores mean little.
- **The query set is small.** Three hand-written paraphrases per category, fixed before running. Do not tune them against the metrics.
- **Long functions are truncated.** The default embedding model reads at most 512 tokens, so the end of a long function may not be embedded.
- **No baselines yet.** The evaluation compares retrievers with each other. It does not yet compare against fixed-size chunking or an LLM with no retrieval, so it cannot yet show how much Solidity-aware chunking helps.

## Roadmap

In this order, each item only after the previous one is measured:

1. Baselines: fixed-size chunking and an LLM-only mode.
2. Structural expansion: pull in callers, callees and parent contracts of the top results.
3. A tree-sitter parser behind the same `Chunk` interface.
4. A reranker.
5. A security knowledge base (SWC Registry first, then public audit reports), split so retrieval never draws on the contracts being evaluated.

## Development

```bash
pip install -e ".[dev]"
pytest
```

The tests use a fake embedder and a fake LLM client, so they need no model download and no API key.

## Data and licensing

The benchmark data comes from [SmartBugs Curated](https://github.com/smartbugs/smartbugs-curated). It is not bundled here. Clone it yourself and follow its license.

