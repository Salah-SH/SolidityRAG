import json

import pytest
from helpers import HashEmbedder
from typer.testing import CliRunner

from solidityrag.cli import app
from solidityrag.eval import evaluate, hit_at_k, is_relevant, load_queries, overall, per_category, precision_at_k
from solidityrag.index.store import Index
from solidityrag.ingest.chunker import chunk_source
from solidityrag.ingest.smartbugs import Annotations
from solidityrag.retrieve import BM25Retriever

SRC = """contract Bank {
    function withdraw() public {
        msg.sender.call.value(balances[msg.sender])();
        balances[msg.sender] = 0;
    }
    function harmless() public { x = 1; }
}"""

ANNOTATIONS = [
    {
        "name": "bank.sol",
        "path": "dataset/reentrancy/bank.sol",
        "vulnerabilities": [{"lines": [3], "category": "reentrancy"}],
    }
]


@pytest.fixture
def annotations(tmp_path):
    p = tmp_path / "vulnerabilities.json"
    p.write_text(json.dumps(ANNOTATIONS))
    return Annotations.from_json(p)


def test_precision_and_hit_math():
    assert precision_at_k([True, False, True], 5) == pytest.approx(0.4)
    assert precision_at_k([True, True], 2) == 1.0
    assert hit_at_k([False, False, True], 3) == 1.0
    assert hit_at_k([False, False, False], 3) == 0.0


def test_annotation_lookup_matches_paths_regardless_of_indexed_root(annotations):
    for label in ("dataset/reentrancy/bank.sol", "reentrancy/bank.sol", "x/y/dataset/reentrancy/bank.sol"):
        assert annotations.for_file(label), label
    assert annotations.for_file("other/file.sol") == []


def test_ambiguous_basename_is_not_guessed():
    from solidityrag.ingest.smartbugs import Annotation

    ann = Annotations(
        [
            Annotation("dataset/a/x.sol", "reentrancy", frozenset({1})),
            Annotation("dataset/b/x.sol", "arithmetic", frozenset({1})),
        ]
    )
    assert ann.for_file("x.sol") == []
    assert ann.for_file("a/x.sol")[0].category == "reentrancy"


def test_relevance_requires_category_and_line_overlap(annotations):
    chunks = chunk_source(SRC, "reentrancy/bank.sol")
    withdraw, harmless = chunks
    assert is_relevant(withdraw, annotations, "reentrancy")
    assert not is_relevant(withdraw, annotations, "arithmetic")
    assert not is_relevant(harmless, annotations, "reentrancy")


def test_evaluate_skips_categories_without_relevant_chunks_and_aggregates(annotations):
    chunks = chunk_source(SRC, "reentrancy/bank.sol")
    index = Index.build(chunks, HashEmbedder())
    queries = {
        "reentrancy": ["call value balances withdraw", "msg sender call"],
        "arithmetic": ["overflow"],  # nothing relevant in this corpus: skipped
    }
    df = evaluate({"bm25": BM25Retriever(index)}, queries, annotations, chunks, k=2)
    assert list(df["category"]) == ["reentrancy"]
    row = df.iloc[0]
    assert row["hit@2"] == 1.0
    assert row["precision@2"] == pytest.approx(0.5)  # one relevant chunk out of k=2
    assert overall(df, k=2).loc["bm25", "hit@2"] == 1.0
    assert per_category(df, "hit@2").loc["reentrancy", "bm25"] == 1.0


def test_bundled_queries_cover_the_smartbugs_categories():
    q = load_queries()
    assert {"reentrancy", "access_control", "arithmetic", "unchecked_low_level_calls"} <= set(q)
    assert all(len(v) >= 2 for v in q.values())


def test_cli_index_then_eval_bm25_only(tmp_path):
    corpus = tmp_path / "dataset" / "reentrancy"
    corpus.mkdir(parents=True)
    (corpus / "bank.sol").write_text(SRC)
    ann = tmp_path / "vulnerabilities.json"
    ann.write_text(json.dumps(ANNOTATIONS))
    out = tmp_path / "idx"

    runner = CliRunner()
    r = runner.invoke(app, ["index", str(tmp_path / "dataset"), "--out", str(out), "--no-embed"])
    assert r.exit_code == 0, r.output
    assert "1 files -> 2 chunks" in r.output

    csv = tmp_path / "res.csv"
    r = runner.invoke(app, ["eval", str(out), "--annotations", str(ann), "--k", "2", "--out", str(csv)])
    assert r.exit_code == 0, r.output
    assert "bm25" in r.output
    assert csv.exists()