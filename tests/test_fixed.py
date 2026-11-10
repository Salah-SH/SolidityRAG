import json

import pytest
from typer.testing import CliRunner

from solidityrag.cli import app
from solidityrag.eval import evaluate
from solidityrag.eval.metrics import lines_covered, vulnerable_lines
from solidityrag.index.store import Index
from solidityrag.ingest import chunk_source_fixed, fixed_chunker, load_chunks
from solidityrag.ingest.sanitize import strip_label_leaks
from solidityrag.ingest.smartbugs import Annotations
from solidityrag.retrieve import BM25Retriever

TEN_LINES = "".join(f"line{i}\n" for i in range(1, 11))


def spans(chunks):
    return [(c.start_line, c.end_line) for c in chunks]


def test_windows_tile_the_file_without_overlap():
    assert spans(chunk_source_fixed(TEN_LINES, window=4)) == [(1, 4), (5, 8), (9, 10)]


def test_overlap_shifts_by_window_minus_overlap_and_stops_at_the_end():
    assert spans(chunk_source_fixed(TEN_LINES, window=4, overlap=2)) == [(1, 4), (3, 6), (5, 8), (7, 10)]


def test_window_text_matches_its_line_range():
    c = chunk_source_fixed(TEN_LINES, "f.sol", window=4)[1]
    assert c.text == "line5\nline6\nline7\nline8\n"
    assert (c.id, c.kind, c.name) == ("f.sol:L5-8", "window", "L5-8")


def test_window_embed_text_has_no_structural_header():
    src = "contract A {\n  function f() public { x = 1; }\n}\n"
    [c] = chunk_source_fixed(src, window=10)
    assert c.contract == "A"
    assert c.embed_text == c.clean_text


def test_contract_owner_is_the_last_declared_before_the_window():
    src = "pragma solidity ^0.4.0;\ncontract A {\n}\ncontract B {\n  uint x;\n}\n"
    assert [c.contract for c in chunk_source_fixed(src, window=2)] == ["?", "A", "B"]


def test_blank_and_annotation_only_windows_are_dropped():
    src = "/*\n * @source: https://x/reentrancy.sol\n * @author: -\n * @vulnerable_at_lines: 3\n\n\ncontract A {}\n"
    chunks = chunk_source_fixed(src, window=2)
    assert all(c.clean_text.strip() for c in chunks)
    assert "reentrancy" not in " ".join(c.clean_text for c in chunks)


@pytest.mark.parametrize("window,overlap", [(0, 0), (4, 4), (4, -1)])
def test_invalid_window_settings_are_rejected(window, overlap):
    with pytest.raises(ValueError):
        chunk_source_fixed(TEN_LINES, window=window, overlap=overlap)


def test_header_source_and_author_lines_are_stripped():
    header = (
        "/*\n"
        " * @source: https://github.com/x/integer_overflow/integer_overflow_1.sol\n"
        " * @author: Suhabe Bugrara\n"
        " * @vulnerable_at_lines: 12\n"
        " */\n"
    )
    assert strip_label_leaks(header) == "/*\n */\n"


def test_loader_accepts_a_chunk_function(tmp_path):
    (tmp_path / "a.sol").write_text(TEN_LINES)
    assert len(load_chunks(tmp_path, fixed_chunker(5)).chunks) == 2


def test_overlapping_windows_count_a_vulnerable_line_once(tmp_path):
    p = tmp_path / "v.json"
    p.write_text(json.dumps([{"path": "dataset/x/a.sol", "vulnerabilities": [{"lines": [5], "category": "c"}]}]))
    ann = Annotations.from_json(p)
    chunks = chunk_source_fixed(TEN_LINES, "x/a.sol", window=4, overlap=2)
    hits = [c for c in chunks if vulnerable_lines(c, ann, "c")]
    assert spans(hits) == [(3, 6), (5, 8)]
    assert len(set().union(*(vulnerable_lines(c, ann, "c") for c in hits))) == 1
    assert len(lines_covered(hits)) == 6  # lines 3..8


def test_evaluate_reports_lines_and_vulnerable_lines(tmp_path):
    p = tmp_path / "v.json"
    p.write_text(json.dumps([{"path": "dataset/x/a.sol", "vulnerabilities": [{"lines": [5], "category": "c"}]}]))
    chunks = chunk_source_fixed(TEN_LINES, "x/a.sol", window=4, overlap=2)
    idx = Index.build(chunks)
    df = evaluate({"bm25": BM25Retriever(idx)}, {"c": ["line5 line6"]}, Annotations.from_json(p), chunks, k=2)
    row = df.iloc[0]
    assert row["vlines@2"] == 1
    assert row["lines@2"] == 6


def test_cli_fixed_chunker_is_recorded_in_the_index(tmp_path):
    corpus = tmp_path / "dataset"
    corpus.mkdir()
    (corpus / "a.sol").write_text("contract A {\n" + TEN_LINES + "}\n")
    out = tmp_path / "idx"
    r = CliRunner().invoke(
        app, ["index", str(corpus), "--out", str(out), "--no-embed", "--chunker", "fixed", "--window", "4", "--overlap", "1"]
    )
    assert r.exit_code == 0, r.output
    assert "fixed:w=4,o=1" in r.output
    assert Index.load(out).chunking == "fixed:w=4,o=1"
