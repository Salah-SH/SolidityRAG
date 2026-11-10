from helpers import FakeLLMClient, HashEmbedder

from solidityrag.generate import ask, build_prompt, check_citations, extract_labels
from solidityrag.index.store import Index
from solidityrag.ingest.chunker import chunk_source
from solidityrag.retrieve import BM25Retriever

SRC = """contract A {
    function w() public { msg.sender.call.value(1)(); balance = 0; }
    function z() public { x = 1; }
}"""


def test_extract_labels_handles_lists_and_adjacent_citations():
    text = "Bad [C1]. Also bad [C2, C3] and [C1][C4]. Not a label [x] or [C]."
    assert extract_labels(text) == ["C1", "C2", "C3", "C1", "C4"]


def test_citation_report_flags_unknown_labels():
    report = check_citations("A [C1] B [C9]", {"C1", "C2"})
    assert report.valid == ["C1"]
    assert report.invalid == ["C9"]
    assert report.validity == 0.5


def test_citation_validity_is_none_without_citations():
    assert check_citations("no citations here", {"C1"}).validity is None


def test_prompt_labels_map_back_to_chunks():
    idx = Index.build(chunk_source(SRC, "a.sol"), HashEmbedder())
    results = BM25Retriever(idx).search("call value", k=2)
    system, user, label_map = build_prompt("Is w reentrant?", results)
    assert "[C1]" in user and "lines=" in user and "Is w reentrant?" in user
    assert label_map["C1"] is results[0].chunk
    assert "Do not invent" in system


def test_ask_uses_the_client_and_checks_citations():
    idx = Index.build(chunk_source(SRC, "a.sol"), HashEmbedder())
    client = FakeLLMClient("Function w sends ether before updating state [C1]. See also [C7].")
    answer = ask("reentrancy in w", BM25Retriever(idx), k=2, client=client)
    assert client.calls[0]["temperature"] == 0.0
    assert "Evidence" in client.calls[0]["messages"][0]["content"]
    assert answer.citations.valid == ["C1"]
    assert answer.citations.invalid == ["C7"]