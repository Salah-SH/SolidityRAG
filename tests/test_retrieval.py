import numpy as np
import pytest
from helpers import HashEmbedder

from solidityrag.index.store import Index
from solidityrag.ingest.chunker import chunk_source
from solidityrag.retrieve import BM25Retriever, DenseRetriever, HybridRetriever, rrf
from solidityrag.tokenize import tokenize

SRC = """contract Wallet {
    mapping(address => uint) balances;
    function withdrawAll() public {
        msg.sender.call.value(balances[msg.sender])();
        balances[msg.sender] = 0;
    }
    function setOwner(address o) public { owner = o; }
    function useDelegate(address lib, bytes data) public { lib.delegatecall(data); }
    function lottery() public { uint r = uint(block.blockhash(block.number - 1)); }
}"""


@pytest.fixture
def index():
    return Index.build(chunk_source(SRC, "w.sol"), HashEmbedder())


def test_tokenizer_splits_camel_and_snake_but_keeps_whole_identifiers():
    toks = tokenize("withdrawAll(); tx.origin; my_var")
    assert {"withdrawall", "withdraw", "all", "tx", "origin", "my_var", "my", "var"} <= set(toks)


def test_rrf_item_ranked_first_by_both_lists_wins_and_singletons_survive():
    fused = rrf([[1, 2, 3], [1, 4]])
    ids = [i for i, _ in fused]
    assert ids[0] == 1
    assert set(ids) == {1, 2, 3, 4}


def test_rrf_scores_follow_the_formula():
    fused = dict(rrf([[7, 8], [8]], c=60))
    assert fused[7] == pytest.approx(1 / 61)
    assert fused[8] == pytest.approx(1 / 62 + 1 / 61)


def test_bm25_finds_exact_token_matches(index):
    top = BM25Retriever(index).search("delegatecall", k=1)[0]
    assert top.chunk.name == "useDelegate"
    assert top.rank == 1


def test_bm25_works_on_a_two_chunk_corpus():
    # stock BM25Okapi gives a term in 1 of 2 documents an IDF of exactly 0
    src = "contract A { function a() public { x.delegatecall(d); } function b() public { y = 1; } }"
    idx = Index.build(chunk_source(src, "a.sol"))
    results = BM25Retriever(idx).search("delegatecall", k=2)
    assert [r.chunk.name for r in results] == ["a"]


def test_bm25_returns_nothing_when_no_term_matches(index):
    assert BM25Retriever(index).search("zzzunseenterm", k=5) == []


def test_dense_returns_k_results_ranked(index):
    results = DenseRetriever(index, HashEmbedder()).search("blockhash lottery", k=3)
    assert len(results) == 3
    assert results[0].chunk.name == "lottery"
    assert [r.rank for r in results] == [1, 2, 3]


def test_hybrid_combines_both_rankings(index):
    bm25 = BM25Retriever(index)
    dense = DenseRetriever(index, HashEmbedder())
    hybrid = HybridRetriever(index, [dense, bm25])
    names = [r.chunk.name for r in hybrid.search("call value balances withdraw", k=3)]
    assert names[0] == "withdrawAll"
    assert len(names) == 3


def test_dense_requires_embeddings():
    idx = Index.build(chunk_source(SRC, "w.sol"))
    with pytest.raises(ValueError):
        DenseRetriever(idx, HashEmbedder())


def test_index_cannot_be_empty():
    with pytest.raises(ValueError):
        Index([])


def test_index_save_and_load_roundtrip(index, tmp_path):
    index.save(tmp_path / "idx")
    loaded = Index.load(tmp_path / "idx")
    assert loaded.chunks == index.chunks
    assert np.allclose(loaded.embeddings, index.embeddings)
    assert loaded.embedder_name == "hash-test"
    assert BM25Retriever(loaded).search("delegatecall", k=1)[0].chunk.name == "useDelegate"


def test_bm25_only_index_has_no_embeddings(tmp_path):
    idx = Index.build(chunk_source(SRC, "w.sol"))
    idx.save(tmp_path / "idx")
    assert Index.load(tmp_path / "idx").embeddings is None