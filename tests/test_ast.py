import pytest

pytest.importorskip("tree_sitter_solidity")

from typer.testing import CliRunner  # noqa: E402

from solidityrag.cli import app  # noqa: E402
from solidityrag.index.store import Index  # noqa: E402
from solidityrag.ingest import chunk_source, chunk_source_ast, chunk_source_ast_functions  # noqa: E402

SRC = """pragma solidity ^0.4.24;

contract Bank {
    mapping(address => uint) balances;
    address owner;

    modifier onlyOwner() { require(msg.sender == owner); _; }

    constructor() public { owner = msg.sender; }

    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }

    function withdrawAll() public {
        uint amount = balances[msg.sender];
        require(amount > 0);
        if (amount > 100) {
            emit Big(amount);
            log0("big");
        }
        msg.sender.call.value(amount)();
        balances[msg.sender] = 0;
    }

    function() public payable { }
}

interface Token { function transfer(address to, uint v) external returns (bool); }
"""


def spans(chunks):
    return [(c.start_line, c.end_line) for c in chunks]


def test_ast_functions_match_the_regex_chunker():
    ast, rx = chunk_source_ast_functions(SRC, "b.sol"), chunk_source(SRC, "b.sol")
    assert [(c.id, c.kind, c.start_line, c.end_line, c.text) for c in ast] == [
        (c.id, c.kind, c.start_line, c.end_line, c.text) for c in rx
    ]


def test_bodyless_interface_functions_are_skipped():
    assert "transfer" not in [c.name for c in chunk_source_ast_functions(SRC)]


@pytest.mark.parametrize("max_lines", [4, 8, 16])
def test_ast_chunks_respect_the_budget_and_cover_every_function(max_lines):
    chunks = chunk_source_ast(SRC, "b.sol", max_lines)
    assert all(c.end_line - c.start_line + 1 <= max_lines for c in chunks)
    covered = {n for c in chunks for n in range(c.start_line, c.end_line + 1)}
    for f in chunk_source(SRC):
        assert set(range(f.start_line, f.end_line + 1)) <= covered, f.name


def test_ast_chunks_never_cut_a_statement():
    # with a 4-line budget withdrawAll must be split, but the if-block (4 lines) stays whole
    chunks = chunk_source_ast(SRC, "b.sol", 4)
    if_start = SRC.splitlines().index("        if (amount > 100) {") + 1
    holder = [c for c in chunks if c.start_line <= if_start <= c.end_line]
    assert len(holder) == 1 and holder[0].end_line >= if_start + 3


def test_split_function_pieces_keep_the_function_name_in_the_header():
    pieces = [c for c in chunk_source_ast(SRC, "b.sol", 4) if c.name == "withdrawAll"]
    assert len(pieces) >= 2
    assert all(c.embed_text.startswith("// Bank.withdrawAll\n") for c in pieces)


def test_whole_contract_fits_in_one_chunk_with_a_large_budget():
    chunks = chunk_source_ast(SRC, "b.sol", 100)
    assert any(c.contract == "Bank" and c.end_line - c.start_line > 20 for c in chunks)


def test_chunks_do_not_merge_across_contracts():
    chunks = chunk_source_ast(SRC, "b.sol", 100)
    assert {c.contract for c in chunks if c.start_line >= SRC.splitlines().index("interface Token { function transfer(address to, uint v) external returns (bool); }") + 1} == {"Token"}


def test_cli_ast_chunker(tmp_path):
    corpus = tmp_path / "dataset"
    corpus.mkdir()
    (corpus / "b.sol").write_text(SRC)
    out = tmp_path / "idx"
    r = CliRunner().invoke(app, ["index", str(corpus), "--out", str(out), "--no-embed", "--chunker", "ast", "--window", "6"])
    assert r.exit_code == 0, r.output
    assert Index.load(out).chunking == "ast:max=6"
