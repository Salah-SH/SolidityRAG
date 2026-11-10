from solidityrag.ingest.chunker import chunk_source


def names(chunks):
    return [(c.contract, c.name) for c in chunks]


def test_plain_function_with_line_range():
    src = (
        "pragma solidity ^0.8.0;\n"
        "contract A {\n"
        "    uint x;\n"
        "    function set(uint v) public {\n"
        "        x = v;\n"
        "    }\n"
        "}\n"
    )
    [c] = chunk_source(src, "a.sol")
    assert (c.contract, c.name, c.kind) == ("A", "set", "function")
    assert (c.start_line, c.end_line) == (4, 6)
    assert c.text.startswith("function set") and c.text.endswith("}")
    assert c.id == "a.sol:A.set:4"


def test_modifier_and_constructor():
    src = """contract A {
    modifier onlyOwner() { require(msg.sender == owner); _; }
    constructor() public { owner = msg.sender; }
    function f() public onlyOwner { }
}"""
    chunks = chunk_source(src)
    assert [(c.kind, c.name) for c in chunks] == [
        ("modifier", "onlyOwner"),
        ("constructor", "constructor"),
        ("function", "f"),
    ]


def test_legacy_unnamed_fallback_is_named_fallback():
    src = "contract A { function() payable { deposits += msg.value; } }"
    [c] = chunk_source(src)
    assert (c.kind, c.name) == ("fallback", "fallback")


def test_new_style_receive_and_fallback():
    src = """contract A {
    receive() external payable { }
    fallback() external payable { }
}"""
    assert [(c.kind, c.name) for c in chunk_source(src)] == [
        ("receive", "receive"),
        ("fallback", "fallback"),
    ]


def test_interface_declarations_without_bodies_are_skipped():
    src = """interface I {
    function a() external;
    function b(uint x) external returns (uint);
}
contract C { function real() public { } }"""
    assert names(chunk_source(src)) == [("C", "real")]


def test_nested_braces_find_the_correct_end():
    src = """contract A {
    function f(uint n) public {
        for (uint i = 0; i < n; i++) {
            if (i == 2) { revert(); }
        }
    }
    function g() public { }
}"""
    f, g = chunk_source(src)
    assert (f.start_line, f.end_line) == (2, 6)
    assert (g.start_line, g.end_line) == (7, 7)


def test_braces_and_keywords_in_comments_and_strings_are_ignored():
    src = """contract A {
    // function fake() { not real
    string s = "}{ function nope() {";
    /* function alsoFake() { */
    function real() public {
        string memory m = "} // not a comment";
    }
}"""
    assert names(chunk_source(src)) == [("A", "real")]


def test_two_contracts_in_one_file_get_the_right_owner():
    src = """contract A { function a() public { } }
library L { function l() internal { } }
contract B { function b() public { } }"""
    assert names(chunk_source(src)) == [("A", "a"), ("L", "l"), ("B", "b")]


def test_no_chunk_for_keywords_nested_inside_a_body():
    src = """contract A {
    function f() public {
        bool fallback = true;
        uint receive = 1;
    }
}"""
    assert names(chunk_source(src)) == [("A", "f")]


def test_unbalanced_braces_do_not_crash():
    assert chunk_source("contract A { function f() public { if (x) { } ") == []


def test_overloaded_functions_get_distinct_ids():
    src = """contract A {
    function f(uint a) public { }
    function f(uint a, uint b) public { }
}"""
    ids = [c.id for c in chunk_source(src, "a.sol")]
    assert len(set(ids)) == 2


def test_label_leaks_are_stripped_from_clean_text_but_kept_in_raw_text():
    src = """contract A {
    function w() public {
        msg.sender.call.value(1)(); // <yes> <report> REENTRANCY
        // <yes> <report> REENTRANCY
        x = 0;
    }
}"""
    [c] = chunk_source(src)
    assert "REENTRANCY" in c.text
    assert "REENTRANCY" not in c.clean_text
    assert "msg.sender.call.value(1)();" in c.clean_text  # the code on that line survives
    assert c.embed_text.startswith("// A.w\n")