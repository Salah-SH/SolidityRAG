"""Remove dataset annotations that would leak the answer into retrieval.

SmartBugs Curated marks vulnerable lines with comments such as
``// <yes> <report> REENTRANCY`` and file headers containing
``@vulnerable_at_lines: 31``. If those stay in the indexed text, BM25 and
embeddings can match the label itself and the benchmark measures nothing.
"""

from __future__ import annotations

import re

_LEAKS = re.compile(
    r"//[ \t]*<yes>[ \t]*<report>[^\n]*"
    r"|^[ \t]*\*?[ \t]*@vulnerable_at_lines[^\n]*\n?",
    re.M,
)


def strip_label_leaks(text: str) -> str:
    return _LEAKS.sub("", text)