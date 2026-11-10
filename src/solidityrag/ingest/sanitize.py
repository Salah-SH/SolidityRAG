"""Remove dataset annotations that would leak the answer into retrieval.

SmartBugs Curated marks vulnerable lines with comments such as
``// <yes> <report> REENTRANCY`` and file headers containing
``@vulnerable_at_lines: 31``, and many ``@source`` header URLs name the
category (``.../integer_overflow_minimal.sol``). ``@author`` lines go too, as
they identify which benchmark suite a file came from. If those stay in the indexed text, BM25 and
embeddings can match the label itself and the benchmark measures nothing.
"""

from __future__ import annotations

import re

_LEAKS = re.compile(
    r"//[ \t]*<yes>[ \t]*<report>[^\n]*"
    r"|^[ \t]*\*?[ \t]*@(?:vulnerable_at_lines|source|author)\b[^\n]*\n?",
    re.M,
)


def strip_label_leaks(text: str) -> str:
    return _LEAKS.sub("", text)