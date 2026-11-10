"""Plot the chunking baseline: the top-k view next to the equal-budget view.

    python scripts/plot_chunking.py benchmarks/v0.2

Reads ``chunking_summary.csv`` and ``chunking_budget.csv`` written by
``chunking_baseline.py`` and saves two PNGs in the same directory:

* ``fig_topk.png``: hit@5 against the lines those 5 chunks cover. Shows that
  function chunks read far more text than 8-line windows at the same k.
* ``fig_budget.png``: hit rate when every chunker reads the same number of lines.
* ``fig_ast.png``: AST chunks against fixed windows of the same size (hybrid
  retrieval, equal line budgets), with function chunks as a reference.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# four series max; categorical slots 1-4 in fixed order, plus distinct markers
SERIES = [
    ("function", "function chunks", "#2a78d6", "o"),
    ("fixed:w=8,o=0", "fixed, 8 lines", "#eb6834", "s"),
    ("fixed:w=16,o=0", "fixed, 16 lines", "#1baf7a", "^"),
    ("fixed:w=32,o=0", "fixed, 32 lines", "#eda100", "D"),
]
RETRIEVERS = ["bm25", "dense", "hybrid"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.labelsize": 9.5,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9.5,
    }
)


def _finish(fig, axes, title: str, subtitle: str, out: Path) -> None:
    for ax in axes:
        ax.set_ylim(0, 0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=INK, fontweight="bold")
    fig.text(0.01, 0.875, subtitle, ha="left", fontsize=9.5, color=INK_2)
    fig.subplots_adjust(top=0.76, bottom=0.24, left=0.055, right=0.99, wspace=0.08)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"saved {out}")


def plot_topk(summary: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    for ax, retriever in zip(axes, RETRIEVERS):
        for key, label, color, marker in SERIES:
            row = summary[(summary.chunking == key) & (summary.retriever == retriever)].iloc[0]
            x, y = row["lines@5"], row["hit@5"]
            ax.scatter(x, y, s=90, color=color, marker=marker, edgecolor=SURFACE, linewidth=2, zorder=3, label=label)
        ax.set_title(retriever, loc="left")
        ax.set_xlim(0, 175)
        ax.set_xlabel("lines covered by the top 5 chunks")
    axes[0].set_ylabel("hit@5 (share of queries)")
    _finish(
        fig, axes,
        "Top-5 scores: function chunks get to read more text",
        "Same k, different amounts of code. Function chunks (mean 8.4 lines) retrieve 76 to 105 lines in their "
        "top 5; 8-line windows retrieve about 39.",
        out,
    )


def plot_budget(budget: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    budgets = sorted(budget.budget.unique())
    for ax, retriever in zip(axes, RETRIEVERS):
        for key, label, color, marker in SERIES:
            rows = budget[(budget.chunking == key) & (budget.retriever == retriever)].sort_values("budget")
            ax.plot(rows.budget, rows.hit, color=color, linewidth=2, marker=marker, markersize=7,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=label, zorder=3)
        ax.set_title(retriever, loc="left")
        ax.set_xscale("log", base=2)
        ax.set_xticks(budgets, [str(b) for b in budgets])
        ax.minorticks_off()
        ax.set_xlabel("line budget (lines read per query)")
    axes[0].set_ylabel("hit rate (share of queries)")
    _finish(
        fig, axes,
        "Equal reading budget: small windows beat whole-function chunks",
        "Each retriever reads ranked chunks until the budget is used up. 27 queries; most gaps are within noise "
        "(see chunking_budget.csv for intervals).",
        out,
    )


def plot_ast(budget: pd.DataFrame, out: Path, retriever: str = "hybrid") -> None:
    sizes = sorted(int(c.split("=")[1]) for c in budget.chunking.unique() if c.startswith("ast:max="))
    fig, axes = plt.subplots(1, len(sizes), figsize=(11, 3.8), sharey=True)
    budgets = sorted(budget.budget.unique())
    for ax, size in zip(axes, sizes):
        series = [
            ("function", "function chunks", "#2a78d6", "o"),
            (f"fixed:w={size},o=0", "fixed window, same size", "#eb6834", "s"),
            (f"ast:max={size}", "AST chunks (cAST), same size", "#1baf7a", "^"),
        ]
        for key, label, color, marker in series:
            rows = budget[(budget.chunking == key) & (budget.retriever == retriever)].sort_values("budget")
            ax.plot(rows.budget, rows.hit, color=color, linewidth=2, marker=marker, markersize=7,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=label, zorder=3)
        ax.set_title(f"chunks of at most {size} lines", loc="left")
        ax.set_xscale("log", base=2)
        ax.set_xticks(budgets, [str(b) for b in budgets])
        ax.minorticks_off()
        ax.set_xlabel("line budget (lines read per query)")
    axes[0].set_ylabel(f"hit rate, {retriever} retrieval")
    _finish(
        fig, axes,
        "Syntax-aware boundaries help most at 16 lines",
        "Same chunk size, different boundaries. AST chunks split at statements and merge small siblings; "
        "windows cut every N lines. 27 queries.",
        out,
    )


def main() -> None:
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "benchmarks/v0.2")
    plot_topk(pd.read_csv(d / "chunking_summary.csv"), d / "fig_topk.png")
    plot_budget(pd.read_csv(d / "chunking_budget.csv"), d / "fig_budget.png")
    plot_ast(pd.read_csv(d / "chunking_budget.csv"), d / "fig_ast.png")


if __name__ == "__main__":
    main()
