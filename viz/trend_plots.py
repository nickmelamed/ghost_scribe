"""Trend plots across flywheel rounds.

Three charts, each on a single axis:

  - the share of essays escalated to Stage 2 and the share sent to a human
  - the false-positive gap on non-native English writing, in percentage points
  - cost per essay in illustrative relative units, stacked by stage

Round 2 appears as a pending slot until results/round2.json exists. The chart
never fills in a value that has not been measured. Cost units are assumptions
(1 for the Stage-1 pass, 5 per Stage-2 escalation, 25 per human review), not
real money.

Run from the repo root:
    python -m viz.trend_plots
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluation.round2 import round1_summary

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
ROUND_LABELS = ["Round 1", "Round 2"]


def load_rounds(results_dir: Path = RESULTS_DIR) -> tuple[dict, dict | None]:
    """Return the round-1 summary and the round-2 summary, or None if round 2 has not run."""
    path = results_dir / "round2.json"
    if path.exists():
        saved = json.loads(path.read_text())
        return saved["round1"], saved["round2"]
    return round1_summary(results_dir), None


def series(round1: dict, round2: dict | None, getter) -> list[float | None]:
    """Return one value per round, with None for a round that has not run."""
    return [getter(round1), getter(round2) if round2 is not None else None]


def style_axes(ax, title: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=TEXT_PRIMARY, pad=12)
    ax.set_ylabel(ylabel, fontsize=9, color=TEXT_SECONDARY)
    ax.set_xlim(-0.5, 1.85)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(ROUND_LABELS, fontsize=9, color=TEXT_SECONDARY)
    ax.tick_params(axis="y", labelsize=8, colors=TEXT_SECONDARY, length=0)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def mark_pending(ax, y: float) -> None:
    ax.text(1, y, "Pending\nafter HITL", ha="center", va="center", fontsize=8, color=TEXT_SECONDARY, style="italic")


def new_figure():
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def plot_lines(ax, values_by_label: dict[str, list[float | None]], colors: list[str], fmt: str) -> None:
    for (label, values), color in zip(values_by_label.items(), colors, strict=True):
        points = [(x, v) for x, v in enumerate(values) if v is not None]
        ax.plot(
            [p[0] for p in points],
            [p[1] for p in points],
            color=color,
            linewidth=2,
            marker="o",
            markersize=8,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
            label=label,
        )
        for x, v in points:
            ax.annotate(fmt.format(v), (x, v), textcoords="offset points", xytext=(10, 0), va="center",
                        fontsize=8.5, color=TEXT_PRIMARY)
    ax.legend(loc="upper left", fontsize=8.5, frameon=False, labelcolor=TEXT_SECONDARY, bbox_to_anchor=(0, 1.0))


def plot_rates(round1: dict, round2: dict | None, path: Path) -> None:
    fig, ax = new_figure()
    data = {
        "Escalated to Stage 2": series(round1, round2, lambda r: 100 * r["escalation_rate"]),
        "Sent to human review": series(round1, round2, lambda r: 100 * r["review_rate"]),
    }
    plot_lines(ax, data, [BLUE, ORANGE], "{:.1f}%")
    style_axes(ax, "Share of essays escalated and reviewed", "Percent of test essays")
    ax.set_ylim(0, max(v for vals in data.values() for v in vals if v is not None) * 1.5)
    if round2 is None:
        mark_pending(ax, ax.get_ylim()[1] * 0.45)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_gap(round1: dict, round2: dict | None, path: Path) -> None:
    fig, ax = new_figure()
    data = {
        "Stage 1 at 0.5 threshold": series(round1, round2, lambda r: 100 * r["gap_stage1"]),
        "Combined pipeline": series(round1, round2, lambda r: 100 * r["gap_combined"]),
    }
    plot_lines(ax, data, [BLUE, ORANGE], "{:+.2f}")
    style_axes(ax, "False-positive gap on non-native English writing", "ESL minus general, percentage points")
    top = max(abs(v) for vals in data.values() for v in vals if v is not None)
    ax.set_ylim(-0.35 * max(top, 0.5), max(top, 0.5) * 1.6)
    ax.axhline(0, color=TEXT_SECONDARY, linewidth=1)
    if round2 is None:
        mark_pending(ax, ax.get_ylim()[1] * 0.4)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_cost(round1: dict, round2: dict | None, path: Path) -> None:
    fig, ax = new_figure()
    parts = [("Stage-1 pass", "stage1", BLUE), ("Stage-2 escalation", "stage2", ORANGE), ("Human review", "review", AQUA)]
    totals: list[float | None] = []
    for x, summary in enumerate([round1, round2]):
        if summary is None:
            totals.append(None)
            continue
        bottom = 0.0
        for label, key, color in parts:
            value = summary["cost"][key]
            ax.bar(x, value, bottom=bottom, width=0.5, color=color, edgecolor=SURFACE, linewidth=2, label=label if x == 0 else None)
            ax.annotate(f"{label} {value:.2f}", (x + 0.27, bottom + value / 2), va="center", fontsize=8.5, color=TEXT_PRIMARY)
            bottom += value
        totals.append(bottom)
        ax.annotate(f"Total {bottom:.2f}", (x, bottom), textcoords="offset points", xytext=(0, 6), ha="center",
                    fontsize=9, color=TEXT_PRIMARY, fontweight="bold")
    style_axes(ax, "Cost per essay in illustrative units", "Relative cost units (assumed)")
    ax.set_xlim(-0.5, 2.1)
    known = [t for t in totals if t is not None]
    ax.set_ylim(0, max(known) * 1.3)
    ax.axhline(1, color=TEXT_SECONDARY, linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate("Stage 1 alone = 1", (1.55, 1), textcoords="offset points", xytext=(0, 4), fontsize=8, color=TEXT_SECONDARY)
    ax.legend(loc="upper right", fontsize=8.5, frameon=False, labelcolor=TEXT_SECONDARY)
    if round2 is None:
        mark_pending(ax, max(known) * 0.5)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main(results_dir: Path = RESULTS_DIR, figures_dir: Path = FIGURES_DIR) -> list[Path]:
    round1, round2 = load_rounds(results_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = [figures_dir / "rates.png", figures_dir / "fairness_gap.png", figures_dir / "cost_per_decision.png"]
    for plot, path in zip((plot_rates, plot_gap, plot_cost), paths, strict=True):
        plot(round1, round2, path)
        print(f"Wrote {path}")
    return paths


if __name__ == "__main__":
    main()
