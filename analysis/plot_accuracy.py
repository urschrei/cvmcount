# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib>=3.9", "numpy>=2"]
# ///
"""Plot the accuracy of CVM estimates against the epsilon guarantee.

Reads the CSV that `cargo run --release --example accuracy` writes, saves a
figure, and prints a Markdown summary table.

    uv run analysis/plot_accuracy.py accuracy.csv cvmcount.png
"""

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter

POINT = "#2f5d8a"
BAND = "#dde6f0"
MODEL = "#b5562b"
INK = "#222222"
MUTED = "#666666"
# Two-sided normal quantiles for 90 % and 95 % intervals.
Z90 = 1.645
Z95 = 1.96


def signed(value: float, digits: int = 1) -> str:
    """Format a signed percentage, with a true minus sign."""
    return f"{value:+.{digits}f}%".replace("-", "−")


def percent(value: float, _position: int) -> str:
    """Format an axis tick as a signed percentage."""
    return "0%" if value == 0 else signed(value, 0)


def buffer_size(epsilon: float, delta: float, stream_size: int) -> int:
    """Return the buffer threshold, as `buffer_size` in src/lib.rs does."""
    return math.ceil(12 / epsilon**2 * math.log2(8 * stream_size / delta))


def expected_level(distinct: int, buffer: int) -> int:
    """Return k for the sampling probability 2**-k at which the expected
    buffer occupancy, distinct * 2**-k, first fits in the buffer."""
    k = 0
    while distinct >> k >= buffer:
        k += 1
    return k


def off_level(estimates: np.ndarray, k: int, buffer: int) -> int:
    """Count the estimates that cannot come from a run that ended at level k.

    Such a run returns |X| * 2**k with |X| < buffer.
    """
    counts = estimates.astype(np.int64)
    return int(np.sum((counts % (1 << k) != 0) | (counts >> k >= buffer)))


def model_sd(distinct: int, k: int) -> float:
    """Return the relative standard deviation, in percent, of |X| / p when
    each distinct element is in X independently with probability p = 2**-k.

    This is the fixed-level sampling model of Chakraborty, Vinodchandran and
    Meel (Claim 5 in arXiv:2301.10191v2).
    """
    p = 2.0**-k
    return 100 * math.sqrt((1 - p) / (distinct * p))


def read(path: Path) -> tuple[dict[float, np.ndarray], int, float, int]:
    estimates: dict[float, list[float]] = defaultdict(list)
    actual, delta, stream_size = 0, 0.0, 0
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            estimates[float(row["epsilon"])].append(float(row["estimate"]))
            actual = int(row["actual"])
            delta = float(row["delta"])
            stream_size = int(row["stream_size"])
    arrays = {e: np.array(v) for e, v in estimates.items()}
    return arrays, actual, delta, stream_size


def main(src: Path, dest: Path) -> None:
    estimates, actual, delta, stream_size = read(src)
    epsilons = sorted(estimates, reverse=True)
    errors = {e: (estimates[e] / actual - 1) * 100 for e in epsilons}
    runs = len(estimates[epsilons[0]])
    misses = sum(int(np.sum(np.abs(errors[e]) > e * 100)) for e in epsilons)
    worst = max(float(np.max(np.abs(errors[e]))) for e in epsilons)
    limit = 5 * math.ceil(worst * 1.3 / 5)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9,
        }
    )
    fig, ax = plt.subplots(figsize=(7, 4.9), layout="constrained")
    # A fixed seed keeps the horizontal jitter the same between renders.
    jitter = np.random.default_rng(0)

    buffers = {e: buffer_size(e, delta, stream_size) for e in epsilons}
    levels = {e: expected_level(actual, buffers[e]) for e in epsilons}
    for e in epsilons:
        off = off_level(estimates[e], levels[e], buffers[e])
        if off:
            print(
                f"warning: at ε = {e:g}, {off} of {runs} runs did not end at "
                f"p = 1/{2 ** levels[e]}; the model does not describe them",
                file=sys.stderr,
            )

    for i, epsilon in enumerate(epsilons):
        band = epsilon * 100
        ax.add_patch(Rectangle((i - 0.42, -band), 0.84, 2 * band, color=BAND, zorder=0))
        err = errors[epsilon]
        x = i + jitter.uniform(-0.3, 0.3, err.size)
        ax.scatter(x, err, s=3, color=POINT, alpha=0.25, linewidths=0, zorder=2)
        spread = Z90 * model_sd(actual, levels[epsilon])
        if spread > 0:
            ax.hlines(
                [-spread, spread],
                i - 0.42,
                i + 0.42,
                color=MODEL,
                lw=1,
                ls=(0, (4, 2)),
                zorder=3,
            )
        # The 95% confidence interval of the mean is too narrow to see at
        # this scale; the printed table reports it.
        ax.scatter(i, float(np.mean(err)), s=14, marker="D", color=INK, zorder=4)
        worst_here = float(np.max(np.abs(err)))
        label = "exact" if worst_here == 0 else f"worst {worst_here:.1f}%"
        ax.text(i, limit * 0.92, label, ha="center", va="top", color=MUTED, fontsize=8)

    ax.axhline(0, color=INK, lw=0.6, zorder=1)
    ax.set_xlim(-0.6, len(epsilons) - 0.4)
    ax.set_ylim(-limit, limit)
    ax.set_xticks(
        range(len(epsilons)),
        [f"ε = {e:g}\n{buffers[e]:,} words" for e in epsilons],
    )
    ax.yaxis.set_major_formatter(FuncFormatter(percent))
    ax.set_xlabel("Accuracy requirement ε, and the buffer size it implies", color=INK)
    ax.set_ylabel("Error relative to the true count", color=INK)
    ax.tick_params(colors=INK, length=0, axis="x")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.suptitle(
        "CVM estimates of distinct words in The King in Yellow",
        x=0.02,
        ha="left",
        fontsize=11,
        color=INK,
    )
    ax.set_title(
        f"{stream_size:,} words, {actual:,} distinct. {runs:,} runs per ε, "
        f"δ = {delta:g}: up to {delta:.0%} of runs may fall outside ±ε; "
        f"{misses} did.",
        loc="left",
        fontsize=8.5,
        color=MUTED,
    )
    fig.legend(
        handles=[
            Patch(color=BAND, label="±ε guarantee"),
            Line2D([], [], ls="", marker="o", ms=3, color=POINT, label="one run"),
            Line2D([], [], ls="", marker="D", ms=3.5, color=INK, label="mean"),
            Line2D(
                [],
                [],
                color=MODEL,
                lw=1,
                ls=(0, (4, 2)),
                label="sampling model: 90% of runs",
            ),
        ],
        loc="outside lower center",
        ncols=4,
        frameon=False,
        fontsize=8,
    )
    fig.savefig(dest, dpi=300, facecolor="white")

    print(
        "| ε | buffer | final p | mean (95% CI) | 5th–95th percentile | model 90% range | worst |"
    )
    print("|---|---|---|---|---|---|---|")
    for e in epsilons:
        err = errors[e]
        mean = float(np.mean(err))
        ci = Z95 * float(np.std(err, ddof=1)) / math.sqrt(err.size)
        lo, hi = np.percentile(err, [5, 95])
        spread = Z90 * model_sd(actual, levels[e])
        print(
            f"| {e:g} | {buffers[e]:,} | 1/{2 ** levels[e]} | {signed(mean, 2)} (±{ci:.2f}%) "
            f"| {signed(lo)} to {signed(hi)} | ±{spread:.1f}% | {np.max(np.abs(err)):.1f}% |"
        )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: plot_accuracy.py ACCURACY_CSV OUTPUT_IMAGE")
    main(Path(sys.argv[1]), Path(sys.argv[2]))
