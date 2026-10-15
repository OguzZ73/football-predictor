"""Matplotlib figures saved as PNG: score heatmap, calibration, final positions."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SERIES = {"home": "#2a78d6", "draw": "#eb6834", "away": "#1baf7a"}
BLUES = LinearSegmentedColormap.from_list("seq_blue", [
    "#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=9)


def score_heatmap(matrix: np.ndarray, home: str, away: str, path: Path | str,
                  subtitle: str = "") -> Path:
    """Annotated heatmap of P(home goals, away goals)."""
    n = matrix.shape[0]
    fig, ax = plt.subplots(figsize=(7, 6.2), facecolor=SURFACE)
    im = ax.imshow(matrix * 100, cmap=BLUES, origin="upper")
    vmax = matrix.max() * 100
    for i in range(n):
        for j in range(n):
            v = matrix[i, j] * 100
            if v >= 0.05:
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8,
                        color="white" if v > 0.55 * vmax else TEXT)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xlabel(f"{away} goals", color=TEXT_2)
    ax.set_ylabel(f"{home} goals", color=TEXT_2)
    ax.tick_params(length=0, colors=TEXT_2)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("probability (%)", color=TEXT_2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=TEXT_2, labelsize=8)
    fig.suptitle(f"{home} vs {away} — score probabilities (%)", color=TEXT,
                 fontsize=12, x=0.02, ha="left")
    if subtitle:
        ax.set_title(subtitle, color=TEXT_2, fontsize=9, loc="left")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def calibration_plot(cal: pd.DataFrame, path: Path | str, title: str = "",
                     min_n: int = 5) -> Path:
    """Small multiples: predicted vs observed frequency per outcome with 95% intervals."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3), facecolor=SURFACE, sharey=True)
    names = {"home": "Home win", "draw": "Draw", "away": "Away win"}
    for ax, (label, color) in zip(axes, SERIES.items()):
        _style(ax)
        d = cal[(cal["outcome"] == label) & (cal["n"] >= min_n)]
        ax.plot([0, 1], [0, 1], color=TEXT_2, lw=1, ls=(0, (3, 3)), zorder=1)
        ax.vlines(d["predicted"], d["lo"], d["hi"], color=color, lw=2, alpha=0.45, zorder=2)
        ax.plot(d["predicted"], d["observed"], color=color, lw=2, marker="o", ms=8,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(color=GRID, lw=0.6)
        ax.set_title(names[label], color=TEXT, fontsize=11, loc="left")
        ax.set_xlabel("predicted probability", color=TEXT_2)
    axes[0].set_ylabel("observed frequency", color=TEXT_2)
    if title:
        fig.suptitle(title, color=TEXT, fontsize=12, x=0.01, ha="left")
    fig.text(0.01, 0.01, f"10 equal-width bins (bins with < {min_n} matches hidden); "
             "vertical bars = 95% Wilson interval; dashed line = perfect calibration",
             color=TEXT_2, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def positions_heatmap(positions: pd.DataFrame, path: Path | str, title: str = "") -> Path:
    """Teams × final position probability heatmap from a season simulation."""
    T = positions.shape[1]
    fig, ax = plt.subplots(figsize=(0.45 * T + 3, 0.32 * len(positions) + 1.5), facecolor=SURFACE)
    vals = positions.to_numpy() * 100
    im = ax.imshow(vals, cmap=BLUES, aspect="auto", vmin=0)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            if vals[i, j] >= 1:
                ax.text(j, i, f"{vals[i, j]:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if vals[i, j] > 0.55 * vals.max() else TEXT)
    ax.set_xticks(range(T), positions.columns)
    ax.set_yticks(range(len(positions)), positions.index)
    ax.tick_params(length=0, colors=TEXT_2, labelsize=8)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("final position", color=TEXT_2)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("probability (%)", color=TEXT_2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=TEXT_2, labelsize=8)
    if title:
        ax.set_title(title, color=TEXT, fontsize=11, loc="left")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def league_strength_plot(lt: pd.DataFrame, path: Path | str) -> Path:
    """Horizontal bars of league strength; unconnected leagues shown as hollow circles."""
    d = lt.sort_values("strength")
    connected = d["uefa_matches"] > 0 if "uefa_matches" in d else pd.Series(True, index=d.index)
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(d) + 1.4), facecolor=SURFACE)
    _style(ax)
    y = np.arange(len(d))
    ax.barh(y[connected.to_numpy()], d.loc[connected, "strength"], color=SERIES["home"], height=0.7)
    ax.scatter(d.loc[~connected, "strength"], y[~connected.to_numpy()], marker="o", s=40,
               facecolor=SURFACE, edgecolor=TEXT_2, linewidth=1.5, zorder=3)
    ax.set_yticks(y, d["name"] if "name" in d else d["league"], fontsize=8)
    ax.axvline(0, color=TEXT_2, lw=0.8)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_xlabel("league strength (attack − defence, log goals; 0 = average)", color=TEXT_2)
    ax.set_title("League strength from UEFA club matches", color=TEXT, fontsize=11, loc="left")
    fig.text(0.01, 0.005, "hollow circle = no UEFA matches in the data: not connected, level stays at the prior (0)",
             color=TEXT_2, fontsize=7)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path
