"""Revised publication figures (Nov 2026): audit dot plot, benchmark MAE vs. detectors, Madrid series with the
reference-free bound. Same style constants as figures.py.

Usage: python -m civic.figures_v2 --results results --out paper/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from civic.figures import CAT, GREY, LIGHT, LANG_NAMES, COL_W, FULL_W, PARTIAL_YEAR, PARTIAL_LABEL, ours, style

INK = "#0b0b0b"
MUTED = "#8a8985"
BLUE, ORANGE, AQUA = CAT[0], CAT[1], CAT[2]
SCEN = [("s1_improve", "S1 improve", BLUE), ("s2_notes", "S2 notes", AQUA), ("s3_persuasive", "S3 persuasive", ORANGE)]
MODEL_NAMES = {"claude-haiku-5-5": "Claude Haiku 5.5", "eurollm-9b": "EuroLLM 9B", "gemma-2-9b-it": "Gemma 2 9B",
               "gpt-5.4-mini": "GPT-5.4 mini", "gpt-oss-20b": "gpt-oss-20b", "llama-3.1-8b": "Llama 3.1 8B",
               "mistral-small-24b": "Mistral Small 24B", "qwen2.5-7b": "Qwen2.5 7B", "salamandra-7b": "Salamandra 7B"}


def _save(fig, out: Path, name: str):
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- audit: one dot per generator
def fig_audit(metrics: pd.DataFrame, homog: pd.DataFrame | None, out: Path):
    panels = [("l1_any_new", "L1 new legal ref.", "% of outputs", 100, "{:.0f}"),
              ("l2_ents_kept", "L2 entities kept", "% of input", 100, "{:.0f}"),
              ("l4_len_ratio", "L4 length ratio", "output / input", 1, "{:.1f}"),
              ("l5_output_grounded", "L5 grounded", "% of output", 100, "{:.0f}")]
    COMM = ["gpt-5.4-mini", "claude-haiku-5-5", "gemini-3.6-flash"]   # only the commercial generators are shown; the bar is the mean over their outputs
    metrics = metrics[metrics.model.isin(COMM)]
    if homog is not None:
        homog = homog[homog.model.isin(COMM)]
    g = metrics.groupby(["mode", "model"]).mean(numeric_only=True)
    pooled = metrics[metrics.model.isin(COMM)].groupby("mode").mean(numeric_only=True)
    if homog is not None:
        g = g.join(homog.groupby(["mode", "model"]).dcos_trunc.mean())
        pooled = pooled.join(homog[homog.model.isin(COMM)].groupby("mode").dcos_trunc.mean())
        panels.insert(2, ("dcos_trunc", "L3 similarity change", "$\\Delta$ cosine", 1, "{:+.2f}"))
    fig, axes = plt.subplots(1, len(panels), figsize=(FULL_W, 2.0), sharey=True)
    rng = np.random.default_rng(7)
    for ax, (col, title, xlabel, scale, fmt) in zip(axes, panels):
        style(ax, grid="x")
        ax.spines["left"].set_visible(False)
        for k, (mode, label, color) in enumerate(SCEN):
            gm = g.loc[mode, col] * scale
            comm = gm.index.isin(COMM)
            y = k + rng.uniform(-0.16, 0.16, len(gm))
            ax.scatter(gm.values[~comm], y[~comm], s=11, facecolors="none", edgecolors=color, linewidths=0.6, alpha=0.75, zorder=3)
            ax.scatter(gm.values[comm], y[comm], s=13, color=color, linewidths=0, zorder=3.5)
            m = pooled.loc[mode, col] * scale
            ax.plot([m, m], [k - 0.3, k + 0.3], color=color, lw=2.2, solid_capstyle="round", zorder=4)
            ax.annotate(fmt.format(0.0 if abs(m) < 0.005 and "+" in fmt else m).replace("+0.00", "0.00"), (m, k - 0.3), xytext=(0, 1.0), textcoords="offset points",
                        ha="center", va="bottom", fontsize=6.5, color=INK)
        if col == "dcos_trunc":
            ax.axvline(0, color=GREY, lw=0.6, zorder=2)
        if col == "l4_len_ratio":
            ax.axvline(1, color=GREY, lw=0.6, zorder=2)
        # 3-5 labelled ticks covering the data in every panel
        from matplotlib.ticker import MaxNLocator
        ticks = {"l1_any_new": [0, 25, 50, 75], "l2_ents_kept": [30, 45, 60, 75], "l5_output_grounded": [30, 45, 60]}
        if col in ticks:
            ax.set_xticks(ticks[col])
        else:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 3, 5, 10], min_n_ticks=3))
        ax.set_title(title, loc="left", fontsize=7.5, color=INK, pad=3)
        ax.set_xlabel(xlabel, fontsize=7, color=GREY)
        ax.set_ylim(2.5, -0.85)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(SCEN)))
    axes[0].set_yticklabels([s[1] for s in SCEN])
    for t, (_, _, c) in zip(axes[0].get_yticklabels(), SCEN):
        t.set_color(INK)
    fig.subplots_adjust(wspace=0.18, top=0.78)
    # legend: scenarios as colour patches; the two mark types drawn exactly as in the panels, in the three colours
    from matplotlib.legend_handler import HandlerTuple
    handles = [Line2D([], [], ls="", marker="s", ms=6, color=c, label=lab) for _, lab, c in SCEN]
    dots = tuple(Line2D([], [], ls="", marker="o", ms=3.8, color=c, mew=0) for _, _, c in SCEN)
    rings = tuple(Line2D([], [], ls="", marker="o", ms=3.6, mfc="none", mec=c, mew=0.6, alpha=0.75) for _, _, c in SCEN)
    bars = tuple(Line2D([], [], ls="", marker="|", ms=9, mew=2.2, color=c) for _, _, c in SCEN)
    fig.legend(handles=handles + [dots, bars],
               labels=[s[1] for s in SCEN] + ["one generator", "mean of the outputs"],
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)}, loc="upper center",
               bbox_to_anchor=(0.5, 1.02), ncol=6, frameon=False, fontsize=7, handlelength=2.2, handletextpad=0.4,
               columnspacing=1.2)
    _save(fig, out, "fig_audit")


# --------------------------------------------------------------------------- benchmark: ours vs corrected detectors
def fig_benchmark(raw: pd.DataFrame, out: Path):
    r = ours(raw)
    r = r[r.scenario == "all"].copy()
    r["err"] = (r.alpha_hat - r.alpha).abs() * 100
    d = r[r.method == "distributional"].groupby(["lang", "heldout_model"]).err.mean()
    dm = d.groupby("lang").mean().sort_values()
    det = [("pangram+rogan_gladen", "Open Pangram + RG", "o"), ("fastdetect+rogan_gladen", "Fast-DetectGPT + RG", "s"),
           ("binoculars+rogan_gladen", "Binoculars + RG", "^")]
    dd = r[r.method.isin([m for m, _, _ in det])].groupby(["method", "lang"]).err.mean()
    langs = list(dm.index)
    fig, ax = plt.subplots(figsize=(COL_W, 2.75))
    style(ax, grid="x")
    ax.spines["left"].set_visible(False)
    rng = np.random.default_rng(3)
    for i, lg in enumerate(langs):
        lo = min(d.loc[lg].min(), min(dd.loc[(m, lg)] for m, _, _ in det))
        hi = max(d.loc[lg].max(), max(dd.loc[(m, lg)] for m, _, _ in det))
        ax.plot([lo, hi], [i, i], color=LIGHT, lw=0.8, zorder=1)
        v = d.loc[lg].values
        ax.scatter(v, i + rng.uniform(-0.13, 0.13, len(v)), s=8, color=BLUE, alpha=0.4, linewidths=0, zorder=3)
        # mean diamond on a white halo so that it stands out from the per-generator dots underneath
        ax.scatter([dm[lg]], [i], s=95, marker="D", color="white", linewidths=0, zorder=5)
        ax.scatter([dm[lg]], [i], s=46, marker="D", color=BLUE, edgecolor="#0d366b", linewidths=0.6, zorder=6)
        # the three detectors are often within a point of each other (e.g. English): dodge them vertically
        for (m, _, mk), dy in zip(det, (-0.2, 0.0, 0.2)):
            ax.scatter([dd.loc[(m, lg)]], [i + dy], s=26, marker=mk, facecolor="white", edgecolor=ORANGE,
                       linewidths=1.1, zorder=4)
    ax.set_yticks(range(len(langs)))
    ax.set_yticklabels([LANG_NAMES[l] for l in langs], color=INK, fontsize=8.3)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=8.3)
    ax.set_ylim(len(langs) - 0.4, -0.6)
    ax.set_xlim(0, None)
    ax.set_xlabel("Mean absolute error of the estimated share (pp)", fontsize=8.5)
    handles = [Line2D([], [], marker="D", ls="", color=BLUE, markeredgecolor="#0d366b", mew=0.6, ms=6, label="Ours, mean"),
               Line2D([], [], marker="o", ls="", color=BLUE, alpha=0.45, ms=3.5, label="Ours, per held-out generator")]
    handles += [Line2D([], [], marker=mk, ls="", markerfacecolor="white", markeredgecolor=ORANGE, mew=1.1, ms=5,
                       label=lab) for _, lab, mk in det]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2, handletextpad=0.3,
              columnspacing=1.0, fontsize=7.4)
    _save(fig, out, "fig_benchmark_mae")


# --------------------------------------------------------------------------- Madrid: estimate and reference-free bound
def fig_timeseries(years: pd.DataFrame, excess: pd.DataFrame, out: Path, instance="decide_madrid", lang="es"):
    y = years[(years.instance == instance) & (years.lang == lang)].sort_values("year")
    e = excess[(excess.instance == instance) & (excess.lang == lang)].sort_values("year")
    fig, ax = plt.subplots(figsize=(COL_W, 1.9))
    style(ax, grid="y")
    ax.axvline(2022.92, color=GREY, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.text(2022.8, 29.5, "ChatGPT", ha="right", va="top", fontsize=6.8, color=GREY)
    ax.fill_between(y.year, y.ci_lo * 100, y.ci_hi * 100, color=BLUE, alpha=0.25, linewidth=0, zorder=2)
    ax.plot(y.year, y.alpha * 100, color=BLUE, lw=1.8, marker="o", ms=3.5, mec="white", mew=0.6, zorder=4,
            label="Distributional estimate")
    part = y[y.year == PARTIAL_YEAR]
    if len(part):  # incomplete last year: open marker and a short note
        ax.plot(part.year, part.alpha * 100, ls="", marker="o", ms=4.2, mfc="white", mec=BLUE, mew=1.2, zorder=5)
        ax.annotate(f"{PARTIAL_YEAR}: {PARTIAL_LABEL}", (PARTIAL_YEAR, part.alpha.iloc[0] * 100), xytext=(-5, 7),
                    textcoords="offset points", ha="right", va="bottom", fontsize=6.3, color=GREY)
    for kind, filled in [("placebo", False), ("post", True)]:
        s = e[e.kind == kind]
        ax.errorbar(s.year + 0.12, s.bound * 100, yerr=[(s.bound - s.ci_lo) * 100, (s.ci_hi - s.bound) * 100],
                    fmt="s", ms=4, color=ORANGE, mfc=ORANGE if filled else "white", mec=ORANGE, mew=1.0,
                    elinewidth=0.8, capsize=0, zorder=3,
                    label="Excess-vocabulary indicator" if filled else "Indicator, placebo years")
    ax.set_ylabel("LLM share (%)")
    ax.set_ylim(-1.5, 31)
    ax.set_xlim(y.year.min() - 0.5, y.year.max() + 0.6)
    ax.set_xticks(range(int(y.year.min()), int(y.year.max()) + 1, 2))
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=6.6, handlelength=1.4, borderaxespad=0.2)
    _save(fig, out, "fig_timeseries")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="paper/figures")
    a = ap.parse_args()
    R, O = Path(a.results), Path(a.out)
    homog = pd.read_csv(R / "audit_homog.csv") if (R / "audit_homog.csv").exists() else None
    fig_audit(pd.read_parquet(R / "audit_metrics.parquet"), homog, O)
    fig_benchmark(pd.read_parquet(R / "benchmark.parquet"), O)
    fig_timeseries(pd.read_csv(R / "alpha_cells_instance_year.csv"), pd.read_csv(R / "excess" / "excess.csv"), O)
    print("revised figures written to", O)


if __name__ == "__main__":
    main()
