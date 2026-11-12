"""Figures of the classifier estimator: benchmark (fig_cls_benchmark) and yearly series of the study (fig_cls_series);
the per-consultation figure is drawn by figures.fig_consultations on the study cells.

  python src/civic/figures_cls.py [--study results/study_cls] [--out paper/figures]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figures import CAT, COL_W, FULL_W, GREY, LIGHT, fig_consultations, style  # noqa: E402

INK = "#0b0b0b"
NAMES = {"decide_madrid": "Decide Madrid", "barcelona": "decidim.barcelona", "participa_gencat": "participa.gencat.cat",
         "participationcitoyenne_ccomptes": "Cour des comptes", "eu_hys_citizens": "Have Your Say, citizens",
         "eu_hys_organisations": "Have Your Say, organisations", "regulations_gov_epa": "Regulations.gov (EPA)",
         "regulations_gov_dot": "Regulations.gov (DOT)", "meta_decidim": "Metadecidim", "jeparticipe_toulouse": "Toulouse",
         "participation_bordeaux_metropole": "Bordeaux Métropole", "oye_participer_lyon": "Lyon",
         "regulations_gov_usda": "Regulations.gov (USDA)", "regulations_gov_doi": "Regulations.gov (DOI)"}
SHORT = {"eu_hys_citizens": "EU portal, citizens", "eu_hys_organisations": "EU portal, organisations",
         "regulations_gov_epa": "Regulations.gov EPA", "regulations_gov_dot": "Regulations.gov DOT"}
LANG = {"es": "Spanish", "ca": "Catalan", "fr": "French", "en": "English", "de": "German", "it": "Italian", "sv": "Swedish",
        "pt": "Portuguese", "fi": "Finnish"}
PANELS = [("decide_madrid", "es"), ("barcelona", "ca"), ("participa_gencat", "ca"), ("participationcitoyenne_ccomptes", "fr"),
          ("eu_hys_citizens", "en"), ("eu_hys_citizens", "fr"), ("eu_hys_citizens", "de"), ("eu_hys_citizens", "it"),
          ("eu_hys_organisations", "en"), ("eu_hys_organisations", "de"), ("regulations_gov_epa", "en"), ("regulations_gov_dot", "en")]


def fig_benchmark(quant: pd.DataFrame, lexical: pd.DataFrame, bino: pd.DataFrame, method: str, out: Path):
    """Forward validation with the generator held out: mean estimate against the true share for the classifier estimator
    and for the lexical estimator (one line per held-out generator, by group), and mean absolute error of the methods."""
    COMM = ["gpt-5.4-mini", "gpt-4o", "gpt-4o-mini", "claude-haiku-5-5", "gemini-3.6-flash"]
    OPEN_C = "#a3a29d"
    q = quant[~quant.heldout_model.str.startswith("group:")]
    fig = plt.figure(figsize=(FULL_W, 2.35))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.35], left=0.06, right=0.985, bottom=0.19, top=0.89, wspace=0.12)
    a0 = fig.add_subplot(gs[0]); a1 = fig.add_subplot(gs[1], sharey=a0)
    gs2 = gs[2].subgridspec(1, 2, width_ratios=[0.95, 1], wspace=0)
    axes = [a0, a1, fig.add_subplot(gs2[1])]
    for ax, d, title in ((axes[0], q[(q.method == method) & (q.exp == "period")], "Classifier estimator"),
                         (axes[1], lexical[(lexical.exp == "period") & (lexical.setting == "adapted|out")], "Lexical estimator")):
        ax.plot([0, 0.5], [0, 0.5], color=INK, lw=0.8, ls=(0, (4, 2)), zorder=5)
        g = d.groupby(["heldout_model", "alpha"]).alpha_hat.mean().unstack(0)
        for m in g.columns:
            comm = m in COMM
            ax.plot(g.index, g[m], color=CAT[0] if comm else OPEN_C, lw=1.0 if comm else 0.8, alpha=0.9, zorder=4 if comm else 3)
        ax.set_xlim(-0.02, 0.52); ax.set_ylim(-0.02, 0.62)
        ax.set_xticks([0, 0.1, 0.2, 0.3, 0.5]); ax.set_xticklabels(["0", "10", "20", "30", "50"])
        ax.set_yticks([0, 0.2, 0.4, 0.6]); ax.set_yticklabels(["0", "20", "40", "60"])
        ax.set_xlabel("true share (%)"); ax.set_title(title, loc="left")
        style(ax, "both")
    axes[1].tick_params(labelleft=False)
    axes[0].set_ylabel("estimated share (%)")
    from matplotlib.lines import Line2D
    axes[0].legend(handles=[Line2D([], [], color=CAT[0], lw=1.0), Line2D([], [], color=OPEN_C, lw=0.8),
                            Line2D([], [], color=INK, lw=0.8, ls=(0, (4, 2)))],
                   labels=["commercial generator held out", "open-weight generator held out", "exact estimate"],
                   loc="upper left", fontsize=6, handlelength=1.6, handletextpad=0.5, borderaxespad=0.1, labelspacing=0.3)
    # mean absolute error by method and protocol: two bars per method
    ax = axes[2]
    q = q.assign(ae=(q.alpha_hat - q.alpha).abs() * 100)
    lexical = lexical.assign(ae=(lexical.alpha_hat - lexical.alpha).abs() * 100)
    bino = bino[~bino.heldout_model.str.startswith("group:")]
    bino = bino.assign(ae=(bino.alpha_hat - bino.alpha).abs() * 100)

    def two(d, m):
        return [d[(d.method == m) & (d.exp == e)].ae.mean() for e in ("instance", "period")]
    rows = [("Classifier + HDy (ours)", two(q, method)), ("Classifier + adjusted count", two(q, "acc_sp_inst")),
            ("Classifier + EM", two(q, "em_inst")), ("Classifier, classify and count", two(q, "cc_inst")),
            ("Lexical estimator", [lexical[(lexical.exp == e) & (lexical.setting == "adapted|out")].ae.mean() for e in ("instance", "period")]),
            ("Binoculars + HDy", two(bino, method if (bino.method == method).any() else "hdyw_inst"))]
    y = np.arange(len(rows))[::-1].astype(float)
    h = 0.3
    for (lab, (a, b)), yy in zip(rows, y):
        for v, off, c in ((a, h / 2 + 0.02, CAT[0]), (b, -h / 2 - 0.02, CAT[1])):
            ax.barh(yy + off, v, height=h, color=c, zorder=3)
            ax.annotate(f"{v:.1f}", xy=(v, yy + off), xytext=(3, 0), textcoords="offset points", va="center", fontsize=6, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, max(max(r[1]) for r in rows) * 1.14); ax.set_ylim(-0.55, len(rows) - 0.45)
    ax.set_xlabel("mean absolute error (pp)"); ax.set_title("Error by method", loc="left")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=CAT[0]), Patch(color=CAT[1])], labels=["instance protocol", "forward protocol"],
              loc="upper right", fontsize=6.5, handlelength=1.0, handleheight=0.8, handletextpad=0.4, borderaxespad=0.1)
    style(ax, "x")
    fig.savefig(out / "fig_cls_benchmark.pdf"); fig.savefig(out / "fig_cls_benchmark.png", dpi=200)
    plt.close(fig)


def fig_series(cy: pd.DataFrame, cyn: pd.DataFrame, out: Path, panels=PANELS):
    """Yearly estimate with the benchmark-calibrated interval, generator-free lower bound and estimate without templated
    letters, for the main instance-language pairs."""
    panels = [p for p in panels if ((cy.instance == p[0]) & (cy.lang == p[1]) & (cy.year >= 2023)).sum() >= 2]
    ncol = 4
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(FULL_W, 1.42 * nrow + 0.35), sharex=True, squeeze=False)
    for ax, (inst, lang) in zip(axes.ravel(), panels):
        g = cy[(cy.instance == inst) & (cy.lang == lang) & (cy.year >= 2016)].sort_values("year")
        n = cyn[(cyn.instance == inst) & (cyn.lang == lang) & (cyn.year >= 2016)].sort_values("year")
        ax.fill_between(g.year, 100 * g.band_lo, 100 * g.band_hi, color=CAT[0], alpha=0.16, lw=0, zorder=2)
        if len(n) and (g.template_share.max() >= 0.05):
            ax.plot(n.year, 100 * n.alpha, color=CAT[1], lw=1.1, marker="s", ms=2.2, zorder=4)
        ax.plot(g.year, 100 * g.lb_lo, color=GREY, lw=0.9, ls=(0, (2, 1.5)), zorder=3)
        ax.plot(g.year, 100 * g.alpha, color=CAT[0], lw=1.4, marker="o", ms=2.6, zorder=5)
        ax.axvline(2022.91, color=GREY, ls="--", lw=0.6, zorder=1)
        top = max(12.0, 100 * g.band_hi.max() * 1.08)
        ax.set_ylim(-0.02 * top, top)
        ax.set_xlim(2015.6, 2026.4)
        ax.set_xticks([2016, 2019, 2022, 2025]); ax.set_xticklabels(["2016", "2019", "2022", "2025"])
        ax.set_title(f"{SHORT.get(inst, NAMES.get(inst, inst))} ({lang})", loc="left", fontsize=7, pad=2.5)
        style(ax)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("share (%)")
    h = [plt.Line2D([], [], color=CAT[0], lw=1.4, marker="o", ms=2.6, label="estimate"),
         plt.Rectangle((0, 0), 1, 1, color=CAT[0], alpha=0.16, lw=0, label="calibrated 95% interval"),
         plt.Line2D([], [], color=GREY, lw=0.9, ls=(0, (2, 1.5)), label="generator-free lower bound"),
         plt.Line2D([], [], color=CAT[1], lw=1.1, marker="s", ms=2.2, label="estimate without templated letters")]
    fig.legend(handles=h, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01), handlelength=1.6, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0.045, 1, 1), h_pad=2.0, w_pad=0.6)   # the title must read as the title of the panel below it
    fig.savefig(out / "fig_cls_series.pdf"); fig.savefig(out / "fig_cls_series.png", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default="results/study_cls")
    ap.add_argument("--quant", default="results/supervised/quant_aug.parquet")
    ap.add_argument("--method", default="hdyv_inst")
    ap.add_argument("--out", default="paper/figures")
    a = ap.parse_args()
    S, O = Path(a.study), Path(a.out)
    fig_benchmark(pd.read_parquet(a.quant), pd.read_parquet("results/robust/robust.parquet"),
                  pd.read_parquet("results/supervised/quant_binoculars.parquet"), a.method, O)
    fig_series(pd.read_csv(S / "cells_instance_year.csv"), pd.read_csv(S / "cells_instance_year_notemplate.csv"), O)
    names = dict(NAMES)
    fig_consultations(pd.read_csv(S / "cells_consultation.csv"), O, names)
    print("figures written to", O)


if __name__ == "__main__":
    main()
