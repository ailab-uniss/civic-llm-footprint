"""Publication figures (PDF, vector) for the paper. Matplotlib only; palette follows the dataviz reference
(categorical: blue, orange, aqua, yellow, magenta, green, violet, red; sequential: one blue ramp).

Usage: python figures.py --results results --out paper/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948", "#7a5c2e"]
SEQ = LinearSegmentedColormap.from_list("seqblue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#e8e8e6", "#f5a673", "#c0390b"])
GREY = "#52514e"
LIGHT = "#d9d9d6"
LANG_NAMES = {"ca": "Catalan", "es": "Spanish", "fr": "French", "it": "Italian", "fi": "Finnish", "en": "English",
              "sv": "Swedish", "pt": "Portuguese", "de": "German"}
METHOD_LABELS = {"distributional": "Distributional (ours)", "binoculars+rogan_gladen": "Binoculars + RG",
                 "fastdetect+rogan_gladen": "Fast-DetectGPT + RG", "pangram+rogan_gladen": "Open Pangram + RG",
                 "pangram+bayes": "Open Pangram + Bayes", "binoculars+bayes": "Binoculars + Bayes",
                 "fastdetect+bayes": "Fast-DetectGPT + Bayes", "pangram+raw": "Open Pangram raw rate",
                 "binoculars+raw": "Binoculars raw rate", "fastdetect+raw": "Fast-DetectGPT raw rate"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 8,
    "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": GREY, "axes.linewidth": 0.6, "xtick.color": GREY, "ytick.color": GREY, "axes.labelcolor": "#0b0b0b",
    "axes.spines.top": False, "axes.spines.right": False, "grid.color": LIGHT, "grid.linewidth": 0.5,
    "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
COL_W, FULL_W = 3.45, 7.1  # IEEE column / full width in inches
PARTIAL_YEAR, PARTIAL_LABEL = 2026, "Jan–Nov"  # last year of data is incomplete (collection ended Nov 2026)


def style(ax, grid="y"):
    ax.grid(axis=grid, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, width=0.5)


# --------------------------------------------------------------------------- Fig: benchmark MAE heatmap
def best_config(raw: pd.DataFrame) -> tuple[str, str]:
    """(selection, likelihood) of the distributional estimator with the lowest MAE over all scenarios and languages."""
    if "likelihood" not in raw.columns:
        raw["likelihood"] = "occurrence"
    raw["likelihood"] = raw.likelihood.fillna("")
    d = raw[raw.method == "distributional"].copy()
    d["err"] = (d.alpha_hat - d.alpha).abs()
    return d.groupby(["selection", "likelihood"]).err.mean().idxmin()


def ours(raw: pd.DataFrame) -> pd.DataFrame:
    sel, lik = best_config(raw)
    return raw[(raw.method != "distributional") | ((raw.selection == sel) & (raw.likelihood == lik))]


def fig_benchmark_heatmap(summary: pd.DataFrame, raw: pd.DataFrame, out: Path):
    raw = ours(raw)
    d = raw[(raw.method == "distributional") & (raw.scenario == "all")].copy()
    d["err"] = (d.alpha_hat - d.alpha).abs()
    m = d.groupby(["lang", "heldout_model"]).err.mean().unstack()
    m = m.rename(index=LANG_NAMES)
    fig, ax = plt.subplots(figsize=(COL_W, 0.32 * len(m) + 1.0))
    # cells are drawn as vector rectangles (an image with masked cells renders unreliably in PDF viewers)
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import FuncFormatter
    vmax = max(0.05, float(np.nanmax(m.values)))
    norm = Normalize(vmin=0, vmax=vmax)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.values[i, j]
            if np.isnan(v):
                continue
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=SEQ(norm(v)), edgecolor="white", lw=1.5))
            ax.text(j, i, f"{100 * v:.1f}", ha="center", va="center", fontsize=6.5,
                    color="white" if v > 0.6 * vmax else "#0b0b0b")
    ax.set_xlim(-0.5, m.shape[1] - 0.5)
    ax.set_ylim(m.shape[0] - 0.5, -0.5)
    ax.set_xticks(range(m.shape[1]))
    ax.set_xticklabels(m.columns, rotation=35, ha="right")
    ax.set_yticks(range(m.shape[0]))
    ax.set_yticklabels(m.index)
    ax.set_xlabel("Held-out generator")
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(False)
    sm = ScalarMappable(norm=norm, cmap=SEQ)
    cb = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02, format=FuncFormatter(lambda t, _: f"{100 * t:.0f}"))
    cb.set_label("MAE of $\\hat\\alpha$ (pp)")
    cb.outline.set_visible(False)
    fig.savefig(out / "fig_benchmark_heatmap.pdf"); fig.savefig(out / "fig_benchmark_heatmap.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- Fig: calibration vs baselines
def fig_calibration(raw: pd.DataFrame, out: Path, langs: list[str] | None = None):
    """Two panels: our estimator and the best detector baseline, one line per language (mean of alpha_hat over
    repetitions and held-out generators against the true alpha). Languages, not methods, carry the colour so
    that cross-language differences are visible at a glance."""
    d = ours(raw)
    d = d[(d.scenario == "all")].copy()
    d["err"] = (d.alpha_hat - d.alpha).abs()
    base = [m for m in ["binoculars+rogan_gladen", "fastdetect+rogan_gladen", "pangram+rogan_gladen"] if m in set(d.method)]
    if base:
        base = [d[d.method.isin(base)].groupby("method").err.mean().idxmin()]
    methods = ["distributional"] + base
    langs = langs or sorted(d.lang.unique())
    # every language gets its own marker (and the two colour-close pairs, German/Portuguese and English/French,
    # also differ in line style) so that the curves stay distinguishable in greyscale
    markers = ["o", "s", "^", "D", "v", "P", "X", "<", "h"]
    dashed = {"pt", "fr"}
    ymax = d[d.method.isin(methods)].groupby(["method", "lang", "alpha"]).alpha_hat.mean().max()
    ytop = max(0.55, float(np.ceil((ymax + 0.04) * 20) / 20))
    fig, axes = plt.subplots(1, len(methods), figsize=(COL_W, 2.45), sharex=True, sharey=True, squeeze=False)
    for k, meth in enumerate(methods):
        ax = axes[0][k]
        ax.plot([0, 1], [0, 1], color=GREY, lw=0.9, ls=(0, (4, 2)), zorder=2)
        for i, lang in enumerate(langs):
            gg = d[(d.lang == lang) & (d.method == meth)].groupby("alpha").alpha_hat.mean()
            if gg.empty:
                continue
            ax.plot(gg.index, gg.values, color=CAT[i % len(CAT)], lw=1.2, marker=markers[i % len(markers)], ms=3.2,
                    mew=0, ls=(0, (3, 1.5)) if lang in dashed else "-", label=LANG_NAMES.get(lang, lang), zorder=3)
        ax.set_title(METHOD_LABELS.get(meth, meth), loc="left")
        style(ax, grid="both")
        ax.set_xlim(-0.01, 0.52)
        ax.set_ylim(-0.01, ytop)
        ax.set_xlabel("True $\\alpha$")
        if k == 0:
            ax.set_ylabel("Estimated $\\hat\\alpha$")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(5, len(l)), bbox_to_anchor=(0.5, -0.12), handlelength=2.0, columnspacing=0.9)
    fig.tight_layout()
    fig.savefig(out / "fig_calibration.pdf"); fig.savefig(out / "fig_calibration.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- Fig: time series of alpha
def fig_timeseries(cells: pd.DataFrame, out: Path, instance_names: dict | None = None, min_cells: int = 4,
                   keep: set | None = None):
    """keep: (instance, lang) pairs to show (e.g. those whose null-calibration floor is small); None = all."""
    cells = cells.dropna(subset=["alpha"]).copy()
    if keep is not None:
        cells = cells[[(i, l) in keep for i, l in zip(cells.instance, cells.lang)]]
    cells["year"] = cells.half.str[:4].astype(int) + np.where(cells.half.str.endswith("2"), 0.75, 0.25)
    # a series whose pre-ChatGPT half-years exceed 10% shows its own false-positive floor rather than LLM use
    pre_max = cells[cells.year < 2022.91].groupby(["instance", "lang"]).alpha.max()
    noisy = {k for k, v in pre_max.items() if v > 0.10}
    cells = cells[[(i, l) not in noisy for i, l in zip(cells.instance, cells.lang)]]
    langs = sorted(cells.lang.unique(), key=lambda l: -len(cells[cells.lang == l]))
    n = len(langs)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    width = COL_W if n <= 2 else FULL_W
    fig, axes = plt.subplots(nrow, ncol, figsize=(width, 2.0 * nrow), sharey=True, squeeze=False)
    for k, lang in enumerate(langs):
        ax = axes[k // ncol][k % ncol]
        g = cells[cells.lang == lang]
        insts = [i for i, c in g.groupby("instance").size().items() if c >= min_cells]
        insts = sorted(insts, key=lambda i: -len(g[g.instance == i]))[:8]
        for i, inst in enumerate(insts):
            gi = g[g.instance == inst].sort_values("year")
            ax.fill_between(gi.year, gi.ci_lo, gi.ci_hi, color=CAT[i], alpha=0.12, lw=0)
            ax.plot(gi.year, gi.alpha, color=CAT[i], lw=1.4, marker="o", ms=2.5,
                    label=(instance_names or {}).get(inst, inst))
        ax.axvline(2022.91, color=GREY, ls="--", lw=0.8)
        ax.annotate("ChatGPT ", xy=(2022.91, 0.97), xycoords=("data", "axes fraction"), color=GREY, fontsize=6.5,
                    ha="right", va="top")
        ax.set_title(LANG_NAMES.get(lang, lang), loc="left")
        style(ax, grid="y")
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        ax.legend(loc="upper center", fontsize=6, ncol=3 if len(insts) > 4 else 2, bbox_to_anchor=(0.5, -0.14), handlelength=1.5)
        if k % ncol == 0:
            ax.set_ylabel("$\\hat\\alpha$ (LLM share)")
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.tight_layout(h_pad=2.2)
    fig.savefig(out / "fig_timeseries.pdf"); fig.savefig(out / "fig_timeseries.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- Fig: consultation portals, per consultation
def fig_consultations(cells: pd.DataFrame, out: Path, instance_names: dict | None = None, min_consultations: int = 10,
                      min_post: int = 10, max_panels: int = 9):
    """One point per process (publication on Have Your Say, docket on Regulations.gov, participatory process on Decidim)
    with at least --min-docs texts, placed at the year of its median text; the line joins the yearly medians."""
    cells = cells.dropna(subset=["alpha"]).copy()
    cells = cells[cells.instance != "eu_hys_unknown"]
    cnt = cells.groupby(["instance", "lang"]).size()
    cnt = cnt[cnt >= min_consultations]
    post = cells[cells.post == 1].groupby(["instance", "lang"]).size().reindex(cnt.index).fillna(0)
    # pre-ChatGPT processes are the false-positive floor: keep pairs whose pre-ChatGPT yearly medians stay at or below 15%
    pre_med = cells[cells.post == 0].groupby(["instance", "lang", "year"]).alpha.median().groupby(level=[0, 1]).max()
    pre_med = pre_med.reindex(cnt.index).fillna(0)
    panels = [k for k in cnt[(post >= min_post) & (pre_med <= 0.15)].sort_values(ascending=False).index][:max_panels]
    n = len(panels)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    # one time convention for points and medians: a process sits at the decimal date of its median text, a yearly
    # median at mid-year (year + 0.5); common x range, ticks and y range across panels so that they are comparable
    if "created_at" in cells:
        t = pd.to_datetime(cells.created_at, utc=True)
        cells["t"] = t.dt.year + (t.dt.dayofyear - 0.5) / np.where(t.dt.is_leap_year, 366, 365)
    else:
        cells["t"] = cells.year + 0.5
    sel = cells[[(i, l) in set(panels) for i, l in zip(cells.instance, cells.lang)]]
    y0 = int(sel.year.min())
    xlo = float(sel.t.min())
    y0 -= (y0 % 2)  # even-year ticks
    y1 = int(sel.year.max())
    ytop = float(np.ceil((sel.alpha.max() + 0.04) * 10) / 10)
    fig, axes = plt.subplots(nrow, ncol, figsize=(FULL_W, 1.9 * nrow + 0.3), sharey=True, squeeze=False)
    rng = np.random.default_rng(0)
    for k, (inst, lang) in enumerate(panels):
        ax = axes[k // ncol][k % ncol]
        g = cells[(cells.instance == inst) & (cells.lang == lang)].copy()
        x = g.t + rng.uniform(-0.04, 0.04, len(g))
        size = 6 + 10 * np.log10(g.n_docs / 100)
        ax.scatter(x, g.alpha, s=size, color=CAT[0], alpha=0.55, lw=0, zorder=3)
        med = g.groupby("year").alpha.median()
        mx = med.index.values + 0.5
        full = med.index.values != PARTIAL_YEAR
        nfull = int(full.sum())
        ax.plot(mx[:nfull], med.values[:nfull], color=CAT[1], lw=1.4, zorder=4)
        ax.plot(mx[full], med.values[full], ls="", marker="o", ms=3.6, color=CAT[1], mec="white", mew=0.6, zorder=5)
        if (~full).any():  # partial last year: dotted link and open marker, as in the Madrid series
            ax.plot(mx[nfull - 1:], med.values[nfull - 1:], color=CAT[1], lw=1.2, ls=(0, (1.5, 1.2)), zorder=4)
            ax.plot(mx[~full], med.values[~full], ls="", marker="o", ms=3.6, mfc="white", mec=CAT[1], mew=1.1, zorder=5)
        ax.axvline(2022.91, color=GREY, ls="--", lw=0.8)
        name = (instance_names or {}).get(inst, inst)
        ax.set_title(f"{name} ({LANG_NAMES.get(lang, lang)})", loc="left", fontsize=7.5)
        style(ax, grid="y")
        ax.set_xticks(range(y0 + 2 * (y0 < xlo), y1 + 1, 2))
        ax.set_xlim(xlo - 0.3, y1 + 1.2)
        ax.set_ylim(-0.02, ytop)
        if k == 0:
            ax.text(0.02, 0.97, f"open marker: {PARTIAL_YEAR} ({PARTIAL_LABEL})", transform=ax.transAxes,
                    ha="left", va="top", fontsize=6.3, color=GREY)
        if k % ncol == 0:
            ax.set_ylabel("$\\hat\\alpha$ (LLM share)")
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.tight_layout(h_pad=1.6)
    fig.savefig(out / "fig_consultations.pdf"); fig.savefig(out / "fig_consultations.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- Fig: pre/post dumbbell
def fig_prepost(cells: pd.DataFrame, out: Path, instance_names: dict | None = None, keep: set | None = None):
    d = cells.dropna(subset=["alpha"])
    if keep is not None:
        d = d[[(i, l) in keep for i, l in zip(d.instance, d.lang)]]
    if "type" in d:   # one contribution type per instance: proposals where they exist, otherwise the largest type
        rows = []
        for (inst, lang), g in d.groupby(["instance", "lang"]):
            typ = "proposal" if (g.type == "proposal").any() else g.groupby("type").n_docs.sum().idxmax()
            rows.append(g[g.type == typ])
        d = pd.concat(rows) if rows else d
    piv = d.pivot_table(index=["instance", "lang"], columns="period", values=["alpha", "ci_lo", "ci_hi"])
    piv = piv.dropna(subset=[("alpha", "pre"), ("alpha", "post")]).sort_values(("alpha", "post"))
    fig, ax = plt.subplots(figsize=(COL_W, 0.22 * len(piv) + 0.8))
    y = np.arange(len(piv))
    for i, ((inst, lang), r) in enumerate(piv.iterrows()):
        ax.plot([r[("alpha", "pre")], r[("alpha", "post")]], [i, i], color=LIGHT, lw=2, zorder=1)
        ax.errorbar(r[("alpha", "pre")], i, xerr=[[r[("alpha", "pre")] - r[("ci_lo", "pre")]], [r[("ci_hi", "pre")] - r[("alpha", "pre")]]],
                    fmt="o", color=CAT[0], ms=4, lw=0.8, capsize=0, zorder=3)
        ax.errorbar(r[("alpha", "post")], i, xerr=[[r[("alpha", "post")] - r[("ci_lo", "post")]], [r[("ci_hi", "post")] - r[("alpha", "post")]]],
                    fmt="o", color=CAT[1], ms=4, lw=0.8, capsize=0, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{(instance_names or {}).get(i, i)} ({l})" for i, l in piv.index])
    ax.set_xlabel("$\\hat\\alpha$ with 95% bootstrap CI")
    ax.plot([], [], "o", color=CAT[0], label="before Dec 2022")
    ax.plot([], [], "o", color=CAT[1], label="after Dec 2022")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    style(ax, grid="x")
    fig.savefig(out / "fig_prepost.pdf"); fig.savefig(out / "fig_prepost.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- Fig: audit heatmap
def fig_audit(metrics: pd.DataFrame, out: Path, cols: list[str], labels: list[str]):
    g = metrics.groupby(["mode", "model"])[cols].mean()
    modes = list(dict.fromkeys(g.index.get_level_values(0)))
    fig, axes = plt.subplots(1, len(modes), figsize=(FULL_W, 0.35 * g.index.get_level_values(1).nunique() + 1.2),
                             sharey=True, squeeze=False)
    for k, mode in enumerate(modes):
        ax = axes[0][k]
        m = g.loc[mode]
        norm = (m - m.min()) / (m.max() - m.min() + 1e-9)
        ax.imshow(norm.values, cmap=SEQ, aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_yticks(range(len(m)))
        ax.set_yticklabels(m.index)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, f"{m.values[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if norm.values[i, j] > 0.6 else "#0b0b0b")
        ax.set_title(mode, loc="left")
        ax.spines[:].set_visible(False)
        ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(out / "fig_audit.pdf"); fig.savefig(out / "fig_audit.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="paper/figures")
    a = ap.parse_args()
    R, O = Path(a.results), Path(a.out)
    O.mkdir(parents=True, exist_ok=True)
    names = {}
    try:
        import yaml
        names = {k: v["name"] for k, v in yaml.safe_load(open("configs/instances.yaml", encoding="utf-8"))["instances"].items()}
    except Exception:  # noqa: BLE001
        pass
    if (R / "benchmark.parquet").exists():
        raw = pd.read_parquet(R / "benchmark.parquet")
        summ = pd.read_csv(R / "benchmark.summary.csv")
        from civic import figures_v2  # revised figures (Nov 2026)
        figures_v2.fig_benchmark(raw, O)
        fig_calibration(raw, O)
    names.update({"eu_hys_citizens": "Have Your Say, citizens", "eu_hys_organisations": "Have Your Say, organisations",
                  "regulations_gov_epa": "Regulations.gov (EPA)", "regulations_gov_dot": "Regulations.gov (DOT)",
                  "regulations_gov_usda": "Regulations.gov (USDA)", "regulations_gov_doi": "Regulations.gov (DOI)"})
    # show only instance-language pairs whose held-out pre-ChatGPT share (null calibration) is at most 5%
    keep = None
    if (R / "null_calibration.csv").exists():
        n = pd.read_csv(R / "null_calibration.csv")
        f = n[(n.slice == "fold1") & (n.reference == "adapted") & (n.instance != "ALL") & (n.alpha <= 0.05)]
        keep = set(zip(f.instance, f.lang)) - {(i, l) for i, l in zip(f.instance, f.lang) if i == "eu_hys_unknown"}
        # ... and whose largest single pre-ChatGPT year (topic drift between years) stays at or below 10%
        yr = n[(n.reference == "adapted") & n.slice.str.startswith("year=") & (n.instance != "ALL")]
        drift = yr.groupby(["instance", "lang"]).alpha.max()
        keep -= {k for k, v in drift.items() if v > 0.10}
        print("time-series/pre-post figures: keeping", len(keep), "instance-language pairs with null floor <= 5% and year floor <= 10%")
    if (R / "alpha_cells_half.csv").exists():
        ch = pd.read_csv(R / "alpha_cells_half.csv")
        if (R / "alpha_cells_consultation.csv").exists():
            # platforms organised in processes/consultations: a half-year is often one process, so the half-year series
            # measures the process of the moment; they are shown per process in fig_consultations, and the half-year
            # series keeps the continuous streams (instances with at most two processes after ChatGPT)
            multi = set()
            if (R / "process_counts.csv").exists():
                pc = pd.read_csv(R / "process_counts.csv")
                pc = pc[pc.period == "post"]
                multi = set(zip(pc[pc.n_processes >= 3].instance, pc[pc.n_processes >= 3].lang))
            else:
                multi = {(i, l) for i, l in zip(ch.instance, ch.lang) if i.startswith(("eu_hys", "regulations_gov"))}
            ch = ch[[(i, l) not in multi for i, l in zip(ch.instance, ch.lang)]]
            print("half-year series: continuous streams only,", ch.groupby(["instance", "lang"]).ngroups, "pairs")
            fig_consultations(pd.read_csv(R / "alpha_cells_consultation.csv"), O, names)
        from civic import figures_v2
        figures_v2.fig_timeseries(pd.read_csv(R / "alpha_cells_instance_year.csv"), pd.read_csv(R / "excess" / "excess.csv"), O)
    if (R / "alpha_cells_period.csv").exists():
        fig_prepost(pd.read_csv(R / "alpha_cells_period.csv"), O, names, keep=keep)
    if (R / "audit_metrics.parquet").exists():
        from civic import figures_v2
        homog = pd.read_csv(R / "audit_homog.csv") if (R / "audit_homog.csv").exists() else None
        figures_v2.fig_audit(pd.read_parquet(R / "audit_metrics.parquet"), homog, O)
    print("figures written to", O)


if __name__ == "__main__":
    main()
