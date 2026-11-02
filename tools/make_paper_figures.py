"""Export the manuscript figures as PDF and PNG from the result files.

Usage:
    uv run python tools/make_paper_figures.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "evaluation" / "results"
LT = RESULTS / "listening_test"
OUT = ROOT / "docs" / "paper" / "figures"

MODEL_ORDER = ["qwen3", "s2pro", "voxtral"]
LABELS = {"qwen3": "Qwen3", "s2pro": "S2 Pro", "voxtral": "Voxtral", "REAL": "Genuine"}
COLORS = {"qwen3": "#4c72b0", "s2pro": "#dd8452", "voxtral": "#55a868", "REAL": "#8c8c8c"}

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "pdf.fonttype": 42})


def fig_nmos(nmos_summary, ratings_long):
    conditions = MODEL_ORDER + ["REAL"]
    row = nmos_summary.set_index("condition").loc[conditions]
    # "repeat" is NaN on the Turing rows, so filter by part before testing it.
    nmos = ratings_long[ratings_long["part"] == "nmos"]
    per_rater = (nmos[nmos["repeat"] == False]  # noqa: E712
                 .groupby(["condition", "rater"])["rating"].mean().reset_index())

    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    x = np.arange(len(conditions))
    rng = np.random.default_rng(0)
    for i, cond in enumerate(conditions):
        vals = per_rater.loc[per_rater["condition"] == cond, "rating"]
        ax.scatter(np.full(len(vals), i) + rng.uniform(-0.12, 0.12, len(vals)), vals,
                   color="black", alpha=0.35, s=10, zorder=3)
    yerr = [row["mean"] - row["ci_low"], row["ci_high"] - row["mean"]]
    ax.bar(x, row["mean"], yerr=yerr, capsize=4, alpha=0.85,
           color=[COLORS[c] for c in conditions])
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in conditions])
    ax.set_ylabel("N-MOS")
    ax.set_ylim(1, 5)
    fig.tight_layout()
    return fig


def fig_turing(turing_summary):
    groups = ["all", "knows_person", "does_not_know_person"]
    group_labels = {"all": "all raters", "knows_person": "knows the patient",
                    "does_not_know_person": "does not know the patient"}
    width = 0.25
    x = np.arange(len(MODEL_ORDER))

    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    for i, group in enumerate(groups):
        sub = (turing_summary[turing_summary["group"] == group]
               .set_index("model").reindex(MODEL_ORDER))
        yerr = [sub["fooling_rate"] - sub["ci_low"], sub["ci_high"] - sub["fooling_rate"]]
        ax.bar(x + (i - 1) * width, sub["fooling_rate"], width, yerr=yerr, capsize=3,
               label=group_labels[group])
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in MODEL_ORDER])
    ax.set_ylabel("fooling rate")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, loc="upper left", frameon=False)
    fig.tight_layout()
    return fig


def fig_sim_vs_nmos(obj_summary, nmos_summary):
    sim = obj_summary[obj_summary["metric"] == "sim"].set_index("model")["median"]
    nmos = nmos_summary.set_index("condition")["mean"]

    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    for cond in MODEL_ORDER + ["REAL"]:
        ax.scatter(sim[cond], nmos[cond], s=70, color=COLORS[cond], zorder=3)
        ax.annotate(LABELS[cond], (sim[cond], nmos[cond]),
                    xytext=(6, 4), textcoords="offset points")
    ax.set_xlabel("speaker similarity (SIM)")
    ax.set_ylabel("naturalness (N-MOS)")
    ax.set_xlim(0.55, 0.85)
    ax.set_ylim(2.5, 4.5)
    fig.tight_layout()
    return fig


def fig_objective(obj_summary):
    panels = [("sim", "SIM", MODEL_ORDER + ["REAL"]),
              ("wer", "WER", MODEL_ORDER + ["REAL"]),
              ("service_rtf", "service RTF", MODEL_ORDER)]

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))
    for ax, (metric, title, conditions) in zip(axes, panels):
        row = (obj_summary[obj_summary["metric"] == metric]
               .set_index("model").loc[conditions])
        yerr = [row["median"] - row["ci_low"], row["ci_high"] - row["median"]]
        x = np.arange(len(conditions))
        ax.bar(x, row["median"], yerr=yerr, capsize=3, alpha=0.85,
               color=[COLORS[c] for c in conditions])
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[c] for c in conditions], rotation=30, ha="right")
        ax.set_title(title)
    axes[0].set_ylim(0, 1)
    axes[2].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[2].set_ylim(0, 1.1)
    axes[2].text(0.02, 1.02, "real time", fontsize=7, transform=axes[2].get_yaxis_transform())
    fig.tight_layout()
    return fig


def main():
    obj_summary = pd.read_csv(RESULTS / "objective_summary.csv")
    nmos_summary = pd.read_csv(LT / "nmos_summary.csv")
    turing_summary = pd.read_csv(LT / "turing_summary.csv")
    ratings_long = pd.read_csv(LT / "ratings_long.csv")

    OUT.mkdir(parents=True, exist_ok=True)
    figures = {
        "fig_nmos.pdf": fig_nmos(nmos_summary, ratings_long),
        "fig_turing.pdf": fig_turing(turing_summary),
        "fig_sim_vs_nmos.pdf": fig_sim_vs_nmos(obj_summary, nmos_summary),
        "fig_objective.pdf": fig_objective(obj_summary),
    }
    for name, fig in figures.items():
        fig.savefig(OUT / name)
        fig.savefig(OUT / name.replace(".pdf", ".png"), dpi=200)
        plt.close(fig)
        print(f"wrote {(OUT / name).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
