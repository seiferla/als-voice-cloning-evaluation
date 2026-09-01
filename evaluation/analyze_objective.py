"""Turns the objective results table into comparisons with intervals.

Usage:
    uv run python evaluation/analyze_objective.py [results.csv]

run_eval.py reports medians. A median answers "what does this system usually
do"; it does not answer "is this system better than that one", which is the
question the report has to defend. Three medians in a table invite a ranking
that the numbers may not support - especially here, where the systems are
compared on only 13-16 texts and one of them has a standard deviation larger
than its mean.

So every comparison here is **paired within text**: the same sentence rendered
by two systems. Pairing removes the largest nuisance - some texts are simply
harder for every system - and is the reason a small number of texts can still
say something. Intervals come from a bootstrap over texts, because texts are
what this study samples; the conclusion has to survive a different set of
sentences, not just these.

Comparisons against the REAL row are the ones that matter most: they say how far
a system is from the human original on the same measure, which is the only way
SIM and WER become readable at all.

Writes objective_summary.csv and objective_pairwise.csv next to the input.
"""

import argparse
import csv
import random
import statistics
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
N_BOOTSTRAP = 4000
SEED = 20260901
CI = 0.95

METRICS = {
    "sim": ("speaker similarity", "higher is better"),
    "wer": ("word error rate", "lower is better"),
    "service_rtf": ("latency / audio length", "lower is better"),
}


def load_rows(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in ("sim", "wer", "service_rtf", "latency_s", "ttfa_s"):
            value = row.get(key)
            row[key] = float(value) if value not in (None, "", "None") else None
    return rows


def per_text(rows, model, metric):
    """{text_id: value}, averaging repeated renders of the same text."""
    grouped = defaultdict(list)
    for row in rows:
        if row["model"] == model and row.get(metric) is not None:
            grouped[row["text_id"]].append(row[metric])
    return {text: statistics.mean(values) for text, values in grouped.items()}


def bootstrap_ci(texts, statistic, n=N_BOOTSTRAP, seed=SEED, ci=CI):
    """Percentile CI under resampling of texts."""
    if len(texts) < 2:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    estimates = []
    for _ in range(n):
        draw = [rng.choice(texts) for _ in texts]
        value = statistic(draw)
        if value is not None:
            estimates.append(value)
    if not estimates:
        return float("nan"), float("nan")
    estimates.sort()
    lo = estimates[max(0, int((1 - ci) / 2 * (len(estimates) - 1)))]
    hi = estimates[min(len(estimates) - 1, int((1 + ci) / 2 * (len(estimates) - 1)))]
    return lo, hi


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("results", nargs="?",
                        default=str(RESULTS_DIR / "results_latest.csv"))
    args = parser.parse_args()

    path = Path(args.results)
    if not path.exists():
        raise SystemExit(f"{path} not found - run evaluation/run_eval.py first")
    rows = load_rows(path)

    models = sorted({r["model"] for r in rows})
    real = "REAL" if "REAL" in models else None
    impostor = "IMPOSTOR" if "IMPOSTOR" in models else None
    systems = [m for m in models if m not in (real, impostor)]
    ordered = systems + [m for m in (real, impostor) if m]

    print(f"{path.name}: {len(rows)} rows, conditions {ordered}\n")
    if real is None:
        print("WARNING: no REAL row. SIM and WER have no anchor and the numbers "
              "below cannot be\n         calibrated - re-run run_eval.py with a "
              "texts_file that has source_audio.\n")

    summary, pairwise = [], []
    for metric, (label, direction) in METRICS.items():
        values = {m: per_text(rows, m, metric) for m in ordered}
        present = [m for m in ordered if values[m]]
        if not present:
            continue

        print(f"=== {metric} — {label} ({direction}) ===")
        for model in present:
            data = values[model]
            texts = sorted(data)
            median = statistics.median(data.values())
            lo, hi = bootstrap_ci(texts, lambda d, v=data: statistics.median(
                [v[t] for t in d]))
            mark = "   <- human anchor" if model == real else (
                "   <- different speaker" if model == impostor else "")
            print(f"  {model:9s} median {median:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  "
                  f"n={len(data)} texts{mark}")
            summary.append({"metric": metric, "model": model,
                            "median": round(median, 4),
                            "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                            "n_texts": len(data)})

        # paired comparisons, Bonferroni-adjusted over the family
        pairs = [(a, b) for i, a in enumerate(present) for b in present[i + 1:]]
        adjusted = 1 - (1 - CI) / max(len(pairs), 1)
        print(f"\n  paired by text, verdict Bonferroni-adjusted over "
              f"{len(pairs)} comparisons:")
        for a, b in pairs:
            shared = sorted(set(values[a]) & set(values[b]))
            if len(shared) < 3:
                continue

            def diff(texts, a=a, b=b):
                deltas = [values[a][t] - values[b][t] for t in texts]
                return statistics.mean(deltas) if deltas else None

            estimate = diff(shared)
            lo, hi = bootstrap_ci(shared, diff)
            alo, ahi = bootstrap_ci(shared, diff, ci=adjusted)
            verdict = ("difference" if (alo > 0 or ahi < 0) else
                       "only at 95%, not after adjustment" if (lo > 0 or hi < 0)
                       else "no difference detected")
            print(f"    {a:9s} - {b:9s} {estimate:+.3f}  "
                  f"CI [{lo:+.3f}, {hi:+.3f}]  adj [{alo:+.3f}, {ahi:+.3f}]  "
                  f"-> {verdict}  (n={len(shared)} texts)")
            pairwise.append({"metric": metric, "model_a": a, "model_b": b,
                             "difference": round(estimate, 4),
                             "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                             "ci_low_adj": round(alo, 4), "ci_high_adj": round(ahi, 4),
                             "excludes_zero_adj": bool(alo > 0 or ahi < 0),
                             "n_texts": len(shared)})
        print()

    for name, data, fields in (
        ("objective_summary.csv", summary,
         ["metric", "model", "median", "ci_low", "ci_high", "n_texts"]),
        ("objective_pairwise.csv", pairwise,
         ["metric", "model_a", "model_b", "difference", "ci_low", "ci_high",
          "ci_low_adj", "ci_high_adj", "excludes_zero_adj", "n_texts"]),
    ):
        with open(path.parent / name, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(data)
    print(f"Written to {path.parent / 'objective_summary.csv'} and "
          f"{path.parent / 'objective_pairwise.csv'}")

    if real:
        print("\nA system whose interval overlaps the REAL row on SIM is at the "
              "ceiling of what\nthis metric resolves - not 'as good as the human', "
              "but 'not separable by ECAPA'.")


if __name__ == "__main__":
    main()
