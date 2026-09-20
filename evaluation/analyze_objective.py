"""Turns the objective results table into comparisons with intervals.

Usage:
    uv run python evaluation/analyze_objective.py [results.csv]

run_eval.py reports medians, which say what a model usually does but not whether
one model is better than another. Every comparison here is paired within text -
the same sentence rendered by two models - with intervals bootstrapped over
texts and Bonferroni-adjusted verdicts.

Writes objective_summary.csv and objective_pairwise.csv.
"""

import argparse
import csv
import json
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
    "latency_s": ("wall-clock until the audio is complete", "lower is better"),
    "ttfa_s": ("time to first audio frame", "lower is better"),
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


TESTSET_PATH = Path(__file__).parent / "testset.json"
RESULTS_DIR = Path(__file__).parent / "results"

AUDIO_EXT = "wav"
REAL_LABEL = "REAL"


def load_case(case_name=None):
    with open(TESTSET_PATH, encoding="utf-8") as f:
        cases = json.load(f)["cases"]
    if case_name is None:
        return cases[0]
    for case in cases:
        if case["name"] == case_name:
            return case
    raise SystemExit(f"Case '{case_name}' not found in {TESTSET_PATH}")


def load_texts(case):
    with open(Path(case["texts_file"]), encoding="utf-8") as f:
        data = json.load(f)
    return [{"id": t.get("id", str(i)), "source_audio": t.get("source_audio")}
            for i, t in enumerate(data["texts"])]


def render_reliability(case_name=None):
    """How often each system produced audio that could be evaluated at all.

    Every metric below, and the whole listening test, is computed on the
    renditions that survived screening. A system that fails on a third of the
    texts is then compared only on the texts it handled - and because a failed
    text is dropped for EVERY system (the balanced design), its failures leave
    no trace in any other number. For a communication aid that reliability gap
    matters more than a small difference in similarity or naturalness, so it is
    reported here rather than left in the build log.

    Counted over all texts in the set, before any exclusion.
    """
    excerpts_path = RESULTS_DIR / "excerpts.json"
    if not excerpts_path.exists():
        return []
    with open(excerpts_path, encoding="utf-8") as f:
        excerpts = json.load(f)
    spans = excerpts["spans"]
    texts = [t["id"] for t in load_texts(load_case(case_name))]

    # Specific reasons where we have them; "not located" is the catch-all for a
    # rendition the aligner could not find the excerpt in at all.
    reasons = defaultdict(dict)
    for entry in excerpts.get("rejected", []):
        reasons[entry["source"]][entry["text_id"]] = entry["reason"]

    exclusions_path = RESULTS_DIR / "exclusions.json"
    if exclusions_path.exists():
        with open(exclusions_path, encoding="utf-8") as f:
            for entry in json.load(f).get("exclude", []):
                reasons[entry["model"]].setdefault(entry["text_id"],
                                                   entry.get("reason", "excluded"))

    rows = []
    for source in sorted({k.split("/", 1)[0] for k in spans} | set(reasons)):
        unlocated = {t: "excerpt not located - the system did not say the text"
                     for t in texts if f"{source}/{t}" not in spans}
        failures = {**unlocated, **reasons.get(source, {})}
        rows.append({"model": source, "n_texts": len(texts),
                     "n_unusable": len(failures),
                     "usable_rate": round(1 - len(failures) / max(len(texts), 1), 3),
                     "failures": failures})
    return rows


def report_render_reliability(rows):
    if not rows:
        return
    print("=== render reliability - did the system produce usable audio at all? ===")
    print("  Over ALL texts, before any exclusion. Everything below is computed on the")
    print("  survivors only, so this is the one place a failure is still visible.\n")
    for row in rows:
        mark = "   <- reference recording, not a system" if row["model"] == REAL_LABEL else ""
        print(f"  {row['model']:9s} usable {row['n_texts'] - row['n_unusable']:2d}/"
              f"{row['n_texts']}  ({row['usable_rate']:.0%}){mark}")
    for row in rows:
        for text_id, reason in sorted(row["failures"].items()):
            print(f"      {row['model']}/{text_id}: {reason}")

    out = RESULTS_DIR / "render_reliability.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "n_texts", "n_unusable", "usable_rate",
                         "failed_text_id", "reason"])
        for row in rows:
            if not row["failures"]:
                writer.writerow([row["model"], row["n_texts"], 0, row["usable_rate"], "", ""])
            for text_id, reason in sorted(row["failures"].items()):
                writer.writerow([row["model"], row["n_texts"], row["n_unusable"],
                                 row["usable_rate"], text_id, reason])
    print(f"\n  -> {out}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("results", nargs="?",
                        default=str(RESULTS_DIR / "results_latest.csv"))
    parser.add_argument("--case", default=None)
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
    report_render_reliability(render_reliability(args.case))
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
