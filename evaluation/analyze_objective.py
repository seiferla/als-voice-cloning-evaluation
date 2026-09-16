"""Turns the objective results table into comparisons with intervals.

Usage:
    uv run python evaluation/analyze_objective.py [results.csv] [--long-form]

run_eval.py reports medians, which say what a model usually does but not whether
one model is better than another. Every comparison here is paired within text -
the same sentence rendered by two models - with intervals bootstrapped over
texts and Bonferroni-adjusted verdicts.

--long-form additionally scores the FULL renders for voice drift, level drift
and speaking rate. The listening test only rates ~10 s excerpts, so anything
that appears over a long utterance is invisible there. That pass needs the audio
and the embedding model, so it is opt-in.

Writes objective_summary.csv and objective_pairwise.csv.
"""

import argparse
import csv
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myecho.evaluation.audio_qc import (  # noqa: E402
    SILENCE_THRESHOLD_DBFS,
    _frame_db,
    load_mono,
)
from myecho.evaluation.metrics import extract_embedding, similarity  # noqa: E402

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
ALIGN_DIR = RESULTS_DIR / "alignments"
LONG_FORM_CSV = RESULTS_DIR / "long_form.csv"

AUDIO_EXT = "wav"
REAL_LABEL = "REAL"

# A third is long enough for a stable ECAPA embedding on these renders (the
# shortest is ~12 s, so ~4 s per third) and short enough to show a trend.
N_PARTS = 3


def load_case(case_name=None):
    with open(TESTSET_PATH, encoding="utf-8") as f:
        cases = json.load(f)["cases"]
    if case_name is None:
        return cases[0]
    for case in cases:
        if case["name"] == case_name:
            return case
    raise SystemExit(f"Case '{case_name}' not found in {TESTSET_PATH}")


def load_long_form_texts(case):
    with open(Path(case["texts_file"]), encoding="utf-8") as f:
        data = json.load(f)
    return [{"id": t.get("id", str(i)), "source_audio": t.get("source_audio")}
            for i, t in enumerate(data["texts"])]


def long_form_real_path(source_audio):
    denoised = Path("data/DeepFilterNet") / source_audio
    return denoised if denoised.exists() else Path("data/wav") / source_audio


def part_similarity(path, ref_embedding, n_parts=N_PARTS):
    """Similarity to the reference for each equal-length part of the render."""
    import soundfile as sf

    x, sr = load_mono(path)
    edges = np.linspace(0, len(x), n_parts + 1).astype(int)
    sims = []
    for start, end in zip(edges[:-1], edges[1:]):
        chunk = x[start:end]
        if len(chunk) < sr:            # under a second is not worth embedding
            sims.append(None)
            continue
        tmp = path.parent / f".part_{path.stem}.wav"
        sf.write(str(tmp), chunk, sr)
        try:
            sims.append(float(similarity(ref_embedding, extract_embedding(tmp))))
        finally:
            tmp.unlink(missing_ok=True)
    return sims


def timing_features(path):
    x, sr = load_mono(path)
    frame_db = _frame_db(x, sr)
    active = frame_db > SILENCE_THRESHOLD_DBFS

    # longest run of inactive frames
    longest, run = 0, 0
    for is_active in active:
        run = 0 if is_active else run + 1
        longest = max(longest, run)

    third = len(frame_db) // 3
    level_drift = (float(np.median(frame_db[-third:][active[-third:]]))
                   - float(np.median(frame_db[:third][active[:third]]))
                   if third > 0 and active[:third].any() and active[-third:].any()
                   else None)

    return {
        "duration_s": round(len(x) / sr, 2),
        "silence_ratio": round(float((~active).mean()), 3),
        "longest_gap_s": round(longest * 0.02, 2),
        "level_drift_db": round(level_drift, 2) if level_drift is not None else None,
    }


def words_per_second(source, text_id, duration_s):
    cache = ALIGN_DIR / source / f"{text_id}.json"
    if not cache.exists() or not duration_s:
        return None
    words = json.loads(cache.read_text(encoding="utf-8"))
    return round(len(words) / duration_s, 2) if words else None



def long_form_report(case_name=None):
    """Objective stability of the FULL renders - what the listening test cannot see.

    The listening test rates ~10 s excerpts, and on this material those sit at
    32-71 % of each render, so ~73 % of what each system produced is never heard
    by a rater. Anything that only appears over a long utterance is therefore
    invisible in the subjective results. This measures it instead of assuming it
    is harmless: does the voice stay the target voice from the start of a render
    to the end, and is the output length a sensible function of the text?

    Needs the audio and the ECAPA model, so it is opt-in (--long-form).
    """
    case = load_case(case_name)
    texts = load_long_form_texts(case)
    case_dir = RESULTS_DIR / case["name"]
    if not case_dir.exists():
        raise SystemExit(f"{case_dir} not found - run evaluation/run_eval.py first")

    print(f"Reference embedding from {case['reference_audio']} ...")
    ref_embedding = extract_embedding(case["reference_audio"])

    models = sorted(p.name for p in case_dir.iterdir() if p.is_dir())
    real_duration, rows = {}, []
    for source in [REAL_LABEL] + models:
        for i, entry in enumerate(texts):
            text_id = entry["id"]
            if source == REAL_LABEL:
                path = (long_form_real_path(entry["source_audio"])
                        if entry["source_audio"] else None)
            else:
                path = case_dir / source / f"{i:02d}_{text_id}.wav"
            if path is None or not path.exists():
                continue
            timing = timing_features(path)
            if source == REAL_LABEL:
                real_duration[text_id] = timing["duration_s"]
            sims = part_similarity(path, ref_embedding)
            first, last = sims[0], sims[-1]
            rows.append({
                "source": source, "text_id": text_id, **timing,
                "words_per_second": words_per_second(source, text_id, timing["duration_s"]),
                "duration_ratio": (round(timing["duration_s"] / real_duration[text_id], 3)
                                   if real_duration.get(text_id) else None),
                "sim_first": round(first, 4) if first is not None else None,
                "sim_last": round(last, 4) if last is not None else None,
                "sim_drift": (round(last - first, 4)
                              if first is not None and last is not None else None),
            })

    fields = ["source", "text_id", "duration_s", "duration_ratio", "words_per_second",
              "silence_ratio", "longest_gap_s", "level_drift_db",
              "sim_first", "sim_last", "sim_drift"]
    with open(LONG_FORM_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    def med(source, key):
        values = [r[key] for r in rows if r["source"] == source and r[key] is not None]
        return statistics.median(values) if values else None

    def fmt(value, digits=3):
        return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "  n/a"

    print(f"\n=== Long-form stability of the full renders ({len(rows)} renders) ===")
    print(f"{'source':9s}{'dur ratio':>11s}{'words/s':>9s}{'silence':>9s}"
          f"{'gap s':>8s}{'level dr':>10s}{'sim 1st':>9s}{'sim 3rd':>9s}{'drift':>9s}")
    for source in [REAL_LABEL] + models:
        print(f"{source:9s}{fmt(med(source, 'duration_ratio')):>11s}"
              f"{fmt(med(source, 'words_per_second'), 2):>9s}"
              f"{fmt(med(source, 'silence_ratio')):>9s}"
              f"{fmt(med(source, 'longest_gap_s'), 2):>8s}"
              f"{fmt(med(source, 'level_drift_db'), 2):>10s}"
              f"{fmt(med(source, 'sim_first')):>9s}"
              f"{fmt(med(source, 'sim_last')):>9s}"
              f"{fmt(med(source, 'sim_drift')):>9s}")
    print(f"\nPer-render values -> {LONG_FORM_CSV}")
    print("sim_drift says whether the voice is still the target voice at the end")
    print("of a long utterance. Read the REAL row as the baseline.")


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
    texts = [t["id"] for t in load_long_form_texts(load_case(case_name))]

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
    parser.add_argument("--long-form", action="store_true",
                        help="also analyse the FULL renders (voice drift, level "
                             "drift, speaking rate). Needs the audio and the "
                             "embedding model, so it takes a few minutes.")
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

    if args.long_form:
        print()
        long_form_report(args.case)
    else:
        print("\nThe listening test only rates ~10 s excerpts. Add --long-form to "
              "also check\nthe full renders for voice drift, level drift and "
              "speaking rate.")


if __name__ == "__main__":
    main()
