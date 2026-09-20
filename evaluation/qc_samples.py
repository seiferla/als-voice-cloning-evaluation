"""Screens every candidate stimulus and flags the broken ones.

Usage:
    uv run python evaluation/qc_samples.py
"""

import csv
import json
import sys
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myecho.evaluation.audio_qc import (  # noqa: E402
    PEAK_CEILING_DBFS,
    TARGET_LUFS,
    load_mono,
    measure,
    normalize,
    to_common_sr,
)

TESTSET_PATH = Path(__file__).parent / "testset.json"
RESULTS_DIR = Path(__file__).parent / "results"
QC_CSV = RESULTS_DIR / "qc_report.csv"
EXCLUSIONS_PATH = RESULTS_DIR / "exclusions.json"

AUDIO_EXT = "wav"

# Flag a stimulus this far from the per-text median of the other models.
SILENCE_EXCESS = 0.20      # +20 percentage points of silence
DURATION_RATIO = 0.30      # +/-30 % duration
NEAR_SILENT_RATIO = 0.95   # 95 % silence = nothing to rate

# Only these mean "technically unusable" and get pre-filled into exclusions.json.
HARD_FLAGS = {"quiet", "near_silent", "clipping"}

CSV_FIELDS = [
    "source", "text_id", "duration_s", "lufs", "rms_dbfs", "peak_dbfs",
    "active_rms_dbfs", "silence_ratio", "clipped_ratio",
    "gain_needed_db", "reaches_target", "flags",
]


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
    return [{"id": t.get("id", str(i)), "source_audio": t["source_audio"]}
            for i, t in enumerate(data["texts"])]


def real_audio_path(source_audio):
    denoised = Path("data/DeepFilterNet") / source_audio
    return denoised if denoised.exists() else Path("data/wav") / source_audio


def candidates(case_dir, texts):
    """[(source_label, text_id, path), ...] for every model render + the real recording."""
    models = sorted(p.name for p in case_dir.iterdir() if p.is_dir())
    out = []
    for model in models:
        for i, entry in enumerate(texts):
            p = case_dir / model / f"{i:02d}_{entry['id']}.{AUDIO_EXT}"
            if p.exists():
                out.append((model, entry["id"], p))
    for entry in texts:
        p = real_audio_path(entry["source_audio"])
        if p.exists():
            out.append(("REAL", entry["id"], p))
    return out


def main():
    case = load_case(sys.argv[1] if len(sys.argv) > 1 else None)
    texts = load_texts(case)
    case_dir = RESULTS_DIR / case["name"]
    if not case_dir.exists():
        raise SystemExit(f"{case_dir} not found - run evaluation/run_eval.py first")

    rows = []
    for source, text_id, path in candidates(case_dir, texts):
        x, sr = load_mono(path)
        x, sr = to_common_sr(x, sr)
        m = measure(x, sr)
        _, norm = normalize(x, sr)
        rows.append({
            "source": source,
            "text_id": text_id,
            **m,
            "gain_needed_db": norm["gain_db"],
            "reaches_target": norm["reached_target"],
            "flags": [],
        })

    # Per-text medians across the model renders; REAL does not set the reference.
    by_text = {}
    for r in rows:
        if r["source"] != "REAL":
            by_text.setdefault(r["text_id"], []).append(r)

    for text_id, group in by_text.items():
        med_dur = median(r["duration_s"] for r in group)
        med_sil = median(r["silence_ratio"] for r in group)
        for r in group:
            if med_dur > 0 and abs(r["duration_s"] - med_dur) / med_dur > DURATION_RATIO:
                r["flags"].append("duration_off")
            if r["silence_ratio"] - med_sil > SILENCE_EXCESS:
                r["flags"].append("silence_heavy")

    for r in rows:
        if not r["reaches_target"]:
            r["flags"].append("quiet")
        if r["clipped_ratio"] > 0.0001:
            r["flags"].append("clipping")
        if r["silence_ratio"] > NEAR_SILENT_RATIO:
            r["flags"].append("near_silent")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(QC_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({**r, "flags": ";".join(r["flags"])})

    print(f"\nMeasured {len(rows)} stimuli -> {QC_CSV}")
    print(f"(target {TARGET_LUFS} LUFS, peak ceiling {PEAK_CEILING_DBFS} dBFS)\n")

    print("Level per source after normalization would be equal; before it is:")
    for source in sorted({r["source"] for r in rows}):
        group = [r for r in rows if r["source"] == source and r["lufs"] is not None]
        lufs = [r["lufs"] for r in group]
        print(f"  {source:8s} n={len(group):2d}  LUFS median={median(lufs):7.1f}  "
              f"min={min(lufs):7.1f}  max={max(lufs):7.1f}  spread={max(lufs) - min(lufs):4.1f} LU")

    flagged = [r for r in rows if r["flags"]]
    hard = [r for r in flagged if HARD_FLAGS & set(r["flags"])]
    soft = [r for r in flagged if not HARD_FLAGS & set(r["flags"])]

    print(f"\n{len(hard)} technically unusable (pre-filled into exclusions.json):")
    for r in hard:
        print(f"  {r['source']:8s} {r['text_id']:28s} {';'.join(r['flags']):28s} "
              f"lufs={r['lufs']}  silence={r['silence_ratio']:.0%}  dur={r['duration_s']}s")
    if not hard:
        print("  (none)")

    print(f"\n{len(soft)} worth listening to, NOT excluded by default:")
    for r in soft:
        print(f"  {r['source']:8s} {r['text_id']:28s} {';'.join(r['flags']):28s} "
              f"lufs={r['lufs']}  silence={r['silence_ratio']:.0%}  dur={r['duration_s']}s")
    if not soft:
        print("  (none)")
    print("  ^ usually just a slower speaking rate or more pauses - a real property\n"
          "    of the voice. Only exclude if you hear an actual synthesis failure.")

    if EXCLUSIONS_PATH.exists():
        print(f"\n{EXCLUSIONS_PATH} exists already - left untouched.")
    else:
        template = {
            "_comment": [
                "Stimuli excluded from the listening test, read by prepare_listening_test.py.",
                "'exclude' is pre-filled with technically unusable stimuli only.",
                "'review_only' is a shortlist to listen to - it is NOT excluded; move an",
                "entry into 'exclude' yourself if you hear a real synthesis failure.",
                "Exclude only technical failures, never because something 'sounds bad' -",
                "that is what the raters are there to judge.",
            ],
            "exclude": [
                {"model": r["source"], "text_id": r["text_id"], "reason": ";".join(r["flags"])}
                for r in hard if r["source"] != "REAL"
            ],
            "review_only": [
                {"model": r["source"], "text_id": r["text_id"], "flags": ";".join(r["flags"])}
                for r in soft if r["source"] != "REAL"
            ],
        }
        with open(EXCLUSIONS_PATH, "w", encoding="utf-8") as f:
            json.dump(template, f, ensure_ascii=False, indent=2)
        print(f"\nWrote exclusion template -> {EXCLUSIONS_PATH} (review it before building the test)")


if __name__ == "__main__":
    main()
