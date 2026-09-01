"""Long-form stability of the full renders - the part the listening test cannot see.

Usage:
    uv run python evaluation/analyze_long_form.py

The listening test rates ~10 s excerpts. That is the right unit for a single
naturalness judgement, but it has a price that has to be paid somewhere: on this
material the excerpts sit between 32 % and 71 % of each render, so about 73 % of
what each model produced is never heard by any rater. Everything that only shows
up over a long utterance - the voice drifting away from the target, the speaking
rate running away, the model trailing off - is invisible in the subjective
results by construction.

So the full renders are analysed objectively instead. No listener is needed for
any of this: the questions are whether the voice stays the same voice from the
beginning of a render to the end, and whether the length of the output is a
sensible function of the length of the text.

Measures, per full render:

    sim_first / sim_last    ECAPA similarity to the reference for the first and
                            the last third of the render
    sim_drift               sim_last - sim_first. Negative means the voice moves
                            away from the target as the utterance goes on. This
                            is the number a 10 s excerpt cannot show.
    level_drift_db          loudness of the last third minus the first third
    duration_ratio          render duration / the real recording of the same text
    words_per_second        speaking rate, from the cached ASR alignment
    silence_ratio           share of frames below the silence threshold
    longest_gap_s           longest single pause - catches a model that stalls

Writes evaluation/results/long_form.csv. Reads the ASR alignments cached by
select_excerpts.py, so it does not re-run Whisper.
"""

import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myecho.evaluation.audio_qc import (  # noqa: E402
    SILENCE_THRESHOLD_DBFS,
    _frame_db,
    load_mono,
)
from myecho.evaluation.metrics import extract_embedding, similarity  # noqa: E402

TESTSET_PATH = Path(__file__).parent / "testset.json"
RESULTS_DIR = Path(__file__).parent / "results"
ALIGN_DIR = RESULTS_DIR / "alignments"
OUT_CSV = RESULTS_DIR / "long_form.csv"

AUDIO_EXT = "wav"
REAL = "REAL"

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


def load_texts(case):
    with open(Path(case["texts_file"]), encoding="utf-8") as f:
        data = json.load(f)
    return [{"id": t.get("id", str(i)), "source_audio": t.get("source_audio")}
            for i, t in enumerate(data["texts"])]


def real_audio_path(source_audio):
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


def main():
    case = load_case(sys.argv[1] if len(sys.argv) > 1 else None)
    texts = load_texts(case)
    case_dir = RESULTS_DIR / case["name"]
    if not case_dir.exists():
        raise SystemExit(f"{case_dir} not found - run evaluation/run_eval.py first")

    reference_audio = case["reference_audio"]
    print(f"Reference embedding from {reference_audio} ...")
    ref_embedding = extract_embedding(reference_audio)

    models = sorted(p.name for p in case_dir.iterdir() if p.is_dir())
    real_duration = {}

    rows = []
    for source in [REAL] + models:
        for i, entry in enumerate(texts):
            text_id = entry["id"]
            if source == REAL:
                path = real_audio_path(entry["source_audio"]) if entry["source_audio"] else None
            else:
                path = case_dir / source / f"{i:02d}_{text_id}.{AUDIO_EXT}"
            if path is None or not path.exists():
                continue

            timing = timing_features(path)
            if source == REAL:
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
            print(f"  {source:8s} {text_id:30s} dur={timing['duration_s']:6.1f}s  "
                  f"sim {first if first is None else round(first, 3)} -> "
                  f"{last if last is None else round(last, 3)}")

    fields = ["source", "text_id", "duration_s", "duration_ratio", "words_per_second",
              "silence_ratio", "longest_gap_s", "level_drift_db",
              "sim_first", "sim_last", "sim_drift"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    def med(source, key):
        values = [r[key] for r in rows if r["source"] == source and r[key] is not None]
        return statistics.median(values) if values else None

    def fmt(value, digits=3):
        return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "  n/a"

    print(f"\n=== Long-form stability (median per source, {len(rows)} full renders) ===")
    print(f"{'source':9s}{'dur ratio':>11s}{'words/s':>9s}{'silence':>9s}"
          f"{'gap s':>8s}{'level dr':>10s}{'sim 1st':>9s}{'sim 3rd':>9s}{'drift':>9s}")
    for source in [REAL] + models:
        print(f"{source:9s}{fmt(med(source, 'duration_ratio')):>11s}"
              f"{fmt(med(source, 'words_per_second'), 2):>9s}"
              f"{fmt(med(source, 'silence_ratio')):>9s}"
              f"{fmt(med(source, 'longest_gap_s'), 2):>8s}"
              f"{fmt(med(source, 'level_drift_db'), 2):>10s}"
              f"{fmt(med(source, 'sim_first')):>9s}"
              f"{fmt(med(source, 'sim_last')):>9s}"
              f"{fmt(med(source, 'sim_drift')):>9s}")

    print(f"\nPer-render values -> {OUT_CSV}")
    print("\nsim_drift is the one the listening test cannot produce: it says whether")
    print("the voice is still the target voice at the end of a long utterance.")
    print("Read the REAL row as the baseline - a real recording also drifts a")
    print("little, because the two thirds are different speech, not because the")
    print("speaker changed.")


if __name__ == "__main__":
    main()
