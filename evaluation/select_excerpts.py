"""Chooses one excerpt per text and locates it in every rendition.

Usage:
    uv run python evaluation/select_excerpts.py

The cut itself happens in prepare_listening_test.py. Output is excerpts.json,
with the ASR pass cached in alignments/ so re-runs are instant.

Two findings are reported and excluded rather than worked around: a rendition
where too little of the excerpt is found did not say it, and one where the words
are all there but stretched over more than twice the per-text median span is a
runaway render.

Why excerpts and how the length is derived: docs/METHODS.md section 3.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myecho.evaluation.excerpt import (  # noqa: E402
    MAX_S,
    MIN_S,
    MIN_MATCH_RATIO,
    SKIP_HEAD_FRAC,
    TARGET_S,
    WORDS_PER_SECOND,
    locate_span,
    select_excerpt,
)
from myecho.evaluation.metrics import WHISPER_MODEL, transcribe_words  # noqa: E402

# A located span far longer than the same excerpt in the other renditions is a
# runaway render, not a slow speaker: the words are all there but stretched
# over several times the time. Cutting it would put a 44 s stimulus into a test
# built to keep stimuli near 10 s, which defeats the whole point.
MAX_SPAN_RATIO = 2.0

TESTSET_PATH = Path(__file__).parent / "testset.json"
RESULTS_DIR = Path(__file__).parent / "results"
EXCERPTS_PATH = RESULTS_DIR / "excerpts.json"
ALIGN_DIR = RESULTS_DIR / "alignments"

AUDIO_EXT = "wav"


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
    """Full text entries: the excerpt is chosen on "text", so it is needed here
    (unlike in qc_samples.py / prepare_listening_test.py, which only need ids)."""
    with open(Path(case["texts_file"]), encoding="utf-8") as f:
        data = json.load(f)
    return [{"id": t.get("id", str(i)), "text": t["text"],
             "source_audio": t["source_audio"]}
            for i, t in enumerate(data["texts"])]


def real_audio_path(source_audio):
    denoised = Path("data/DeepFilterNet") / source_audio
    return denoised if denoised.exists() else Path("data/wav") / source_audio


def candidates(case_dir, texts):
    """[(source_label, text_id, path), ...] for every model render + the real one."""
    models = sorted(p.name for p in case_dir.iterdir() if p.is_dir())
    out = []
    for model in models:
        for i, entry in enumerate(texts):
            path = case_dir / model / f"{i:02d}_{entry['id']}.{AUDIO_EXT}"
            if path.exists():
                out.append((model, entry["id"], path))
    for entry in texts:
        path = real_audio_path(entry["source_audio"])
        if path.exists():
            out.append(("REAL", entry["id"], path))
    return out


def aligned_words(source, text_id, audio_path):
    """Word timestamps for one file, cached on disk.

    The ASR pass is the expensive part (minutes per file on CPU), and nothing
    about a finished render changes, so it is written once and reused.
    """
    cache = ALIGN_DIR / source / f"{text_id}.json"
    if cache.exists():
        with open(cache, encoding="utf-8") as f:
            return json.load(f), True

    words = transcribe_words(audio_path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(words, f, ensure_ascii=False)
    return words, False


def reject_runaway_spans(spans, max_ratio=MAX_SPAN_RATIO):
    """Drop spans far longer than the same excerpt in the other renditions.

    locate_span only asks whether the words are there, not how long the model
    took to say them. A rendition that says the right words over 4x the time
    passes the match check and would then become a 44 s stimulus in a test
    designed around 10 s ones - and `voxtral/AUDIO-2021-07-30-06-44-26` in this
    material does exactly that. Compared against the per-text median across
    renditions, so a text that is simply long for everyone is not penalized.

    Returns (kept, rejected) where rejected is [(source, text_id, reason), ...].
    """
    by_text = {}
    for key, span in spans.items():
        by_text.setdefault(key.split("/", 1)[1], []).append(span["duration"])

    kept, rejected = {}, []
    for key, span in spans.items():
        source, text_id = key.split("/", 1)
        limit = median(by_text[text_id]) * max_ratio
        if span["duration"] > limit:
            rejected.append((source, text_id,
                             f"span {span['duration']:.1f}s vs {limit / max_ratio:.1f}s "
                             f"median for this text - runaway render"))
        else:
            kept[key] = span
    return kept, rejected


def main():
    case = load_case(sys.argv[1] if len(sys.argv) > 1 else None)
    texts = load_texts(case)
    case_dir = RESULTS_DIR / case["name"]
    if not case_dir.exists():
        raise SystemExit(f"{case_dir} not found - run evaluation/run_eval.py first")

    excerpts = {t["id"]: select_excerpt(t["text"]) for t in texts}
    print(f"Case: {case['name']}  texts: {len(texts)}")
    print(f"Target {TARGET_S:.0f}s (accepted {MIN_S:.0f}-{MAX_S:.0f}s), "
          f"head {SKIP_HEAD_FRAC:.0%} skipped, ASR '{WHISPER_MODEL}'\n")
    for text_id, ex in excerpts.items():
        print(f"  {text_id:30s} {ex['n_words']:3d} words  ends at {ex['boundary']}")

    work = candidates(case_dir, texts)
    print(f"\nAligning {len(work)} files (cached ones are instant)...")

    spans = {}
    failures = []
    for source, text_id, path in work:
        words, cached = aligned_words(source, text_id, path)
        span = locate_span(words, excerpts[text_id]["text"])
        key = f"{source}/{text_id}"
        if span is None:
            failures.append((source, text_id))
            print(f"  {'cache' if cached else 'asr  '}  {key:40s} NOT FOUND")
            continue
        span["duration"] = round(span["end"] - span["start"], 2)
        spans[key] = span
        print(f"  {'cache' if cached else 'asr  '}  {key:40s} "
              f"{span['start']:6.2f}-{span['end']:6.2f}s "
              f"({span['duration']:5.2f}s, match {span['match_ratio']:.2f})")

    spans, rejected = reject_runaway_spans(spans)
    for source, text_id, reason in rejected:
        failures.append((source, text_id))
        print(f"  REJECTED  {source}/{text_id:32s} {reason}")

    payload = {
        "case": case["name"],
        "created": datetime.now().isoformat(timespec="seconds"),
        "params": {
            "target_s": TARGET_S, "min_s": MIN_S, "max_s": MAX_S,
            "skip_head_frac": SKIP_HEAD_FRAC,
            "words_per_second": WORDS_PER_SECOND,
            "min_match_ratio": MIN_MATCH_RATIO,
            "max_span_ratio": MAX_SPAN_RATIO,
            "whisper_model": WHISPER_MODEL,
        },
        "excerpts": excerpts,
        "spans": spans,
        "rejected": [{"source": s_, "text_id": t_, "reason": r_}
                     for s_, t_, r_ in rejected],
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(EXCERPTS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    durations = sorted(s["duration"] for s in spans.values())
    print(f"\n{len(spans)}/{len(work)} located -> {EXCERPTS_PATH}")
    if durations:
        mid = durations[len(durations) // 2]
        print(f"Excerpt duration: {durations[0]:.1f}-{durations[-1]:.1f}s "
              f"(median {mid:.1f}s), vs. 12-92s for the full renders.")
    if failures:
        print(f"\n{len(failures)} rendition(s) unusable (excerpt not found, or found "
              "but stretched over a runaway span) - listen to these, they are "
              "usually a real synthesis failure:")
        for source, text_id in failures:
            print(f"  {source}/{text_id}")


if __name__ == "__main__":
    main()
