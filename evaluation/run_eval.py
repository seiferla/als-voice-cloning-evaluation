"""Objective evaluation: synthesizes each text with each model and computes
SIM / WER / latency, alongside the anchors those numbers need to be readable.

Usage:
    uv run python evaluation/run_eval.py
    uv run python evaluation/run_eval.py --repeats 3      # recommended final run

Fill in evaluation/testset.json first (one or more cases, each with its own
reference audio + texts to synthesize).

Voice IDs come from .env by default (S2PRO_REFERENCE_ID, VOXTRAL_VOICE_ID,
QWEN3_VOICE_ID) and can be overridden per case in testset.json. A model whose
voice ID is missing is skipped, so a partial run is still possible.

A case's texts come from either an inline "texts" list of strings, or a
"texts_file" pointing to a JSON file with {"texts": [{"id": ..., "text": ...}, ...]}.
Use texts_file for real/personal texts so they can live under data/ (gitignored)
instead of in this tracked testset.json. The listening-test scripts additionally
need a "source_audio" per text, so prefer texts_file.

Anchors: why the plain numbers are not enough
---------------------------------------------
SIM is a cosine similarity between ECAPA embeddings. 0.70 is not good or bad by
itself - the scale has no fixed meaning, it depends on the speaker, the
recording and the checkpoint. WER is measured by an ASR system that also makes
mistakes on real speech, and the test texts here are transcripts of spontaneous
voice memos, which are harder to recognize than anything a TTS model reads out.

So this runner also evaluates two extra conditions, written into the same table:

    REAL       the speaker's other recordings, scored exactly like a model.
               Its SIM is the upper bound (how similar the same person is to
               themselves across two recordings) and its WER is the floor (how
               well the ASR recovers the target text from real speech).
               Measured here: WER floor ~0.18 with whisper-small, while the
               better clones score ~0.09 - i.e. the clones are EASIER to
               transcribe than the human original, because reading a transcript
               produces cleaner speech than speaking spontaneously. Without
               this row, that inversion is invisible and WER looks like
               intelligibility when it is closer to "distance from read speech".

    IMPOSTOR   a different speaker, optional. Its SIM is the lower bound: what
               the metric returns for someone who is not the target at all.
               Set "impostor_audio" in the case to a file or a glob (e.g. a few
               Common Voice DE clips). Without it, the bottom of the SIM scale
               is unknown and the reported similarity cannot be calibrated.

Repetitions
-----------
These are sampling TTS systems: the same text gives a different render every
time. With --repeats 1 the model-internal variance is invisible and a single
unlucky render looks like a property of the model. --repeats 3 is the
recommended setting for the run you report; the default stays 1 so an
exploratory run does not silently triple the API bill.

Output
------
evaluation/results/results_<timestamp>.csv  per render, plus the REAL/IMPOSTOR rows
evaluation/results/results_latest.csv       a copy, for scripts
evaluation/results/eval_run_meta.json       git commit, seed, input hashes, settings
"""

import argparse
import asyncio
import csv
import hashlib
import json
import statistics
import subprocess
from datetime import datetime
from pathlib import Path

from myecho.config import (
    QWEN3_VOICE_ID,
    S2PRO_REFERENCE_ID,
    VOXTRAL_VOICE_ID,
)
from myecho.evaluation.metrics import (
    WHISPER_MODEL,
    extract_embedding,
    similarity,
    transcribe,
    wer,
)
from myecho.models.qwen3 import Qwen3TTS
from myecho.models.s2_pro import S2ProTTS
from myecho.models.voxtral import VoxtralTTS

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTSET_PATH = Path(__file__).parent / "testset.json"
RESULTS_DIR = Path(__file__).parent / "results"

# All three models now write WAV, so the eval reads one lossless format only.
AUDIO_EXT = "wav"

REAL = "REAL"
IMPOSTOR = "IMPOSTOR"

CSV_FIELDS = [
    "case", "model", "text_id", "repeat", "sim", "wer",
    # Renamed from "rtf": this is wall-clock API latency divided by the length of
    # the audio it returned, measured over the internet against three different
    # services (one of them non-streaming, one hosted in Singapore). Real RTF is
    # compute time over audio duration on known hardware, which is not what any
    # of these APIs let you observe.
    "latency_s", "ttfa_s", "service_rtf", "audio_duration_s", "connect_s",
    "note", "audio_file",
]

PLACEHOLDERS = ("REPLACE", "your_")


def load_testset():
    with open(TESTSET_PATH, encoding="utf-8") as f:
        return json.load(f)


def is_configured(value):
    """False for None, empty, or a leftover template placeholder."""
    if not value:
        return False
    return not any(p in value for p in PLACEHOLDERS)


def load_texts(case):
    """Resolve a case's texts to [{"id", "text", "source_audio"}, ...]."""
    if "texts_file" in case:
        with open(Path(case["texts_file"]), encoding="utf-8") as f:
            data = json.load(f)
        return [{"id": t.get("id", str(i)), "text": t["text"],
                 "source_audio": t.get("source_audio")}
                for i, t in enumerate(data["texts"])], data

    return ([{"id": str(i), "text": text, "source_audio": None}
             for i, text in enumerate(case["texts"])], {})


def real_audio_path(source_audio):
    """The speaker's own recording of this text, denoised where available."""
    if not source_audio:
        return None
    denoised = Path("data/DeepFilterNet") / source_audio
    if denoised.exists():
        return denoised
    raw = Path("data/wav") / source_audio
    return raw if raw.exists() else None


def build_models(case):
    """Resolve which models are runnable for this case: [(name, model), ...]."""
    models = []

    s2pro_id = case.get("s2pro_reference_id") or S2PRO_REFERENCE_ID
    if is_configured(s2pro_id):
        models.append(("s2pro", S2ProTTS(reference_id=s2pro_id)))
    else:
        print("  skip s2pro: no reference_id (S2PRO_REFERENCE_ID or case override)")

    voxtral_id = case.get("voxtral_voice_id") or VOXTRAL_VOICE_ID
    if is_configured(voxtral_id):
        # Reuse the existing voice instead of calling create_voice() per run -
        # a fresh clone every run would make SIM values non-reproducible.
        models.append(("voxtral", VoxtralTTS(voice_id=voxtral_id)))
    else:
        print("  skip voxtral: no voice_id (VOXTRAL_VOICE_ID or case override)")

    qwen3_id = case.get("qwen3_voice_id") or QWEN3_VOICE_ID
    if is_configured(qwen3_id):
        models.append(("qwen3", Qwen3TTS(voice_id=qwen3_id)))
    else:
        print("  skip qwen3: no voice_id (QWEN3_VOICE_ID or case override)")

    return models


def rel(path):
    """Repo-relative path, so a results CSV can be attached to a report without
    leaking the absolute path of somebody's home directory."""
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def score(audio_path, text, ref_embedding):
    """(sim, wer) for one audio file against the reference voice and target text."""
    clone_embedding = extract_embedding(audio_path)
    sim_score = similarity(ref_embedding, clone_embedding)
    wer_score = wer(text, transcribe(audio_path))
    return round(sim_score, 4), round(wer_score, 4)


async def run_model(name, model, texts, ref_embedding, out_dir, repeats):
    rows = []
    for i, entry in enumerate(texts):
        text_id, text = entry["id"], entry["text"]
        for repeat in range(repeats):
            suffix = "" if repeats == 1 else f"_r{repeat + 1}"
            output_path = out_dir / f"{i:02d}_{text_id}{suffix}.{AUDIO_EXT}"
            try:
                result = await model.synthesize(text, output_path=output_path)
            except Exception as e:
                print(f"    [{name}] {text_id}{suffix} SYNTHESIS FAILED: "
                      f"{type(e).__name__}: {e}")
                continue

            try:
                sim_score, wer_score = score(output_path, text, ref_embedding)
            except Exception as e:
                print(f"    [{name}] {text_id}{suffix} SCORING FAILED: "
                      f"{type(e).__name__}: {e}")
                continue

            m = result["metrics"]
            rows.append({
                "model": name, "text_id": text_id, "repeat": repeat + 1,
                "sim": sim_score, "wer": wer_score,
                "latency_s": m.get("latency"),
                "ttfa_s": m.get("ttfa"),
                "service_rtf": m.get("rtf"),
                "audio_duration_s": m.get("audio_duration_s"),
                "connect_s": m.get("connect_s"),
                "note": "",
                "audio_file": rel(output_path),
            })
            print(f"    [{name}] {text_id}{suffix}: SIM={sim_score:.3f}  "
                  f"WER={wer_score:.3f}  service_RTF={_fmt(m.get('rtf'))}")
    return rows


def run_anchors(texts, meta, reference_audio, ref_embedding, impostor_paths):
    """The REAL and IMPOSTOR rows: the same metrics on non-synthesized audio.

    REAL is the speaker's own recording of each text. Two of them are special
    and get flagged rather than dropped, because both are still informative:
    the recording the voices were cloned from is not held out, and the recording
    used as the SIM reference would score 1.0 against itself, so that one is
    skipped entirely.
    """
    rows = []
    reference_name = Path(reference_audio).name
    clone_source = meta.get("cloning_source_audio")

    for entry in texts:
        path = real_audio_path(entry.get("source_audio"))
        if path is None:
            continue
        if Path(entry["source_audio"]).name == reference_name:
            print(f"    [{REAL}] {entry['id']}: skipped - this recording IS the "
                  "SIM reference, comparing it with itself would score 1.0")
            continue
        try:
            sim_score, wer_score = score(path, entry["text"], ref_embedding)
        except Exception as e:
            print(f"    [{REAL}] {entry['id']} SCORING FAILED: {type(e).__name__}: {e}")
            continue
        note = "not held out: this is the voice-cloning source" \
            if clone_source and Path(entry["source_audio"]).name == clone_source else ""
        rows.append({
            "model": REAL, "text_id": entry["id"], "repeat": 1,
            "sim": sim_score, "wer": wer_score,
            "latency_s": None, "ttfa_s": None, "service_rtf": None,
            "audio_duration_s": None, "connect_s": None,
            "note": note, "audio_file": rel(path),
        })
        print(f"    [{REAL}] {entry['id']}: SIM={sim_score:.3f}  WER={wer_score:.3f}"
              f"{'  (' + note + ')' if note else ''}")

    for path in impostor_paths:
        try:
            sim_score = round(similarity(ref_embedding, extract_embedding(path)), 4)
        except Exception as e:
            print(f"    [{IMPOSTOR}] {path.name} FAILED: {type(e).__name__}: {e}")
            continue
        rows.append({
            "model": IMPOSTOR, "text_id": path.stem, "repeat": 1,
            "sim": sim_score, "wer": None,
            "latency_s": None, "ttfa_s": None, "service_rtf": None,
            "audio_duration_s": None, "connect_s": None,
            "note": "different speaker - SIM lower bound",
            "audio_file": rel(path),
        })
        print(f"    [{IMPOSTOR}] {path.name}: SIM={sim_score:.3f}")

    return rows


def resolve_impostors(case):
    """Files for the SIM lower bound, from "impostor_audio" (a path or a glob)."""
    pattern = case.get("impostor_audio")
    if not pattern:
        return []
    path = Path(pattern)
    if path.exists() and path.is_file():
        return [path]
    matches = sorted(Path().glob(pattern))
    if not matches:
        print(f"  note: impostor_audio '{pattern}' matched no files")
    return matches


def _fmt(value, digits=3):
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "n/a"


async def run_case(case, repeats):
    name = case["name"]
    texts, meta = load_texts(case)
    reference_audio = case["reference_audio"]

    print(f"\n=== Case: {name} ===")
    if not Path(reference_audio).exists():
        print(f"  reference_audio not found: {reference_audio} - case skipped")
        return []

    print("Extracting reference speaker embedding...")
    ref_embedding = extract_embedding(reference_audio)

    case_dir = RESULTS_DIR / name
    rows = []
    for model_name, model in build_models(case):
        print(f"\nRunning {model_name} ({repeats} render(s) per text)...")
        out_dir = case_dir / model_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for row in await run_model(model_name, model, texts, ref_embedding,
                                   out_dir, repeats):
            rows.append({"case": name, **row})

    print(f"\nAnchors (REAL upper bound / WER floor, IMPOSTOR lower bound)...")
    impostors = resolve_impostors(case)
    if not impostors:
        print("  no impostor_audio configured - the SIM lower bound stays unknown, "
              "so an absolute\n  SIM value cannot be calibrated. Add a few clips of "
              "a different speaker\n  (e.g. Common Voice DE) as \"impostor_audio\" "
              "in testset.json.")
    for row in run_anchors(texts, meta, reference_audio, ref_embedding, impostors):
        rows.append({"case": name, **row})

    return rows


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def summarize(all_rows):
    """Median and IQR per case/model, not the mean.

    A single runaway or truncated render moves a mean of 16 items a long way -
    voxtral's WER standard deviation on this material is larger than its mean,
    which means one or two total failures rather than uniformly worse output.
    The median says what the model usually does; the IQR says how consistent it
    is; the failures are visible as rows in the CSV.
    """
    print("\n=== Summary (median [IQR] per case/model) ===")
    for case_name, model_name in sorted({(r["case"], r["model"]) for r in all_rows}):
        rows = [r for r in all_rows if r["case"] == case_name and r["model"] == model_name]
        parts = []
        for key in ("sim", "wer", "service_rtf", "ttfa_s", "latency_s"):
            values = sorted(v for v in (r.get(key) for r in rows)
                            if isinstance(v, (int, float)))
            if not values:
                continue
            median = statistics.median(values)
            if len(values) >= 4:
                q1, q3 = (statistics.median(values[:len(values) // 2]),
                          statistics.median(values[(len(values) + 1) // 2:]))
                parts.append(f"{key}={median:.3f} [{q1:.3f}-{q3:.3f}]")
            else:
                parts.append(f"{key}={median:.3f}")
        print(f"{case_name}/{model_name} (n={len(rows)}): " + "  ".join(parts))


async def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--repeats", type=int, default=1,
                        help="Renders per text per model. 3 is recommended for the "
                             "run you report; these APIs sample, so 1 render hides "
                             "the model's own variance.")
    args = parser.parse_args()

    testset = load_testset()
    cases = testset["cases"]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.repeats == 1:
        print("note: --repeats 1. Each text is rendered once, so the spread you see "
              "is\n      text difficulty only, not the model's own run-to-run "
              "variance.\n      Use --repeats 3 for the final run.")

    all_rows = []
    for case in cases:
        all_rows += await run_case(case, args.repeats)

    if not all_rows:
        print("\nNo results produced.")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"results_{stamp}.csv"
    for path in (csv_path, RESULTS_DIR / "results_latest.csv"):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)

    meta = {
        "run": stamp,
        "git_commit": git_commit(),
        "repeats": args.repeats,
        "whisper_model": WHISPER_MODEL,
        "speaker_embedding": "speechbrain/spkrec-ecapa-voxceleb",
        "cases": [
            {
                "name": case["name"],
                "reference_audio": case.get("reference_audio"),
                "texts_file": case.get("texts_file"),
                "texts_file_sha256": (
                    hashlib.sha256(Path(case["texts_file"]).read_bytes()).hexdigest()
                    if case.get("texts_file") and Path(case["texts_file"]).exists()
                    else None
                ),
                "impostor_audio": case.get("impostor_audio"),
            }
            for case in cases
        ],
        "n_rows": len(all_rows),
    }
    (RESULTS_DIR / "eval_run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nSaved {csv_path} (and results_latest.csv)")
    print(f"Run provenance -> {RESULTS_DIR / 'eval_run_meta.json'}")
    summarize(all_rows)
    print("\nRead SIM against the anchors, never on its own: REAL is the upper "
          "bound (the\nsame person across two recordings), IMPOSTOR the lower "
          "bound. Read WER against\nthe REAL row, which is this ASR's error rate "
          "on the human original.")


if __name__ == "__main__":
    asyncio.run(main())
