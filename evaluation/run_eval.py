"""Evaluation runner: synthesizes each text with each model and computes SIM/WER/latency/RTF.

Usage:
    uv run python evaluation/run_eval.py

Fill in evaluation/testset.json first (one or more cases, each with its own
reference audio + texts to synthesize).

Voice IDs come from .env by default (S2PRO_REFERENCE_ID, VOXTRAL_VOICE_ID,
QWEN3_VOICE_ID) and can be overridden per case in testset.json. A model whose
voice ID is missing is skipped, so a partial run is still possible.

A case's texts come from either an inline "texts" list of strings, or a
"texts_file" pointing to a JSON file with {"texts": [{"id": ..., "text": ...}, ...]}.
Use texts_file for real/personal texts so they can live under data/ (gitignored)
instead of in this tracked testset.json.
"""

import asyncio
import csv
import json
from pathlib import Path

from myecho.config import (
    QWEN3_VOICE_ID,
    S2PRO_REFERENCE_ID,
    VOXTRAL_VOICE_ID,
)
from myecho.evaluation.metrics import extract_embedding, similarity, transcribe, wer
from myecho.models.qwen3 import Qwen3TTS
from myecho.models.s2_pro import S2ProTTS
from myecho.models.voxtral import VoxtralTTS

TESTSET_PATH = Path(__file__).parent / "testset.json"
RESULTS_DIR = Path(__file__).parent / "results"

# All three models now write WAV, so the eval reads one lossless format only.
AUDIO_EXT = "wav"

CSV_FIELDS = [
    "case", "model", "text_id", "sim", "wer",
    "latency", "ttfa", "rtf", "audio_duration_s", "connect_s", "audio_file",
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
    """Resolve a case's texts to [{"id": ..., "text": ...}, ...].

    Either inline "texts" (list of strings, id = index) or "texts_file"
    (a JSON file with {"texts": [{"id": ..., "text": ...}, ...]}).
    """
    if "texts_file" in case:
        texts_path = Path(case["texts_file"])
        with open(texts_path, encoding="utf-8") as f:
            data = json.load(f)
        return [{"id": t.get("id", str(i)), "text": t["text"]} for i, t in enumerate(data["texts"])]

    return [{"id": str(i), "text": text} for i, text in enumerate(case["texts"])]


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


async def run_model(name, model, texts, ref_embedding, out_dir):
    rows = []
    for i, entry in enumerate(texts):
        text_id, text = entry["id"], entry["text"]
        output_path = out_dir / f"{i:02d}_{text_id}.{AUDIO_EXT}"
        try:
            result = await model.synthesize(text, output_path=output_path)
        except Exception as e:
            print(f"    [{name}] {text_id} SYNTHESIS FAILED: {type(e).__name__}: {e}")
            continue

        try:
            clone_embedding = extract_embedding(output_path)
            sim_score = similarity(ref_embedding, clone_embedding)
            asr_text = transcribe(output_path)
            wer_score = wer(text, asr_text)
        except Exception as e:
            print(f"    [{name}] {text_id} SCORING FAILED: {type(e).__name__}: {e}")
            continue

        m = result["metrics"]
        rows.append({
            "model": name,
            "text_id": text_id,
            "sim": round(sim_score, 4),
            "wer": round(wer_score, 4),
            "latency": m.get("latency"),
            "ttfa": m.get("ttfa"),
            "rtf": m.get("rtf"),
            "audio_duration_s": m.get("audio_duration_s"),
            "connect_s": m.get("connect_s"),
            "audio_file": str(output_path),
        })
        print(f"    [{name}] {text_id}: SIM={sim_score:.3f}  WER={wer_score:.3f}  RTF={_fmt(m.get('rtf'))}")

    return rows


def _fmt(value, digits=3):
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "n/a"


async def run_case(case):
    name = case["name"]
    texts = load_texts(case)
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
        print(f"\nRunning {model_name}...")
        out_dir = case_dir / model_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for row in await run_model(model_name, model, texts, ref_embedding, out_dir):
            rows.append({"case": name, **row})

    return rows


async def main():
    testset = load_testset()
    cases = testset["cases"]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for case in cases:
        all_rows += await run_case(case)

    if not all_rows:
        print("\nNo results produced.")
        return

    csv_path = RESULTS_DIR / "results_1.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSaved results to {csv_path}")

    print("\n=== Summary (mean per case/model) ===")
    for case_name, model_name in sorted({(r["case"], r["model"]) for r in all_rows}):
        rows = [r for r in all_rows if r["case"] == case_name and r["model"] == model_name]
        print(
            f"{case_name}/{model_name} (n={len(rows)}): "
            f"SIM={_fmt(_mean(rows, 'sim'))}  "
            f"WER={_fmt(_mean(rows, 'wer'))}  "
            f"RTF={_fmt(_mean(rows, 'rtf'))}  "
            f"TTFA={_fmt(_mean(rows, 'ttfa'))}  "
            f"latency={_fmt(_mean(rows, 'latency'))}"
        )


def _mean(rows, key):
    values = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return sum(values) / len(values) if values else None


if __name__ == "__main__":
    asyncio.run(main())
