"""Evaluation runner: synthesizes each text with each model and computes SIM/WER/latency/RTF.

Usage:
    uv run python evaluation/run_eval.py

Fill in evaluation/testset.json first (reference audio + texts to synthesize).
"""

import asyncio
import csv
import json
from pathlib import Path

from myecho.models.s2_pro import S2ProTTS
from myecho.models.voxtral import VoxtralTTS
from myecho.evaluation.metrics import similarity, wer, extract_embedding, transcribe

TESTSET_PATH = Path(__file__).parent / "testset.json"
RESULTS_DIR = Path(__file__).parent / "results"


def load_testset():
    with open(TESTSET_PATH, encoding="utf-8") as f:
        return json.load(f)


async def run_model(name, model, texts, ref_embedding, out_dir):
    rows = []
    for i, text in enumerate(texts):
        output_path = out_dir / f"{i:02d}.wav"
        try:
            result = await model.synthesize(text, output_path=output_path)
        except Exception as e:
            print(f"  [{name}] text {i} FAILED: {e}")
            continue

        clone_embedding = extract_embedding(output_path)
        sim_score = similarity(ref_embedding, clone_embedding)
        asr_text = transcribe(output_path)
        wer_score = wer(text, asr_text)

        m = result["metrics"]
        row = {
            "model": name,
            "text_id": i,
            "sim": round(sim_score, 4),
            "wer": round(wer_score, 4),
            "latency": m.get("latency"),
            "rtf": m.get("rtf"),
            "ttfa": m.get("ttfa"),
        }
        rows.append(row)
        print(f"  [{name}] {i}: SIM={sim_score:.3f}  WER={wer_score:.3f}  RTF={m.get('rtf')}")

    return rows


async def main():
    testset = load_testset()
    texts = testset["texts"]
    reference_audio = testset["reference_audio"]

    print("Extracting reference speaker embedding...")
    ref_embedding = extract_embedding(reference_audio)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []

    # S2-Pro (needs a reference_id already registered on Fish Audio)
    reference_id = testset.get("s2pro_reference_id")
    if reference_id and "REPLACE" not in reference_id:
        print("\nRunning S2-Pro...")
        out_dir = RESULTS_DIR / "s2pro"
        out_dir.mkdir(exist_ok=True)
        model = S2ProTTS(reference_id=reference_id)
        all_rows += await run_model("s2pro", model, texts, ref_embedding, out_dir)
    else:
        print("\nSkipping S2-Pro (set 's2pro_reference_id' in testset.json)")

    # Voxtral (creates a voice clone from the reference audio)
    print("\nRunning Voxtral...")
    out_dir = RESULTS_DIR / "voxtral"
    out_dir.mkdir(exist_ok=True)
    model = VoxtralTTS()
    model.create_voice(reference_audio)
    all_rows += await run_model("voxtral", model, texts, ref_embedding, out_dir)

    # Qwen3 (optional — only runs once configured with a DashScope token)
    try:
        from myecho.models.qwen3 import Qwen3TTS

        print("\nRunning Qwen3...")
        out_dir = RESULTS_DIR / "qwen3"
        out_dir.mkdir(exist_ok=True)
        model = Qwen3TTS()
        model.create_voice(reference_audio)
        all_rows += await run_model("qwen3", model, texts, ref_embedding, out_dir)
    except Exception as e:
        print(f"\nSkipping Qwen3 (not configured yet): {e}")

    if not all_rows:
        print("\nNo results produced.")
        return

    csv_path = RESULTS_DIR / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSaved results to {csv_path}")

    print("\n=== Summary (mean per model) ===")
    models = sorted(set(r["model"] for r in all_rows))
    for m in models:
        rows = [r for r in all_rows if r["model"] == m]
        sim_mean = sum(r["sim"] for r in rows) / len(rows)
        wer_mean = sum(r["wer"] for r in rows) / len(rows)
        rtf_vals = [r["rtf"] for r in rows if r["rtf"] is not None]
        rtf_mean = sum(rtf_vals) / len(rtf_vals) if rtf_vals else None
        print(f"{m}: SIM={sim_mean:.3f}  WER={wer_mean:.3f}  RTF={rtf_mean}")


if __name__ == "__main__":
    asyncio.run(main())
