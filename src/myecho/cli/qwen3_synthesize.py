"""Qwen3 (DashScope) TTS synthesize runner (CLI)."""

import argparse
import asyncio
import sys
from pathlib import Path

from myecho.config import QWEN3_API_KEY, QWEN3_VOICE_ID
from myecho.models.qwen3 import Qwen3TTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3 (DashScope) TTS Synthese-Runner")
    parser.add_argument(
        "--text",
        default="Hallo, das ist ein automatisierter Qwen3 Synthese-Test.",
        help="Zu synthetisierender Text.",
    )
    parser.add_argument(
        "--voice-id",
        default=QWEN3_VOICE_ID,
        help="Voice ID (default: aus .env QWEN3_VOICE_ID).",
    )
    parser.add_argument(
        "--out",
        default="outputs/qwen3_test.wav",
        help="Ausgabe-Pfad für die WAV-Datei.",
    )
    return parser.parse_args()


async def _run(text: str, voice_id: str, out_path: str) -> dict:
    tts = Qwen3TTS(voice_id=voice_id)
    return await tts.synthesize(text, output_path=out_path)


def main() -> int:
    args = parse_args()

    if not QWEN3_API_KEY:
        print("ERROR: QWEN3_API_KEY fehlt in .env", file=sys.stderr)
        return 2
    if not args.voice_id:
        print("ERROR: QWEN3_VOICE_ID fehlt (in .env oder --voice-id)", file=sys.stderr)
        return 2

    result = asyncio.run(_run(args.text, args.voice_id, args.out))

    out = Path(args.out)
    size = out.stat().st_size if out.exists() else 0
    ok = out.exists() and size > 1000

    m = result["metrics"]
    ttfa_s = m.get("ttfa")
    latency_s = m.get("latency")
    duration_s = m.get("audio_duration_s")
    rtf = m.get("rtf")

    print(f"file={out} exists={out.exists()} size={size} bytes")
    parts = []
    if ttfa_s is not None:
        parts.append(f"ttfa={ttfa_s:.3f} s")
    if latency_s is not None:
        parts.append(f"latency={latency_s:.3f} s")
    if duration_s is not None:
        parts.append(f"audio_duration={duration_s:.3f} s")
    if rtf is not None:
        parts.append(f"rtf={rtf:.3f} (processing_time / audio_duration, dimensionslos)")
    print("metrics: " + " | ".join(parts) if parts else f"metrics={m}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
