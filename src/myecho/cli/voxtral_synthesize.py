"""Voxtral TTS synthesize runner (CLI)."""

import argparse
import asyncio
import sys
from pathlib import Path

from myecho.config import MISTRAL_API_KEY, VOXTRAL_VOICE_ID
from myecho.models.voxtral import VoxtralTTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Voxtral TTS Synthese-Runner")
    parser.add_argument(
        "--text",
        default="Hallo, das ist ein automatisierter Voxtral Synthese-Test.",
        help="Zu synthetisierender Text.",
    )
    parser.add_argument(
        "--voice-id",
        default=VOXTRAL_VOICE_ID,
        help="Voice ID (default: aus .env VOXTRAL_VOICE_ID).",
    )
    parser.add_argument(
        "--out",
        default="outputs/voxtral_test.wav",
        help="Ausgabe-Pfad für die WAV-Datei.",
    )
    return parser.parse_args()


async def _run(text: str, voice_id: str, out_path: str) -> dict:
    tts = VoxtralTTS(voice_id=voice_id)
    return await tts.synthesize(text, output_path=out_path)


def main() -> int:
    args = parse_args()

    if not MISTRAL_API_KEY:
        print("ERROR: MISTRAL_API_KEY fehlt in .env", file=sys.stderr)
        return 2
    if not args.voice_id:
        print("ERROR: VOXTRAL_VOICE_ID fehlt (in .env oder --voice-id)", file=sys.stderr)
        return 2

    result = asyncio.run(_run(args.text, args.voice_id, args.out))

    out = Path(args.out)
    size = out.stat().st_size if out.exists() else 0
    ok = out.exists() and size > 1000

    m = result["metrics"]
    latency_s = m.get("latency")
    rtf = m.get("rtf")
    duration_s = m.get("audio_duration_s")

    print(f"file={out} exists={out.exists()} size={size} bytes")
    print(
        "metrics: "
        f"latency={latency_s:.3f} s | "
        f"audio_duration={duration_s:.3f} s | "
        f"rtf={rtf:.3f} (processing_time / audio_duration, dimensionslos)"
        if latency_s is not None and duration_s is not None and rtf is not None
        else f"metrics={m}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

