"""S2-Pro (Fish Audio) TTS synthesize runner (CLI)."""

import argparse
import asyncio
import sys
from pathlib import Path

from myecho.config import FISH_API_KEY, FISH_REFERENCE_ID
from myecho.models.s2_pro import S2ProTTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="S2-Pro (Fish Audio) TTS Synthese-Runner")
    parser.add_argument(
        "--text",
        default="Hallo, das ist ein automatisierter S2-Pro Synthese-Test.",
        help="Zu synthetisierender Text.",
    )
    parser.add_argument(
        "--reference-id",
        default=FISH_REFERENCE_ID,
        help="Reference/Model ID (default: aus .env FISH_REFERENCE_ID).",
    )
    parser.add_argument(
        "--out",
        default="outputs/s2pro_test.mp3",
        help="Ausgabe-Pfad für die MP3-Datei.",
    )
    parser.add_argument("--speed", type=float, default=1.0, help="Sprechgeschwindigkeit.")
    parser.add_argument("--volume", type=float, default=0.0, help="Lautstärke-Offset (dB).")
    return parser.parse_args()


async def _run(text: str, reference_id: str, out_path: str, speed: float, volume: float) -> dict:
    tts = S2ProTTS(reference_id=reference_id)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    return await tts.synthesize(text, output_path=out_path, speed=speed, volume=volume)


def main() -> int:
    args = parse_args()

    if not FISH_API_KEY:
        print("ERROR: FISH_API_KEY fehlt in .env", file=sys.stderr)
        return 2
    if not args.reference_id:
        print("ERROR: FISH_REFERENCE_ID fehlt (in .env oder --reference-id)", file=sys.stderr)
        return 2

    result = asyncio.run(_run(args.text, args.reference_id, args.out, args.speed, args.volume))

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

