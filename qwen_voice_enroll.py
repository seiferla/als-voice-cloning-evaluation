#!/usr/bin/env python3
"""Voice-Enrollment für Qwen3-TTS-VC-Realtime (Model Studio, Region Singapur).

Legt einen Voice-Clone an, der mit dem Realtime-WebSocket-Pfad
(/api-ws/v1/realtime) nutzbar ist, und listet/löscht bestehende Voices.

Voraussetzung:
    export DASHSCOPE_API_KEY="sk-..."     # Singapur-Key!

Beispiele:
    python qwen_voice_enroll.py create --audio ref.wav --name larsrt
    python qwen_voice_enroll.py list
    python qwen_voice_enroll.py delete --voice qwen-tts-vc-...
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import pathlib
import sys

import requests

WORKSPACE_ID = os.getenv("DASHSCOPE_WORKSPACE_ID", "ws-nv1jxpvb33k9l469")
BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    f"https://{WORKSPACE_ID}.ap-southeast-1.maas.aliyuncs.com/api/v1",
)
CUSTOMIZATION_PATH = "/services/audio/tts/customization"

# Enrollment-Modell ist fix; target_model bestimmt, wogegen die Voice bindet.
ENROLLMENT_MODEL = "qwen-voice-enrollment"
TARGET_MODEL_REALTIME = "qwen3-tts-vc-realtime-2026-01-15"

MIME_BY_SUFFIX = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4"}


def _post(payload: dict) -> dict:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        sys.exit("DASHSCOPE_API_KEY ist nicht gesetzt.")

    resp = requests.post(
        f"{BASE_URL.rstrip('/')}{CUSTOMIZATION_PATH}",
        json=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=180,
    )
    if resp.status_code != 200:
        sys.exit(f"HTTP {resp.status_code}: {resp.text}")
    return resp.json()


def _audio_to_data_uri(path: pathlib.Path) -> str:
    if not path.exists():
        sys.exit(f"Audiodatei nicht gefunden: {path}")

    size_mb = path.stat().st_size / 1024 / 1024
    if size_mb > 10:
        sys.exit(f"Datei ist {size_mb:.1f} MB, erlaubt sind maximal 10 MB.")

    mime = MIME_BY_SUFFIX.get(path.suffix.lower()) or mimetypes.guess_type(path)[0]
    if not mime:
        sys.exit(f"Unbekanntes Audioformat: {path.suffix}")

    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{b64}"


def cmd_create(args: argparse.Namespace) -> None:
    audio_field = (
        args.url if args.url else _audio_to_data_uri(pathlib.Path(args.audio))
    )
    result = _post(
        {
            "model": ENROLLMENT_MODEL,
            "input": {
                "action": "create",
                "target_model": args.target_model,
                "preferred_name": args.name,
                "audio": {"data": audio_field},
            },
        }
    )
    voice = result.get("output", {}).get("voice")
    if not voice:
        sys.exit(f"Kein voice-Feld in der Antwort:\n{json.dumps(result, indent=2)}")

    print(f"voice      = {voice}")
    print(f"target_model = {args.target_model}")
    print("\nIn der Realtime-Session als update_session(voice=...) setzen.")


def cmd_list(args: argparse.Namespace) -> None:
    result = _post({"model": ENROLLMENT_MODEL, "input": {"action": "list"}})
    print(json.dumps(result, indent=2, ensure_ascii=False))


def cmd_delete(args: argparse.Namespace) -> None:
    result = _post(
        {
            "model": ENROLLMENT_MODEL,
            "input": {"action": "delete", "voice": args.voice},
        }
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_create = sub.add_parser("create", help="Neue Voice anlegen")
    src = p_create.add_mutually_exclusive_group(required=True)
    src.add_argument("--audio", help="Lokale Referenzdatei (wav/mp3/m4a)")
    src.add_argument("--url", help="Öffentlich erreichbare Audio-URL")
    p_create.add_argument("--name", required=True, help="preferred_name (Präfix)")
    p_create.add_argument("--target-model", default=TARGET_MODEL_REALTIME)
    p_create.set_defaults(func=cmd_create)

    p_list = sub.add_parser("list", help="Vorhandene Voices auflisten")
    p_list.set_defaults(func=cmd_list)

    p_delete = sub.add_parser("delete", help="Voice löschen")
    p_delete.add_argument("--voice", required=True)
    p_delete.set_defaults(func=cmd_delete)

    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
