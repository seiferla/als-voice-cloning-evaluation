#!/usr/bin/env python3
"""Voice enrollment for Qwen3-TTS-VC-Realtime (Model Studio, Singapore region)."""

import argparse
import base64
import json
import mimetypes
import os
import pathlib
import sys

import requests
from dotenv import load_dotenv

load_dotenv()

# The workspace id is account-specific, so it lives in .env instead of here.
WORKSPACE_ID = os.getenv("QWEN3_WORKSPACE_ID") or os.getenv("DASHSCOPE_WORKSPACE_ID")
BASE_URL = os.getenv("QWEN3_BASE_URL") or os.getenv("DASHSCOPE_BASE_URL") or (
    f"https://{WORKSPACE_ID}.ap-southeast-1.maas.aliyuncs.com/api/v1"
    if WORKSPACE_ID else None
)
CUSTOMIZATION_PATH = "/services/audio/tts/customization"

# The enrollment model is fixed; target_model decides which model the voice binds to.
ENROLLMENT_MODEL = "qwen-voice-enrollment"
TARGET_MODEL_REALTIME = "qwen3-tts-vc-realtime-2026-01-15"

MIME_BY_SUFFIX = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4"}


def _post(payload: dict) -> dict:
    api_key = os.getenv("QWEN3_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        sys.exit("QWEN3_API_KEY is not set (see .env.example).")
    if not BASE_URL:
        sys.exit(
            "QWEN3_WORKSPACE_ID is not set (see .env.example).\n"
            "Find it in the DashScope console; it is the ws-... part of your "
            "workspace host.\nAlternatively set QWEN3_BASE_URL to the full "
            "endpoint."
        )

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
        sys.exit(f"Audio file not found: {path}")

    size_mb = path.stat().st_size / 1024 / 1024
    if size_mb > 10:
        sys.exit(f"File is {size_mb:.1f} MB, the limit is 10 MB.")

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
        sys.exit(f"No voice field in the response:\n{json.dumps(result, indent=2)}")

    print(f"voice      = {voice}")
    print(f"target_model = {args.target_model}")
    print("\nUse it in the realtime session as update_session(voice=...).")


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
    src.add_argument("--audio", help="Local reference file (wav/mp3/m4a)")
    src.add_argument("--url", help="Publicly reachable audio URL")
    p_create.add_argument("--name", required=True, help="preferred_name (prefix)")
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
