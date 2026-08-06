
"""Configuration from environment variables."""

import os
from dotenv import load_dotenv

load_dotenv()

FISH_API_KEY = os.getenv("FISH_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not FISH_API_KEY:
    raise ValueError("FISH_API_KEY not set in .env")

S2PRO_WS_URI = "wss://api.fish.audio/v1/tts/live"
S2PRO_MODEL = "s2-pro"
VOXTRAL_MODEL = "voxtral-mini-tts-2603"
QWEN3_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
QWEN3_DEVICE = "cuda:0"
