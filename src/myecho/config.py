"""Configuration from environment variables."""

import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
S2PRO_API_KEY = os.getenv("S2PRO_API_KEY")
VOXTRAL_API_KEY = os.getenv("VOXTRAL_API_KEY")
QWEN3_API_KEY = os.getenv("QWEN3_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Voice IDs
VOXTRAL_VOICE_ID = os.getenv("VOXTRAL_VOICE_ID")
S2PRO_REFERENCE_ID = os.getenv("S2PRO_REFERENCE_ID")
QWEN3_VOICE_ID = os.getenv("QWEN3_VOICE_ID")

# Model configs
S2PRO_WS_URI = "wss://api.fish.audio/v1/tts/live"
S2PRO_MODEL = "s2-pro"
VOXTRAL_MODEL = "voxtral-mini-tts-2603"
QWEN3_MODEL = "qwen-audio-3.0-tts-flash"

# DashScope (Qwen3) URLs
QWEN3_BASE_HTTP_API_URL = os.getenv("QWEN3_BASE_HTTP_API_URL", "https://dashscope.aliyuncs.com/api/v1")
QWEN3_BASE_WS_API_URL = os.getenv("QWEN3_BASE_WS_API_URL", "wss://dashscope.aliyuncs.com/api/v1")
