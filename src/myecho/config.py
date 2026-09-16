"""Configuration from environment variables."""

import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
S2PRO_API_KEY = os.getenv("S2PRO_API_KEY")
VOXTRAL_API_KEY = os.getenv("VOXTRAL_API_KEY")
QWEN3_API_KEY = os.getenv("QWEN3_API_KEY")

# Voice IDs
VOXTRAL_VOICE_ID = os.getenv("VOXTRAL_VOICE_ID")
S2PRO_REFERENCE_ID = os.getenv("S2PRO_REFERENCE_ID")
QWEN3_VOICE_ID = os.getenv("QWEN3_VOICE_ID")

# Model configs
S2PRO_WS_URI = "wss://api.fish.audio/v1/tts/live"
S2PRO_MODEL = "s2-pro"
# "balanced" really streams the audio (several frames) -> TTFA is measurable.
# "normal" returns everything in one frame, which makes TTFA == latency.
S2PRO_LATENCY_MODE = os.getenv("S2PRO_LATENCY_MODE", "balanced")
VOXTRAL_MODEL = "voxtral-mini-tts-2603"
QWEN3_MODEL = "qwen-audio-3.0-tts-flash"

# Realtime voice-clone model. A voice enrolled with this target_model can ONLY
# be used over /api-ws/v1/realtime - not over the tts_v2 /api-ws/v1/inference path.
QWEN3_REALTIME_MODEL = os.getenv("QWEN3_REALTIME_MODEL", "qwen3-tts-vc-realtime-2026-01-15")

# DashScope (Qwen3) host - dedicated per-workspace endpoint, not the shared dashscope.aliyuncs.com
QWEN3_API_HOST = os.getenv("QWEN3_API_HOST", "dashscope.aliyuncs.com")
QWEN3_BASE_HTTP_API_URL = f"https://{QWEN3_API_HOST}/api/v1"
QWEN3_BASE_WS_API_URL = f"wss://{QWEN3_API_HOST}/api-ws/v1/inference"
QWEN3_BASE_WS_REALTIME_URL = f"wss://{QWEN3_API_HOST}/api-ws/v1/realtime"

# Optional: only needed when the key is not already scoped to the workspace.
QWEN3_WORKSPACE_ID = os.getenv("QWEN3_WORKSPACE_ID") or os.getenv("DASHSCOPE_WORKSPACE_ID") or None

# Language hint for synthesis ("de", "en", ... or None for auto-detect)
QWEN3_LANGUAGE_TYPE = os.getenv("QWEN3_LANGUAGE_TYPE", "de")
