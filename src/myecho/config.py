
"""Configuration from environment variables."""

import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
FISH_API_KEY = os.getenv("FISH_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Model configs
S2PRO_WS_URI = "wss://api.fish.audio/v1/tts/live"
S2PRO_MODEL = "s2-pro"
VOXTRAL_MODEL = "voxtral-mini-tts-2603"
QWEN3_MODEL = "qwen-audio-3.0-tts-flash"

# DashScope (Qwen3) URLs
DASHSCOPE_BASE_HTTP_API_URL = os.getenv("DASHSCOPE_BASE_HTTP_API_URL", "https://dashscope.aliyuncs.com/api/v1")
DASHSCOPE_BASE_WS_API_URL = os.getenv("DASHSCOPE_BASE_WS_API_URL", "wss://dashscope.aliyuncs.com/api/v1")
