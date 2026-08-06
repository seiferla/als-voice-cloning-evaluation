# Voice Cloning Evaluation Framework

Evaluation framework for open-source TTS and voice cloning models. Includes audio preprocessing, model inference, and evaluation metrics.

## Quick Start

```bash
# Install dependencies
uv sync --extra notebook

# Setup environment
cp .env.example .env
# Add FISH_API_KEY to .env

# Run notebooks
uv run jupyter notebook notebooks/data_analysis.ipynb
```

## Project Structure

```
src/myecho/
├── config.py                    # Load API keys from .env
├── models/
│   └── s2_pro.py               # S2-Pro TTS (Fish Audio API)
└── evaluation/
    └── metrics.py              # SIM, WER metrics

notebooks/
├── data_preprocessing.ipynb     # Audio preprocessing (LUFS, silence removal, WhisperX)
├── data_analysis.ipynb          # Analysis & visualization
└── evaluation.ipynb             # Metric evaluation

data/                            # Audio files (gitignored)
├── wav/
├── m4a/
└── DeepFilterNet/
```

## Usage

### Python API

```python
from myecho.models.s2_pro import S2ProTTS
import asyncio

tts = S2ProTTS(reference_id="your_reference_id")
result = asyncio.run(tts.synthesize("Hallo Welt"))

print(f"TTFA: {result['metrics']['ttfa']*1000:.0f} ms")
print(f"RTF: {result['metrics']['rtf']:.3f}")
```

### Evaluation Metrics

```python
from myecho.evaluation.metrics import similarity, wer
import numpy as np

# Speaker similarity
sim_score = similarity(ref_embedding, cloned_embedding)

# Word error rate
error_rate = wer("reference text", "asr output")
```

## Dependencies

**Core:**
- `websockets`, `msgpack` — S2-Pro WebSocket communication
- `pydub`, `pedalboard` — Audio processing & effects
- `librosa`, `numpy`, `scipy` — Audio analysis
- `python-dotenv` — Environment variables

**Optional (evaluation):**
```bash
uv sync --extra eval  # Adds: jiwer, openai-whisper, whisperx
```

**Optional (notebooks):**
```bash
uv sync --extra notebook  # Adds: jupyter, matplotlib, plotly, pandas, sklearn, transformers, torch, etc.
```

## Models Evaluated

- **S2-Pro** (Fish Audio API) — WebSocket streaming TTS with reference voice
- **Voxtral** (Open Source) — To be implemented
- **Qwen3** (Open Source) — To be implemented

## Metrics

**Objective:**
- **SIM** — Speaker Similarity (cosine distance of embeddings)
- **WER** — Word Error Rate (via Whisper/WhisperX)

**Performance:**
- **RTF** — Real-Time Factor
- **TTFA** — Time To First Audio (streaming latency)

## Setup Notes

- API keys are loaded from `.env` file (not in repository)
- `.env.example` shows required variables
- All audio data is gitignored (privacy/size)
- Notebooks require CUDA for transcription/embeddings (CPU slower)

## Files

- `pyproject.toml` — Dependencies and project metadata
- `.env.example` — Template for environment variables
- `src/myecho/` — Main package code
