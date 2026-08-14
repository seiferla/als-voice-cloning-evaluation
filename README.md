# Voice Cloning Evaluation Framework

Evaluation framework for voice cloning models (TTS, speaker similarity, intelligibility, performance metrics).

## Quick Start

```bash
# Install core dependencies
uv sync

# Setup API keys
cp .env.example .env
# Fill in: S2PRO_API_KEY, VOXTRAL_API_KEY (optional: QWEN3_API_KEY for Qwen3)

# Install evaluation tools (for SIM/WER/ASR/embedding extraction)
uv sync --extra eval

# Run evaluation pipeline
uv run python evaluation/run_eval.py
```

## Evaluation Pipeline

1. **Prepare testset** — edit `evaluation/testset.json`:
   ```json
   {
     "reference_audio": "data/wav/your_reference.wav",
     "s2pro_reference_id": "fish-audio-reference-id",
     "texts": ["Text 1 to synthesize", "Text 2 to synthesize"]
   }
   ```

2. **Run evaluation**:
   ```bash
   uv run python evaluation/run_eval.py
   ```

3. **Results** — saved to `evaluation/results/`:
   - `results.csv` — per-text metrics (SIM, WER, latency, RTF)
   - Model subdirectories with generated audio files

## Models Tested

| Model | Type | Status |
|---|---|---|
| **S2-Pro** | Fish Audio WebSocket API | ✓ Implemented |
| **Voxtral** | Mistral API (custom voice) | ✓ Implemented |
| **Qwen3** | Alibaba DashScope API | ✓ Implemented (requires DashScope token) |

## Metrics

**Quality Metrics:**
- **SIM** — Speaker Similarity (cosine distance of ECAPA embeddings) — how much does it sound like the reference?
- **WER** — Word Error Rate (via Whisper ASR) — is the text intelligible?

**Performance Metrics:**
- **Latency** — total API response time
- **RTF** — Real-Time Factor (latency / audio_duration) — can it run in realtime? (RTF < 1 = yes)
- **TTFA** — Time To First Audio (S2-Pro only, streaming latency)

## Project Structure

```
src/myecho/
├── config.py                    # API keys, model configs
├── models/
│   ├── s2_pro.py               # Fish Audio WebSocket TTS
│   ├── voxtral.py              # Mistral voice cloning
│   └── qwen3.py                # Alibaba DashScope voice cloning
└── evaluation/
    └── metrics.py              # SIM (ECAPA), WER (Whisper), ASR transcription

evaluation/
├── testset.json                 # Input: reference audio + texts to evaluate
├── run_eval.py                  # Main evaluation pipeline
└── results/                     # Output: CSV + generated audio

notebooks/                       # Jupyter for preprocessing, analysis, visualization
data/                            # Audio files (gitignored)
```

## Dependencies

**Core** (uv sync):
- TTS models: `websockets`, `msgpack`, `mistralai`, `soundfile`
- Audio processing: `pydub`, `pedalboard`, `librosa`, `numpy`, `scipy`
- Config: `python-dotenv`

**Evaluation** (uv sync --extra eval):
- Metrics: `jiwer` (WER), `openai-whisper` (ASR), `speechbrain` (ECAPA embeddings)
- Audio I/O: `torch`, `torchaudio`

**Notebooks** (uv sync --extra notebook):
- Jupyter, matplotlib, plotly, pandas, scikit-learn, transformers

**Qwen3** (uv sync --extra qwen3):
- `qwen-tts` (local inference) — optional; DashScope API version is what's currently implemented

## Usage Examples

### Synthesize with S2-Pro

```python
from myecho.models.s2_pro import S2ProTTS
import asyncio

tts = S2ProTTS(reference_id="your-reference-id")
result = asyncio.run(tts.synthesize("Hallo Welt"))
print(f"RTF: {result['metrics']['rtf']:.3f}")
```

### Synthesize with Voxtral

```python
from myecho.models.voxtral import VoxtralTTS
import asyncio

tts = VoxtralTTS()
tts.create_voice("reference_audio.wav")
result = asyncio.run(tts.synthesize("Hallo Welt"))
```

### Extract Speaker Similarity

```python
from myecho.evaluation.metrics import extract_embedding, similarity

ref_emb = extract_embedding("reference.wav")
clone_emb = extract_embedding("synthesized.wav")
sim = similarity(ref_emb, clone_emb)
print(f"Speaker similarity: {sim:.3f}")
```

### Get Word Error Rate

```python
from myecho.evaluation.metrics import transcribe, wer

asr_text = transcribe("synthesized.wav", language="de")
error_rate = wer("original text", asr_text)
print(f"WER: {error_rate:.3f}")
```

## Configuration

### .env Template

```bash
S2PRO_API_KEY=your-key-here                          # S2-Pro
VOXTRAL_API_KEY=your-key-here                       # Voxtral
QWEN3_API_KEY=your-key-here                     # Qwen3 (optional)
QWEN3_BASE_HTTP_API_URL=https://dashscope.aliyuncs.com/api/v1
QWEN3_BASE_WS_API_URL=wss://dashscope.aliyuncs.com/api/v1
```

## Notes

- All audio data and results are gitignored (privacy/size)
- Embeddings and ASR require CUDA for speed; CPU works but is slow
- S2-Pro requires a pre-registered voice on Fish Audio
- Voxtral creates a custom voice from your reference audio (API call)
- Qwen3 can run local or via DashScope API (currently configured for API)
