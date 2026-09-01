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

Two halves. The **objective** part (SIM, WER, RTF, TTFA) runs unattended. The
**subjective** part (N-MOS, Turing test) needs human listeners — see
[Subjective evaluation](#subjective-evaluation-n-mos--turing-test) below.

### Objective metrics

1. **Prepare testset** — edit `evaluation/testset.json`:
   ```json
   {
     "cases": [
       {
         "name": "case_01",
         "reference_audio": "data/wav/held_out_reference.wav",
         "texts": ["Text 1 to synthesize", "Text 2 to synthesize"]
       }
     ]
   }
   ```
   Voice IDs come from `.env` (`S2PRO_REFERENCE_ID`, `VOXTRAL_VOICE_ID`,
   `QWEN3_VOICE_ID`); a per-case `s2pro_reference_id` / `voxtral_voice_id` /
   `qwen3_voice_id` overrides them. A model without a voice ID is skipped.

   Two rules for trustworthy numbers:
   - `reference_audio` should be a **held-out** recording of the speaker, not the
     file the voices were cloned from — otherwise SIM is too optimistic.
   - Write numbers as **words** in the texts. WER normalization expands digits
     (`2021` → `zweitausendeinundzwanzig`), but German ordinals inflect
     (`dritte` vs. `dritten`), which still costs one substitution.

2. **Run evaluation**:
   ```bash
   uv run python evaluation/run_eval.py
   ```

3. **Results** — saved to `evaluation/results/`:
   - `results_1.csv` — per-text metrics (SIM, WER, latency, RTF)
   - Model subdirectories with generated audio files

## Subjective evaluation (N-MOS + Turing test)

Runs on the audio the objective pipeline already produced. Designed for one
operator working through many raters in person: you play the samples, the rater
tells you the score, you click it, and the results are saved locally.

### Step 1 — Screen the samples (once)

```bash
uv sync --extra eval                        # needs pyloudnorm for BS.1770 loudness
uv run python evaluation/qc_samples.py
```

Measures every candidate sample (duration, loudness, peak, silence ratio,
clipping) into `evaluation/results/qc_report.csv` and flags the broken ones.
It writes `evaluation/results/exclusions.json`, pre-filled with **technical
failures only**:

- `quiet` / `near_silent` / `clipping` → pre-filled for exclusion
- `duration_off` / `silence_heavy` → **not** excluded, only a shortlist to listen
  to. These usually mean the model just speaks slower or pauses more, which is a
  real property of the voice that the raters should judge.

**Listen to every flagged sample, then edit `exclusions.json`.** Exclude only
actual synthesis failures — never because something "sounds bad", since that is
exactly what you are measuring. The file is your written record of what was
removed and why.

**Balanced design:** if a text is excluded for one model, that text is dropped
for **all** models, so every model is scored on identical material. Texts differ
in difficulty, so otherwise each model's mean would average over a slightly
different text set. With 16 texts and 1 exclusion this gives 15 texts × 3 models
= **45 N-MOS samples and 15 Turing pairs (5 per model)**. Set
`"keep_unbalanced": true` in `exclusions.json` to keep the extra items instead.

### Step 2 — Build the test (once, then never again)

```bash
uv run python evaluation/prepare_listening_test.py
```

Produces `evaluation/results/listening_test/`:

| File | What it is |
|---|---|
| `index.html` | the rating page |
| `samples/` | anonymized audio, no model name in any filename |
| `key.json` | **hidden** mapping sample → model. Never show this to a rater |
| `stimuli.csv` | measured level/gain per stimulus, for the write-up |
| `share.zip` | `index.html` + `samples/` only, for remote raters |

Every stimulus is resampled to 24 kHz and loudness-normalized to −23 LUFS, so
neither level nor bandwidth can stand in for quality. This matters: qwen3 renders
~7 dB quieter than the other models, which would have cost it naturalness points
for the wrong reason, and voxtral clones were louder than the real recordings,
which leaked a "the louder one is fake" cue into the Turing test.

⚠️ **Do not run this script again once you have started collecting ratings.** It
reshuffles the samples and rewrites `key.json`, which silently desyncs the answers
you already have. It refuses to run if `responses/` has data (`--force` overrides).

Also: **don't open `key.json` yourself** before the sessions. You run them, and
knowing which side is real can leak through how you react.

### Step 3 — Run the sessions

```bash
uv run python evaluation/listening_test_server.py
```

Leave it running for the whole study, then for each rater:

1. Open <http://localhost:8765>
2. Enter name/abbreviation, age, and whether they know the person personally
3. Play each sample, the rater says the score, you click it
   - Part 1: naturalness 1–5 per sample (N-MOS)
   - Part 2: per pair, which of A/B is the real voice (Turing test)
4. Click **"Ergebnisse speichern"** → appended to
   `responses/all_ratings.jsonl` and the summary CSVs update immediately
5. Click **"Für nächste Person zurücksetzen"** → empty form for the next rater

The **stimulus set is the same for everyone**, but the page **randomizes the
order and the A/B sides per rater**. So you can spread the 30 raters over as many
days as you like without losing comparability — just keep the playback conditions
constant (same headphones, same volume) and avoid scheduling all the people who
know the speaker on one day and all strangers on another.

### Step 4 — Read the results

Updated automatically after every rater, or on demand:

```bash
uv run python evaluation/analyze_listening_test.py
```

- `listening_test/nmos_summary.csv` — mean/std naturalness per model
- `listening_test/turing_summary.csv` — fooling rate per model with 95% CI
  against the 0.50 chance level, split by whether the rater knows the speaker
  (familiarity is a strong confound: people who know the voice detect clones
  much more easily)

For the final write-up, prefer a mixed-effects model
(`rating ~ model + (1|rater) + (1|text)`) over the plain means — the reported CI
treats each answer as independent, which it is not.

### Remote raters

Share `share.zip` via a cloud link (~130 MB, too big for email). They unzip it,
open `index.html`, rate, and send back the downloaded `ratings_*.json`. Drop that
file into `responses/` — the analysis picks it up alongside the server-collected
ones.

## Models Tested

| Model | Type | Status |
|---|---|---|
| **S2-Pro** | Fish Audio WebSocket API | ✓ Implemented |
| **Voxtral** | Mistral API (custom voice) | ✓ Implemented |
| **Qwen3** | Alibaba DashScope realtime WS | ✓ Implemented (requires DashScope token) |

All three write **WAV** so SIM/WER/N-MOS compare like with like:

| Model | Protocol | Output | TTFA |
|---|---|---|---|
| S2-Pro | Fish Audio WebSocket (streaming) | WAV 44.1 kHz | ✓ (needs `S2PRO_LATENCY_MODE=balanced`) |
| Voxtral | Mistral HTTP, one shot | WAV 24 kHz | ✗ not streaming |
| Qwen3 | DashScope `/api-ws/v1/realtime` | WAV 24 kHz | ✓ |

## Metrics

**Quality Metrics (objective):**
- **SIM** — Speaker Similarity (cosine distance of ECAPA embeddings) — how much does it sound like the reference?
- **WER** — Word Error Rate (via Whisper ASR) — is the text intelligible?

**Quality Metrics (subjective, human raters):**
- **N-MOS** — Naturalness Mean Opinion Score, 1–5 per sample
- **Turing test** — fooling rate: how often a clone is mistaken for the real
  recording (0.50 = indistinguishable from guessing)

**Performance Metrics:**
- **Latency** — total API response time
- **RTF** — Real-Time Factor (latency / audio_duration) — can it run in realtime? (RTF < 1 = yes)
- **TTFA** — Time To First Audio (S2-Pro + Qwen3; Voxtral is not streaming)

All timings start **before** the connection is opened, so latency and TTFA are
comparable across the three APIs. `connect_s` reports the handshake share
separately. Note Qwen3 runs in Singapore — its RTF carries extra network time.

## Project Structure

```
src/myecho/
├── config.py                    # API keys, model configs
├── models/
│   ├── s2_pro.py               # Fish Audio WebSocket TTS
│   ├── voxtral.py              # Mistral voice cloning
│   └── qwen3.py                # Alibaba DashScope voice cloning
└── evaluation/
    ├── metrics.py              # SIM (ECAPA), WER (Whisper), ASR transcription
    └── audio_qc.py             # BS.1770 loudness measurement + normalization

evaluation/
├── testset.json                       # Input: reference audio + texts to evaluate
├── run_eval.py                        # Objective pipeline (SIM/WER/RTF/TTFA)
├── qc_samples.py                      # Screen samples, flag technical failures
├── prepare_listening_test.py          # Build the blinded listening test
├── listening_test_template.html       # Rating page (order randomized per rater)
├── listening_test_server.py           # Serve it + auto-save each rater's result
├── analyze_listening_test.py          # N-MOS + Turing test results
└── results/                           # Output: CSV, generated audio, listening test

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
- Listening test: `pyloudnorm` (BS.1770 loudness normalization)

The listening test itself needs nothing else — the rating page is plain
HTML/JS and the server uses only the Python standard library.

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

# Reuse an existing voice (what the eval does - a fresh clone per run would
# make SIM non-reproducible):
tts = VoxtralTTS(voice_id="your-voice-id")
result = asyncio.run(tts.synthesize("Hallo Welt"))

# Or create one once from a reference recording:
# tts = VoxtralTTS(); tts.create_voice("reference_audio.wav")
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
QWEN3_API_HOST=dashscope.aliyuncs.com           # or your dedicated per-workspace host
QWEN3_REALTIME_MODEL=qwen3-tts-vc-realtime-2026-01-15

S2PRO_REFERENCE_ID=...                          # pre-registered voice
S2PRO_LATENCY_MODE=balanced                     # balanced = streaming, normal = best quality
VOXTRAL_VOICE_ID=...
QWEN3_VOICE_ID=...
```

## Notes

- All audio data and results are gitignored (privacy/size)
- Embeddings and ASR require CUDA for speed; CPU works but is slow
- S2-Pro requires a pre-registered voice on Fish Audio
- Voxtral creates a custom voice from your reference audio (API call)
- Qwen3 can run local or via DashScope API (currently configured for API)
- A Qwen3 voice enrolled against `qwen3-tts-vc-realtime-*` works **only** over
  `/api-ws/v1/realtime`, not over the older `tts_v2` path. Enroll local files with
  `python qwen_voice_enroll.py create --audio ref.wav --name myvoice`
- N-MOS and the Turing test are implemented as a local listening test — see
  [Subjective evaluation](#subjective-evaluation-n-mos--turing-test)
- Listening-test stimuli are loudness-normalized (−23 LUFS) and resampled to
  24 kHz, because level and bandwidth differences between models otherwise show
  up as fake quality differences. Native sample rate and measured loudness are
  reported separately in `qc_report.csv` / `stimuli.csv`
