# myEcho — Voice Cloning Evaluation

Part of **myEcho**, a project on voice cloning for an ALS patient. This
repository compares three commercial voice-cloning APIs on the voice of one
German speaker, using recordings the person made themselves in 2021.

Three services are evaluated: **s2pro** (Fish Audio), **voxtral** (Mistral) and
**qwen3** (Alibaba DashScope). Each one clones the voice, renders 16 texts, and
is scored on four things:

| | metric | how |
|---|---|---|
| speaker similarity | SIM | cosine similarity of ECAPA-TDNN embeddings |
| intelligibility | WER | Whisper ASR vs. the target text |
| naturalness | ACR 1–5 | listening test with ~30 human raters |
| detectability | 2AFC | real vs. clone, "which is the real voice" |

plus service latency (API response time, time to first audio).

The speaker's own recordings are carried through the whole evaluation as a
**fourth condition**. That is the central design choice: SIM = 0.70 and
WER = 0.11 mean nothing on their own, so every number is reported against what
the real voice scores on the same measure.

**Headline result so far** (objective part; the listening test is not yet run):

| condition | SIM | WER |
|---|---|---|
| **real voice (reference)** | **0.79** (0.67–0.86) | **0.18** |
| qwen3 | 0.743 | 0.079 |
| s2pro | 0.707 | 0.100 |
| voxtral | 0.600 | 0.112 |

qwen3 and s2pro land *inside* the range the similarity metric returns for two
different recordings of the same person — the metric cannot separate their clones
from the real speaker. And all three clones are *easier* to transcribe than the
human original, because reading a transcript produces cleaner speech than
speaking spontaneously into a phone.

📄 **[docs/METHODS.md](docs/METHODS.md)** explains why the evaluation is built
this way, with the measured numbers behind each decision.
📄 **[PREREGISTRATION.md](PREREGISTRATION.md)** records the design that was fixed
before any rater was asked.

---

## How to run it

### 1. Prerequisites

- **[uv](https://docs.astral.sh/uv/)** and Python 3.10+.
- **The recordings.** `data/` is gitignored, so a fresh clone has no audio. The
  pipeline expects:

  ```
  data/
  ├── wav/                 the speaker's own recordings, one WAV per text
  ├── DeepFilterNet/       denoised copies, same filenames as in wav/
  ├── eval_texts.json      one entry per text: id, source_audio, text
  └── enrollment/          created by `make enrollment`
  ```

  `DeepFilterNet/` is optional — without it the raw files from `wav/` are used.

- **API keys and one enrolled voice per service.** Only needed for the
  synthesis step; everything else runs on the audio that step produced.

### 2. Install

```bash
uv sync                  # core dependencies
uv sync --extra eval     # metrics, ASR, embeddings, statistics
cp .env.example .env     # then fill it in
```

`.env` needs, per service you want to evaluate:

| service | key | voice |
|---|---|---|
| s2pro | `S2PRO_API_KEY` | `S2PRO_REFERENCE_ID`, plus `S2PRO_LATENCY_MODE=balanced` for TTFA |
| voxtral | `VOXTRAL_API_KEY` | `VOXTRAL_VOICE_ID` |
| qwen3 | `QWEN3_API_KEY` | `QWEN3_VOICE_ID`, `QWEN3_WORKSPACE_ID` |

A service with no voice ID is skipped, so a partial run works.

### 3. Point it at your data

`evaluation/testset.json`:

```json
{"cases": [{
  "name": "case_2021_07",
  "reference_audio": "data/DeepFilterNet/held_out_reference.wav",
  "texts_file": "data/eval_texts.json",
  "impostor_audio": null
}]}
```

`texts_file` holds the text to synthesize plus the speaker's own recording of it:

```json
{"texts": [{"id": "AUDIO-2021-08-16", "source_audio": "AUDIO-2021-08-16.wav",
            "text": "Hi Lars, ich bin's nochmal ..."}]}
```

Three things to get right, explained in
[METHODS section 2](docs/METHODS.md#2-objective-metrics):

- `reference_audio` must be **held out** — not the file the voices were cloned
  from, or SIM comes out too optimistic.
- `impostor_audio` should point at clips of a **different** speaker. Without it
  the similarity scale has no lower end.
- Write numbers as words in the texts; German ordinal inflection otherwise costs
  a WER substitution.

### 4. Run the pipeline

`make help` lists every target. In order:

```bash
make enrollment SOURCE=data/wav/<reference>.wav   # optional, see below
make objective                                    # synthesize + SIM/WER/latency
make stimuli                                      # build the listening test
make check                                        # verify it — before any rater
make sessions                                     # run the rater sessions
make results                                      # naturalness, 2AFC, comparisons
```

| target | what it writes | time | how you know it worked |
|---|---|---|---|
| `enrollment` | `data/enrollment/` + a hash manifest | seconds | prints the reference file and its sha256 |
| `objective` | `results_<stamp>.csv`, `results_latest.csv`, the rendered audio | minutes, **costs API money** | one SIM/WER line per text per service |
| `stimuli` | `results/listening_test/` — samples, `index.html`, `key.json` | ~15 s (first run several minutes: one ASR pass per file, then cached) | "N texts x M conditions = … stimuli", plus the per-rater audio length |
| `check` | `cue_audit.csv` | ~2 s | "All checks passed. Safe to start the sessions." — exits non-zero otherwise |
| `sessions` | `responses/all_ratings.jsonl` | as long as your raters take | prints a line per submitted rating |
| `results` | `nmos_summary.csv`, `turing_summary.csv`, `objective_*.csv`, … | ~30 s | tables with means, fooling rates and intervals |

`make enrollment` is only needed once, and only if you want the three services
enrolled from one identical file — which you do, because otherwise a similarity
difference between them cannot be attributed to the service. The three uploads
themselves are manual; the script prints the per-service checklist.

Two more targets, neither needing raters or API access:

```bash
make long-form     # stability of the FULL renders: voice drift, level drift, rate
make coverage      # does the reported interval cover? what can it detect?
make test          # unit tests, ~1 s
```

### 5. Running the sessions

```bash
make sessions      # serves http://localhost:8765
```

Leave it running for the whole study. For each rater: open the URL, hand them
the keyboard — **they enter their own scores**, not you — and when they submit,
the button turns into a reset for the next person. Append `?plan=2` to force a
specific rater plan if you need to even out the last few sessions.

For remote raters, send them `results/listening_test/share.zip` (~21 MB, no
answer key). They unzip, open `index.html`, rate, and send back the downloaded
`ratings_*.json`; drop it into `responses/` and `make results` picks it up.

### 6. Important during the study

The stimulus set is **frozen from the first rating onwards**. `make stimuli`
reshuffles the samples and rewrites the answer key, which silently desyncs
ratings already collected — the script refuses to run once `responses/` has data.
Run `make check` before the first session; it fails loudly on everything that
cannot be repaired afterwards. Do not open `key.json` yourself.

### If you have no API keys

Everything except `make objective` runs on audio that already exists under
`evaluation/results/`. So with a populated `results/` folder you can build the
listening test, verify it, run raters and analyse the outcome without touching a
service.

### If something fails

| message | cause |
|---|---|
| `<dir> not found - run evaluation/run_eval.py first` | no rendered audio yet; run `make objective` |
| `excerpts.json not found` | run `make stimuli`, which runs the excerpt step first |
| `No usable ratings found in .../responses` | no ratings collected yet — expected before the sessions |
| `responses already has collected ratings` | protecting your data; do not use `--force` mid-study |
| `skip <service>: no voice_id` | that service has no voice ID in `.env` and is skipped |
| a `check` line says **STRONG CUE** | a low-level property still separates real from clone; see [METHODS section 5](docs/METHODS.md#5-confounds) |

---

## Repository structure

```
src/myecho/                       importable package
├── config.py                     API keys and model configuration
├── models/
│   ├── s2_pro.py                 Fish Audio WebSocket TTS
│   ├── voxtral.py                Mistral voice cloning
│   └── qwen3.py                  Alibaba DashScope voice cloning
└── evaluation/
    ├── metrics.py                SIM (ECAPA), WER (Whisper), ASR
    ├── excerpt.py                choosing, locating and cutting excerpts
    └── audio_qc.py               loudness and pause-floor matching

evaluation/                       the pipeline, in running order
├── testset.json                  input: reference audio + texts
├── prepare_enrollment.py         one canonical voice sample for all 3 services
├── run_eval.py                   synthesize + objective metrics
├── select_excerpts.py            choose the ~10 s excerpts
├── qc_samples.py                 screen for technical failures
├── prepare_listening_test.py     build the blinded listening test
├── check_listening_test.py       verify blinding, design and level matching
├── listening_test_server.py      serve the test, save each rater
├── analyze_listening_test.py     naturalness + 2AFC results
├── analyze_objective.py          SIM/WER comparisons (--long-form for renders)
└── results/                      all output (gitignored)

tools/check_ci_coverage.py        does the reported interval cover? what can it detect?
tests/                            unit tests (uv run pytest)
notebooks/                        data preprocessing and exploratory analysis
docs/METHODS.md                   methodology and reasoning
data/                             audio (gitignored — personal recordings)
```

### The three services

| service | protocol | output | streaming |
|---|---|---|---|
| s2pro | Fish Audio WebSocket | WAV 44.1 kHz | yes (`S2PRO_LATENCY_MODE=balanced`) |
| voxtral | Mistral HTTP | WAV 24 kHz | no |
| qwen3 | DashScope `/api-ws/v1/realtime` | WAV 24 kHz | yes |

⚠️ The three were originally enrolled from **different** inputs (qwen3 with
server-side denoising, voxtral with a gender hint, s2pro via a web upload), so
the current similarity ranking is confounded with the enrollment. The code
defaults are now neutral; `make enrollment` produces one canonical sample with a
hash manifest so all three can be re-enrolled from the same file. See
[METHODS §5](docs/METHODS.md#5-confounds).

---

## Notes

- All audio and results are gitignored, for privacy and size.
- ASR and embeddings run on CPU but are much faster with CUDA.
- `WHISPER_MODEL` in `metrics.py` is `small`. A German-finetuned model would be
  better — but if you change it, every WER value **including the reference
  floor** has to be recomputed together.
- A qwen3 voice enrolled against `qwen3-tts-vc-realtime-*` works only over
  `/api-ws/v1/realtime`. Enroll local files with
  `python qwen_voice_enroll.py create --audio ref.wav --name myvoice`.
