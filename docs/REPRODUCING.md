# Reproducing the evaluation

How to run the pipeline and where things live. Results are in the [README](../README.md); methodology in [METHODS.md](METHODS.md).

## Running it

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+. `data/` is
gitignored; the pipeline expects

```
data/
├── wav/                 the speaker's recordings, one WAV per text
├── DeepFilterNet/       denoised copies, same names (optional)
├── eval_texts.json      per text: id, source_audio, text
└── enrollment/          written by `make enrollment`
```

```bash
uv sync --extra eval
cp .env.example .env     # API keys and one voice ID per service
```

| service | key | voice |
|---|---|---|
| s2pro | `S2PRO_API_KEY` | `S2PRO_REFERENCE_ID`, `S2PRO_LATENCY_MODE=balanced` for TTFA |
| voxtral | `VOXTRAL_API_KEY` | `VOXTRAL_VOICE_ID` |
| qwen3 | `QWEN3_API_KEY` | `QWEN3_VOICE_ID`, `QWEN3_WORKSPACE_ID` |

A service without a voice ID is skipped. `evaluation/testset.json` points at the
reference audio and the texts file; `reference_audio` must be a recording that
was not used for enrollment.

```bash
make enrollment SOURCE=data/wav/<memo>.wav   # one reference file for all 3 services
make objective                               # synthesize + SIM/WER/latency (calls the APIs)
make stimuli                                 # excerpts, screening, blinded listening test
make check                                   # design and blinding gates, exits non-zero on failure
make sessions                                # serve the test at localhost:8765
make results                                 # naturalness, 2AFC, objective comparisons
make report                                  # tables and figures for docs/paper/
```

Only `make objective` calls the APIs; everything else runs on its output. The
stimulus set is frozen once `responses/` has data; `make stimuli` refuses to
rebuild it. Below five raters `make results` suppresses the comparison verdicts.

Single renders without the pipeline, e.g. to check a key after enrolling:

```bash
uv run voxtral-synthesize --text "Guten Morgen" --out outputs/test.wav
uv run qwen3-synthesize   --text "Guten Morgen" --out outputs/test.wav
uv run s2pro-synthesize   --text "Guten Morgen" --out outputs/test.mp3
```

qwen3 voices are enrolled with
`uv run python tools/qwen3_voice_enroll.py create --audio ref.wav --name myvoice`;
voxtral and s2pro through their APIs or web UI, using the file `make enrollment`
writes.

## Layout (paths from the repository root)

```
src/myecho/
├── config.py                     keys and endpoints from .env
├── cli/                          the three synthesize commands
├── models/                       s2_pro.py, voxtral.py, qwen3.py
└── evaluation/
    ├── metrics.py                SIM, WER, ASR
    ├── excerpt.py                choosing and locating excerpts
    └── audio_qc.py               loudness and pause-floor matching

evaluation/                       one script per Makefile stage
├── testset.json
├── prepare_enrollment.py         make enrollment
├── run_eval.py                   make objective
├── select_excerpts.py            make stimuli
├── qc_samples.py
├── prepare_listening_test.py
├── listening_test_template.html
├── check_listening_test.py       make check
├── listening_test_server.py      make sessions
├── analyze_listening_test.py     make results
├── analyze_objective.py
└── results/                      audio and rater-level answers gitignored;
                                  the summary CSVs are committed

tools/
├── check_ci_coverage.py          interval coverage and power by simulation
├── make_paper_tables.py          report tables from the CSVs, checks quoted numbers
├── make_paper_figures.py         report figures
└── qwen3_voice_enroll.py

tests/                            uv run --extra dev pytest
notebooks/                        data_analysis, data_preprocessing, results_analysis
docs/
├── METHODS.md
└── paper/                        report sections, tables, figures
PREREGISTRATION.md                hypotheses and design, committed before the first rater
```

## Notes

- s2pro streams over WebSocket and returns 44.1 kHz WAV; voxtral is HTTP,
  non-streaming, 24 kHz; qwen3 streams over `/api-ws/v1/realtime`, 24 kHz. A
  qwen3 voice enrolled for `qwen3-tts-vc-realtime-*` only works over that path.
- ASR and embeddings run on CPU; CUDA is much faster.
- `WHISPER_MODEL` is `small`. Changing it means recomputing every WER value
  including the REAL floor.
- All audio and rater-level data stay out of the repository.
