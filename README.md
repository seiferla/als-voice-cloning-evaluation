# Voice Cloning Evaluation Framework

Evaluation framework for voice cloning: speaker similarity, intelligibility,
naturalness, detectability and service latency, for three commercial TTS APIs
cloning one German speaker.

Part of **myEcho** — voice cloning for an ALS patient. The design commitments
made before the listening test are in [PREREGISTRATION.md](PREREGISTRATION.md);
read that first if you want to know what the numbers are allowed to mean.

## Quick Start

```bash
uv sync                       # core dependencies
uv sync --extra eval          # metrics, ASR, embeddings, loudness, statistics
cp .env.example .env          # then fill in the API keys and voice IDs
```

The pipeline runs in order. Each step reads what the previous one wrote.

```bash
uv run python evaluation/run_eval.py --repeats 3   # 1. synthesize + objective metrics
uv run python evaluation/select_excerpts.py        # 2. pick + locate ~10 s excerpts
uv run python evaluation/qc_samples.py             # 3. screen for technical failures
uv run python evaluation/prepare_listening_test.py # 4. build the blinded test (ONCE)
uv run python evaluation/verify_build.py           # 5. gate: blinding + design invariants
uv run python evaluation/audit_stimulus_cues.py    # 6. check for giveaway cues
uv run python evaluation/listening_test_server.py  # 7. run the rater sessions
uv run python evaluation/analyze_listening_test.py # 8. results
```

Two more, neither of which needs raters or API access:

```bash
uv run python evaluation/analyze_long_form.py      # stability of the FULL renders
uv run python evaluation/check_ci_coverage.py      # does the reported CI cover? what can it detect?
```

Steps 2–6 need no API access — they work on the audio step 1 produced.
**Run step 5 before the first rater.** It fails loudly on anything that cannot
be repaired once data collection starts.

---

## 1 · Objective metrics

### Configure the testset

`evaluation/testset.json`:

```json
{
  "cases": [
    {
      "name": "case_2021_07",
      "reference_audio": "data/DeepFilterNet/held_out_reference.wav",
      "texts_file": "data/eval_texts.json",
      "impostor_audio": null
    }
  ]
}
```

`texts_file` points to a JSON file with one entry per text:

```json
{"texts": [{"id": "AUDIO-2021-08-16", "source_audio": "AUDIO-2021-08-16.wav",
            "text": "Hi Lars, ich bin's nochmal ..."}]}
```

`source_audio` is required — it is the speaker's own recording of that text, used
as the REAL condition and as the real side of the 2AFC pairs. Personal texts live
under `data/` (gitignored) rather than in this tracked file.

Voice IDs come from `.env` (`S2PRO_REFERENCE_ID`, `VOXTRAL_VOICE_ID`,
`QWEN3_VOICE_ID`); a per-case `s2pro_reference_id` / `voxtral_voice_id` /
`qwen3_voice_id` overrides them. A model without a voice ID is skipped.

Three rules for numbers you can defend:

- `reference_audio` must be a **held-out** recording, not the file the voices were
  cloned from — otherwise SIM is too optimistic. The runner skips the text whose
  own recording is the reference (it would score 1.0 against itself) and flags
  the cloning source.
- Write numbers as **words** in the texts. WER normalization expands digits
  (`2021` → `zweitausendeinundzwanzig`), but German ordinals inflect (`dritte`
  vs. `dritten`), which still costs one substitution.
- Set `impostor_audio` to clips of a **different** speaker. See "Read SIM against
  its anchors" below — without it the bottom of the SIM scale is unknown.

### Run

```bash
uv run python evaluation/run_eval.py --repeats 3
```

`--repeats` is the number of renders per text. These APIs sample, so with 1
render the model's own run-to-run variance is invisible and a single unlucky
render looks like a property of the model. The default is 1 so an exploratory run
does not triple the API bill; use 3 for the run you report.

Output in `evaluation/results/`:

| File | What |
|---|---|
| `results_<timestamp>.csv` | one row per render, plus the REAL and IMPOSTOR rows |
| `results_latest.csv` | a copy of the newest run, for scripts |
| `eval_run_meta.json` | git commit, ASR/embedding checkpoints, input hashes |
| `<case>/<model>/` | the generated audio |

### Read SIM against its anchors, never on its own

SIM is a cosine **similarity** between ECAPA-TDNN speaker embeddings. The scale
has no absolute meaning — it depends on the speaker, the recording and the
checkpoint. So `run_eval.py` scores two extra conditions into the same table:

- **REAL** — the speaker's own other recordings. Its SIM is the ceiling: how
  similar the same person is to *themselves* across two recordings.
- **IMPOSTOR** — a different speaker (needs `impostor_audio`). Its SIM is the
  floor: what the metric returns for someone who is not the target at all.

Measured on this material:

| condition | SIM (median) | WER (median) |
|---|---|---|
| **REAL (anchor)** | **0.79** (range 0.67–0.86) | **0.18** |
| qwen3 | 0.746 | 0.091 |
| s2pro | 0.702 | 0.110 |
| voxtral | 0.599 | 0.209 |
| IMPOSTOR (anchor) | not yet measured | — |

Two conclusions that are invisible without the anchor row:

1. qwen3 and s2pro fall **inside** the range this metric returns for two
   recordings of the same person. Their SIM values are at the ceiling of what it
   can resolve, and the gap between them is not interpretable as a similarity
   difference. voxtral sits clearly below.
2. The clones have a **lower** word error rate than the human original (0.09 vs
   0.18). Reading a transcript produces cleaner speech than speaking
   spontaneously into a phone. WER here measures distance from read speech, not
   intelligibility.

### `service_rtf` is not RTF

`service_rtf` = wall-clock API latency ÷ length of the returned audio, measured
over the internet against three different services — one of them non-streaming,
one hosted in Singapore. Real RTF is compute time ÷ audio duration on known
hardware, which none of these APIs let you observe. `connect_s` reports the
handshake separately, as an RTT proxy. All timings start **before** the
connection is opened, so they are comparable across the three APIs.

---

## 2 · Subjective evaluation (naturalness + 2AFC "which is real")

Runs on the audio the objective pipeline produced. Designed for one operator
working through many raters in person.

### Step 1 — Pick the excerpts

```bash
uv run python evaluation/select_excerpts.py
```

Stimuli are ~10 s excerpts, not the 12–92 s full renders. The reasoning and the
measured cost are in the [stimulus length appendix](#appendix--stimulus-length)
— read it before changing `TARGET_S`.

Writes `excerpts.json`, with the ASR pass cached in `alignments/` so re-runs are
instant. Two kinds of finding are reported and excluded rather than papered over:

- under 60 % of the excerpt's words found → the system did not say the passage
- found, but stretched over more than 2× the per-text median → a runaway render

### Step 2 — Screen the samples

```bash
uv run python evaluation/qc_samples.py
```

Measures every candidate (duration, BS.1770 loudness, peak, silence ratio,
clipping) into `qc_report.csv` and writes `exclusions.json`, pre-filled with
**technical failures only**:

- `quiet` / `near_silent` / `clipping` → pre-filled for exclusion
- `duration_off` / `silence_heavy` → **not** excluded, only a shortlist to listen
  to. These usually mean the system speaks slower or pauses more, which is a real
  property of the voice that raters should judge.

**Listen to every flagged sample, then edit `exclusions.json`.** Exclude only
actual synthesis failures — never because something "sounds bad", since that is
exactly what you are measuring. The file is your written record.

**Balanced design:** if a text is excluded for one system it is dropped for
**all** systems, so every condition is scored on identical material. Set
`"keep_unbalanced": true` to keep the extra items instead.

### Step 3 — Build the test (once, then never again)

```bash
uv run python evaluation/prepare_listening_test.py
```

Produces `evaluation/results/listening_test/`:

| File | What it is |
|---|---|
| `index.html` | the rating page |
| `samples/` | the stimuli, named `s_001.wav` … and nothing else |
| `key.json` | **hidden** mapping stimulus → condition. Never show a rater |
| `run_meta.json` | seed, git commit, input hashes |
| `stimuli.csv` | measured level, pause floor and excerpt span per stimulus |
| `share.zip` | `index.html` + `samples/` only (~21 MB), for remote raters |

Three design points worth knowing:

**The real voice is a condition, not just the answer key.** It is rated for
naturalness alongside the three systems. Without it, "qwen3 scores 4.1" has no
ceiling to compare against.

**Every rater gets a plan.** The build enumerates 6 plans (2 role groups × 3
model rotations) and each rater draws one at random. A plan sends half the texts
to part 1 and half to part 2, so **no rater meets the same text in both parts** —
with content-matched excerpts and the real voice in part 1, they would otherwise
hear the answer to part 2. Across raters, every text is used in both roles and
every system is paired with every text. On this material: 13 texts × 4 conditions
= 52 naturalness items, 39 pairs (13 per system), 24–28 items and 6–7 pairs per
rater, **~6.5 min of audio per rater**.

**Nothing in the shipped files says what anything is.** Names are assigned after
the shuffle. `share.zip` contains no key.

All stimuli go through **identical** processing: resample to 24 kHz → cut →
loudness-normalize to −23 LUFS → match the pause floor to speech −26 dB. The last
step matters more than it sounds. The real recordings are phone voice memos put
through DeepFilterNet, the clones are generated clean, and before matching a
single threshold on the pause fraction or the pause noise floor separated real
from clone with **96 % accuracy** — the 2AFC test could have been won without
judging a voice. Pause *structure* is left untouched, because how often a voice
stops is a property of the system that raters should hear.

⚠️ **Do not run this script again once you have started collecting ratings.** It
reshuffles and rewrites `key.json`, which silently desyncs the answers you
already have. It refuses to run if `responses/` has data (`--force` overrides).

Also: **don't open `key.json` yourself** before the sessions. You run them, and
knowing which side is real can leak through how you react.

### Step 4 — Audit the stimuli for giveaway cues

```bash
uv run python evaluation/audit_stimulus_cues.py
```

Measures cheap low-level features on the stimuli that were actually built, and
reports how well the best single threshold separates real from clone. A cue at
the guessing baseline is gone; a cue clearly above it must be mitigated or
reported. On the current set:

| feature | accuracy | baseline 0.750 |
|---|---|---|
| pause noise floor | 0.745 | gone (was 0.96) |
| pause fraction | 0.788 | weak |
| duration | 0.788 | weak |
| **energy 4–8 kHz** | **0.885** | **residual cue** |

The brightness cue is deliberately not equalized: all three systems render
duller than the phone-recorded original, which is partly the recording chain and
partly a real failure to reproduce the source timbre. EQ-ing it would erase what
raters are there to hear. Report the 2AFC numbers together with this figure.

### Step 5 — Run the sessions

```bash
uv run python evaluation/listening_test_server.py
```

Leave it running for the whole study, then for each rater:

1. Open <http://localhost:8765> (add `?plan=2` to force a plan)
2. They enter age and whether they know the speaker. **The rater clicks, not
   you** — you are the author of the system, and entering their scores for them
   is a demand effect. Their id is generated, no names.
3. Three unscored training items first, then part 1 (naturalness, five verbal
   ACR labels), then part 2 (which of A/B is real) which unlocks when part 1 is
   done. Two items in part 1 are silently repeated, for a rating-noise estimate.
4. **"Ergebnisse speichern"** → appended to `responses/all_ratings.jsonl`, and
   the summaries update in the background
5. **"Für nächste Person zurücksetzen"** → fresh page for the next rater

The stimulus set is the same for everyone; the page randomizes the plan, the
order and the A/B sides per rater. So you can spread raters over as many days as
you like — just keep the playback conditions constant (same headphones, same
volume) and avoid scheduling all the people who know the speaker on one day.

### Step 6 — Read the results

```bash
uv run python evaluation/analyze_listening_test.py
```

| File | What |
|---|---|
| `nmos_summary.csv` | mean naturalness per condition, including REAL, with CI |
| `nmos_pairwise.csv` | condition differences in MOS points, with CI |
| `turing_summary.csv` | fooling rate per system, with CI, split by familiarity |
| `turing_familiarity.csv` | does knowing the speaker change detection — the contrast, with CI |
| `reliability.csv` | intra-rater agreement from the hidden repeats |
| `ratings_long.csv` | one row per answer, for your own analysis |

Intervals come from a **two-way cluster bootstrap** over raters and texts. The
answers are not independent — each rater answers many items and each item is
answered by many raters — so pooling them as independent observations gives
intervals that are too narrow. Pairwise model comparisons are paired within
rater, which removes each rater's own scale offset, and their verdict uses a
Bonferroni-adjusted level: with 4 conditions there are 6 comparisons, so at a
nominal 5 % each the chance of at least one false positive is about 26 %.

`check_ci_coverage.py` measures whether that interval actually covers, by
simulating this exact design. Result: **0.99 against a nominal 0.95** — the
interval is conservative, too wide rather than too narrow. Resampling raters only
gave 0.93. The conservative direction is kept on purpose and reported rather than
tuned away: a wide interval can only weaken a claim, a narrow one manufactures
findings.

If `statsmodels` is installed, a mixed model (`rating ~ condition`, random rater
and text) and a rater-clustered logit for the 2AFC answers are fitted as a
confirmatory check. The bootstrap stays the headline number at this sample size.

The **reliability** number is the yardstick: a difference between conditions
smaller than one rater's own test-retest noise does not mean anything.

### Remote raters

Share `share.zip` (~21 MB). They unzip it, open `index.html`, rate, and send back
the downloaded `ratings_*.json`. Drop that into `responses/` — the analysis picks
it up and deduplicates on (rater id, timestamp).

---

## Appendix — stimulus length

### How the length is determined

The length is **not set directly**. A target length is converted into a word
count, a passage of that many words is chosen on the text, and the resulting
duration is whatever that passage takes in each voice.

**1. Target length → target word count.** Using the speaking rate measured on
this material:

```
words_per_second = 2.94          # median across the three systems
target = round(TARGET_S × 2.94)  # 10.0 s → 29 words
lo     = round(MIN_S    × 2.94)  #  8.0 s → 24 words
hi     = round(MAX_S    × 2.94)  # 13.0 s → 38 words
```

**2. Choose the passage on the text, not on the audio.** Skip the first
`SKIP_HEAD_FRAC = 15 %` of the words — the opening of a voice memo is a greeting,
and the first seconds of a render carry onset artifacts that differ per system.
Then take the run of **consecutive whole sentences** whose word count falls in
`[lo, hi]` and lies closest to `target`. If a single sentence is longer than `hi`
(these transcripts are spontaneous speech; one text has a 49-word "sentence"),
split it at commas and merge chunks instead. Only if neither works, take a plain
word window and flag the ragged end.

This step is deterministic and touches no audio, so **the choice cannot favour
any system**. Result on the 16 texts: 14 end at a sentence boundary, 2 at a
comma, none mid-sentence.

**3. Locate the same passage in every rendition.** Whisper word-level timestamps
→ fuzzy match of the normalized token sequence (`difflib`, longest common
subsequence). The first and last matched token give `start` and `end`. Below a
`MIN_MATCH_RATIO = 0.6` token match the system did not say the passage; that is
reported and excluded, never replaced by a time window.

**4. Cut.** `start − 0.2 s` to `end + 0.3 s` (padding absorbs aligner error and
keeps the release of the last word), 20 ms fades at both edges (a hard cut clicks,
and a click is heard and rated as an artifact).

**Final duration is therefore free.** Measured on the built set: **5.8–14.8 s,
median 9.6 s**, 55 stimuli, 9.0 min of audio in total. The content is identical
across systems; the duration differs because the systems speak at different
rates, which is a real property of the voice that raters should hear.

### Why the length was reduced — with the numbers

Not "P.800 says so". Five separate reasons, each measurable on this material:

**1. The unit of judgement must match the unit of analysis.** ACR asks for one
number per stimulus. A 36 s render contains several distinguishable quality
events — a clean opening, a glitch at 0:20, odd prosody at 0:30 — and the rater
compresses them into a single score with an unknown weighting. The score is then
not "the naturalness of this system", it is an unrecorded mixture. At ~10 s the
stimulus is short enough to be held as one auditory impression. ITU-T P.800
Annex B uses stimuli of roughly this length for exactly this reason.

**2. Session length and fatigue.** The previous design was 45 N-MOS items plus 15
pairs of full renders = **~46 minutes of pure audio**, i.e. 60–75 minutes per
session with replays and clicking. Ratings from the second half of such a session
are dominated by fatigue, not by the stimuli. The current design is **6.1–6.5 min
of audio per rater** and roughly a 20-minute session.

**3. Statistical power per unit of rater time.** This is the strongest argument
and it is arithmetic, not opinion. For a fixed session budget `T`, the number of
items is `n ≈ T / duration`, and the standard error of a condition mean falls
with `1/√n`. Going from a 36 s median to a 9.6 s median is **3.75× the items for
the same rater time**, i.e. intervals about **1.9× narrower** — for free, without
recruiting a single extra person. Measured on the built design by simulation
(`check_ci_coverage.py`): the study can detect a **0.30 MOS** difference with 80 %
power at 30 raters.

**4. More items means more clusters for the interval to resample.** The reported
CI resamples texts as well as raters, because the study generalizes over
sentences too. That resampling is only trustworthy with enough distinct texts.
Short stimuli are what makes it affordable to keep all 13 texts × 4 conditions in
the design instead of a handful.

**5. Content matching only becomes possible at this length.** Every system is
judged on the *same passage*. With full renders, a rater comparing systems is
also comparing which parts of a 90-second voicemail they happened to pay
attention to.

### Why not simply "the first 10 seconds"

The systems speak the same text at different rates — the per-text spread is
1.16× at the median and up to **1.53×**. A fixed time window therefore holds a
different *amount of content* per system and lands mid-word more often for the
slower one. Both push the slower system's rating down for a reason that has
nothing to do with its voice. On this material that would have penalized voxtral,
which is already the weakest — a bias pointing in the direction of the
conclusion, which is the worst kind.

### What shortening costs, and how that cost is covered

The excerpts sit between **32 % and 71 %** of each render, so about **73 % of
what each system produced is never heard by any rater**. Anything that only
appears over a long utterance is invisible in the subjective results by
construction. That is a real objection and it is answered with measurement, not
with an assurance:

```bash
uv run python evaluation/analyze_long_form.py    # -> results/long_form.csv
```

It scores every **full** render objectively: ECAPA similarity of the first third
vs the last third (does the voice stay the target voice?), level drift, speaking
rate, duration ratio and the longest gap. On this material:

| source | duration ratio | words/s | level drift | sim 1st → 3rd | drift |
|---|---|---|---|---|---|
| REAL | 1.000 | 2.53 | +0.8 dB | 0.758 → 0.773 | +0.030 |
| qwen3 | 0.901 | 2.92 | −1.2 dB | 0.715 → 0.717 | +0.004 |
| s2pro | 0.845 | 3.00 | −1.7 dB | 0.670 → 0.672 | +0.011 |
| voxtral | 0.966 | 2.71 | −2.8 dB | 0.586 → 0.598 | +0.008 |

- **No system drifts away from the target voice over a long utterance**
  (correlation between render length and voice drift: r = −0.05, n = 48). The
  largest single drift in the whole set belongs to a *real* recording. So
  excerpting does not hide a speaker-identity problem — that is now a measured
  statement instead of an assumption.
- **voxtral fades out**: up to **−11.4 dB** between the first and last third of
  one render, and −5 to −7 dB in several others. This is a genuine long-form
  defect and it is exactly the kind of thing the loudness-normalized 10 s
  excerpts hide, which is why it is reported here.
- **All three systems speak faster than the person** (2.7–3.0 vs 2.53 words/s).

Report both halves together: the listening test says how good ~10 s sounds, and
`long_form.csv` says whether a 90-second message would survive.

---

## Models tested

| Model | Type | Protocol | Output | TTFA |
|---|---|---|---|---|
| **S2-Pro** | Fish Audio | WebSocket (streaming) | WAV 44.1 kHz | ✓ (needs `S2PRO_LATENCY_MODE=balanced`) |
| **Voxtral** | Mistral | HTTP, one shot | WAV 24 kHz | ✗ not streaming |
| **Qwen3** | Alibaba DashScope | `/api-ws/v1/realtime` | WAV 24 kHz | ✓ |

⚠️ **The three were not enrolled from identical input** — qwen3 with server-side
preprocessing, voxtral with raw bytes plus a hardcoded gender hint, s2pro through
a web UI. That is an alternative explanation for the SIM ranking and the single
most valuable remaining fix: re-enroll all three from one identical, identically
preprocessed file and record the file, its sha256 and the parameters.

## Metrics

**Objective**

- **SIM** — speaker similarity, cosine similarity of ECAPA-TDNN embeddings. Read
  against the REAL and IMPOSTOR anchors.
- **WER** — word error rate via Whisper ASR. Read against the REAL row, which is
  the ASR's own error rate on the human original.
- **latency_s / ttfa_s / service_rtf / connect_s** — service timings, see above.

**Subjective**

- **Naturalness (ACR 1–5)** per stimulus, with the real voice as a condition.
- **2AFC detectability** — fooling rate: how often a clone is mistaken for the
  real recording. 0.50 = indistinguishable from guessing.

## Project structure

```
src/myecho/
├── config.py                     # API keys, model configs
├── models/{s2_pro,voxtral,qwen3}.py
└── evaluation/
    ├── metrics.py                # SIM (ECAPA), WER, ASR + word timestamps
    ├── excerpt.py                # excerpt selection, location, cutting
    └── audio_qc.py               # BS.1770 loudness + pause-floor matching

evaluation/
├── testset.json                  # input: reference audio + texts
├── run_eval.py                   # objective metrics + REAL/IMPOSTOR anchors
├── select_excerpts.py            # choose + locate the ~10 s excerpts
├── qc_samples.py                 # screen for technical failures
├── prepare_listening_test.py     # build the blinded test
├── listening_test_template.html  # rating page
├── listening_test_server.py      # serve it, save each rater
├── verify_build.py               # gate: blinding, role split, coverage, levels
├── audit_stimulus_cues.py        # check the built stimuli for giveaway cues
├── analyze_listening_test.py     # naturalness + 2AFC results
├── analyze_long_form.py          # stability of the full renders
├── check_ci_coverage.py          # CI coverage + power, by simulation
└── results/                      # output (gitignored)

tests/                            # pytest: uv run pytest
scripts/, qwen_voice_enroll.py    # CLI runners, Qwen3 voice enrollment
notebooks/                        # preprocessing, EDA, visualization
data/                             # audio (gitignored)
```

## Dependencies

**Core** (`uv sync`): `websockets`, `msgpack`, `mistralai`, `dashscope`,
`soundfile`, `numpy`, `scipy`, `num2words`, `python-dotenv`.

**Evaluation** (`uv sync --extra eval`): `jiwer`, `openai-whisper`,
`speechbrain`, `torch`, `torchaudio`, `pyloudnorm`, and `statsmodels` + `pandas`
for the confirmatory models. The rating page and server need nothing beyond the
standard library.

**Notebooks** (`uv sync --extra notebook`), **dev** (`uv sync --extra dev`,
pytest).

## Notes

- All audio and results are gitignored (privacy and size).
- Embeddings and ASR are much faster on CUDA; CPU works.
- `WHISPER_MODEL` in `metrics.py` is `small`. A German-finetuned model would be
  better — if you swap it, every WER number **including the REAL floor** must be
  recomputed together, or the comparison breaks.
- S2-Pro needs a pre-registered voice on Fish Audio. Voxtral creates a custom
  voice from reference audio via API. A Qwen3 voice enrolled against
  `qwen3-tts-vc-realtime-*` works **only** over `/api-ws/v1/realtime`, not the
  older `tts_v2` path — enroll local files with
  `python qwen_voice_enroll.py create --audio ref.wav --name myvoice`.
