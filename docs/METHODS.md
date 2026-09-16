# Evaluation methodology

Why the evaluation is built the way it is, with the measured numbers behind each
decision. The README says how to run it; this says why. Structured to map onto
the *Evaluation/Experimente* chapter of the report.

Contents: [1 Setup](#1-setup) · [2 Objective metrics](#2-objective-metrics) ·
[3 Stimulus length](#3-stimulus-length) ·
[4 Audio processing](#4-audio-processing) · [5 Confounds](#5-confounds) ·
[6 Listening-test design](#6-listening-test-design) ·
[7 Statistics](#7-statistics) · [8 Limitations](#8-limitations)

---

## 1 Setup

One German speaker (an ALS patient), 16 texts that are transcripts of their own
2021 voice memos, three commercial cloning services: **s2pro** (Fish Audio),
**voxtral** (Mistral), **qwen3** (Alibaba DashScope). Each service renders each
text; the speaker's own recording of the same text is kept as a fourth
condition, **REAL**.

Four conditions rather than three is the central design choice. Every metric
below is a number without a scale until there is something to compare it to, and
the only honest reference is the person themselves.

---

## 2 Objective metrics

### SIM — speaker similarity

Cosine similarity between ECAPA-TDNN embeddings (`speechbrain/spkrec-ecapa-voxceleb`)
of the reference recording and the rendered audio.

An ECAPA cosine has no absolute meaning — it depends on the speaker, the
recording and the checkpoint. So it is bracketed:

- **upper bound**: the speaker against *themselves*, i.e. the reference recording
  vs. their other real recordings
- **lower bound**: the reference vs. a *different* speaker (`impostor_audio`)

| condition | SIM (median) | 95 % CI |
|---|---|---|
| **REAL (upper bound)** | **0.79** | range 0.67–0.86 |
| qwen3 | 0.743 | [0.729, 0.758] |
| s2pro | 0.707 | [0.693, 0.721] |
| voxtral | 0.600 | [0.572, 0.630] |
| IMPOSTOR (lower bound) | not yet measured | — |

**qwen3 and s2pro fall inside the range the metric returns for two recordings of
the same person.** ECAPA cannot separate their clones from the real speaker, so
the difference between them is resolvable by the metric but is *not* evidence
that one clone is closer to the person. voxtral is clearly below. Without the
upper bound, "SIM = 0.70" is uninterpretable and this conclusion is unavailable.

The lower bound is the one number that cannot be produced from this project's own
data. A stand-in test using another recording of the *same* speaker scored 0.733
— inside the clone range — which is exactly why a genuinely different voice is
needed before absolute SIM values are quoted.

Caveat: ECAPA-VoxCeleb is trained on 16 kHz English celebrity speech. It is out
of domain for German phone voice memos, and that is a limit on all SIM values
here, not just on the differences.

### WER — word error rate

Whisper ASR transcribes the rendered audio; `jiwer` compares it to the target
text. Both sides are normalized identically (case, punctuation, umlaut
transliteration, hyphens, digits → German number words), so the metric does not
measure the ASR's formatting habits.

| condition | WER (median) | 95 % CI |
|---|---|---|
| **REAL (floor)** | **0.18** | — |
| qwen3 | 0.079 | [0.058, 0.104] |
| s2pro | 0.100 | [0.087, 0.125] |
| voxtral | 0.112 | [0.098, 0.171] |

**The clones score better than the human original.** Reading a transcript
produces cleaner, more canonical speech than speaking spontaneously into a
phone. So WER here does not measure intelligibility — it measures *distance from
read speech*, and a value below the human floor is not a quality win.

Second consequence: the measurement floor (0.18) is larger than the differences
between systems (0.02–0.11). Paired by text, the **qwen3 vs. s2pro WER
difference is not detectable** (−0.019, adjusted CI [−0.051, +0.014]). A
German-finetuned ASR would tighten this; if it is swapped in, every WER value
including the floor must be recomputed in one pass.

### Service latency

`service_rtf` = wall-clock API latency ÷ length of the returned audio. Named
`service_rtf` and not `RTF` on purpose: real RTF is compute time ÷ audio
duration on known hardware, and none of these APIs expose that. What is measured
is three different services over the internet — one non-streaming, one hosted in
Singapore. `connect_s` reports the handshake separately as an RTT proxy. All
timings start before the connection is opened, so they are comparable.

---

## 3 Stimulus length

### How it is computed

The duration is not set directly. A target duration becomes a word count, a
passage of that length is chosen **on the text**, and the resulting audio length
is whatever that passage takes in each voice.

```
words_per_second = 2.94             # measured median across the three systems
target = round(10.0 × 2.94) = 29 words
band   = [round(8.0 × 2.94), round(13.0 × 2.94)] = [24, 38] words
```

1. Skip the first `SKIP_HEAD_FRAC = 15 %` of the words — the opening of a voice
   memo is a greeting, and the first seconds of a render carry onset artifacts
   that differ per system.
2. Take the run of **consecutive whole sentences** whose word count falls in the
   band and lies closest to `target`. If a single sentence exceeds the band
   (these are spontaneous-speech transcripts; one text has a 49-word
   "sentence"), split it at commas instead. Only if neither works, use a plain
   word window and flag the ragged end.
3. Locate that passage in every rendition via Whisper word-level timestamps and
   a fuzzy token match (`difflib`, longest common subsequence). Below a 0.6 token
   match the system did not say the passage — reported and excluded, never
   replaced by a time window. Only matched runs of **≥ 3 tokens** may set the
   start and end (`MIN_ANCHOR_TOKENS`); the ratio still counts every matched
   token. Without that rule a single common word ("das") matching elsewhere in
   the transcript stretches the span across everything in between: it inflated
   18 of 64 excerpts by 0.5–5.2 s, and turned two correct 8.5 s and 8.2 s
   passages into 31.2 s and 20.6 s spans that were then discarded as runaway
   renders. An inflated span is the worse failure of the two — it is cut and
   used, so the conditions silently stop holding matched content.
4. Cut `start − 0.2 s` to `end + 0.3 s`, with 20 ms fades (a hard cut clicks, and
   a click is heard and rated as an artifact).

Steps 1–2 touch no audio, so the choice cannot favour any system. Result on 16
texts: 14 end at a sentence boundary, 2 at a comma, none mid-sentence. Final
durations **5.9–14.4 s, median 8.8 s**; 55 stimuli, 8.5 min of audio.

### Why ~10 s and not the full render

Five reasons, each with a number.

1. **The unit of judgement must match the unit of analysis.** ACR asks for one
   score per stimulus. A 36 s render contains several distinguishable quality
   events, and the rater compresses them with an unknown weighting — so the
   score is an unrecorded mixture, not "the naturalness of this system". ITU-T
   P.800 Annex B uses stimuli of roughly this length for the same reason.
2. **Session length.** Full renders meant ~46 min of pure audio per rater, i.e.
   60–75 min sessions; the second half of such a session measures fatigue. Now
   5.9–6.2 min of audio, ~20 min per session.
3. **Power per unit of rater time.** For a fixed budget `T`, items `n ≈ T/duration`
   and the standard error of a mean falls with `1/√n`. Going from a 36 s median
   to 9.6 s is **3.75× the items** for the same rater time, i.e. intervals about
   **1.9× narrower** — without recruiting anyone.
4. **More items means more text clusters** for the interval to resample over
   (see §6), which is what makes keeping all 13 texts × 4 conditions affordable.
5. **Content matching becomes possible.** Every system is judged on the same
   passage; with full renders a rater also compares which parts of a 90-second
   voicemail they happened to attend to.

### Why not "the first 10 seconds"

The systems speak the same text at different rates — 1.16× spread at the median,
up to **1.53×**. A fixed window therefore holds a different *amount of content*
per system and lands mid-word more often for the slower one. On this material
that would have penalized voxtral, which is already the weakest — a bias
pointing in the direction of the conclusion.

### What it costs, and the answer

The excerpts sit at **32–71 %** of each render: about **73 % of what each system
produced is never heard by a rater**. Measured on the full renders instead
(`analyze_objective.py --long-form`):

| source | dur. ratio | words/s | level drift | sim 1st → 3rd | drift |
|---|---|---|---|---|---|
| REAL | 1.000 | 2.53 | +0.8 dB | 0.758 → 0.773 | +0.030 |
| qwen3 | 0.901 | 2.92 | −1.2 dB | 0.715 → 0.717 | +0.004 |
| s2pro | 0.845 | 3.00 | −1.7 dB | 0.670 → 0.672 | +0.011 |
| voxtral | 0.966 | 2.71 | −2.8 dB | 0.586 → 0.598 | +0.008 |

- **No system drifts away from the target voice over a long utterance**
  (correlation of render length with drift: r = −0.05, n = 48). The largest
  single drift in the whole set belongs to a *real* recording. So excerpting
  hides no speaker-identity problem — a measured statement, not an assumption.
- **voxtral fades out**, up to −11.4 dB across one render. A genuine long-form
  defect that loudness-normalized 10 s excerpts cannot show.
- **All three speak faster than the person** (2.7–3.0 vs 2.53 words/s).

---

## 4 Audio processing

Two separate paths. The **objective** path measures the renders as the services
returned them — nothing is normalized, because SIM and WER should see what the
service actually produced. The **subjective** path prepares stimuli for human
ears, and there every condition has to be put through exactly the same treatment,
or a rater ends up judging the recording chain instead of the voice.

```mermaid
flowchart TD
    subgraph SRC[" "]
        REF["reference recording<br/>(one held-out voice memo)"]
        TXT["16 texts<br/>(transcripts of the memos)"]
        RAW["the speaker's own recordings<br/>data/wav → DeepFilterNet"]
    end

    REF -->|enrollment, same file for all 3| API
    TXT --> API["s2pro · voxtral · qwen3<br/>synthesis"]
    API --> RENDER["full renders<br/>12–92 s, 24 / 44.1 kHz"]

    RENDER --> OBJ
    RAW --> OBJ
    subgraph OBJ["objective path — audio kept as produced"]
        direction TB
        E["ECAPA-TDNN embedding<br/>→ SIM vs. the reference"]
        W["Whisper ASR + text normalization<br/>→ WER vs. the target text"]
        L["timings from the API call<br/>→ latency, TTFA, service_rtf"]
        D["first third vs. last third<br/>→ voice drift, level drift"]
    end

    RENDER --> SUB
    RAW --> SUB
    subgraph SUB["subjective path — identical treatment for every condition"]
        direction TB
        S1["1 · resample to 24 kHz<br/><i>lowest common rate — no system<br/>gets more bandwidth</i>"]
        S2["2 · cut the excerpt at word boundaries<br/><i>~10 s, same passage in every<br/>rendition, ±0.2/0.3 s, 20 ms fades</i>"]
        S3["3 · normalize to −23 LUFS<br/><i>BS.1770 — a quieter sample is rated<br/>less natural regardless of quality</i>"]
        S4["4 · match the pause floor<br/><i>add 1/f noise to speech −26 dB —<br/>removes the recording-chain cue</i>"]
        S1 --> S2 --> S3 --> S4
    end

    SUB --> STIM["55 stimuli, s_001.wav …<br/>opaque names, key held back"]
    STIM --> TEST["listening test<br/>naturalness + 2AFC"]
```

**Why that order.** The cut runs *before* normalization: BS.1770 integrated
loudness is measured over the whole signal, so normalizing a 42 s render and then
cutting 10 s out of it would leave the excerpts at different levels again — the
exact confound step 3 exists to remove. The pause floor is matched *after*
normalization, so the added noise sits at a fixed distance below a speech level
that is already the same everywhere.

**One file through all four steps** (s2pro, `AUDIO-2021-08-30-06-40-21`):

| stage | length | rate | loudness | peak | speech − pause |
|---|---|---|---|---|---|
| source render | 42.66 s | 44 100 Hz | −21.57 LUFS | −4.90 dBFS | 37.1 dB |
| 1 resample | 42.66 s | 24 000 Hz | −21.57 LUFS | −4.92 dBFS | 37.1 dB |
| 2 cut (30.72–40.36 s, match 0.87) | 10.14 s | 24 000 Hz | −21.52 LUFS | −4.92 dBFS | 29.0 dB |
| 3 normalize (−1.5 dB) | 10.14 s | 24 000 Hz | **−23.00 LUFS** | −6.40 dBFS | 28.6 dB |
| 4 pause floor (noise at −57 dBFS) | 10.14 s | 24 000 Hz | −23.00 LUFS | −6.37 dBFS | **25.8 dB** |

After step 4 every stimulus in the test sits at the same loudness and the same
distance from its own noise floor, whatever it came from. What survives is the
voice, the prosody and the pause *structure* — which is what raters are there to
judge.

**What is deliberately not done:** no denoising of the renders, no equalization,
no de-clicking, no trimming of leading or trailing silence beyond the excerpt
cut. Each of those would hide a defect that belongs in the rating. The real
recordings are denoised (DeepFilterNet), but that happens before this pipeline
and is the reason step 4 exists at all — see §5.

---

## 5 Confounds

Every stimulus goes through identical processing: resample to 24 kHz → cut →
loudness-normalize to −23 LUFS (BS.1770) → match the pause floor to speech
−26 dB.

### Level

qwen3 renders ~7 dB quieter than the other two, and a quieter sample is rated
less natural almost regardless of its quality. Loudness (not RMS) because
BS.1770 gating ignores silence, so a gappy render is measured on its speech.

### Recording chain — the important one

The real recordings are phone voice memos put through DeepFilterNet; the clones
are generated clean. Before matching:

| | pause fraction | speech − pause floor |
|---|---|---|
| REAL | 0.22 | 39.4 dB |
| s2pro | 0.08 | 33.0 dB |
| voxtral | 0.15 | 28.7 dB |
| qwen3 | 0.10 | 29.6 dB |

A single threshold on either column identifies the real recording with **96 %
accuracy**. The cue is self-inflicted: denoising moves the real recordings from
0.132 / 30.6 dB to 0.219 / 36.8 dB, i.e. it gives them *digitally silent* pauses
that no TTS model produces. A rater noticing "the one with dead-silent gaps is
real" wins without judging a voice.

Fix: raise every stimulus's pause floor to one common value by adding 1/f-shaped
noise — only ever adding, because lowering a floor would mean denoising one
condition and not the others. Deterministic (seed from the filename). Pause
*structure* is left untouched, since how often a voice stops is a property of
the system.

### How the cues are measured — and why the earlier numbers were too kind

Earlier versions of this section quoted accuracies against a "guessing
baseline" of 0.750: one threshold on one feature, classifying each stimulus on
its own, compared against the base rate of always answering "clone". That is
the wrong baseline for the task. A rater in the Turing part hears **two** files
of the same passage side by side and picks one, so chance is **0.500**, and the
only thing that matters is which of the two is larger. The base rate also moves
with the design — the number of rated conditions changes it — which makes those
numbers incomparable between builds.

`check_listening_test.py` now reports both. The paired column is the one to
read; it fits a direction only, never a threshold, so there is nothing to
overfit. Measured on the current set (13 texts, 39 pairs):

| feature | single stimulus | **paired (chance 0.500)** | reading |
|---|---|---|---|
| `silence_ratio` | 0.846 | **0.821** | more silence = real |
| `hf_tilt_db` | 0.846 | **0.795** | brighter = real |
| `duration_s` | 0.769 | **0.718** | longer = real |
| `lf_tilt_db` | 0.788 | **0.692** | less low end = real |
| `pause_snr_db` | 0.769 | **0.564** | cleaner pauses = real |
| `level_std_db` | 0.750 | 0.513 | no usable cue |

Read these to one decimal, not three. Each paired figure counts wins over 33
pairs, so a single pair flipping moves it by 0.03 — and rebuilding the set with
a different stimulus-id assignment (which reseeds the pause-floor noise) moved
`silence_ratio` between 0.82 and 0.91 across builds with no design change.

### Residual cues, reported not removed

Four features stay usable after matching. None is removed, and each for a
reason that has to be stated rather than hidden:

- **Pause structure** (0.786, the strongest). The real recordings pause more
  often than any system. This is deliberate: how a voice distributes its stops
  is part of what a rater is there to judge, and equalizing it would remove a
  genuine difference between the systems. It is nonetheless the single largest
  route to answering the Turing part without listening to voice quality.
- **Brightness**, energy 4–8 kHz (0.738). All three systems render duller than
  the phone-recorded original — partly the recording chain, partly a real
  failure to reproduce the source timbre. Equalizing it would erase what
  raters are there to hear.
- **Duration** (0.667). The speaker is slower than every system (2.53 vs
  2.7–3.0 words/s), so a content-matched excerpt runs longer in her voice. It
  cannot be normalized away without either changing the content — which breaks
  the matched-passage design — or time-stretching, which is an audible
  artifact. See §3.
- **Pause floor spread** (0.641). Three voxtral stimuli sit 11 dB below the
  common target and could not be lifted: their own noise floor never drops far
  enough for a floor to be measured at all (`pause_measured: fallback` in
  `stimuli.csv`), and the matching only ever adds noise. They are kept rather
  than excluded — dropping them would remove exactly the material on which
  voxtral performed worst, which flatters it. The effect runs in two directions
  and both belong in the report: hiss depresses voxtral's naturalness, and hiss
  also sounds like a phone recording, which can raise its fooling rate.

All six features are written per stimulus to `cue_audit.csv` and must be
reported next to the Turing results.

### Enrollment — not yet removed

qwen3 was enrolled with `enable_preprocess=True` (server-side denoising), voxtral
with raw bytes plus a hardcoded `gender="female"`, s2pro through a web UI. **The
three did not receive the same input**, so a SIM difference can come from the
model or from one model having been handed cleaner audio. This is the largest
single threat to the conclusion. Code defaults are now neutral and
`prepare_enrollment.py` produces one canonical reference with a hash manifest,
but the numbers in §2 predate that. Either re-enroll and re-run, or state in the
report that the ranking is confounded with the enrollment input.

---

## 6 Listening-test design

### Two tasks

- **Naturalness (ACR 1–5)**, five verbal labels per ITU-T P.800, all four
  conditions including REAL.
- **2AFC detectability**: one real recording vs. one clone of the same passage,
  "which is the real voice". Fooling rate 0.50 = indistinguishable from guessing.

### Rater plans

6 plans = 2 role groups × 3 model rotations; each rater draws one at random.
A plan sends half the texts to the naturalness part and half to the 2AFC part.

Two problems this solves at once:

1. **No rater meets the same text in both parts.** With content-matched excerpts
   and REAL as a rated condition, part 1 would otherwise contain the answer to
   part 2.
2. **All text × model pairs get used.** The earlier design assigned one model per
   text, giving the 2AFC test 5 distinct pairs per model — and more raters could
   not fix that, because the interval was over 5 items however many people
   answered. Now 39 pairs, 13 per model, fully balanced.

Plans are enumerated rather than drawn per rater, so cell coverage does not
depend on luck at n ≈ 30.

### Per rater

Order shuffled; A/B sides flipped and logged; three unscored training items
first; part 2 locked until part 1 is complete; two hidden repeats at least 6
positions from their original (naively shuffled, 7 % landed back-to-back, where
the two ratings agree from memory and the reliability estimate comes out too
good). Generated anonymous id, no names. The **rater** enters the scores, not the
operator — the operator is the author of the system.

Counts: 13 usable texts × 4 conditions = 52 naturalness items, 39 pairs, 24–28
items and 6–7 pairs per rater.

No low anchor. An ITU-T P.810 MNRU anchor was built and then dropped: it degrades
a *recording* with noise, while the question put to the rater is about the
naturalness of a *voice*, so it anchored the wrong axis — and it cost a fifth of
every session. The consequence is recorded in §8.

### Blinding

All stimuli are named `s_001.wav`, assigned after the shuffle; `key.json` holds
the only mapping and is not shipped. An earlier version wrote
`turing_03_real.wav` into the page manifest, where any rater could read it.
`check_listening_test.py` gates 17 such properties and exits non-zero.

### Exclusions, fixed in advance

Technical failures only, never "this sounds bad" — that is what raters judge.

- cannot reach target loudness without clipping; essentially no speech; clipped
- excerpt not found in the rendition (< 0.6 token match): the system did not say it
- excerpt found but stretched over > 2× the per-text median span: a runaway render
- no measurable pause floor: not one frame falls below the silence threshold, so
  `match_noise_floor` cannot equalize the item and it would sit 7–12 dB away from
  the common floor — the exact cue §5 exists to remove

`duration_off` / `silence_heavy` alone are **not** exclusions — they usually mean
the system speaks slower or pauses more, which raters should hear.

If a text is excluded for one system it is dropped for all, so every condition is
scored on identical material. On this set: three voxtral renditions, hence 3 of 16
texts, leaving 13, all three for having no measurable pause floor.
**Every exclusion in this set is a voxtral rendition**; qwen3 and s2pro produced
usable audio for all 16 texts.

Two further voxtral renditions were excluded as runaway renders in an earlier
build and are **not** failures: the aligner had mislocated their excerpts (§3),
reporting a 31.2 s and a 20.6 s span for passages that are actually 8.5 s and
8.2 s. Both renders are within 5 % of the length the other systems produced.

That asymmetry is a result, and the balanced-design rule is what hides it: the
surviving 13 texts are the ones voxtral handled, so every system is then scored
on material selected by one system's success. `analyze_objective.py` therefore
reports the per-system usable-render rate (`render_reliability.csv`) before any
quality metric, so a system that fails outright cannot look good by being absent.

---

## 7 Statistics

**Naturalness**: mean of per-rater means (so every person carries equal weight
despite different plan sizes). **2AFC**: fooling rate, same aggregation.

**Intervals** come from a two-way cluster bootstrap: each replicate resamples
raters *and* texts with replacement. Answers are not independent — each rater
answers many items and each item is answered by many raters — so pooling them as
independent observations gives intervals that are too narrow.

**Coverage was checked by simulation** on this exact design, before any data
existed (`tools/check_ci_coverage.py`):

| method | coverage of the nominal 95 % CI |
|---|---|
| two-way (used) | **0.99** |
| raters only | 0.93 |

The interval is **conservative** — too wide, not too narrow. That is kept and
reported rather than tuned away: a wide interval can only weaken a claim, a
narrow one manufactures findings. The cost is sensitivity, so a null result
should be quoted with this figure.

**Power**, same simulation: a 0.30 MOS difference is detected with 80 % power at
30 raters (0.20 → 35 %, 0.40 → 98 %). So 30 raters is roughly the right number,
and a smaller difference will be reported as *not detectable*, not as absent.

**Pairwise comparisons** are paired within rater, which removes each rater's
scale offset — the largest noise source in a MOS study — and their verdict uses a
Bonferroni-adjusted level: 4 conditions means 6 comparisons, and at a nominal 5 %
each the family-wise error rate would be ~26 %.

**Familiarity** is reported as the contrast with its own interval, not as two
subgroup intervals to compare by eye: two overlapping intervals can hide a real
difference and two separated ones can appear without one.

**Rating noise** comes from the hidden repeats: the mean absolute difference
between two ratings of the same stimulus by the same rater. A condition
difference smaller than that is inside one person's own scale noise — real,
perhaps, but not meaningful.

If `statsmodels` is installed, a mixed model (`rating ~ condition`, random rater
and text) and a rater-clustered logit are fitted as a confirmatory check. The
bootstrap stays the headline at this sample size.

---

## 8 Limitations

1. **One speaker.** This is a case study. Intervals generalize over texts and
   raters, not over speakers. Nothing here says how these systems behave on a
   different voice.
2. **Enrollment confound** (§5) — the largest one, and still present in the
   numbers.
3. **SIM lower bound missing**, so absolute similarity values are uncalibrated.
4. **WER floor larger than the effects** (§2), measured with whisper-small.
5. **73 % of each render is not rated subjectively** (§3), answered objectively
   rather than assumed away.
6. **Four residual cues in the Turing part** (§5), measured paired against
   chance 0.500: pause structure ≈0.82, brightness ≈0.80, duration ≈0.72,
   low-end ≈0.69. A rater who notices any of them can answer above chance
   without judging voice quality, so the Turing result is a lower bound on how
   distinguishable the clones are — not a measure of voice similarity alone.
   Pause structure is strong enough that the fooling rate should be read as
   "detectable, largely via prosody and pausing", not "the voice is
   distinguishable".
7. **The exclusions are not missing at random.** All five excluded renditions are
   voxtral (§6), so the 13 surviving texts are the ones voxtral rendered usably
   and every system is compared on a text sample conditioned on one system's
   success. Read the usable-render rate — voxtral 13/16, qwen3 and s2pro 16/16 —
   next to any quality number, and expect voxtral's naturalness here to be
   flattered by the absence of its worst cases.
8. **Spontaneous vs. read speech.** The real side is spontaneous with
   hesitations; the clones read a transcript of it. Disfluencies are in the text,
   but the delivery differs — and that is part of what makes a clone detectable.
9. **The operator is the author.** Rater clicks, key unopened, sides re-flipped
   per rater — but this is not a blind experimenter.
10. **Convenience rater sample**, no hearing screening, and the familiarity
    subgroup will be small unless recruited to a quota.
11. **No low anchor** (§6), so nothing in the set forces a rater to use the
    bottom of the scale. Absolute MOS values are therefore interpretable only
    within this study and must not be compared with MOS numbers from other
    work — a restriction that applies to MOS across studies anyway, since it
    depends on the rater pool, the instructions, the language and the material.
    Within-study comparisons (system vs. system, system vs. REAL) are unaffected,
    and the two hidden repeats still measure whether a rater scores
    consistently.
