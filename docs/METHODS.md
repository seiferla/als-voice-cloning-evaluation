# Evaluation methodology

Design decisions and the measured numbers behind them. The README says how to
run the pipeline; the report in `docs/paper/` is the written-up version. All
numbers below come from `evaluation/results/*.csv` (objective run
`20260908_194424`, listening test with 30 raters).

Contents: [1 Setup](#1-setup) · [2 Objective metrics](#2-objective-metrics) ·
[3 Excerpts](#3-excerpts) · [4 Stimulus processing](#4-stimulus-processing) ·
[5 Cues](#5-cues) · [6 Listening test](#6-listening-test) ·
[7 Statistics](#7-statistics) · [8 Results](#8-results) · [9 Limitations](#9-limitations)

## 1 Setup

One German speaker (an ALS patient), 16 texts transcribed from her own 2021
voice memos, three commercial cloning services: **s2pro** (Fish Audio),
**voxtral** (Mistral), **qwen3** (Alibaba DashScope). Each service renders each
text. The speaker's own recording of the same text is kept as a fourth
condition, **REAL**, and every metric is reported against it, because none of
the metrics has an absolute scale.

All three services were enrolled from one file: a 30 s excerpt of one held-out
memo, mono, 24 kHz, normalized to −1 dBFS peak, no denoising, no provider
options such as gender hints. `data/enrollment/enrollment.json` records the
source and output hashes (created 2026-09-02, before the final run). The SIM
reference is a different held-out memo, so no clone is compared against the
file it was cloned from.

## 2 Objective metrics

### SIM

Cosine similarity of ECAPA-TDNN embeddings (`speechbrain/spkrec-ecapa-voxceleb`)
between the reference recording and each render. REAL is the speaker's other
recordings against the same reference, i.e. the upper bound the metric can reach
for this person.

| condition | SIM (median) | 95 % CI | vs. REAL (adjusted) |
|---|---|---|---|
| **REAL** | **0.787** | [0.760, 0.825] | |
| qwen3 | 0.755 | [0.738, 0.769] | −0.031 [−0.068, +0.004], not separable |
| s2pro | 0.703 | [0.687, 0.713] | −0.088 [−0.116, −0.056] |
| voxtral | 0.609 | [0.602, 0.627] | −0.174 [−0.207, −0.140] |

qwen3 is the only system the metric cannot separate from the real voice. A lower
bound (a different speaker in `impostor_audio`) was not measured, so absolute
SIM values remain uncalibrated. ECAPA-VoxCeleb is trained on 16 kHz English
speech and is out of domain for German phone memos.

### WER

Whisper `small` transcribes each render; `jiwer` compares it to the target text.
Both sides are normalized identically (case, punctuation, umlauts, hyphens,
digits → German number words).

| condition | WER (median) | 95 % CI | vs. REAL (adjusted) |
|---|---|---|---|
| **REAL** | **0.190** | [0.147, 0.228] | |
| qwen3 | 0.097 | [0.075, 0.115] | −0.090 [−0.131, −0.059] |
| s2pro | 0.119 | [0.101, 0.142] | −0.060 [−0.098, −0.031] |
| voxtral | 0.116 | [0.107, 0.141] | −0.036 [−0.077, +0.034], not separable |

All clones score below the human original: reading a transcript produces
cleaner speech than talking into a phone. WER here measures distance from read
speech, not intelligibility, and the reference floor (0.19) is larger than the
differences between systems.

### Latency

`latency_s` is wall-clock time of the API call, `service_rtf` is latency divided
by the length of the returned audio, `ttfa_s` is time to the first audio frame
(streaming services only). Medians over 16 texts: latency qwen3 13.8 s, s2pro
7.8 s, voxtral 5.7 s; service_rtf 0.39 / 0.23 / 0.14; TTFA qwen3 1.55 s, s2pro
1.21 s. These are network measurements of hosted services, not compute RTF.

### Usable renders

voxtral produced a usable render for 13 of 16 texts, qwen3 and s2pro for all 16
(`render_reliability.csv`). The three failures had no measurable pause floor
(§6, exclusions). Under the balanced-design rule the whole text is dropped for
every condition, so the listening test uses 13 texts.

## 3 Excerpts

Raters hear ~10 s excerpts, not full renders. The excerpt is chosen on the text,
so the choice cannot favour a system, then located in each render.

1. Target 10 s = 29 words (2.94 words/s measured), band 24–38 words. Skip the
   first 15 % (greeting, onset artifacts).
2. Take the run of whole sentences closest to the target inside the band; split
   an over-long sentence at commas; otherwise a plain word window, flagged.
3. Locate the passage in each render via Whisper word timestamps and a fuzzy
   token match (`difflib`). Below 0.6 matched tokens the system did not say the
   passage and the render is excluded. Only matched runs of ≥ 3 tokens set the
   span boundaries; without that rule a stray common word stretched 18 of 64
   spans by 0.5–5.2 s.
4. Cut start − 0.2 s to end + 0.3 s with 20 ms fades.

Result: 14 of 16 excerpts end at a sentence boundary, 2 at a comma. Durations
5.9–14.4 s, median 8.8 s.

Why excerpts: ACR gives one score per stimulus, and a 36 s render mixes several
quality events into one unrecorded average. Full renders would mean ~46 min of
audio per rater; excerpts give 3.75× the items for the same rater time and let
every system be judged on the same passage. A fixed 10 s window was rejected
because the systems differ in speaking rate by up to 1.53×, so it would hold
different content per system.

The excerpts cover 32–71 % of each render. To check that excerpting hides
nothing, the full renders were scored once (script in the history at `4129a4b`):

| source | dur. ratio | words/s | level drift | SIM first → last third |
|---|---|---|---|---|
| REAL | 1.000 | 2.53 | +0.8 dB | 0.758 → 0.773 |
| qwen3 | 0.901 | 2.92 | −1.2 dB | 0.715 → 0.717 |
| s2pro | 0.845 | 3.00 | −1.7 dB | 0.670 → 0.672 |
| voxtral | 0.966 | 2.71 | −2.8 dB | 0.586 → 0.598 |

No system drifts away from the target voice over a long render (r = −0.05
between length and drift, n = 48). voxtral fades out by up to −11.4 dB across a
render, which normalized excerpts cannot show. All three speak faster than the
person.

## 4 Stimulus processing

The objective path measures the renders as returned. The subjective path puts
every condition through the same four steps:

1. Resample to 24 kHz, the lowest native rate (s2pro renders at 44.1 kHz).
2. Cut the excerpt (§3).
3. Normalize to −23 LUFS (BS.1770). Cut before normalizing, because integrated
   loudness is measured over the whole signal.
4. Raise the pause floor to speech − 26 dB by adding 1/f noise (§5). After
   normalizing, so the added noise sits at a fixed distance below a level that
   is already equal everywhere.

One file through the four steps (s2pro, `AUDIO-2021-08-30-06-40-21`):

| stage | length | loudness | speech − pause |
|---|---|---|---|
| source render | 42.66 s | −21.57 LUFS | 37.1 dB |
| 2 cut (match 0.87) | 10.14 s | −21.52 LUFS | 29.0 dB |
| 3 normalize | 10.14 s | −23.00 LUFS | 28.6 dB |
| 4 pause floor | 10.14 s | −23.00 LUFS | 25.8 dB |

Not done: denoising, equalization, de-clicking, silence trimming beyond the cut.
Each would hide a defect that belongs in the rating.

## 5 Cues

A rater in the 2AFC part can win without judging the voice if some low-level
property separates real from clone. Two were removed, four remain and are
reported.

**Level.** qwen3 renders ~7 dB quieter than the others; step 3 removes this.

**Recording chain.** The real recordings are phone memos put through
DeepFilterNet, which gives them digitally silent pauses that no TTS produces.
Before matching, a single threshold on the pause floor identified the real
recording with 96 % accuracy:

| | pause fraction | speech − pause floor |
|---|---|---|
| REAL | 0.22 | 39.4 dB |
| s2pro | 0.08 | 33.0 dB |
| voxtral | 0.15 | 28.7 dB |
| qwen3 | 0.10 | 29.6 dB |

Step 4 raises every floor to one common value. Noise is only added; lowering a
floor would mean denoising one condition. Pause structure is left as is.

**Residual cues**, measured by `check_listening_test.py` as the share of 2AFC
pairs in which the feature points to the real recording (chance 0.50, 39 pairs,
read to one decimal):

| feature | paired accuracy | reading | why not removed |
|---|---|---|---|
| `silence_ratio` | ≈ 0.8 | more pauses = real | pausing is part of the voice |
| `hf_tilt_db` | ≈ 0.8 | brighter = real | equalizing would erase a real timbre failure |
| `duration_s` | ≈ 0.7 | longer = real | speaker is slower; content is matched, so duration differs |
| `pause_snr_db` | ≈ 0.6 | cleaner pauses = real | three voxtral files could not be lifted (see below) |

Three voxtral stimuli sit ~11 dB below the common floor: their own noise never
drops far enough for a floor to be measured (`pause_measured: fallback`), and
the matching only adds. They are kept, because dropping them would remove
exactly the material voxtral handled worst. All six features are written per
stimulus to `cue_audit.csv`.

## 6 Listening test

**Two tasks.** Naturalness on a 5-point ACR scale (ITU-T P.800), all four
conditions including REAL. Then 2AFC: one real recording and one clone of the
same passage, "which is the real voice". Fooling rate 0.50 means guessing.

**Rater plans.** 6 plans (2 role groups × 3 model rotations), one drawn per
rater. A plan sends half the texts to the naturalness part and half to the 2AFC
part, so no rater hears the same text in both (part 1 would otherwise contain
the answer to part 2). Every text × model pair is used: 39 pairs, 13 per system.

**Per rater.** Order shuffled, A/B sides flipped and logged, three unscored
training items, part 2 locked until part 1 is done, two hidden repeats at least
6 positions apart. Anonymous generated id. 24–28 items and 6–7 pairs per rater,
~6.5 min of audio. The rater enters the scores.

**Blinding.** Stimuli are `s_001.wav` … assigned after the shuffle; `key.json`
holds the only mapping and is not shipped. `check_listening_test.py` gates 17
properties (no condition names or text ids in the page, no positional pattern,
level and floor matching, plan coverage) and exits non-zero on any failure.

**Exclusions**, fixed before any rating: cannot reach −23 LUFS without
clipping; essentially silent or clipped; excerpt not found (< 0.6 match);
excerpt span > 2× the per-text median; no measurable pause floor. Never
"sounds bad". If a text is excluded for one system it is dropped for all. On
this set: three voxtral renders, all for an unmeasurable pause floor, so 13 of
16 texts remain.

No low anchor. An MNRU anchor was built and dropped: it degrades a recording
with noise while the question is about the naturalness of a voice.

## 7 Statistics

Naturalness is the mean of per-rater means; the fooling rate likewise, so every
rater carries equal weight. Intervals come from a two-way cluster bootstrap that
resamples raters and texts, since answers within a rater and within a text are
not independent.

Coverage was checked by simulation on this design before data collection
(`tools/check_ci_coverage.py`): the two-way interval covers a true 0.30 MOS
difference in 0.99 of simulated studies (nominal 0.95), resampling raters only
gives 0.93. The interval is conservative. Power at 30 raters: a 0.30 MOS
difference is detected in 80 % of studies, 0.20 in 35 %, 0.40 in 98 %.

Pairwise differences are paired within rater and judged at a Bonferroni-adjusted
level over the 6 comparisons. Familiarity (raters who know the speaker vs. those
who do not) is reported as a contrast with its own interval. The hidden repeats
give the rating noise: a difference smaller than one rater's own repeat spread
is not meaningful. A mixed model (`rating ~ condition`, random rater and text)
is fitted as a confirmatory check when `statsmodels` is installed.

## 8 Results

30 raters, 13 texts, 52 naturalness items, 39 pairs. 15 raters know the speaker.

**Naturalness** (`nmos_summary.csv`, `nmos_pairwise.csv`):

| condition | mean | 95 % CI | vs. REAL (adjusted) |
|---|---|---|---|
| **REAL** | **4.16** | [3.81, 4.46] | |
| voxtral | 4.06 | [3.79, 4.34] | −0.10 [−0.57, +0.48], not separable |
| s2pro | 3.92 | [3.64, 4.22] | −0.23 [−0.72, +0.44], not separable |
| qwen3 | 3.04 | [2.71, 3.37] | −1.12 [−1.65, −0.51] |

qwen3 is below every other condition (−0.88 to s2pro, −1.01 to voxtral, both
adjusted intervals exclude zero). s2pro and voxtral are not separable from each
other or from REAL. Rating noise from 60 repeat pairs: mean absolute difference
0.55 points, 92 % within one point. The s2pro–voxtral gap (0.13) is inside that
noise; the qwen3 gaps are not.

**2AFC** (`turing_summary.csv`), share of pairs in which the clone was picked
as the real voice:

| system | all raters | knows the speaker | does not |
|---|---|---|---|
| voxtral | 0.41 [0.23, 0.62], chance | 0.53 | 0.29 |
| s2pro | 0.25 [0.12, 0.40] | 0.30 | 0.19 |
| qwen3 | 0.14 [0.00, 0.25] | 0.20 | 0.07 |

voxtral is the only system raters could not tell from the real voice above
chance. The familiarity contrast is not separable from zero for any system
(qwen3 +0.13 [−0.10, +0.33]), and runs opposite to H4: raters who know the
speaker were fooled more often, not less.

**Against the preregistration.** H1 (all systems below REAL on naturalness)
holds only for qwen3. H2 (naturalness follows similarity: qwen3 ≥ s2pro >
voxtral) is reversed: the system with the highest embedding similarity is rated
least natural and detected most often, and the system with the lowest similarity
is rated closest to the real voice. H3 (all detected above chance) fails for
voxtral. H4 is not supported.

## 9 Limitations

1. One speaker. Intervals generalize over texts and raters, not over speakers.
2. No SIM lower bound, so absolute similarity values are uncalibrated.
3. WER floor (0.19) larger than the effects, measured with whisper-small.
4. 73 % of each render is not rated subjectively (§3).
5. Four residual cues in the 2AFC part (§5). The fooling rate is a lower bound
   on how distinguishable the clones are, largely via pausing and brightness,
   not a measure of voice similarity alone.
6. The three excluded texts are all voxtral failures, so every system is scored
   on material selected by voxtral's success. Read 13/16 next to its
   naturalness.
7. Spontaneous vs. read speech: the real side has hesitations and a phone
   recording chain; the clones read a transcript.
8. The operator is the author. Raters click, the key stays closed, but this is
   not a blind experimenter.
9. Convenience sample of raters, no hearing screening, 15/15 split on
   familiarity by chance rather than quota.
10. No low anchor, so absolute MOS values are comparable only within this
    study. Within-study comparisons are unaffected.
