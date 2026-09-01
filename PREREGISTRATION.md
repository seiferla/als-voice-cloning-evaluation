# Preregistration — subjective evaluation

**Draft. Review it, then commit it before the first rater.** The point is the
commit timestamp: it separates "decided what counts, then measured" from the
other order. Everything here is already implemented; this describes a frozen
design. Reasoning and measured numbers: [docs/METHODS.md](docs/METHODS.md).

| | |
|---|---|
| Speaker | one German speaker (ALS patient), 2021 voice memos |
| Systems | s2pro (Fish Audio), voxtral (Mistral), qwen3 (Alibaba DashScope) |
| Reference condition | REAL — the speaker's own recordings |
| Frozen at | commit + seed in `evaluation/results/listening_test/run_meta.json` |
| Planned raters | 30 |

## TODO — only you can answer these

1. **Consent and data protection.** The study clones a real, identifiable
   person's voice, plays it to ~30 listeners, and enrollment sent that voice to
   three companies. Record here: who consented to what and when, who decided if
   the person cannot consent themselves, what happens to the recordings, voices
   and ratings at the end of the project, and what the three providers do with
   the uploaded audio. Raters give age and familiarity, and are identified only
   by a generated id — no names.
2. **SIM lower bound.** Needs clips of a *different* speaker (e.g. Common Voice
   DE) in `impostor_audio`. Until then, absolute similarity values are
   uncalibrated and the report must say so.

## Hypotheses

Stated before any rating exists. The objective results in the table below were
already known, so H2 is a prediction *across measurements*, not a fresh guess.

- **H1** All three systems score below REAL on naturalness.
- **H2** The naturalness ranking follows the similarity ranking: qwen3 ≥ s2pro > voxtral.
- **H3** All three are detected above chance (fooling rate < 0.50).
- **H4** Raters who know the speaker detect the clones more often.

## Design, as built

13 usable texts × 4 conditions = 52 naturalness items, 39 2AFC pairs (13 per
system, all text × system cells). 6 rater plans (2 role groups × 3 model
rotations), drawn at random; per rater 24–28 items, 6–7 pairs, ~6.5 min of audio.
No rater meets the same text in both parts. Stimuli are ~10 s content-matched
excerpts, all put through identical processing (24 kHz → cut → −23 LUFS → common
pause floor). Order, A/B sides and two hidden repeats are randomized per rater;
the rater enters their own scores.

## Exclusions, fixed in advance

Technical failures only, never "sounds bad" — that is what raters judge. Cannot
reach target loudness without clipping; near-silent; clipped; excerpt not found
in the rendition (< 0.6 token match); excerpt stretched over > 2× the per-text
median span. `duration_off` / `silence_heavy` alone are **not** exclusions. If a
text is excluded for one system it is dropped for all. On this set: three voxtral
renditions, hence 3 of 16 texts. No rater exclusion rule.

## Analysis, fixed in advance

Mean of per-rater means; 95 % intervals from a two-way cluster bootstrap over
raters **and** texts; pairwise comparisons paired within rater with a
Bonferroni-adjusted verdict over the 6 comparisons; familiarity as a contrast
with its own interval. Intra-rater noise from the hidden repeats sets the floor
for what counts as a meaningful difference. Confirmatory mixed model if
`statsmodels` is available. No interim analysis decides when to stop.

Interval validity and power were checked by simulation on this design before any
data existed: coverage 0.99 against a nominal 0.95 (conservative), and a 0.30 MOS
difference is detected with 80 % power at 30 raters. Smaller differences will be
reported as *not detectable*, not as absent.

## Objective results, already fixed

| condition | SIM | WER |
|---|---|---|
| **REAL (anchor)** | **0.79** (0.67–0.86) | **0.18** |
| qwen3 | 0.743 | 0.079 |
| s2pro | 0.707 | 0.100 |
| voxtral | 0.600 | 0.112 |

To be stated regardless of how the listening test turns out: qwen3 and s2pro sit
inside the human-to-human similarity range, so that difference is not evidence
that one clone is closer to the person; the clones have a *lower* word error rate
than the human original; and the qwen3–s2pro WER difference is not detectable
(−0.019, adjusted CI [−0.051, +0.014]).

## Limitations to state in the report

One speaker (a case study, not a general result) · the enrollment confound, still
present in the numbers above · missing SIM lower bound · a WER floor larger than
the differences · 73 % of each render not rated subjectively (answered
objectively instead) · a residual brightness cue (0.885 vs a 0.750 baseline) ·
spontaneous vs. read speech · a non-blind operator. Details in
[docs/METHODS.md §8](docs/METHODS.md#8-limitations).

---
*Written before data collection. Change history is in git.*
