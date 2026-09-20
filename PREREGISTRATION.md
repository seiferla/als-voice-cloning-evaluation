# Preregistration of the subjective evaluation

Hypotheses, design and analysis were fixed and committed before the first rater
answered; the commit timestamp is in `git log`. Reasoning and measured numbers:
[docs/METHODS.md](docs/METHODS.md).

| | |
|---|---|
| Speaker | one German speaker (ALS patient), 2021 voice memos |
| Systems | s2pro (Fish Audio), voxtral (Mistral), qwen3 (Alibaba DashScope) |
| Reference condition | REAL, the speaker's own recordings |
| Frozen at | commit and seed in `evaluation/results/listening_test/run_meta.json` |
| Planned raters | 30 |

## Data protection

The speaker's recordings and the cloned voices are used with her consent.
Raters give only age, headphone use and whether they know the speaker, and are
identified by a generated id. Audio, rater-level answers and voice IDs are not
part of this repository. Enrollment sent one 30 s recording to each of the three
providers under their respective terms.

## Hypotheses

Stated before any rating existed. The objective results were already known, so
H2 is a prediction across measurements.

- **H1** All three systems score below REAL on naturalness.
- **H2** The naturalness ranking follows the similarity ranking: qwen3 ≥ s2pro > voxtral.
- **H3** All three are detected above chance (fooling rate < 0.50).
- **H4** Raters who know the speaker detect the clones more often.

## Design

13 usable texts × 4 conditions = 52 naturalness items, 39 2AFC pairs (13 per
system, all text × system cells). 6 rater plans (2 role groups × 3 model
rotations), drawn at random; per rater 24–28 items, 6–7 pairs, ~6.5 min of audio.
No rater meets the same text in both parts. Stimuli are ~10 s content-matched
excerpts, all put through identical processing (24 kHz, cut, −23 LUFS, common
pause floor). Order, A/B sides and two hidden repeats are randomized per rater;
the rater enters their own scores.

## Exclusions

Technical failures only, never perceived quality: cannot reach target loudness
without clipping; near-silent; clipped; excerpt not found in the rendition
(< 0.6 token match); excerpt stretched over > 2× the per-text median span; no
measurable pause floor. If a text is excluded for one system it is dropped for
all. On this set: three voxtral renditions, hence 3 of 16 texts. No rater
exclusion rule.

## Analysis

Mean of per-rater means; 95 % intervals from a two-way cluster bootstrap over
raters and texts; pairwise comparisons paired within rater with a
Bonferroni-adjusted verdict over the 6 comparisons; familiarity as a contrast
with its own interval. Intra-rater noise from the hidden repeats sets the floor
for a meaningful difference. Confirmatory mixed model if `statsmodels` is
available. No interim analysis.

Interval validity and power were checked by simulation on this design before
data collection: coverage 0.99 against a nominal 0.95, and a 0.30 MOS difference
is detected with 80 % power at 30 raters. Smaller differences are reported as
not detectable, not as absent.

## Objective results known at the time

SIM ranking qwen3 > s2pro > voxtral, WER of all three below the REAL floor,
qwen3 vs. s2pro WER difference not detectable. Final values: METHODS §2.

## Outcome

Filled in after the study, see METHODS §8: H1 holds numerically for all three
but is statistically confirmed only for qwen3; H2 is reversed; H3 fails for
voxtral; H4 points the opposite way for all three systems but is not
significant at this sample size.
