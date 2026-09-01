# Preregistration — subjective evaluation of three voice-cloning services

**Status: DRAFT. Read it, change what you disagree with, then commit it before
the first rater.** The point of this file is the commit timestamp: it separates
"I measured something and then decided what counts" from "I decided what counts
and then measured". Everything below is already implemented in the repo, so this
is a description of a frozen design, not a plan.

Two sections are marked **TODO (only you can answer)**. Fill them in before you
collect data.

| | |
|---|---|
| Study | myEcho — voice cloning for an ALS patient |
| Speaker | one German speaker, from 2021 voice memos |
| Systems | s2pro (Fish Audio), voxtral (Mistral), qwen3 (Alibaba DashScope) |
| Reference condition | REAL — the speaker's own recordings |
| Stimulus set frozen at | commit + seed recorded in `evaluation/results/listening_test/run_meta.json` |
| Planned raters | 30 |

---

## 1 · TODO (only you can answer): consent and data protection

The evaluation uses the recorded voice of a real, identifiable person, and the
listening test plays that voice to about 30 people. Before the first session,
write down here:

- who consented, to what exactly (cloning, playback to third parties,
  publication of the report, keeping the audio after the project), in what form,
  and when
- whether the person is able to consent themselves, and if not, who did
- what happens to the recordings, the cloned voices and the collected ratings at
  the end of the project
- whether the three API providers keep the uploaded reference audio, and for how
  long — voice enrollment sends the speaker's voice to three companies

Raters are told what they are listening to (a naturalness test and a
real-vs-synthetic test), give age and whether they know the speaker, and are
identified only by a generated id (`R-XXXXXX`). No names are collected.

## 2 · TODO (only you can answer): the SIM lower bound

`SIM` is a cosine similarity between ECAPA speaker embeddings. The upper bound is
measured (see §5); the lower bound needs recordings of a **different** speaker,
which this repo does not contain. Pick a corpus (e.g. Common Voice DE), put a
handful of clips somewhere under `data/`, and set `"impostor_audio"` in
`evaluation/testset.json`. Note the corpus and its licence here.

Until this is done, absolute SIM values cannot be calibrated and the report must
say so.

---

## 3 · Hypotheses

Directional, stated before the ratings exist.

- **H1 (naturalness).** All three systems score below the REAL condition on
  naturalness (ACR 1–5).
- **H2 (ranking).** The naturalness ranking follows the objective similarity
  ranking already measured on the same material: qwen3 ≥ s2pro > voxtral.
- **H3 (detectability).** In the 2AFC test, all three systems are detected above
  chance, i.e. the fooling rate is below 0.50.
- **H4 (familiarity).** Raters who know the speaker personally detect the clones
  more often than raters who do not.

H2 is the one that can fail cleanly: naturalness and speaker similarity are
different things, and a system can sound pleasant without sounding like the
target.

## 4 · Design, as built

**Stimuli.** ~10 s excerpts, not full renders. One excerpt is chosen per text
**on the text**, before any audio is touched, so the choice cannot favour a
system; it is then located in every rendition by word-level ASR timestamps and
cut at word boundaries. Content is identical across systems; duration is free to
differ, because speaking rate is a property of the voice. A fixed time window was
rejected: the systems differ in speaking rate by up to 1.53× on the same text, so
"the first 10 seconds" would compare different amounts of content and cut
mid-word more often for the slower system.

**Processing, identical for every condition.** Resample to 24 kHz → cut →
loudness-normalize to −23 LUFS (BS.1770) → match the pause floor to speech
−26 dB by adding 1/f noise. The last step exists because the real recordings are
phone voice memos put through DeepFilterNet while the clones are generated clean:
before matching, a single threshold on the pause fraction or the pause noise
floor separated real from clone with 96 % accuracy, so the 2AFC test could have
been won without judging a voice at all.

**Counts** (13 usable texts × 4 conditions):

| | |
|---|---|
| Naturalness items | 52 (13 per condition, incl. REAL) |
| 2AFC pairs | 39 (13 per system, all text × system cells) |
| Per rater | 24–28 naturalness items, 6–7 pairs, ~6.5 min of audio |
| Rater plans | 6 (2 role groups × 3 model rotations), drawn at random |

**Per rater**, the page fixes: the plan, the item order, the A/B side of every
pair, and two hidden repeats. No rater meets the same text in both parts — with
content-matched excerpts and REAL as a condition, part 1 would otherwise hand
them the answer to part 2. Part 2 stays locked until part 1 is complete.

**The rater enters their own scores.** Naturalness uses the five verbal ACR
labels of ITU-T P.800, not only labelled endpoints. Three unscored training items
precede the scored ones.

## 5 · Objective measurements, already collected

Fixed before the listening test, from `results_latest.csv` (16 texts, 1 render
per text — the reported run will use `--repeats 3`):

| condition | SIM (median) | WER (median) |
|---|---|---|
| REAL (anchor) | **0.79** (0.67–0.86) | **0.18** |
| qwen3 | 0.746 | 0.091 |
| s2pro | 0.702 | 0.110 |
| voxtral | 0.599 | 0.209 |

Two things follow and will be stated in the report regardless of how the
listening test turns out:

- qwen3 and s2pro fall **inside** the range this metric returns for two
  recordings of the same person, so their SIM values are at the ceiling of what
  it can resolve. Differences between them are not interpretable as similarity
  differences.
- The clones have a **lower** word error rate than the human original. WER here
  measures how close the output is to read speech, not how intelligible it is.

## 6 · Exclusions, fixed in advance

Only technical failures, never "this sounds bad" — that is what the raters
judge. Screened by `qc_samples.py` and `select_excerpts.py` before the test is
built, recorded in `exclusions.json` and `key.json`:

- cannot reach the target loudness without clipping (`quiet`)
- essentially no speech (`near_silent`), or already clipped (`clipping`)
- the excerpt is not found in the rendition (< 0.6 token match) — the system did
  not say the passage
- the excerpt is found but stretched over more than 2× the per-text median span —
  a runaway render

`duration_off` and `silence_heavy` alone are **not** exclusions: they usually
mean the system speaks slower or pauses more, which raters should hear.

If a text is excluded for one system it is dropped for **all** systems, so every
condition is scored on identical material. Excluded on this set: three voxtral
renditions, hence three of sixteen texts, leaving 13. Two of the three were
independently flagged by both the level screening and the alignment.

No rater exclusion rule is planned. Sessions with missing answers are analysed
as they are; the mixed/bootstrap analysis handles unbalanced data.

## 7 · Analysis, fixed in advance

- **Naturalness.** Mean of per-rater means per condition. 95 % intervals from a
  two-way cluster bootstrap over raters **and** texts. Pairwise comparisons are
  paired within rater, which removes each rater's scale offset.
- **2AFC.** Fooling rate per system, same bootstrap, compared against 0.50.
  Reported for all raters and split by whether the rater knows the speaker.
- **Confirmatory.** `rating ~ condition` with random intercepts for rater and
  text; `fooled ~ model` as a rater-clustered logit. The bootstrap is the
  headline number at this sample size.
- **Rating noise.** Mean absolute difference between the two ratings of the
  hidden repeats. A condition difference smaller than this is inside one rater's
  own scale noise and will be reported as such.
- No interim analysis decides when to stop. 30 raters, or as many as can be
  recruited, and the number is reported either way.

## 8 · Known limitations, to be stated in the report

1. **One speaker.** Every interval generalizes over texts and raters, not over
   speakers. Nothing here says how these systems behave on a different voice.
2. **The enrollment was not controlled.** qwen3 received server-side
   preprocessing (`enable_preprocess=True`), voxtral raw bytes plus a hardcoded
   `gender` hint, s2pro was uploaded through a web UI. **The three systems did
   not get the same input**, which is an alternative explanation for the SIM
   ranking. Re-enrolling all three from one identical, identically preprocessed
   file is the single most valuable remaining fix.
3. **A residual brightness cue.** After matching level and pause floor, one
   feature still separates real from clone above the guessing baseline: energy in
   4–8 kHz (0.885 vs a 0.750 baseline, `cue_audit.csv`). All three systems render
   duller than the phone-recorded original. This was deliberately **not**
   equalized: it is partly the recording chain and partly a real failure of the
   clones to reproduce the source timbre, and EQ-ing it would also erase what
   raters are supposed to hear. The 2AFC results must be read together with this
   number.
4. **Spontaneous vs read speech.** The real side is spontaneous speech with
   hesitations; the clones read a transcript of it. Disfluencies are in the text,
   but the delivery differs, and that is part of what makes a clone detectable.
5. **WER via whisper-small.** A German-finetuned model would be better; if it is
   swapped in, every WER number including the floor must be recomputed together.
6. **The operator knows the study.** Sessions are run by the author. The rater
   clicks, the key is not opened beforehand, and the A/B sides are re-flipped per
   rater, but this is not a blind experimenter.

---

*Written before data collection. Change history is in git.*
