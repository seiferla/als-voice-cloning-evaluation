"""Choosing one excerpt per evaluation text and locating it in every rendition.

The excerpt is picked on the TEXT, before any audio is touched, so the choice
cannot favour a model. It is then found in each rendition by word-level ASR
timestamps and cut at word boundaries, which keeps the content identical across
models and lets the duration differ - speaking rate is a property of the voice.

A fixed time window is not used: the models differ in speaking rate by up to
1.53x on the same text, so a fixed window would hold different content per model
and cut mid-word more often for the slower one.

Excerpts are whole sentences where possible. These transcripts are ASR output of
spontaneous speech, so sentences are short and occasionally very long; runs of
sentences are merged up to the target and an over-long one is split at commas.

Details: docs/METHODS.md section 3.
"""

import difflib
import re

import numpy as np

from myecho.evaluation.metrics import normalize_text

# Measured median across the three models on this material (words / second).
# Used only to express the target excerpt length in seconds instead of words.
WORDS_PER_SECOND = 2.94

TARGET_S = 10.0
MIN_S = 8.0
MAX_S = 13.0

# The opening of a voice memo is a greeting, and the first seconds of a render
# carry onset artifacts that differ per model. Skip that head before choosing.
SKIP_HEAD_FRAC = 0.15

# Below this share of the excerpt's words matched in the audio, the rendition
# did not actually say the excerpt (skipped or garbled it) - report, never cut.
MIN_MATCH_RATIO = 0.6

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _words(seconds):
    return round(seconds * WORDS_PER_SECOND)


def split_sentences(text):
    """Split on sentence-final punctuation. ASR output rarely contains
    abbreviations like "z.B.", so the simple rule is enough here."""
    return [s.strip() for s in _SENTENCE_END.split(text.strip()) if s.strip()]


def _from_sentences(sentences, counts, start_at, lo, hi, target):
    """Best run of consecutive whole sentences starting at or after start_at."""
    best = None
    consumed = 0
    for i in range(len(sentences)):
        if consumed >= start_at:
            n = 0
            for j in range(i, len(sentences)):
                n += counts[j]
                if n < lo:
                    continue
                if n <= hi:
                    score = abs(n - target)
                    if best is None or score < best[0]:
                        best = (score, i, j, n)
                break  # first run reaching lo decides this start index
        consumed += counts[i]
    return best


def _from_commas(sentences, counts, start_at, lo, hi, target):
    """Fallback for a run-on sentence longer than the whole target window:
    split it at commas and merge those chunks instead."""
    consumed = 0
    for i, sentence in enumerate(sentences):
        if consumed >= start_at and counts[i] > hi:
            chunks = [c.strip() for c in sentence.split(",") if c.strip()]
            n = 0
            picked = []
            for chunk in chunks:
                picked.append(chunk)
                n += len(chunk.split())
                if n >= lo:
                    break
            if lo <= n <= hi:
                return ", ".join(picked), n, i
        consumed += counts[i]
    return None


def select_excerpt(text, target_s=TARGET_S, min_s=MIN_S, max_s=MAX_S,
                   skip_head_frac=SKIP_HEAD_FRAC):
    """Choose the excerpt for one text. Deterministic - no randomness, so the
    same text always yields the same excerpt and the choice is reproducible.

    Returns {"text", "n_words", "boundary", "sentence_range"}. "boundary" says
    where the excerpt ends: a sentence end is ideal, a comma acceptable, and
    "word" means the text offered neither and the cut is mid-sentence.
    """
    target, lo, hi = _words(target_s), _words(min_s), _words(max_s)
    sentences = split_sentences(text)
    counts = [len(s.split()) for s in sentences]
    start_at = skip_head_frac * sum(counts)

    for head in (start_at, 0):  # retry from the top if the head skip left nothing
        best = _from_sentences(sentences, counts, head, lo, hi, target)
        if best:
            _, i, j, n = best
            return {"text": " ".join(sentences[i:j + 1]), "n_words": n,
                    "boundary": "sentence", "sentence_range": [i, j]}

        commas = _from_commas(sentences, counts, head, lo, hi, target)
        if commas:
            chunk, n, i = commas
            return {"text": chunk, "n_words": n,
                    "boundary": "comma", "sentence_range": [i, i]}

    # Nothing splits cleanly (a long sentence without commas): take the target
    # number of words from the head-skipped position and flag the ragged end.
    words = text.split()
    offset = min(int(start_at), max(0, len(words) - target))
    picked = words[offset:offset + target]
    return {"text": " ".join(picked), "n_words": len(picked),
            "boundary": "word", "sentence_range": None}


def _timed_tokens(word_times):
    """Flatten ASR words into (normalized token, start, end).

    normalize_text is the same normalization the WER uses, so a word it expands
    into several tokens ("2021") contributes each token with that word's timing.
    """
    out = []
    for w in word_times:
        for token in normalize_text(w["word"]).split():
            out.append((token, float(w["start"]), float(w["end"])))
    return out


def locate_span(word_times, excerpt_text, min_match=MIN_MATCH_RATIO):
    """Find the excerpt's time span inside one rendition.

    Matching is fuzzy on purpose: the ASR of a synthesized file will not
    reproduce the text exactly, so an exact search would fail on good audio.
    difflib finds the longest common subsequence of tokens, and the first and
    last matched token give the span.

    Returns {"start", "end", "match_ratio"}, or None when too little of the
    excerpt was found - which means the model did not say it, and is a finding
    rather than something to paper over.
    """
    timed = _timed_tokens(word_times)
    if not timed:
        return None

    hypothesis = [t for t, _, _ in timed]
    target = normalize_text(excerpt_text).split()
    if not target:
        return None

    matcher = difflib.SequenceMatcher(None, hypothesis, target, autojunk=False)
    blocks = [b for b in matcher.get_matching_blocks() if b.size > 0]
    if not blocks:
        return None

    ratio = sum(b.size for b in blocks) / len(target)
    if ratio < min_match:
        return None

    first, last = blocks[0], blocks[-1]
    return {
        "start": timed[first.a][1],
        "end": timed[last.a + last.size - 1][2],
        "match_ratio": round(ratio, 3),
    }


def cut(x, sr, start, end, pad_before=0.20, pad_after=0.30, fade_s=0.02):
    """Cut [start, end] out of a signal, with padding and short fades.

    The padding absorbs the timestamp error of the aligner (typically well
    under 200 ms) and keeps the natural release of the last word. The fades
    remove the click of a hard cut - a click is heard as an artifact and would
    be rated as one.
    """
    i0 = max(0, int(round((start - pad_before) * sr)))
    i1 = min(len(x), int(round((end + pad_after) * sr)))
    y = np.array(x[i0:i1], dtype=np.float32, copy=True)

    fade = min(int(fade_s * sr), len(y) // 2)
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        y[:fade] *= ramp
        y[-fade:] *= ramp[::-1]
    return y
