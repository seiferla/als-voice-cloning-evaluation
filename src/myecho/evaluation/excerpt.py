"""Choose one excerpt per text and locate it in every rendition.

The excerpt is picked on the text, then found in each rendition via word-level
ASR timestamps and cut at word boundaries. Same content per model, duration
free to differ. See docs/METHODS.md section 3.
"""

import difflib
import re

import numpy as np

from myecho.evaluation.metrics import normalize_text

WORDS_PER_SECOND = 2.94  # measured median across the three models

TARGET_S = 10.0
MIN_S = 8.0
MAX_S = 13.0

SKIP_HEAD_FRAC = 0.15  # the opening is a greeting plus onset artifacts

MIN_MATCH_RATIO = 0.6  # below this the rendition did not say the excerpt

# A matching block needs this many tokens before it may set the span start/end;
# a single stray "das" elsewhere in the transcript otherwise stretches the span.
MIN_ANCHOR_TOKENS = 3

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

# run_eval.py writes "<idx>_<text>_r1.wav" when --repeats > 1. Repeat 1 by rule.
FIRST_REPEAT_SUFFIX = "_r1"


def render_path(case_dir, model, index, text_id, ext="wav"):
    plain = case_dir / model / f"{index:02d}_{text_id}.{ext}"
    if plain.exists():
        return plain
    return case_dir / model / f"{index:02d}_{text_id}{FIRST_REPEAT_SUFFIX}.{ext}"


def _words(seconds):
    return round(seconds * WORDS_PER_SECOND)


def split_sentences(text):
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
                break
        consumed += counts[i]
    return best


def _from_commas(sentences, counts, start_at, lo, hi, target):
    """Fallback for a run-on sentence: split at commas and merge chunks."""
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
    """Deterministic excerpt choice. Returns {"text", "n_words", "boundary", "sentence_range"}.

    "boundary" is "sentence", "comma", or "word" (mid-sentence cut, flagged).
    """
    target, lo, hi = _words(target_s), _words(min_s), _words(max_s)
    sentences = split_sentences(text)
    counts = [len(s.split()) for s in sentences]
    start_at = skip_head_frac * sum(counts)

    for head in (start_at, 0):
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

    words = text.split()
    offset = min(int(start_at), max(0, len(words) - target))
    picked = words[offset:offset + target]
    return {"text": " ".join(picked), "n_words": len(picked),
            "boundary": "word", "sentence_range": None}


def _timed_tokens(word_times):
    """(normalized token, start, end) per ASR word, using the WER normalization."""
    out = []
    for w in word_times:
        for token in normalize_text(w["word"]).split():
            out.append((token, float(w["start"]), float(w["end"])))
    return out


def locate_span(word_times, excerpt_text, min_match=MIN_MATCH_RATIO):
    """Fuzzy-match the excerpt in one rendition. Returns {"start", "end", "match_ratio"} or None."""
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

    anchors = [b for b in blocks if b.size >= MIN_ANCHOR_TOKENS] or blocks
    first, last = anchors[0], anchors[-1]
    return {
        "start": timed[first.a][1],
        "end": timed[last.a + last.size - 1][2],
        "match_ratio": round(ratio, 3),
    }


def cut(x, sr, start, end, pad_before=0.20, pad_after=0.30, fade_s=0.02):
    """Cut [start, end] with padding for aligner error and short fades against clicks."""
    i0 = max(0, int(round((start - pad_before) * sr)))
    i1 = min(len(x), int(round((end + pad_after) * sr)))
    y = np.array(x[i0:i1], dtype=np.float32, copy=True)

    fade = min(int(fade_s * sr), len(y) // 2)
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        y[:fade] *= ramp
        y[-fade:] *= ramp[::-1]
    return y
