"""Evaluation metrics for voice cloning."""

import os
import re

import numpy as np

# Whisper's word_timestamps path runs its DTW through numba with parallel=True,
# which starts a second OpenMP runtime next to the one torch already loaded. On
# macOS with a conda-based interpreter that combination aborts the process
# (OMP Error #15) or segfaults mid-inference. Pinning numba to one thread keeps
# the alignment on a single runtime; the DTW is a negligible share of the ASR
# cost, so nothing measurable is lost. setdefault, so an explicit value wins.
# Must run before torch/numba are imported, i.e. at import time of this module.
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_classifier = None
_whisper_model = None

# One place to change the ASR size. "small" is the current baseline; the EDA
# notebook uses a German-finetuned large model, which is the better choice once
# the WER numbers are re-run.
WHISPER_MODEL = "small"


def similarity(ref_embedding, clone_embedding):
    """Cosine similarity between speaker embeddings."""
    ref = np.array(ref_embedding)
    clone = np.array(clone_embedding)
    return float(np.dot(ref, clone) / (np.linalg.norm(ref) * np.linalg.norm(clone)))


# Punctuation that Whisper adds or omits freely - never a real recognition error.
_PUNCT = str.maketrans({c: " " for c in ".,;:!?\u2026\"'\u201c\u201d\u201e\u2018\u2019()[]{}<>/\\|*_+=&%$#@~`\u2013\u2014"})


def _spell_numbers_de(text):
    """Replace digit groups with German number words.

    "2021" -> "zweitausendeinundzwanzig", "3." -> "dritte". Whisper writes
    numbers sometimes as digits, sometimes as words; applying the same
    expansion to BOTH sides removes that source of fake errors.

    Residual limitation: German ordinals inflect ("am dritten Mai"), while
    num2words returns the base form ("dritte") - that leaves one substitution.
    Best avoided by writing numbers as words in the test texts.

    No-op if num2words is missing.
    """
    try:
        from num2words import num2words
    except ImportError:
        return text

    def replace(match):
        number = int(match.group(1))
        is_ordinal = bool(match.group(2))
        try:
            return num2words(number, lang="de", to="ordinal" if is_ordinal else "cardinal")
        except Exception:
            return match.group(0)

    # A dot directly after digits and followed by a space marks an ordinal
    # ("am 3. Mai"); a dot at the end of a sentence does not.
    return re.sub(r"(\d+)(\.(?=\s))?", replace, text)


def normalize_text(text):
    """Normalize text for WER: casing, punctuation, umlauts, numbers, whitespace.

    Without this, WER measures the ASR's formatting choices instead of the
    TTS model's intelligibility.
    """
    text = text.lower()
    text = _spell_numbers_de(text)
    # Umlauts and sharp s: transliterate on both sides so an ASR that writes
    # "schoen" instead of "schön" does not count as a recognition error.
    for src, dst in (("\u00df", "ss"), ("\u00e4", "ae"), ("\u00f6", "oe"), ("\u00fc", "ue")):
        text = text.replace(src, dst)
    text = text.translate(_PUNCT)
    # Hyphens join compounds; treat them as word boundaries on both sides.
    text = text.replace("-", " ")
    return " ".join(text.split())


def wer(reference_text, asr_output):
    """Word error rate between texts (both normalized first)."""
    try:
        from jiwer import wer as jiwer_wer
    except ImportError:
        raise ImportError("jiwer not installed: uv sync --extra eval")

    reference = normalize_text(reference_text)
    hypothesis = normalize_text(asr_output)
    if not reference:
        raise ValueError("Reference text is empty after normalization")
    if not hypothesis:
        # jiwer would raise on an empty hypothesis; a silent clone is 100% error.
        return 1.0
    return float(jiwer_wer(reference, hypothesis))


def extract_embedding(audio_path):
    """Speaker embedding (ECAPA-TDNN) for SIM comparison."""
    global _classifier
    try:

        import soundfile as sf
        import torch
        import torchaudio
        from speechbrain.inference import EncoderClassifier
    except ImportError:
        raise ImportError("speechbrain/torchaudio not installed: uv sync --extra eval")

    if _classifier is None:
        _classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

    # soundfile instead of torchaudio.load(): recent torchaudio versions need
    # torchcodec/FFmpeg just to read a file, which isn't installed here.
    data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
    signal = torch.from_numpy(data.T)  # (channels, samples)
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)
    if sr != 16000:
        signal = torchaudio.functional.resample(signal, sr, 16000)

    embedding = _classifier.encode_batch(signal)
    return embedding.squeeze().detach().cpu().numpy()


def _load_whisper():
    global _whisper_model
    try:
        import whisper
    except ImportError:
        raise ImportError("openai-whisper not installed: uv sync --extra eval")

    if _whisper_model is None:
        _whisper_model = whisper.load_model(WHISPER_MODEL)
    return _whisper_model


def transcribe(audio_path, language="de"):
    """Transcribe audio with Whisper (produces the ASR output for wer())."""
    result = _load_whisper().transcribe(str(audio_path), language=language)
    return result["text"]


def transcribe_words(audio_path, language="de"):
    """Word-level timestamps: [{"word", "start", "end"}, ...].

    Used to locate an excerpt inside a rendition (see excerpt.py) so stimuli
    can be cut at word boundaries instead of at a fixed time offset.
    """
    result = _load_whisper().transcribe(
        str(audio_path), language=language, word_timestamps=True
    )
    return [
        {"word": w["word"].strip(), "start": float(w["start"]), "end": float(w["end"])}
        for segment in result["segments"]
        for w in segment.get("words", [])
    ]
