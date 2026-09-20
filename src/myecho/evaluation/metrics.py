"""SIM, WER and ASR helpers."""

import os
import re

import numpy as np

# Whisper's word-timestamp DTW runs numba with a second OpenMP runtime next to
# torch's, which aborts on macOS (OMP Error #15). One numba thread avoids it.
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_classifier = None
_whisper_model = None

WHISPER_MODEL = "small"


def similarity(ref_embedding, clone_embedding):
    """Cosine similarity between speaker embeddings."""
    ref = np.array(ref_embedding)
    clone = np.array(clone_embedding)
    return float(np.dot(ref, clone) / (np.linalg.norm(ref) * np.linalg.norm(clone)))


_PUNCT = str.maketrans({c: " " for c in ".,;:!?…\"'“”„‘’()[]{}<>/\\|*_+=&%$#@~`–—"})


def _spell_numbers_de(text):
    """Digits to German number words, on both sides of the WER ("2021", "3." -> ordinal)."""
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

    return re.sub(r"(\d+)(\.(?=\s))?", replace, text)


def normalize_text(text):
    """Lowercase, spell out numbers, transliterate umlauts, drop punctuation and hyphens."""
    text = text.lower()
    text = _spell_numbers_de(text)
    for src, dst in (("ß", "ss"), ("ä", "ae"), ("ö", "oe"), ("ü", "ue")):
        text = text.replace(src, dst)
    text = text.translate(_PUNCT)
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
        return 1.0  # jiwer raises on an empty hypothesis; a silent render is 100 % error
    return float(jiwer_wer(reference, hypothesis))


def extract_embedding(audio_path):
    """ECAPA-TDNN speaker embedding."""
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

    # soundfile instead of torchaudio.load(), which needs torchcodec/FFmpeg.
    data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
    signal = torch.from_numpy(data.T)
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
    result = _load_whisper().transcribe(str(audio_path), language=language)
    return result["text"]


def transcribe_words(audio_path, language="de"):
    """Word-level timestamps: [{"word", "start", "end"}, ...]."""
    result = _load_whisper().transcribe(
        str(audio_path), language=language, word_timestamps=True
    )
    return [
        {"word": w["word"].strip(), "start": float(w["start"]), "end": float(w["end"])}
        for segment in result["segments"]
        for w in segment.get("words", [])
    ]
