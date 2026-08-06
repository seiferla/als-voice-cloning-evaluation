"""Evaluation metrics for voice cloning."""

import numpy as np

_classifier = None
_whisper_model = None


def similarity(ref_embedding, clone_embedding):
    """Cosine similarity between speaker embeddings."""
    ref = np.array(ref_embedding)
    clone = np.array(clone_embedding)
    return float(np.dot(ref, clone) / (np.linalg.norm(ref) * np.linalg.norm(clone)))


def wer(reference_text, asr_output):
    """Word error rate between texts."""
    try:
        from jiwer import wer as jiwer_wer
        return float(jiwer_wer(reference_text.lower(), asr_output.lower()))
    except ImportError:
        raise ImportError("jiwer not installed: uv sync --extra eval")


def extract_embedding(audio_path):
    """Speaker embedding (ECAPA-TDNN) for SIM comparison."""
    global _classifier
    try:
        import torchaudio
        from speechbrain.inference import EncoderClassifier
    except ImportError:
        raise ImportError("speechbrain/torchaudio not installed: uv sync --extra eval")

    if _classifier is None:
        _classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

    signal, sr = torchaudio.load(str(audio_path))
    if sr != 16000:
        signal = torchaudio.functional.resample(signal, sr, 16000)

    embedding = _classifier.encode_batch(signal)
    return embedding.squeeze().detach().cpu().numpy()


def transcribe(audio_path, language="de"):
    """Transcribe audio with Whisper (produces the ASR output for wer())."""
    global _whisper_model
    try:
        import whisper
    except ImportError:
        raise ImportError("openai-whisper not installed: uv sync --extra eval")

    if _whisper_model is None:
        _whisper_model = whisper.load_model("small")

    result = _whisper_model.transcribe(str(audio_path), language=language)
    return result["text"]
