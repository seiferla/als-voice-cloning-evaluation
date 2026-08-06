"""Evaluation metrics for voice cloning."""

import numpy as np


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
