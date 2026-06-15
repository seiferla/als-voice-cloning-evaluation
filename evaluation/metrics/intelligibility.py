"""WER — Verständlichkeit via deutschem Whisper + jiwer.

Transkribiert das generierte Audio und vergleicht gegen den Ground-Truth-Text
(händisch korrigiertes WhisperX-Transkript).
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Union

import jiwer
import torch
from transformers import pipeline

from .. import config
from ..audio_utils import load_audio

# Umlaute/ß bleiben erhalten; Satzzeichen werden entfernt, Whitespace normiert.
_PUNCT = re.compile(r"[^\wäöüßÄÖÜ\s]", flags=re.UNICODE)
_WS = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    """Lowercase, Satzzeichen entfernen, Whitespace zusammenfassen."""
    s = s.lower().strip()
    s = _PUNCT.sub(" ", s)
    return _WS.sub(" ", s).strip()


def _pipeline_device():
    if config.DEVICE == "cuda":
        return 0
    return config.DEVICE  # "mps" / "cpu" werden von aktuellen transformers akzeptiert


@lru_cache(maxsize=1)
def _load_asr():
    dtype = torch.float16 if config.DEVICE == "cuda" else torch.float32
    return pipeline(
        "automatic-speech-recognition",
        model=config.ASR_MODEL_ID,
        torch_dtype=dtype,
        device=_pipeline_device(),
        chunk_length_s=30,
        generate_kwargs={"language": "german", "task": "transcribe"},
    )


def transcribe(path: Union[str, Path]) -> str:
    asr = _load_asr()
    wav = load_audio(path, config.ASR_SAMPLE_RATE)
    out = asr({"array": wav, "sampling_rate": config.ASR_SAMPLE_RATE})
    return out["text"].strip()


def score(reference_text: str, generated_path: Union[str, Path]) -> dict:
    """Transkribiert das Audio und berechnet die WER gegen den Ground-Truth-Text."""
    hypothesis = transcribe(generated_path)
    ref = normalize_text(reference_text)
    hyp = normalize_text(hypothesis)
    if not ref:
        return {"hypothesis": hypothesis, "wer": float("nan")}
    return {
        "hypothesis": hypothesis,
        "wer": jiwer.wer(ref, hyp),
    }
