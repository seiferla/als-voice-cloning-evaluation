"""SIM — Speaker-Encoder-Cosine-Similarity via WavLM-SV.

Bettet Referenz- und generiertes Audio mit ``microsoft/wavlm-base-plus-sv``
ein und berechnet die Cosine-Similarity. Höher = sprecherähnlicher.

Dieses WavLM-SV-Modell ist der De-facto-Standard für die SIM-Metrik in
aktuellen Zero-Shot-TTS/Voice-Cloning-Papern (z.B. VALL-E).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Union

import torch
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, WavLMForXVector

from .. import config
from ..audio_utils import load_audio


@lru_cache(maxsize=1)
def _load_model():
    extractor = AutoFeatureExtractor.from_pretrained(config.SV_MODEL_ID)
    model = WavLMForXVector.from_pretrained(config.SV_MODEL_ID)
    model = model.to(config.DEVICE).eval()
    return extractor, model


@lru_cache(maxsize=512)
def _embed_cached(path_str: str) -> torch.Tensor:
    """L2-normiertes Sprecher-Embedding (gecacht — Referenzen werden mehrfach genutzt)."""
    extractor, model = _load_model()
    wav = load_audio(path_str, config.SV_SAMPLE_RATE)
    inputs = extractor(
        [wav], sampling_rate=config.SV_SAMPLE_RATE, return_tensors="pt", padding=True
    )
    inputs = {k: v.to(config.DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        emb = model(**inputs).embeddings
    return F.normalize(emb, dim=-1).squeeze(0).cpu()


def embed(path: Union[str, Path]) -> torch.Tensor:
    return _embed_cached(str(path))


def similarity(reference_path: Union[str, Path], generated_path: Union[str, Path]) -> float:
    """Cosine-Similarity zwischen Referenz- und generiertem Audio (Bereich ~[-1, 1])."""
    ref = embed(reference_path)
    gen = embed(generated_path)
    return float(torch.dot(ref, gen))  # beide L2-normiert -> Dot = Cosine
