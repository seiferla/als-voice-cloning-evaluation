"""Audio-Laden: Mono, Zielabtastrate, float32 in [-1, 1]."""
from __future__ import annotations

from pathlib import Path
from typing import Union

import librosa
import numpy as np


def load_audio(path: Union[str, Path], sample_rate: int) -> np.ndarray:
    """Lädt eine Audiodatei als Mono-float32-Array mit der gewünschten Rate."""
    wav, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    return wav.astype("float32")


def duration_seconds(path: Union[str, Path]) -> float:
    """Dauer einer Audiodatei in Sekunden (ohne sie komplett zu laden)."""
    return float(librosa.get_duration(path=str(path)))
