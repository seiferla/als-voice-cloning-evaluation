"""Zentrale Konfiguration für das Evaluations-Framework.

Pfade, Modell-IDs und Geräte-Auswahl an einer Stelle, damit Notebook und CLI
dieselben Einstellungen teilen.
"""
from __future__ import annotations

from pathlib import Path

import torch

# evaluation/ liegt eine Ebene unter dem Projekt-Root
EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent

# --- Pfade (Audio bleibt lokal/gitignored, siehe evaluation/README.md) -------
DATA_DIR = EVAL_DIR / "data"
REFERENCE_DIR = DATA_DIR / "reference"          # reference/<id>.wav
GENERATED_DIR = DATA_DIR / "generated"          # generated/<model>/<id>.wav
METADATA_CSV = DATA_DIR / "metadata.csv"        # Spalten: id, text
RESULTS_DIR = EVAL_DIR / "results"              # results.csv etc.

# --- Evaluierte Modelle = Unterordner-Namen in data/generated/ ---------------
MODELS = ["qwen3", "fish_speech_s2", "voxtral"]

# --- Modell-IDs --------------------------------------------------------------
ASR_MODEL_ID = "primeline/whisper-large-v3-german"   # WER (deutsch)
SV_MODEL_ID = "microsoft/wavlm-base-plus-sv"          # SIM (Speaker-Verification)

# --- Zielabtastraten ---------------------------------------------------------
SV_SAMPLE_RATE = 16_000
ASR_SAMPLE_RATE = 16_000


def get_device() -> str:
    """Bestes verfügbares Gerät: CUDA > MPS (Apple) > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


DEVICE = get_device()
