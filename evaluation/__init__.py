"""myEcho — Evaluations-Framework für Voice-Cloning-Modelle.

Bewertet generierte Audios von Qwen3, Fish-Speech-S2 und Voxtral entlang der
objektiven Metriken:
  - SIM : WavLM-SV Cosine-Similarity
  - WER : Word Error Rate (deutscher Whisper)

(Subjektive Metriken N-MOS/SMOS/Turing und Performance RTF/TTFA folgen später.)

Die schweren Metrik-Module (transformers/jiwer/librosa) werden bewusst NICHT
beim Paket-Import geladen — so funktioniert ``from evaluation import config``
auch ohne die volle Modell-Toolchain. Zum Scoren:
``from evaluation.run_eval import run``.
"""
from . import config  # noqa: F401

__all__ = ["config"]
