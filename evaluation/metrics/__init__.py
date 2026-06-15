"""Einzelne Metrik-Implementierungen (lazy zu importieren).

- ``similarity``      : SIM (WavLM-SV Cosine-Similarity)
- ``intelligibility`` : WER (deutscher Whisper + jiwer)

Bewusst keine Eager-Imports: jedes Submodul zieht schwere Abhängigkeiten
(transformers/jiwer/librosa) und wird erst bei Bedarf importiert.
"""
