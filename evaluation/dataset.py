"""Daten-Pairing: verbindet Referenz-Audio, generierte Audios und Ground-Truth-Text.

Erwartete Verzeichnisstruktur (siehe evaluation/README.md):

    evaluation/data/
    ├── metadata.csv               # Spalten: id, text
    ├── reference/<id>.wav         # sauberes Original der Zielsprecherin (für SIM)
    └── generated/
        ├── qwen3/<id>.wav
        ├── fish_speech_s2/<id>.wav
        └── voxtral/<id>.wav

Der Ground-Truth-Text (Spalte `text`) ist das händisch korrigierte
WhisperX-Transkript des Original-Audios. Da die Modelle denselben Text
nachgesprochen haben (Resynthese), wird die WER gegen diesen Text berechnet.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence

import pandas as pd

from . import config


@dataclass
class Sample:
    """Ein zu bewertendes generiertes Audio inkl. Referenz und Ground-Truth-Text."""

    id: str
    model: str
    text: str               # Ground-Truth (händisch korrigierter WhisperX-Text)
    generated_path: Path    # zu bewertendes Audio
    reference_path: Path    # Referenz der Zielsprecherin (für SIM)


def load_metadata(path: Path = config.METADATA_CSV) -> pd.DataFrame:
    """Lädt metadata.csv und prüft die Pflichtspalten."""
    if not path.exists():
        raise FileNotFoundError(
            f"metadata.csv nicht gefunden: {path}\n"
            "Lege sie mit den Spalten 'id,text' an (siehe evaluation/README.md)."
        )
    df = pd.read_csv(path)
    missing = {"id", "text"} - set(df.columns)
    if missing:
        raise ValueError(f"metadata.csv fehlen Spalten: {missing}")
    df["id"] = df["id"].astype(str)
    return df


def iter_samples(
    models: Optional[Sequence[str]] = None,
    metadata: Optional[pd.DataFrame] = None,
) -> Iterator[Sample]:
    """Iteriert über alle (id × Modell)-Kombinationen, für die ein Audio existiert."""
    models = list(models) if models is not None else config.MODELS
    df = metadata if metadata is not None else load_metadata()
    for _, row in df.iterrows():
        ref = config.REFERENCE_DIR / f"{row['id']}.wav"
        for model in models:
            gen = config.GENERATED_DIR / model / f"{row['id']}.wav"
            if not gen.exists():
                continue  # dieses Modell hat das Sample nicht erzeugt
            yield Sample(
                id=row["id"],
                model=model,
                text=row["text"],
                generated_path=gen,
                reference_path=ref,
            )


def coverage_report(
    models: Optional[Sequence[str]] = None,
    metadata: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Übersicht, welche (id × Modell) vorhanden sind und ob die Referenz existiert.

    Praktisch, um vor dem Lauf fehlende Dateien zu finden.
    """
    models = list(models) if models is not None else config.MODELS
    df = metadata if metadata is not None else load_metadata()
    rows = []
    for _, row in df.iterrows():
        _id = row["id"]
        ref_ok = (config.REFERENCE_DIR / f"{_id}.wav").exists()
        entry = {"id": _id, "reference": ref_ok}
        for model in models:
            entry[model] = (config.GENERATED_DIR / model / f"{_id}.wav").exists()
        rows.append(entry)
    return pd.DataFrame(rows)
