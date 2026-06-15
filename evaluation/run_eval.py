"""Offline-Scoring: bewertet alle bereits generierten Audios und schreibt results.csv.

Berechnet pro (id × Modell):
  - SIM : WavLM-SV Cosine-Similarity zur Referenz
  - WER : Word Error Rate (deutscher Whisper vs. Ground-Truth-Text)

Usage:
    python -m evaluation.run_eval                  # alle Modelle
    python -m evaluation.run_eval --models qwen3 voxtral
"""
from __future__ import annotations

import argparse
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from . import config
from .dataset import iter_samples
from .metrics import intelligibility, similarity


def run(
    models: Optional[Sequence[str]] = None,
    progress: bool = True,
) -> pd.DataFrame:
    """Bewertet alle vorhandenen Samples und speichert results.csv."""
    samples = list(iter_samples(models=models))
    if not samples:
        raise RuntimeError(
            "Keine Samples gefunden. Liegen Audios unter "
            f"{config.GENERATED_DIR}/<modell>/<id>.wav und ist metadata.csv gepflegt?"
        )

    rows = []
    for i, s in enumerate(samples, 1):
        if progress:
            print(f"[{i}/{len(samples)}] {s.model} :: {s.id}")
        row = {"id": s.id, "model": s.model, "text": s.text}

        # SIM
        row["sim"] = (
            similarity.similarity(s.reference_path, s.generated_path)
            if s.reference_path.exists()
            else float("nan")
        )

        # WER
        intel = intelligibility.score(s.text, s.generated_path)
        row["wer"] = intel["wer"]
        row["hypothesis"] = intel["hypothesis"]

        rows.append(row)

    df = pd.DataFrame(rows)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.RESULTS_DIR / "results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nGespeichert: {out_path}  ({len(df)} Zeilen)")
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Mittelwert, 95%-Konfidenzintervall und n je Modell und Metrik."""
    metrics = [c for c in ("sim", "wer") if c in df.columns]

    def ci95(series: pd.Series) -> float:
        x = series.dropna()
        if len(x) < 2:
            return float("nan")
        return 1.96 * x.std(ddof=1) / np.sqrt(len(x))

    out = {}
    for m in metrics:
        grp = df.groupby("model")[m]
        out[(m, "mean")] = grp.mean()
        out[(m, "ci95")] = grp.apply(ci95)
        out[(m, "n")] = grp.count()
    result = pd.DataFrame(out)
    result.columns = pd.MultiIndex.from_tuples(result.columns)
    return result


def _main() -> None:
    parser = argparse.ArgumentParser(description="Offline-Scoring der generierten Audios.")
    parser.add_argument("--models", nargs="*", default=None, help="Teilmenge der Modelle.")
    args = parser.parse_args()

    df = run(models=args.models)
    print("\n=== Aggregat pro Modell (mean | 95%-CI | n) ===")
    with pd.option_context("display.width", 120):
        print(aggregate(df))


if __name__ == "__main__":
    _main()
