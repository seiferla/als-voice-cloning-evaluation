# Evaluations-Framework

Bewertet die von **Qwen3**, **Fish-Speech-S2** und **Voxtral** generierten Audios
gegen das Original der Zielsprecherin.

> Erster Schritt: nur die objektiven Metriken **SIM** und **WER**.
> Subjektiv (N-MOS/SMOS/Turing) und Performance (RTF/TTFA) folgen in einem späteren Schritt.

## Metriken

| Achse | Metrik | Modul | Wie |
|---|---|---|---|
| Objektiv | **SIM** | `metrics/similarity.py` | Cosine-Similarity der WavLM-SV-Embeddings (Referenz ↔ generiert) |
| Objektiv | **WER** | `metrics/intelligibility.py` | dt. Whisper transkribiert → jiwer vs. Ground-Truth-Text |

## Verzeichnis-Konvention

```
evaluation/data/                      (gitignored — Audio & Transkripte bleiben lokal)
├── metadata.csv                      # Spalten: id, text  (händisch korrigiertes Transkript)
├── reference/
│   └── <id>.wav                      # sauberes Original der Zielsprecherin (für SIM)
└── generated/
    ├── qwen3/<id>.wav
    ├── fish_speech_s2/<id>.wav
    └── voxtral/<id>.wav
```

- **`<id>`** ist der gemeinsame Schlüssel. Gleiches `<id>.wav` in `reference/` und in jedem `generated/<modell>/` gehört zusammen.
- **`text`** in `metadata.csv` = der Text, den die Modelle nachgesprochen haben (Resynthese). Die WER wird dagegen berechnet.
- Fehlt für eine `id` das Audio eines Modells, wird dieses Paar einfach übersprungen.

## Setup

```bash
pip install -r evaluation/requirements-eval.txt
```

## Nutzung

**Offline-Scoring (SIM, WER):**
```bash
python -m evaluation.run_eval                 # alle Modelle
python -m evaluation.run_eval --models qwen3 voxtral
```
→ schreibt `evaluation/results/results.csv` (pro-Utterance) und druckt das Aggregat
(Mittelwert ± 95 %-CI je Modell).

**Aus Python / Notebook:**
```python
from evaluation.run_eval import run, aggregate
df = run()
aggregate(df)
```

Siehe `notebooks/evaluation.ipynb` für Scoring + Visualisierung.

## Methodik-Notizen (für die Thesis)

- **SIM-Modell:** `microsoft/wavlm-base-plus-sv` — De-facto-Standard in aktuellen
  Zero-Shot-TTS/Voice-Cloning-Papern.
- **ASR für WER:** `primeline/whisper-large-v3-german`.
- **Text-Normalisierung:** Kleinschreibung, Satzzeichen entfernt, Whitespace normiert,
  Umlaute/ß erhalten (relevant für Deutsch).
- **Statistik:** Ergebnisse werden pro Utterance gespeichert; Aggregate als
  Mittelwert ± 95 %-CI. Für Modellvergleiche paired Wilcoxon/t-Test auf den
  pro-Utterance-Werten empfehlenswert.
- **Reproduzierbarkeit:** Modell-IDs zentral in `config.py`.
