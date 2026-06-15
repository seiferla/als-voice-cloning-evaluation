# myEcho — ALS Voice Cloning Evaluation

Dieses Repository enthält die Evaluation verschiedener Voice-Cloning-Modelle im Rahmen des **myEcho**-Projekts. Ziel ist es, eine geeignete Lösung zu finden, um die Stimme einer deutschen ALS-Patientin anhand weniger Legacy-Audiodaten (Videos, Sprachnachrichten) zu klonen.

---

## Evaluierte Modelle

| Modell | Typ | Beschreibung |
|---|---|---|
| **S2-pro** | API (Fish Audio) | WebSocket-basiertes Streaming-TTS mit Reference-Voice |
| **Voxtral** | Open Source | — |
| **Qwen3** | Open Source | — |

---

## Evaluationsmetriken

### Objektiv
| Metrik | Beschreibung |
|---|---|
| **SIM** | Speaker Similarity — cosine similarity zwischen Sprecher-Embeddings (z.B. WavLM, ECAPA) |
| **WER** | Word Error Rate — Transkriptionsgenauigkeit via ASR |

### Subjektiv
| Metrik | Beschreibung |
|---|---|
| **N-MOS** | Naturalness Mean Opinion Score — menschliche Bewertung der Natürlichkeit (1–5) |
| **Turing Test** | Können Hörer Original und Klon unterscheiden? |

### Performance
| Metrik | Beschreibung |
|---|---|
| **RTF** | Real-Time Factor — Verhältnis Inferenzzeit zu Audiodauer (< 1 = Echtzeit) |
| **TTFA** | Time To First Audio — Latenz bis zum ersten Audiopaket (Streaming) |

---

## Projektstruktur

```
├── api/
│   └── s2_pro.py                  — S2-pro WebSocket TTS Client (Fish Audio)
├── notebooks/
│   ├── data_preprocessing.ipynb  — Audio-Vorverarbeitung (WhisperX, pydub, Pedalboard)
│   └── data_analysis.ipynb       — Analyse der Audiodaten
└── data/                          — Lokale Audiodaten (gitignored)
```

---

## Datenpipeline

1. **Preprocessing** (`data_preprocessing.ipynb`): Stille-Entfernung, LUFS-Normalisierung, Limiter, Transkription & Alignment mit WhisperX
2. **Modell-Inferenz**: Klonierung per jeweiligem Modell
3. **Evaluation**: Berechnung der o.g. Metriken

---

## Setup

```bash
pip install websockets msgpack mutagen pydub pedalboard whisperx librosa
```

> Für S2-pro wird ein API-Key von [fish.audio](https://fish.audio) benötigt. Diesen als Umgebungsvariable `FISH_API_KEY` setzen.
