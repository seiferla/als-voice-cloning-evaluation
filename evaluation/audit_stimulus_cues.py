"""Checks the finished stimulus set for low-level cues that give the answer away.

Usage:
    uv run python evaluation/prepare_listening_test.py
    uv run python evaluation/audit_stimulus_cues.py

A Turing test is only about voice quality if the two sides cannot be told apart
by something else. On the first stimulus set here they could: the real
recordings are 2021 phone voice memos put through DeepFilterNet, the clones are
generated clean, and a single threshold on the pause fraction or on the pause
noise floor separated real from clone with 96 % accuracy. A rater who noticed
that would score near the ceiling without ever judging a voice.

So this script measures a handful of cheap features on the stimuli that were
actually built, and reports, per feature, how well the best single threshold
separates the real recordings from the clones. It is a lower bound on what a
listener could exploit - a machine with one number is a weak attacker - but it
is enough to catch a cue that dominates the test.

Read the output like this:

    at or near the baseline   the cue is gone
    clearly above             a residual cue: mitigate it, or report the
                              Turing numbers together with this figure

The baseline is the accuracy of always answering "clone", since the conditions
are unbalanced (one real per N models).

Writes evaluation/results/listening_test/cue_audit.csv - keep it, it belongs in
the limitations section of the write-up.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myecho.evaluation.audio_qc import (  # noqa: E402
    SILENCE_THRESHOLD_DBFS,
    _frame_db,
    load_mono,
)

OUT_DIR = Path(__file__).parent / "results" / "listening_test"
KEY_PATH = OUT_DIR / "key.json"
AUDIT_CSV = OUT_DIR / "cue_audit.csv"

FEATURES = {
    "silence_ratio": "share of frames below the silence threshold (pause structure)",
    "pause_snr_db": "speech level minus pause floor (recording chain)",
    "hf_tilt_db": "energy 4-8 kHz relative to the whole band (brightness)",
    "lf_tilt_db": "energy 50-300 Hz relative to the whole band (body)",
    "duration_s": "stimulus length (speaking rate)",
    "level_std_db": "spread of the frame levels (dynamics)",
}


def features(path):
    x, sr = load_mono(path)
    frame_db = _frame_db(x, sr)
    active = frame_db > SILENCE_THRESHOLD_DBFS

    frame = max(int(0.02 * sr), 1)
    frames = x[: len(x) // frame * frame].reshape(-1, frame)[active]
    spectrum = np.abs(np.fft.rfft(frames * np.hanning(frame), axis=1)) ** 2
    freqs = np.fft.rfftfreq(frame, 1 / sr)
    total = 10 * np.log10(max(spectrum.mean(), 1e-20))

    def band(lo, hi):
        mask = (freqs >= lo) & (freqs < hi)
        return 10 * np.log10(max(spectrum[:, mask].mean(), 1e-20)) - total

    speech = float(np.median(frame_db[active])) if active.any() else float("nan")
    pause = float(np.median(frame_db[~active])) if (~active).sum() >= 3 else float("nan")

    return {
        "silence_ratio": float((~active).mean()),
        "pause_snr_db": speech - pause,
        "hf_tilt_db": band(4000, 8000),
        "lf_tilt_db": band(50, 300),
        "duration_s": len(x) / sr,
        "level_std_db": float(np.std(frame_db[active])) if active.any() else float("nan"),
    }


def best_threshold_accuracy(values, is_real):
    """Accuracy of the best single threshold, and its direction.

    Both directions are tried, so the number answers "could one number
    separate these at all", not "does a high value mean real".
    """
    values = np.asarray(values, dtype=float)
    is_real = np.asarray(is_real, dtype=int)
    usable = np.isfinite(values)
    values, is_real = values[usable], is_real[usable]
    if len(values) == 0:
        return float("nan"), 0
    best, direction = 0.0, 0
    for threshold in np.unique(values):
        for sign in (1, -1):
            accuracy = (((sign * (values - threshold) > 0).astype(int)) == is_real).mean()
            if accuracy > best:
                best, direction = float(accuracy), sign
    return best, direction


def main():
    if not KEY_PATH.exists():
        raise SystemExit(f"{KEY_PATH} not found - run prepare_listening_test.py first")
    key = json.loads(KEY_PATH.read_text(encoding="utf-8"))
    real_condition = key.get("real_condition", "REAL")

    rows = []
    for stimulus_id, info in sorted(key["stimuli"].items()):
        if info["role"] != "scored":
            continue
        path = OUT_DIR / "samples" / f"{stimulus_id}.wav"
        if not path.exists():
            continue
        rows.append({"stimulus": stimulus_id, "condition": info["condition"],
                     "text_id": info["text_id"], **features(path)})

    if not rows:
        raise SystemExit("No scored stimuli found.")

    with open(AUDIT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    conditions = sorted({r["condition"] for r in rows})
    print(f"{len(rows)} scored stimuli, conditions: {conditions}\n")

    print("Median per condition:")
    header = "  " + f"{'condition':10s}" + "".join(f"{name:>16s}" for name in FEATURES)
    print(header)
    for condition in conditions:
        group = [r for r in rows if r["condition"] == condition]
        line = "  " + f"{condition:10s}"
        for name in FEATURES:
            values = [r[name] for r in group if np.isfinite(r[name])]
            line += f"{np.median(values):16.2f}" if values else f"{'n/a':>16s}"
        print(line)

    is_real = [r["condition"] == real_condition for r in rows]
    baseline = max(np.mean(is_real), 1 - np.mean(is_real))
    print("\nCan one number tell the real recordings from the clones?")
    print(f"(baseline = always answer 'clone' = {baseline:.3f})\n")
    for name, description in FEATURES.items():
        accuracy, _ = best_threshold_accuracy([r[name] for r in rows], is_real)
        margin = accuracy - baseline
        verdict = ("at baseline - no usable cue" if margin <= 0.03 else
                   "weak residual cue" if margin <= 0.10 else
                   "STRONG CUE - mitigate or report it")
        print(f"  {name:16s} acc={accuracy:.3f}  ({margin:+.3f})  {verdict}")
        print(f"  {'':16s} {description}")

    print(f"\nPer-stimulus features -> {AUDIT_CSV}")
    print("\nThis is a lower bound: one threshold on one feature is a weak attacker.")
    print("A cue at baseline here can still be audible; a cue well above it is")
    print("certainly audible and must be reported alongside the Turing results.")


if __name__ == "__main__":
    main()
