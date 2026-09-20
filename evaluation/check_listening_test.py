"""Checks the built listening test before the first rater.

Usage:
    uv run python evaluation/check_listening_test.py
"""

import csv
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myecho.evaluation.audio_qc import (  # noqa: E402
    PEAK_CEILING_DBFS,
    SILENCE_THRESHOLD_DBFS,
    _frame_db,
    load_mono,
    measure,
)

OUT_DIR = Path(__file__).parent / "results" / "listening_test"
AUDIT_CSV = OUT_DIR / "cue_audit.csv"

LOUDNESS_TOLERANCE_LU = 1.0
PAUSE_SNR_TOLERANCE_DB = 4.0

# Words that must never reach a rater, matched as whole words.
ROLE_WORDS = ("real", "clone", "fake", "echt", "synth")


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
    """Accuracy of the best single threshold, and its direction."""
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


def paired_cue_accuracy(feature_by_stimulus, pairs):
    """Accuracy of "the larger (or smaller) of the two is the real one"."""
    wins = 0
    total = 0
    for pair in pairs.values():
        a = feature_by_stimulus.get(pair["real_id"])
        b = feature_by_stimulus.get(pair["clone_id"])
        if a is None or b is None or not (np.isfinite(a) and np.isfinite(b)):
            continue
        total += 1
        if a > b:
            wins += 1
    if total == 0:
        return float("nan"), 0
    share = wins / total
    # Whichever direction is the better rule; 0.5 means the feature is useless.
    return (share, 1) if share >= 0.5 else (1.0 - share, -1)


def run_cue_audit(key):
    """Report whether one low-level measurement separates real from clone."""
    real_condition = key.get("real_condition", "REAL")
    rows = []
    for stimulus_id, info in sorted(key["stimuli"].items()):
        if info["role"] != "scored":
            continue
        path = OUT_DIR / "samples" / f"{stimulus_id}.wav"
        if path.exists():
            rows.append({"stimulus": stimulus_id, "condition": info["condition"],
                         "text_id": info["text_id"], **features(path)})
    if not rows:
        print("  (no scored stimuli found)")
        return

    with open(AUDIT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    is_real = [r["condition"] == real_condition for r in rows]
    baseline = max(np.mean(is_real), 1 - np.mean(is_real))
    by_id = {r["stimulus"]: r for r in rows}
    pairs = key.get("pairs", {})

    print(f"  single stimulus, base rate 'always answer clone' = {baseline:.3f}")
    print(f"  paired within a 2AFC pair, chance = 0.500  <- this is the rater's task")
    print(f"  {'feature':16s} {'single':>8s} {'paired':>8s}  verdict")
    for name, description in FEATURES.items():
        accuracy, _ = best_threshold_accuracy([r[name] for r in rows], is_real)
        paired, direction = paired_cue_accuracy(
            {sid: row[name] for sid, row in by_id.items()}, pairs)
        # Judged on the paired number: that is the strategy a rater actually has.
        margin = (paired - 0.5) if np.isfinite(paired) else 0.0
        verdict = ("no usable cue" if margin <= 0.05 else
                   "weak residual cue" if margin <= 0.15 else
                   "STRONG CUE - mitigate or report it")
        arrow = "" if margin <= 0.05 else (" (larger=real)" if direction > 0
                                           else " (smaller=real)")
        print(f"  {name:16s} {accuracy:8.3f} {paired:8.3f}  {verdict}{arrow}")
    print(f"  n pairs = {len(pairs)}; per-stimulus features -> {AUDIT_CSV}")


class Report:
    def __init__(self):
        self.failures = []

    def check(self, ok, name, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
        if not ok:
            self.failures.append(name)
        return ok


def manifest_from(html):
    match = re.search(r"const MANIFEST = (\{.*?\});\n", html, re.S)
    if match is None:
        raise SystemExit("Could not find MANIFEST in index.html")
    return json.loads(match.group(1))


def check_blinding(report, key, html, samples, share):
    # "training" is visible on purpose; the secrets are condition and text id.
    secrets = set(key["conditions"])
    secrets |= {info["text_id"] for info in key["stimuli"].values()}
    secrets |= {info["role"] for info in key["stimuli"].values()} - {"training"}

    # Only the shipped payload matters, not the source comments in the template.
    manifest_json = re.search(r"const MANIFEST = (\{.*?\});\n", html, re.S).group(1)
    leaked = sorted(s for s in secrets if s and s in manifest_json)
    report.check(not leaked, "blinding: manifest carries no condition or text id",
                 f"leaked: {leaked}" if leaked else "")

    names = [p.name for p in samples.iterdir()]
    bad = sorted(n for n in names if any(s and s in n for s in secrets))
    report.check(not bad, "blinding: sample filenames are opaque",
                 f"{bad[:3]}" if bad else f"{len(names)} files, e.g. {sorted(names)[0]}")

    role_hits = sorted(n for n in names
                       if any(re.search(rf"\b{w}\b", n, re.I) for w in ROLE_WORDS))
    report.check(not role_hits, "blinding: no filename says real/clone",
                 f"{role_hits[:3]}" if role_hits else "")

    zip_names = zipfile.ZipFile(share).namelist() if share.exists() else []
    report.check("key.json" not in zip_names, "no key leak: share.zip has no key.json")
    zip_bad = sorted(n for n in zip_names if any(s and s in n for s in secrets))
    report.check(not zip_bad, "no key leak: share.zip filenames are opaque",
                 f"{zip_bad[:3]}" if zip_bad else f"{len(zip_names)} entries")


def check_design(report, key, manifest):
    stimuli, pairs = key["stimuli"], key["pairs"]
    real = key.get("real_condition", "REAL")
    anchor = key.get("anchor_condition")
    # The anchor is rated but never paired, so it is not a model here.
    models = [c for c in key["conditions"] if c not in (real, anchor)]

    overlaps = []
    for plan in manifest["plans"]:
        part1 = {stimuli[s]["text_id"] for s in plan["nmos"]}
        part2 = {pairs[q["id"]]["text_id"] for q in plan["turing"]}
        if part1 & part2:
            overlaps.append((plan["id"], sorted(part1 & part2)))
    report.check(not overlaps, "role split: no plan uses a text in both parts",
                 f"{overlaps}" if overlaps else f"{len(manifest['plans'])} plans clean")

    texts = {info["text_id"] for info in stimuli.values() if info["role"] == "scored"}
    cells = {(p["text_id"], p["model"]) for p in pairs.values()}
    expected = len(texts) * len(models)
    report.check(len(cells) == expected, "coverage: every text x model cell has a pair",
                 f"{len(cells)}/{expected}")

    per_model = Counter(p["model"] for p in pairs.values())
    report.check(len(set(per_model.values())) <= 1,
                 "balance: each model has the same number of pairs", dict(per_model))

    per_condition = Counter(info["condition"] for info in stimuli.values()
                            if info["role"] == "scored")
    report.check(len(set(per_condition.values())) <= 1,
                 "balance: each condition has the same number of items",
                 dict(per_condition))
    report.check(real in per_condition,
                 "the real voice is a rated condition, not only the answer key",
                 f"{per_condition.get(real, 0)} items")

    plan_models = Counter(pairs[q["id"]]["model"]
                          for plan in manifest["plans"] for q in plan["turing"])
    report.check(len(set(plan_models.values())) <= 1,
                 "balance: each model appears equally often across plans",
                 dict(plan_models))

    # A positional pattern (position i = condition i % n) would de-blind the test.
    n_cond = len(key["conditions"])
    positional = []
    for plan in manifest["plans"]:
        for residue in range(n_cond):
            at_residue = {stimuli[s]["condition"]
                          for i, s in enumerate(plan["nmos"]) if i % n_cond == residue}
            if len(at_residue) == 1 and len(plan["nmos"]) > n_cond:
                positional.append((plan["id"], residue, at_residue.pop()))
    report.check(not positional,
                 "blinding: position in the manifest does not reveal the condition",
                 f"leaked: {positional[:4]}" if positional
                 else f"{len(manifest['plans'])} plans, no positional pattern")


def check_files(report, manifest, key, samples):
    referenced = set(manifest["files"])
    on_disk = {p.stem for p in samples.glob("*.wav")}
    missing = sorted(referenced - on_disk)
    orphan = sorted(on_disk - referenced)
    report.check(not missing, "files: every manifest id exists on disk", f"{missing[:5]}")
    report.check(not orphan, "files: no unreferenced sample left over",
                 f"{orphan[:5]}" if orphan else f"{len(on_disk)} files")

    used_in_plans = set()
    for plan in manifest["plans"]:
        used_in_plans.update(plan["nmos"])
        for q in plan["turing"]:
            used_in_plans.update((q["a"], q["b"]))
    used_in_plans.update(manifest["training"])
    unused = sorted(referenced - used_in_plans)
    report.check(not unused, "files: every stimulus is used by some plan",
                 f"{unused[:5]}" if unused else "")

    report.check(bool(manifest["training"]),
                 "training items exist, so the scale is used once before it counts",
                 f"{len(manifest['training'])} items")


def measured_lufs(stimulus_id):
    """Integrated loudness of the file as written, or None if unreadable."""
    path = OUT_DIR / "samples" / f"{stimulus_id}.wav"
    if not path.exists():
        return None
    import pyloudnorm

    x, sr = load_mono(path)
    if len(x) / sr < 0.4:  # BS.1770 needs one 400 ms block
        return None
    value = pyloudnorm.Meter(sr).integrated_loudness(x)
    return float(value) if np.isfinite(value) else None


def check_levels(report, key, target_lufs, target_pause_snr):
    achieved = [(sid, info["levels"]) for sid, info in key["stimuli"].items()]

    # Measured from the written files, not from the key's predicted values.
    off_level = []
    for sid, _ in achieved:
        value = measured_lufs(sid)
        if value is not None and abs(value - target_lufs) > LOUDNESS_TOLERANCE_LU:
            off_level.append((sid, round(value, 2)))
    report.check(not off_level, f"levels: all stimuli within {LOUDNESS_TOLERANCE_LU} LU "
                 f"of {target_lufs} LUFS (measured on the written files)",
                 f"{off_level[:3]}" if off_level
                 else f"{len(achieved)} files measured")

    # A pause floor that could not be measured at all is not "already fine"
    unmeasured = [(sid, lv.get("n_silent_frames")) for sid, lv in achieved
                  if lv.get("pause_measured") == "fallback"]
    report.check(not unmeasured,
                 "levels: every stimulus has a measurable pause floor",
                 f"fallback used for {unmeasured[:3]}" if unmeasured
                 else f"{len(achieved)} stimuli")

    snrs = [(sid, lv["pause_snr_after_db"]) for sid, lv in achieved
            if lv.get("pause_snr_after_db") is not None]

    # Cleaner pauses than the target mean the floor matching failed.
    too_clean = [(sid, round(v, 1)) for sid, v in snrs
                 if v > target_pause_snr + PAUSE_SNR_TOLERANCE_DB]
    noisier = [(sid, round(v, 1)) for sid, v in snrs
               if v < target_pause_snr - PAUSE_SNR_TOLERANCE_DB]
    spread = (max(v for _, v in snrs) - min(v for _, v in snrs)) if snrs else 0.0

    report.check(not too_clean,
                 f"levels: no stimulus has cleaner pauses than speech "
                 f"-{target_pause_snr:.0f} dB (the giveaway direction)",
                 f"unmatched: {too_clean[:3]}" if too_clean
                 else f"spread {spread:.1f} dB across {len(snrs)} stimuli")

    # Noisier than the target is still a cue, even though lifting a floor is refused.
    report.check(spread <= 2 * PAUSE_SNR_TOLERANCE_DB,
                 f"levels: pause floors sit within {2 * PAUSE_SNR_TOLERANCE_DB:.0f} dB "
                 f"of each other",
                 f"spread {spread:.1f} dB - report this next to the 2AFC results; "
                 f"noisiest: {sorted(noisier)[:3]}" if spread > 2 * PAUSE_SNR_TOLERANCE_DB
                 else f"spread {spread:.1f} dB")
    for sid, value in noisier:
        info = key["stimuli"][sid]
        print(f"         {sid} = {info['condition']}/{info['text_id']} at {value} dB")


def measured_peak_clip(stimulus_id):
    """(peak_dbfs, clipped_ratio) of the file as written, or (None, None)."""
    path = OUT_DIR / "samples" / f"{stimulus_id}.wav"
    if not path.exists():
        return None, None
    x, sr = load_mono(path)
    m = measure(x, sr)
    return m["peak_dbfs"], m["clipped_ratio"]


def check_clipping(report, key, peak_ceiling_dbfs):
    # Added floor noise can push a peak over the ceiling; check the written file.
    stats = {sid: measured_peak_clip(sid) for sid in key["stimuli"]}
    measured = {sid: v for sid, v in stats.items() if v[0] is not None}

    clipped = [(sid, ratio) for sid, (_, ratio) in measured.items() if ratio > 0]
    report.check(not clipped, "clipping: no stimulus has any sample at full scale",
                 f"{sorted(clipped)[:5]}" if clipped else f"{len(measured)} files measured")

    over_ceiling = [(sid, peak) for sid, (peak, _) in measured.items() if peak > peak_ceiling_dbfs]
    report.check(not over_ceiling,
                 f"clipping: no stimulus exceeds the {peak_ceiling_dbfs} dBFS peak ceiling",
                 f"{sorted(over_ceiling)[:5]}" if over_ceiling else f"{len(measured)} files measured")


def main():
    if not (OUT_DIR / "key.json").exists():
        raise SystemExit(f"{OUT_DIR} has no key.json - run prepare_listening_test.py first")

    key = json.loads((OUT_DIR / "key.json").read_text(encoding="utf-8"))
    html = (OUT_DIR / "index.html").read_text(encoding="utf-8")
    manifest = manifest_from(html)

    meta_path = OUT_DIR / "run_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    print(f"Verifying {OUT_DIR}")
    print(f"  built {key.get('stimulus_set_built')} at commit "
          f"{(meta.get('git_commit') or '?')[:7]}, seed {meta.get('seed')}\n")

    report = Report()
    check_blinding(report, key, html, OUT_DIR / "samples", OUT_DIR / "share.zip")
    check_design(report, key, manifest)
    check_files(report, manifest, key, OUT_DIR / "samples")
    check_levels(report, key, key.get("target_lufs", -23.0),
                 key.get("target_pause_snr_db", 26.0))
    check_clipping(report, key, PEAK_CEILING_DBFS)

    print("\nCue audit - could one measurement give the answer away?")
    run_cue_audit(key)

    responses = OUT_DIR / "responses"
    collected = (any(responses.glob("*.json")) or (responses / "all_ratings.jsonl").exists()
                 if responses.exists() else False)
    print(f"\n  ratings already collected: {'YES' if collected else 'no'}"
          + ("  <- the set is frozen; do not rebuild" if collected else ""))

    if report.failures:
        print(f"\n{len(report.failures)} check(s) FAILED: {', '.join(report.failures)}")
        print("Fix these before collecting any ratings - none of them can be "
              "repaired afterwards.")
        sys.exit(1)
    print("\nAll checks passed. Safe to start the sessions.")


if __name__ == "__main__":
    main()
