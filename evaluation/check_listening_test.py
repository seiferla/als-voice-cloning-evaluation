"""Checks the built listening test before the first rater.

Usage:
    uv run python evaluation/check_listening_test.py

Two parts:

1. Invariants - properties the stimulus set must have. Exits non-zero if any
   fails, so it can gate a script. None of them can be repaired once ratings
   have been collected.
2. Cue audit - can a single low-level measurement tell the real recordings from
   the clones? A cue near the guessing baseline is gone; a cue clearly above it
   has to be mitigated or reported with the results.

Reasoning and measured numbers: docs/METHODS.md section 5.
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
    SILENCE_THRESHOLD_DBFS,
    _frame_db,
    load_mono,
)

OUT_DIR = Path(__file__).parent / "results" / "listening_test"
AUDIT_CSV = OUT_DIR / "cue_audit.csv"

LOUDNESS_TOLERANCE_LU = 1.0
PAUSE_SNR_TOLERANCE_DB = 4.0

# Words that must never reach a rater. "real"/"clone"/"fake" are checked as whole
# words so an unrelated substring in the page text cannot trip the check.
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


def run_cue_audit(key):
    """Report whether one low-level measurement separates real from clone.

    A lower bound on what a listener could exploit - one threshold on one feature
    is a weak attacker - but enough to catch a cue that dominates the test.
    """
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
    print(f"  baseline (always answer 'clone') = {baseline:.3f}")
    for name, description in FEATURES.items():
        accuracy, _ = best_threshold_accuracy([r[name] for r in rows], is_real)
        margin = accuracy - baseline
        verdict = ("no usable cue" if margin <= 0.03 else
                   "weak residual cue" if margin <= 0.10 else
                   "STRONG CUE - mitigate or report it")
        print(f"  {name:16s} acc={accuracy:.3f} ({margin:+.3f})  {verdict}")
    print(f"  per-stimulus features -> {AUDIT_CSV}")


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
    # "training" is deliberately visible - the page labels those items as
    # practice, so the rater already knows. The secrets are the condition of a
    # stimulus and which text it is.
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
    models = [c for c in key["conditions"] if c != real]

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


def check_levels(report, key, target_lufs, target_pause_snr):
    achieved = [(sid, info["levels"]) for sid, info in key["stimuli"].items()]
    off_level = [(sid, lv["achieved_lufs"]) for sid, lv in achieved
                 if lv.get("achieved_lufs") is not None
                 and abs(lv["achieved_lufs"] - target_lufs) > LOUDNESS_TOLERANCE_LU]
    report.check(not off_level, f"levels: all stimuli within {LOUDNESS_TOLERANCE_LU} LU "
                 f"of {target_lufs} LUFS", f"{off_level[:3]}" if off_level else "")

    snrs = [(sid, lv["pause_snr_after_db"]) for sid, lv in achieved
            if lv.get("pause_snr_after_db") is not None]

    # The two directions mean opposite things and only one is a defect.
    #
    # CLEANER pauses than the target means the matching did not do its job, and
    # a stimulus with unnaturally silent gaps is exactly the cue that made the
    # first version of this test winnable without listening to a voice.
    #
    # NOISIER pauses than the target is expected and is left alone on purpose:
    # the source itself generates noise in its gaps. Lowering it would mean
    # denoising one condition and not the others. It also cannot help a rater
    # find the real recording, since the real side sits at the target.
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
    if noisier:
        print(f"  [note] {len(noisier)} stimulus/stimuli noisier than the target "
              f"by design (own noise floor): {noisier[:3]}")
        for sid, _ in noisier:
            info = key["stimuli"][sid]
            print(f"         {sid} = {info['condition']}/{info['text_id']} - a "
                  "property of that render, left untouched")


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
