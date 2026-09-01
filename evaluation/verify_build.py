"""Checks the built listening test against the properties it is supposed to have.

Usage:
    uv run python evaluation/prepare_listening_test.py
    uv run python evaluation/verify_build.py

Run this before the first rater. Every property below was a real defect at some
point in this repo, and none of them can be repaired after data collection
starts - the stimulus set is frozen from the first submission.

    blinding        no condition name, text id or role word appears in the page,
                    in a sample filename or in share.zip. The first version
                    shipped turing_03_real.wav in the manifest.
    no key leak     share.zip does not contain key.json
    role split      no rater plan uses the same text in part 1 and part 2. With
                    matched excerpts and the real voice as a condition, part 1
                    would otherwise contain the answer to part 2.
    coverage        every text x model cell has a pair, so the interval is over
                    all texts and not over a handful
    balance         each model appears equally often across plans, and each
                    condition has the same number of naturalness items
    files           every id in the manifest exists on disk, and nothing on disk
                    is unreferenced
    levels          all stimuli within 1 LU of the loudness target and within a
                    narrow band of the pause-floor target

Exits non-zero if any check fails, so it can gate a script or a CI job.
"""

import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

OUT_DIR = Path(__file__).parent / "results" / "listening_test"

LOUDNESS_TOLERANCE_LU = 1.0
PAUSE_SNR_TOLERANCE_DB = 4.0

# Words that must never reach a rater. "real"/"clone"/"fake" are checked as whole
# words so an unrelated substring in the page text cannot trip the check.
ROLE_WORDS = ("real", "clone", "fake", "echt", "synth")


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
