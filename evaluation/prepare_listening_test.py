"""Builds the blinded listening test from the audio in evaluation/results/.

Usage:
    uv run python evaluation/run_eval.py
    uv run python evaluation/select_excerpts.py
    uv run python evaluation/qc_samples.py
    uv run python evaluation/prepare_listening_test.py     # run ONCE
    uv run python evaluation/check_listening_test.py

Three design points, explained in docs/METHODS.md section 6:

- The real voice is a rated condition, not only the answer key, so naturalness
  has an upper reference.
- Each rater draws one of the enumerated plans. A plan splits the texts between
  the two parts, so no rater meets the same text twice, and across raters every
  model is paired with every text.
- Nothing in the shipped files says what anything is. Stimuli are named
  s_001.wav, assigned after the shuffle; key.json holds the only mapping.

Output goes to evaluation/results/listening_test/ (gitignored): samples/,
index.html, key.json (hidden), run_meta.json, stimuli.csv, responses/ and
share.zip for remote raters.

Re-running reshuffles everything and overwrites key.json, which desyncs ratings
already collected. It refuses to run once responses/ has data (--force
overrides).
"""

import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myecho.evaluation.audio_qc import (  # noqa: E402
    COMMON_SR,
    TARGET_LUFS,
    TARGET_PAUSE_SNR_DB,
    prepare_stimulus,
)

TESTSET_PATH = Path(__file__).parent / "testset.json"
RESULTS_DIR = Path(__file__).parent / "results"
TEMPLATE_PATH = Path(__file__).parent / "listening_test_template.html"
OUT_DIR = RESULTS_DIR / "listening_test"
EXCLUSIONS_PATH = RESULTS_DIR / "exclusions.json"
EXCERPTS_PATH = RESULTS_DIR / "excerpts.json"

AUDIO_EXT = "wav"

# The real recording is a condition in the N-MOS part, not only the answer in
# the Turing part. Named so it cannot collide with a model directory name.
REAL = "REAL"

# Texts are split into this many role groups. 2 = half the texts go to the
# N-MOS part for a given rater and half to the Turing part, so nobody meets the
# same text twice. Raising it shortens the session and lowers coverage per
# rater; the plan count is ROLE_GROUPS x len(models).
ROLE_GROUPS = 2

# Unscored items shown before the real ones so the rater has heard the range of
# the material before the first score that counts. Taken from texts that are
# dropped from the analysis anyway.
N_TRAINING = 3


def load_case(case_name=None):
    with open(TESTSET_PATH, encoding="utf-8") as f:
        testset = json.load(f)
    cases = testset["cases"]
    if case_name is None:
        return cases[0]
    for case in cases:
        if case["name"] == case_name:
            return case
    raise SystemExit(f"Case '{case_name}' not found in {TESTSET_PATH}")


def load_texts(case):
    texts_path = Path(case["texts_file"])
    with open(texts_path, encoding="utf-8") as f:
        data = json.load(f)
    return [{"id": t.get("id", str(i)), "source_audio": t["source_audio"]}
            for i, t in enumerate(data["texts"])]


def real_audio_path(source_audio):
    """Prefer the denoised recording.

    Note this is only half a fix on its own: denoising removes the background
    noise that would identify the real samples, but it also gives them
    digitally silent pauses that no TTS model produces, which identifies them
    just as well. audio_qc.match_noise_floor closes that second gap.
    """
    denoised = Path("data/DeepFilterNet") / source_audio
    if denoised.exists():
        return denoised
    return Path("data/wav") / source_audio


def find_models(case_dir):
    return sorted(p.name for p in case_dir.iterdir() if p.is_dir())


def clone_audio_path(case_dir, model, index, text_id):
    return case_dir / model / f"{index:02d}_{text_id}.{AUDIO_EXT}"


def load_exclusions():
    """({(model, text_id): reason}, keep_unbalanced) from exclusions.json."""
    if not EXCLUSIONS_PATH.exists():
        return {}, False
    with open(EXCLUSIONS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    excluded = {(e["model"], e["text_id"]): e.get("reason", "") for e in data.get("exclude", [])}
    return excluded, bool(data.get("keep_unbalanced", False))


def load_excerpts():
    """The excerpt spans written by select_excerpts.py."""
    if not EXCERPTS_PATH.exists():
        raise SystemExit(
            f"{EXCERPTS_PATH} not found - run evaluation/select_excerpts.py first.\n"
            "It picks one ~10 s excerpt per text and locates it in every rendition, "
            "so raters judge matched content instead of the 12-92 s full renders."
        )
    with open(EXCERPTS_PATH, encoding="utf-8") as f:
        return json.load(f)


def unlocated(spans, models, texts):
    """Renditions whose excerpt was not found, in the shape of exclusions.json.

    Reusing that shape is deliberate: the balanced-design rule then applies to
    them too, so the text is dropped for every model rather than leaving one
    model scored on material the others were not.
    """
    return {
        (model, entry["id"]): "excerpt not found in this rendition"
        for model in models
        for entry in texts
        if f"{model}/{entry['id']}" not in spans
    }


def dropped_texts(excluded, keep_unbalanced):
    """Texts to drop for EVERY model, not just the one that failed.

    Excluding one model's render of a text leaves that text rated for the
    other models only. Texts differ in difficulty, so each model's mean would
    then be an average over a slightly different text set - a small bias that
    is free to avoid: drop the whole text and every model is scored on
    identical material (a balanced design).

    Set "keep_unbalanced": true in exclusions.json to keep the extra items
    instead, e.g. if you would otherwise lose too much material.
    """
    if keep_unbalanced:
        return set()
    return {text_id for _, text_id in excluded}


def build_items(case_dir, conditions, texts, spans, index_of, excluded=()):
    """One stimulus per (condition, text): {"cond", "text_id", "src", "span"}.

    A condition with no located span for a text is skipped rather than looked up
    blindly. With the default balanced design that case never arises - the text
    is already dropped for everyone - but with "keep_unbalanced": true it does,
    and a KeyError in the middle of building a frozen stimulus set is the worst
    possible time to find out.
    """
    items = []
    for entry in texts:
        text_id = entry["id"]
        for cond in conditions:
            if (cond, text_id) in excluded:
                continue
            span = spans.get(f"{cond}/{text_id}")
            if span is None:
                print(f"  skip {cond}/{text_id}: no located excerpt")
                continue
            if cond == REAL:
                src = real_audio_path(entry["source_audio"])
            else:
                src = clone_audio_path(case_dir, cond, index_of[text_id], text_id)
            if src is None or not src.exists():
                print(f"  skip {cond}/{text_id}: audio file missing")
                continue
            items.append({"cond": cond, "text_id": text_id, "src": src,
                          "span": span})
    return items


def build_plans(texts, models, role_groups=ROLE_GROUPS):
    """Enumerate the rater plans: (role group) x (model rotation).

    Plan (g, r) sends text slot i to the Turing part when i % role_groups == g,
    and pairs it with models[(i + r) % len(models)]. Every other text goes to
    the N-MOS part with all conditions.

    Enumerated rather than drawn per rater: with ~30 raters, random assignment
    would leave the coverage of some text x model cells to luck. Here every
    cell is covered as soon as each plan has been used once.
    """
    plans = []
    for g in range(role_groups):
        for r in range(len(models)):
            turing, nmos = [], []
            for i, entry in enumerate(texts):
                if i % role_groups == g:
                    turing.append((entry["id"], models[(i + r) % len(models)]))
                else:
                    nmos.append(entry["id"])
            plans.append({"role_group": g, "model_rotation": r,
                          "nmos_texts": nmos, "turing": turing})
    return plans


def training_items(case_dir, conditions, texts, skip_texts, spans, index_of, n=N_TRAINING):
    """Unscored warm-up stimuli, taken from texts dropped from the analysis.

    Dropped texts are the natural source: they are already outside the results,
    and reusing them means the warm-up sounds exactly like the real task. One
    real recording first, then clones from different models, so the rater has
    heard both ends of the material before the first scored item.
    """
    picked, used_models = [], set()
    for entry in texts:
        if entry["id"] not in skip_texts:
            continue
        for cond in [REAL] + [c for c in conditions if c != REAL]:
            if len(picked) >= n:
                return picked
            if cond != REAL and cond in used_models:
                continue
            if f"{cond}/{entry['id']}" not in spans:
                continue
            src = (real_audio_path(entry["source_audio"]) if cond == REAL
                   else clone_audio_path(case_dir, cond, index_of[entry["id"]], entry["id"]))
            if not src.exists():
                continue
            picked.append({"cond": cond, "text_id": entry["id"], "src": src,
                           "span": spans[f"{cond}/{entry['id']}"]})  # checked above
            if cond != REAL:
                used_models.add(cond)
    return picked


def has_existing_responses():
    responses_dir = OUT_DIR / "responses"
    if not responses_dir.exists():
        return False
    return any(responses_dir.glob("*.json")) or (responses_dir / "all_ratings.jsonl").exists()


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("case", nargs="?", default=None)
    parser.add_argument("--seed", type=int, default=20260901,
                        help="Seed for stimulus id assignment. Recorded in run_meta.json; "
                             "the same seed rebuilds the identical test.")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even though ratings were already collected. "
                             "This invalidates them.")
    args = parser.parse_args()

    if has_existing_responses() and not args.force:
        raise SystemExit(
            f"{OUT_DIR / 'responses'} already has collected ratings.\n"
            "Re-running this would reshuffle the samples and overwrite key.json, "
            "which desyncs those ratings from what they actually answered.\n"
            "Finish analyzing the current round first (or move responses/ aside "
            "yourself), or pass --force if you really mean to start a fresh test."
        )

    rng = random.Random(args.seed)

    case = load_case(args.case)
    texts = load_texts(case)
    index_of = {entry["id"]: i for i, entry in enumerate(texts)}
    case_dir = RESULTS_DIR / case["name"]
    if not case_dir.exists():
        raise SystemExit(f"{case_dir} not found - run evaluation/run_eval.py first")

    models = find_models(case_dir)
    conditions = models + [REAL]
    excerpts = load_excerpts()
    spans = excerpts["spans"]

    excluded, keep_unbalanced = load_exclusions()
    excluded = {**unlocated(spans, models, texts), **excluded}
    skip_texts = dropped_texts(excluded, keep_unbalanced)
    # A text whose REAL recording has no located excerpt cannot form a Turing
    # pair and has no anchor, so it goes entirely - independent of balancing.
    skip_texts |= {e["id"] for e in texts if f"{REAL}/{e['id']}" not in spans}

    print(f"Case: {case['name']}  conditions: {conditions}  texts: {len(texts)}")
    if excluded:
        print(f"\nExcluding {len(excluded)} rendition(s):")
        for (model, text_id), reason in sorted(excluded.items()):
            print(f"  {model}/{text_id}: {reason}")
    if skip_texts:
        print(f"\nDropping {len(skip_texts)} text(s) for ALL conditions, so every "
              "condition is scored on identical material (balanced design):")
        for text_id in sorted(skip_texts):
            print(f"  {text_id}")

    kept = [e for e in texts if e["id"] not in skip_texts]
    if len(kept) < 4:
        raise SystemExit(f"Only {len(kept)} usable text(s) left - too few for a test.")

    plans = build_plans(kept, models)
    items = build_items(case_dir, conditions, kept, spans, index_of, excluded)
    training = training_items(case_dir, conditions, texts, skip_texts, spans, index_of)

    # --- render every stimulus under an opaque name -------------------------
    samples_dir = OUT_DIR / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    for stale in samples_dir.glob(f"*.{AUDIO_EXT}"):
        stale.unlink()

    # A Turing pair reuses the two stimuli that already exist for its text: the
    # clone of that model and the real recording. Both sides are therefore cut
    # from the same excerpt and processed identically.
    by_key = {}
    all_stimuli = items + training
    order = list(range(len(all_stimuli)))
    rng.shuffle(order)

    print(f"\nCutting excerpts, normalizing to {TARGET_LUFS} LUFS @ {COMMON_SR} Hz, "
          f"matching the pause floor to speech -{TARGET_PAUSE_SNR_DB:.0f} dB...")
    stimuli_rows, key_stimuli, files = [], {}, {}
    for anon_index, item_index in enumerate(order, start=1):
        item = all_stimuli[item_index]
        anon_id = f"s_{anon_index:03d}"
        dest = samples_dir / f"{anon_id}.{AUDIO_EXT}"
        info = prepare_stimulus(item["src"], dest, span=item["span"])
        role = "training" if item_index >= len(items) else "scored"
        by_key[(item["cond"], item["text_id"], role)] = anon_id
        files[anon_id] = f"samples/{dest.name}"
        key_stimuli[anon_id] = {"condition": item["cond"], "text_id": item["text_id"],
                                "role": role, "levels": info}
        stimuli_rows.append({
            "stimulus": anon_id, "role": role, "condition": item["cond"],
            "text_id": item["text_id"],
            "excerpt_start_s": round(item["span"]["start"], 2),
            "excerpt_end_s": round(item["span"]["end"], 2),
            "match_ratio": item["span"]["match_ratio"], **info,
        })
        if not info["reached_target"]:
            print(f"  WARNING {anon_id} ({item['cond']}/{item['text_id']}): {info['reason']}")
        if info["pause_snr_before_db"] is not None and not info["raised"]:
            print(f"  NOTE {anon_id} ({item['cond']}/{item['text_id']}): pause floor "
                  f"already above target ({info['pause_snr_before_db']} dB) - not lowered")

    # --- pairs and plans, in terms of those opaque ids ----------------------
    pairs, key_pairs = {}, {}
    pair_index = 0
    for plan in plans:
        plan_pairs = []
        for text_id, model in plan["turing"]:
            clone_id = by_key.get((model, text_id, "scored"))
            real_id = by_key.get((REAL, text_id, "scored"))
            if clone_id is None or real_id is None:
                continue
            existing = pairs.get((text_id, model))
            if existing is None:
                pair_index += 1
                existing = f"q_{pair_index:03d}"
                pairs[(text_id, model)] = existing
                # Baseline side only; the page re-flips it per rater.
                real_side = rng.choice(["a", "b"])
                key_pairs[existing] = {"model": model, "text_id": text_id,
                                       "real_side": real_side,
                                       "real_id": real_id, "clone_id": clone_id}
            info = key_pairs[existing]
            plan_pairs.append({
                "id": existing,
                "a": info["real_id"] if info["real_side"] == "a" else info["clone_id"],
                "b": info["clone_id"] if info["real_side"] == "a" else info["real_id"],
            })
        plan["pairs"] = plan_pairs

    manifest_plans = []
    for i, plan in enumerate(plans):
        nmos_ids = [by_key[(cond, text_id, "scored")]
                    for text_id in plan["nmos_texts"]
                    for cond in conditions
                    if (cond, text_id, "scored") in by_key]
        manifest_plans.append({"id": i, "nmos": nmos_ids, "turing": plan["pairs"]})

    training_ids = [by_key[(t["cond"], t["text_id"], "training")] for t in training]

    manifest = {
        "files": files,
        "training": training_ids,
        "plans": manifest_plans,
    }

    with open(OUT_DIR / "stimuli.csv", "w", newline="", encoding="utf-8") as f:
        fields = ["stimulus", "role", "condition", "text_id", "duration_s",
                  "excerpt_start_s", "excerpt_end_s", "match_ratio",
                  "gain_db", "achieved_lufs", "reached_target", "reason",
                  "pause_snr_before_db", "pause_snr_after_db", "noise_added_dbfs",
                  "raised", "sample_rate"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(stimuli_rows)

    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    html = template.replace("__MANIFEST_JSON__", json.dumps(manifest, ensure_ascii=False))
    (OUT_DIR / "index.html").write_text(html, encoding="utf-8")

    key = {
        "case": case["name"],
        "stimulus_set_built": datetime.now().isoformat(timespec="seconds"),
        "conditions": conditions,
        "real_condition": REAL,
        "target_lufs": TARGET_LUFS,
        "target_pause_snr_db": TARGET_PAUSE_SNR_DB,
        "sample_rate": COMMON_SR,
        "excerpt_params": excerpts["params"],
        "excluded": [{"model": m, "text_id": t, "reason": r} for (m, t), r in excluded.items()],
        "dropped_texts": sorted(skip_texts),
        "stimuli": key_stimuli,
        "pairs": key_pairs,
        "plans": [{"id": i, "role_group": p["role_group"],
                   "model_rotation": p["model_rotation"]} for i, p in enumerate(plans)],
    }
    (OUT_DIR / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=2),
                                      encoding="utf-8")

    run_meta = {
        "built": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "git_commit": git_commit(),
        "case": case["name"],
        "texts_file": case["texts_file"],
        "texts_file_sha256": sha256(case["texts_file"]),
        "excerpts_created": excerpts.get("created"),
        "excerpt_params": excerpts["params"],
        "conditions": conditions,
        "n_texts_total": len(texts),
        "n_texts_used": len(kept),
        "n_stimuli": len(key_stimuli),
        "n_pairs": len(key_pairs),
        "n_plans": len(plans),
        "target_lufs": TARGET_LUFS,
        "target_pause_snr_db": TARGET_PAUSE_SNR_DB,
        "sample_rate": COMMON_SR,
    }
    (OUT_DIR / "run_meta.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2),
                                           encoding="utf-8")

    (OUT_DIR / "responses").mkdir(exist_ok=True)

    share_zip = OUT_DIR / "share.zip"
    with zipfile.ZipFile(share_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(OUT_DIR / "index.html", "index.html")
        for f in sorted(samples_dir.iterdir()):
            zf.write(f, f"samples/{f.name}")

    per_plan_nmos = [len(p["nmos"]) for p in manifest_plans]
    per_plan_turing = [len(p["turing"]) for p in manifest_plans]
    seconds = {r["stimulus"]: r["duration_s"] for r in stimuli_rows}
    plan_minutes = [
        (sum(seconds[i] for i in p["nmos"])
         + sum(seconds[q["a"]] + seconds[q["b"]] for q in p["turing"])) / 60
        for p in manifest_plans
    ]

    print(f"\n{len(kept)} texts x {len(conditions)} conditions = {len(items)} scored "
          f"stimuli, {len(key_pairs)} Turing pairs, {len(training_ids)} training items.")
    print(f"{len(plans)} rater plans ({ROLE_GROUPS} role groups x {len(models)} model "
          f"rotations); each rater draws one at random.")
    print(f"  per rater: {min(per_plan_nmos)}-{max(per_plan_nmos)} N-MOS items, "
          f"{min(per_plan_turing)}-{max(per_plan_turing)} Turing pairs, "
          f"{min(plan_minutes):.1f}-{max(plan_minutes):.1f} min of audio")
    print(f"Per-stimulus levels, pause floors and excerpt spans -> {OUT_DIR / 'stimuli.csv'}")
    print(f"Build provenance (seed, commit, input hashes) -> {OUT_DIR / 'run_meta.json'}")
    print("\nThis stimulus set is now FROZEN. Do not re-run this script mid-study.")
    print(f"\nStart sessions with: uv run python evaluation/listening_test_server.py")
    print(f"For remote raters: share {share_zip} (does NOT contain key.json).")
    print("Do NOT open key.json yourself before the sessions - you run them, and")
    print("knowing which side is real can leak through how you react.")


if __name__ == "__main__":
    main()
