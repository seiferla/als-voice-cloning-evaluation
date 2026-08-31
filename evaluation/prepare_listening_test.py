"""Builds a blinded, randomized listening test (N-MOS + Turing test) from
already-synthesized audio in evaluation/results/.

Usage (in-person, one machine, back-to-back raters):
    uv run python evaluation/run_eval.py               # produce audio first
    uv run python evaluation/prepare_listening_test.py  # build the test (run ONCE - see below)
    uv run python evaluation/listening_test_server.py   # serve it, then open http://localhost:8765 per rater

Re-running prepare_listening_test.py reshuffles everything and overwrites
key.json, which desyncs any ratings already collected - it refuses to run
once evaluation/results/listening_test/responses/ has data (pass --force to
override once you're truly starting a fresh test).

Output goes to evaluation/results/listening_test/ (gitignored, same as the
rest of evaluation/results/):
    samples/       - anonymized copies of the audio (no model name in the filename)
    index.html     - self-contained rating UI (served locally, or opened directly
                     in a browser as a fallback - see listening_test_server.py)
    key.json       - HIDDEN mapping anon_id -> model/text. Needed for analysis,
                     must NOT be shown to raters (it would unblind the test).
    responses/     - one line per rater in all_ratings.jsonl, written by the
                     server as each rater finishes; nmos_summary.csv and
                     turing_summary.csv are regenerated after every submission.
    share.zip      - index.html + samples/ only (no key.json), for raters who
                     can't come in person. It's ~200MB (80 WAV files), so share
                     it via a cloud drive link rather than email; they rate on
                     their own machine (no server, falls back to a manual
                     ratings_*.json download) and send that file back to you -
                     drop it straight into responses/, analyze_listening_test.py
                     picks it up alongside the server-collected ones.

N-MOS part: every synthesized sample (all models x all texts), rated 1-5 for
naturalness, order shuffled.

Turing test part: one real recording vs. one clone per text, paired up so a
rater has to say which of A/B is the real voice. The clone model is assigned
round-robin across texts so every model gets roughly equal coverage; A/B side
and pair order are both randomized to avoid position bias.

To recompute results.csv / print a full report at any time (also runs
automatically after every submission through the server):
    uv run python evaluation/analyze_listening_test.py
"""

import csv
import json
import random
import sys
import zipfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myecho.evaluation.audio_qc import (  # noqa: E402
    COMMON_SR,
    TARGET_LUFS,
    prepare_stimulus,
)

TESTSET_PATH = Path(__file__).parent / "testset.json"
RESULTS_DIR = Path(__file__).parent / "results"
TEMPLATE_PATH = Path(__file__).parent / "listening_test_template.html"
OUT_DIR = RESULTS_DIR / "listening_test"
EXCLUSIONS_PATH = RESULTS_DIR / "exclusions.json"

AUDIO_EXT = "wav"


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
    """Prefer the denoised recording so 'real' samples aren't trivially
    identifiable by background noise alone."""
    denoised = Path("data/DeepFilterNet") / source_audio
    if denoised.exists():
        return denoised
    return Path("data/wav") / source_audio


def find_models(case_dir):
    return sorted(p.name for p in case_dir.iterdir() if p.is_dir())


def clone_audio_path(case_dir, model, index, text_id):
    return case_dir / model / f"{index:02d}_{text_id}.{AUDIO_EXT}"


def load_exclusions():
    """{(model, text_id): reason} from exclusions.json, written by qc_samples.py."""
    if not EXCLUSIONS_PATH.exists():
        return {}
    with open(EXCLUSIONS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return {(e["model"], e["text_id"]): e.get("reason", "") for e in data.get("exclude", [])}


def build_nmos(case_dir, models, texts, excluded):
    samples = []
    for model in models:
        for i, entry in enumerate(texts):
            if (model, entry["id"]) in excluded:
                continue
            src = clone_audio_path(case_dir, model, i, entry["id"])
            if src.exists():
                samples.append({"model": model, "text_id": entry["id"], "src": src})
    # Order is deliberately NOT fixed here beyond assigning ids: the page
    # shuffles per rater at load time (see listening_test_template.html), so
    # order effects stay noise instead of favouring particular items.
    random.shuffle(samples)
    for i, s in enumerate(samples, start=1):
        s["anon_id"] = f"nmos_{i:02d}"
    return samples


def build_turing(case_dir, models, texts, excluded):
    pairs = []
    for i, entry in enumerate(texts):
        model = models[i % len(models)]
        if (model, entry["id"]) in excluded:
            continue
        clone_src = clone_audio_path(case_dir, model, i, entry["id"])
        real_src = real_audio_path(entry["source_audio"])
        if not (clone_src.exists() and real_src.exists()):
            continue
        # Which side holds the real recording is re-randomized per rater in the
        # page; this is only the baseline the page flips against.
        real_position = random.choice(["A", "B"])
        pairs.append({
            "model": model,
            "text_id": entry["id"],
            "real_position": real_position,
            "real_src": real_src,
            "clone_src": clone_src,
        })
    random.shuffle(pairs)
    for i, p in enumerate(pairs, start=1):
        p["pair_id"] = f"turing_{i:02d}"
    return pairs


def has_existing_responses():
    responses_dir = OUT_DIR / "responses"
    if not responses_dir.exists():
        return False
    return any(responses_dir.glob("*.json")) or (responses_dir / "all_ratings.jsonl").exists()


def main():
    force = "--force" in sys.argv
    if force:
        sys.argv.remove("--force")

    if has_existing_responses() and not force:
        raise SystemExit(
            f"{OUT_DIR / 'responses'} already has collected ratings.\n"
            "Re-running this would reshuffle the samples and overwrite key.json, "
            "which desyncs those ratings from what they actually answered.\n"
            "Finish analyzing the current round first (or move responses/ aside "
            "yourself), or pass --force if you really mean to start a fresh test."
        )

    case_name = sys.argv[1] if len(sys.argv) > 1 else None
    case = load_case(case_name)
    texts = load_texts(case)
    case_dir = RESULTS_DIR / case["name"]
    if not case_dir.exists():
        raise SystemExit(f"{case_dir} not found - run evaluation/run_eval.py first")

    models = find_models(case_dir)
    excluded = load_exclusions()
    print(f"Case: {case['name']}  models: {models}  texts: {len(texts)}")
    if excluded:
        print(f"Excluding {len(excluded)} stimuli from {EXCLUSIONS_PATH.name}:")
        for (model, text_id), reason in excluded.items():
            print(f"  {model}/{text_id}: {reason}")
    else:
        print(f"No exclusions ({EXCLUSIONS_PATH.name} missing or empty) - "
              "run evaluation/qc_samples.py first to screen the stimuli.")

    nmos = build_nmos(case_dir, models, texts, excluded)
    turing = build_turing(case_dir, models, texts, excluded)

    samples_dir = OUT_DIR / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # Every stimulus is resampled to one rate and loudness-normalized, so
    # neither level nor bandwidth can stand in for quality. See audio_qc.py.
    print(f"\nNormalizing stimuli to {TARGET_LUFS} LUFS @ {COMMON_SR} Hz...")
    stimuli_rows = []

    manifest_nmos = []
    key_nmos = []
    for s in nmos:
        dest = samples_dir / f"{s['anon_id']}.{AUDIO_EXT}"
        info = prepare_stimulus(s["src"], dest)
        manifest_nmos.append({"id": s["anon_id"], "file": f"samples/{dest.name}"})
        key_nmos.append({"anon_id": s["anon_id"], "model": s["model"],
                         "text_id": s["text_id"], "loudness": info})
        stimuli_rows.append({"stimulus": s["anon_id"], "role": "nmos", "model": s["model"],
                             "text_id": s["text_id"], **info})
        if not info["reached_target"]:
            print(f"  WARNING {s['anon_id']} ({s['model']}/{s['text_id']}): {info['reason']}")

    manifest_turing = []
    key_turing = []
    for p in turing:
        dest_real = samples_dir / f"{p['pair_id']}_real.{AUDIO_EXT}"
        dest_clone = samples_dir / f"{p['pair_id']}_clone.{AUDIO_EXT}"
        info_real = prepare_stimulus(p["real_src"], dest_real)
        info_clone = prepare_stimulus(p["clone_src"], dest_clone)
        file_a = dest_real.name if p["real_position"] == "A" else dest_clone.name
        file_b = dest_clone.name if p["real_position"] == "A" else dest_real.name
        manifest_turing.append({
            "id": p["pair_id"],
            "file_a": f"samples/{file_a}",
            "file_b": f"samples/{file_b}",
        })
        key_turing.append({
            "pair_id": p["pair_id"],
            "model": p["model"],
            "text_id": p["text_id"],
            "real_position": p["real_position"],
            "loudness_real": info_real,
            "loudness_clone": info_clone,
        })
        stimuli_rows.append({"stimulus": f"{p['pair_id']}_real", "role": "turing_real",
                             "model": "REAL", "text_id": p["text_id"], **info_real})
        stimuli_rows.append({"stimulus": f"{p['pair_id']}_clone", "role": "turing_clone",
                             "model": p["model"], "text_id": p["text_id"], **info_clone})
        for label, info in (("real", info_real), ("clone", info_clone)):
            if not info["reached_target"]:
                print(f"  WARNING {p['pair_id']}_{label} ({p['text_id']}): {info['reason']}")

    with open(OUT_DIR / "stimuli.csv", "w", newline="", encoding="utf-8") as f:
        fields = ["stimulus", "role", "model", "text_id", "gain_db", "achieved_lufs",
                  "reached_target", "reason", "sample_rate"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(stimuli_rows)

    manifest = {"nmos": manifest_nmos, "turing": manifest_turing}
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    html = template.replace("__MANIFEST_JSON__", json.dumps(manifest, ensure_ascii=False))
    (OUT_DIR / "index.html").write_text(html, encoding="utf-8")

    key = {
        "case": case["name"],
        "stimulus_set_built": datetime.now().isoformat(timespec="seconds"),
        "target_lufs": TARGET_LUFS,
        "sample_rate": COMMON_SR,
        "excluded": [{"model": m, "text_id": t, "reason": r} for (m, t), r in excluded.items()],
        "nmos": key_nmos,
        "turing": key_turing,
    }
    (OUT_DIR / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")

    (OUT_DIR / "responses").mkdir(exist_ok=True)

    share_zip = OUT_DIR / "share.zip"
    with zipfile.ZipFile(share_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(OUT_DIR / "index.html", "index.html")
        for f in samples_dir.iterdir():
            zf.write(f, f"samples/{f.name}")

    print(f"\n{len(nmos)} N-MOS samples, {len(turing)} Turing pairs.")
    print(f"Per-stimulus levels -> {OUT_DIR / 'stimuli.csv'}")
    print("\nThis stimulus set is now FROZEN. Every rater rates these same items;")
    print("the page shuffles their order (and the A/B sides) per rater, so sessions")
    print("on different days stay comparable. Do not re-run this script mid-study.")
    print(f"\nStart sessions with: uv run python evaluation/listening_test_server.py")
    print(f"For remote raters: share {share_zip} (does NOT contain key.json).")
    print("Do NOT open key.json yourself before the sessions - you run them, and")
    print("knowing which side is real can leak through how you react.")


if __name__ == "__main__":
    main()