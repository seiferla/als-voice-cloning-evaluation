"""Builds a blinded, randomized listening test (N-MOS + Turing test) from
already-synthesized audio in evaluation/results/.

Usage:
    uv run python evaluation/run_eval.py                    # produce audio first
    uv run python evaluation/prepare_listening_test.py       # then build the test

Output goes to evaluation/results/listening_test/ (gitignored, same as the
rest of evaluation/results/):
    samples/       - anonymized copies of the audio (no model name in the filename)
    index.html     - self-contained rating UI, open directly in a browser
    key.json       - HIDDEN mapping anon_id -> model/text. Needed for analysis,
                     must NOT be shown to raters (it would unblind the test).
    share.zip      - index.html + samples/ only (no key.json) - send this to raters.
                     It's ~200MB (80 WAV files), so share it via a cloud drive link
                     rather than email. Each rater unzips it, opens index.html in
                     their browser, rates, and sends back the downloaded
                     ratings_*.json (not the whole zip).

N-MOS part: every synthesized sample (all models x all texts), rated 1-5 for
naturalness, order shuffled.

Turing test part: one real recording vs. one clone per text, paired up so a
rater has to say which of A/B is the real voice. The clone model is assigned
round-robin across texts so every model gets roughly equal coverage; A/B side
and pair order are both randomized to avoid position bias.

After collecting ratings*.json (downloaded from index.html) into
evaluation/results/listening_test/responses/, run:
    uv run python evaluation/analyze_listening_test.py
"""

import json
import random
import shutil
import sys
import zipfile
from pathlib import Path

TESTSET_PATH = Path(__file__).parent / "testset.json"
RESULTS_DIR = Path(__file__).parent / "results"
TEMPLATE_PATH = Path(__file__).parent / "listening_test_template.html"
OUT_DIR = RESULTS_DIR / "listening_test"

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


def build_nmos(case_dir, models, texts):
    samples = []
    for model in models:
        for i, entry in enumerate(texts):
            src = clone_audio_path(case_dir, model, i, entry["id"])
            if src.exists():
                samples.append({"model": model, "text_id": entry["id"], "src": src})
    random.shuffle(samples)
    for i, s in enumerate(samples, start=1):
        s["anon_id"] = f"nmos_{i:02d}"
    return samples


def build_turing(case_dir, models, texts):
    pairs = []
    for i, entry in enumerate(texts):
        model = models[i % len(models)]
        clone_src = clone_audio_path(case_dir, model, i, entry["id"])
        real_src = real_audio_path(entry["source_audio"])
        if not (clone_src.exists() and real_src.exists()):
            continue
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


def main():
    case_name = sys.argv[1] if len(sys.argv) > 1 else None
    case = load_case(case_name)
    texts = load_texts(case)
    case_dir = RESULTS_DIR / case["name"]
    if not case_dir.exists():
        raise SystemExit(f"{case_dir} not found - run evaluation/run_eval.py first")

    models = find_models(case_dir)
    print(f"Case: {case['name']}  models: {models}  texts: {len(texts)}")

    nmos = build_nmos(case_dir, models, texts)
    turing = build_turing(case_dir, models, texts)

    samples_dir = OUT_DIR / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    manifest_nmos = []
    key_nmos = []
    for s in nmos:
        dest = samples_dir / f"{s['anon_id']}.{AUDIO_EXT}"
        shutil.copyfile(s["src"], dest)
        manifest_nmos.append({"id": s["anon_id"], "file": f"samples/{dest.name}"})
        key_nmos.append({"anon_id": s["anon_id"], "model": s["model"], "text_id": s["text_id"]})

    manifest_turing = []
    key_turing = []
    for p in turing:
        dest_real = samples_dir / f"{p['pair_id']}_real.{AUDIO_EXT}"
        dest_clone = samples_dir / f"{p['pair_id']}_clone.{AUDIO_EXT}"
        shutil.copyfile(p["real_src"], dest_real)
        shutil.copyfile(p["clone_src"], dest_clone)
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
        })

    manifest = {"nmos": manifest_nmos, "turing": manifest_turing}
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    html = template.replace("__MANIFEST_JSON__", json.dumps(manifest, ensure_ascii=False))
    (OUT_DIR / "index.html").write_text(html, encoding="utf-8")

    key = {"case": case["name"], "nmos": key_nmos, "turing": key_turing}
    (OUT_DIR / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")

    (OUT_DIR / "responses").mkdir(exist_ok=True)

    share_zip = OUT_DIR / "share.zip"
    with zipfile.ZipFile(share_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(OUT_DIR / "index.html", "index.html")
        for f in samples_dir.iterdir():
            zf.write(f, f"samples/{f.name}")

    print(f"\n{len(nmos)} N-MOS samples, {len(turing)} Turing pairs.")
    print(f"Share {share_zip} with raters (does NOT contain key.json).")
    print(f"Save their downloaded ratings_*.json into {OUT_DIR / 'responses'}/")
    print("Do NOT show key.json to raters - it unblinds the test.")


if __name__ == "__main__":
    main()