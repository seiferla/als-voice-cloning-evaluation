"""Builds ONE canonical enrollment file and records exactly what went into it.

Usage:
    uv run python evaluation/prepare_enrollment.py data/wav/AUDIO-2021-07-31-14-49-50.wav

Why this exists
---------------
The three systems were not cloned from the same input:

    qwen3     enable_preprocess=True  -> server-side denoising and normalization
    voxtral   raw file bytes          + a hardcoded gender="female" hint
    s2pro     uploaded by hand through the Fish Audio web UI

So the comparison "which system clones this voice best" is confounded at the
first step. A difference in SIM can come from the model or from the fact that one
model was handed cleaner audio than another, and nothing in the results can tell
those apart. This is the largest single threat to the conclusion, larger than
anything in the listening test.

The fix is one reference file, preprocessed once, here, under your control, and
handed to all three with every provider-side preprocessing switched OFF. Then the
only thing that differs between the systems is the system.

What this script does
---------------------
Produces `data/enrollment/reference_<stem>.wav` - mono, 24 kHz, loudness
normalized, optionally trimmed to a fixed duration - plus `enrollment.json`
recording the source file, its sha256, the output sha256, the exact parameters
and the provider settings each system must be given. Deterministic: the same
input and parameters produce a byte-identical file.

It deliberately does NOT call the three APIs. Enrollment costs money, two of the
three keep the audio, and s2pro has no API for it at all - that has to stay a
decision you make, not a side effect of running a script. What it does give you
is the file and the checklist, so the three enrollments are reproducible and
documented instead of remembered.

After running it
----------------
    qwen3    python qwen_voice_enroll.py create --audio <output> --name myecho_v2
             then set enable_preprocess=False (src/myecho/models/qwen3.py) -
             it defaults to True, which is the confound
    voxtral  VoxtralTTS().create_voice(<output>)  - and drop the gender hint
             unless you can pass the same hint to all three
    s2pro    upload <output> in the Fish Audio web UI, no trimming, no cleanup

Then put the three new ids in .env, re-run run_eval.py, and note the
enrollment.json hash in the report.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myecho.evaluation.audio_qc import (  # noqa: E402
    load_mono,
    measure,
    normalize,
    to_common_sr,
)

OUT_DIR = Path("data/enrollment")
MANIFEST = OUT_DIR / "enrollment.json"

# 24 kHz is the lowest native rate among the three systems, so every provider
# gets the same bandwidth rather than one of them getting more.
TARGET_SR = 24000
# Enrollment audio is normally wanted at a normal speech level, not at the
# -23 LUFS broadcast target used for listening-test stimuli.
TARGET_LUFS = -20.0

PROVIDER_SETTINGS = {
    "qwen3": {
        "enable_preprocess": False,
        "why": "True runs server-side denoising, so qwen3 would receive different "
               "audio from the other two",
        "how": "python qwen_voice_enroll.py create --audio {output} --name {name}",
    },
    "voxtral": {
        "gender_hint": None,
        "why": "a hardcoded gender hint is an extra input the other two never got",
        "how": "VoxtralTTS().create_voice('{output}')",
    },
    "s2pro": {
        "web_ui": True,
        "why": "no enrollment API - upload by hand, and do not use any of the "
               "cleanup or trim options offered by the UI",
        "how": "upload {output} at fish.audio, then put the reference id in .env",
    },
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("source", help="the ONE recording all three systems get")
    parser.add_argument("--name", default="myecho", help="voice name to enroll under")
    parser.add_argument("--seconds", type=float, default=None,
                        help="trim to this many seconds from the start "
                             "(providers differ in what they accept; trimming here "
                             "means all three get the same excerpt)")
    parser.add_argument("--target-lufs", type=float, default=TARGET_LUFS)
    parser.add_argument("--target-sr", type=int, default=TARGET_SR)
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        raise SystemExit(f"{source} not found")

    x, sr = load_mono(source)
    before = measure(x, sr)
    x, sr = to_common_sr(x, sr, args.target_sr)
    if args.seconds:
        x = x[: int(args.seconds * sr)]
    y, level = normalize(x, sr, args.target_lufs)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUT_DIR / f"reference_{source.stem}.wav"
    sf.write(str(output), y, sr, subtype="PCM_16")
    after = measure(y, sr)

    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "source": str(source),
        "source_sha256": sha256(source),
        "source_measured": before,
        "output": str(output),
        "output_sha256": sha256(output),
        "output_measured": after,
        "level": level,
        "parameters": {
            "target_sample_rate": args.target_sr,
            "target_lufs": args.target_lufs,
            "trim_seconds": args.seconds,
            "denoising": "none - applied by nobody, so all three get the same signal",
        },
        "voice_name": args.name,
        "provider_settings": {
            name: {**cfg, "how": cfg["how"].format(output=output, name=args.name)}
            for name, cfg in PROVIDER_SETTINGS.items()
        },
        "enrolled": {name: None for name in PROVIDER_SETTINGS},
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    print(f"Canonical enrollment reference -> {output}")
    print(f"  {after['duration_s']} s, {sr} Hz mono, "
          f"{after['lufs']} LUFS (was {before['lufs']}), gain {level['gain_db']:+.1f} dB")
    print(f"  sha256 {manifest['output_sha256'][:16]}...")
    if not level["reached_target"]:
        print(f"  WARNING: could not reach {args.target_lufs} LUFS - {level['reason']}. "
              "The file is usable but quieter than asked for; note it in the report.")
    print(f"\nManifest -> {MANIFEST}")
    print("\nEnroll all three from THIS file, with provider preprocessing off:\n")
    for name, cfg in manifest["provider_settings"].items():
        print(f"  {name}")
        print(f"    {cfg['how']}")
        print(f"    why it matters: {cfg['why']}")
    print("\nThen record the three new voice ids under \"enrolled\" in the manifest,")
    print("put them in .env, and re-run evaluation/run_eval.py. Until all three come")
    print("from this one file, a SIM difference between systems cannot be attributed")
    print("to the systems.")


if __name__ == "__main__":
    main()
