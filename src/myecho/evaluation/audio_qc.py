"""Level measurement and normalization for listening-test stimuli.

Why this exists: loudness differences confound subjective ratings. A quieter
sample is rated less natural almost regardless of its actual quality, so any
systematic level difference between models turns into a fake quality
difference. This material has one: qwen3 renders about 7 dB quieter than
s2pro and voxtral, and the real reference recordings sit between them (which
would also leak a "the louder one is fake" cue into the Turing test).

So every stimulus gets normalized to one common loudness before a listening
test. Loudness is ITU-R BS.1770-4 integrated loudness (pyloudnorm), not plain
RMS, because its gating ignores silence - a broken render that is half
silence is then measured on its speech instead of its gaps.

Normalization only removes the gain confound. Genuine defects (gappy or
breathy speech, artifacts) survive it and still get heard and rated, which is
what we want. A file that cannot reach the target without clipping is
reported, not silently rescued - see qc_samples.py.
"""

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

# EBU R128 target. Sits mid-range for this material, so most files need only
# a small gain change and clipping stays unlikely.
TARGET_LUFS = -23.0
PEAK_CEILING_DBFS = -1.0

# Frames below this count as silence for silence_ratio / active_rms.
SILENCE_THRESHOLD_DBFS = -50.0

# Lowest native rate across the models (qwen3/voxtral 24k, s2pro 44.1k).
# Everything is resampled to this so audio bandwidth can't differ per system.
COMMON_SR = 24000

_FRAME_S = 0.02


def load_mono(path):
    """Read any audio file as float32 mono."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return data.mean(axis=1), sr


def _db(x):
    return 20.0 * np.log10(max(float(x), 1e-12))


def measure(x, sr):
    """Level/QC measurements for one signal."""
    try:
        import pyloudnorm
    except ImportError:
        raise SystemExit("pyloudnorm missing - run: uv sync --extra eval")

    duration_s = len(x) / sr
    # BS.1770 needs at least one 400 ms block.
    if duration_s >= 0.4:
        lufs = pyloudnorm.Meter(sr).integrated_loudness(x)
    else:
        lufs = float("-inf")

    frame = max(int(_FRAME_S * sr), 1)
    frames = x[: len(x) // frame * frame].reshape(-1, frame) if len(x) >= frame else x.reshape(1, -1)
    frame_db = np.array([_db(np.sqrt((f ** 2).mean())) for f in frames])
    active = frame_db > SILENCE_THRESHOLD_DBFS

    return {
        "duration_s": round(duration_s, 2),
        "lufs": round(lufs, 2) if np.isfinite(lufs) else None,
        "peak_dbfs": round(_db(np.abs(x).max() if len(x) else 0.0), 2),
        "rms_dbfs": round(_db(np.sqrt((x ** 2).mean()) if len(x) else 0.0), 2),
        "active_rms_dbfs": round(float(frame_db[active].mean()), 2) if active.any() else None,
        "silence_ratio": round(float((~active).mean()), 3),
        "clipped_ratio": round(float((np.abs(x) >= 0.999).mean()), 5),
    }


def to_common_sr(x, sr, target_sr=COMMON_SR):
    """Resample so every system's stimuli share one audio bandwidth."""
    if sr == target_sr:
        return x, sr
    from math import gcd

    g = gcd(int(sr), int(target_sr))
    return resample_poly(x, int(target_sr // g), int(sr // g)).astype(np.float32), target_sr


def normalize(x, sr, target_lufs=TARGET_LUFS, peak_ceiling_dbfs=PEAK_CEILING_DBFS):
    """Scale to target loudness, but never above the peak ceiling.

    Returns (signal, info). info["reached_target"] is False when the peak
    ceiling forced a lower gain - such a file is still too quiet relative to
    the others and should be excluded rather than used as-is.
    """
    before = measure(x, sr)
    if before["lufs"] is None:
        return x, {"gain_db": 0.0, "achieved_lufs": None, "reached_target": False,
                   "reason": "too short to measure"}

    gain_db = target_lufs - before["lufs"]
    gain = 10.0 ** (gain_db / 20.0)

    peak = float(np.abs(x).max()) if len(x) else 0.0
    ceiling = 10.0 ** (peak_ceiling_dbfs / 20.0)
    reached = True
    if peak * gain > ceiling:
        gain = ceiling / peak if peak > 0 else 1.0
        gain_db = _db(gain)
        reached = False

    y = (x * gain).astype(np.float32)
    return y, {
        "gain_db": round(gain_db, 2),
        "achieved_lufs": round(before["lufs"] + gain_db, 2),
        "reached_target": reached,
        "reason": "" if reached else f"would clip: needs +{target_lufs - before['lufs']:.1f} dB",
    }


def prepare_stimulus(src_path, dest_path, target_lufs=TARGET_LUFS, target_sr=COMMON_SR):
    """Read, resample to the common rate, loudness-normalize, write."""
    x, sr = load_mono(src_path)
    x, sr = to_common_sr(x, sr, target_sr)
    y, info = normalize(x, sr, target_lufs)
    sf.write(str(dest_path), y, sr, subtype="PCM_16")
    info["sample_rate"] = sr
    return info
