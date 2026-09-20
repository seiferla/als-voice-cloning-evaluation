"""Loudness and pause-floor matching for listening-test stimuli.

Level is equalized because a quieter sample is rated less natural. The pause
floor is equalized because the real recordings are denoised phone memos and the
clones are generated clean, which makes the gaps a giveaway. Noise is only ever
added, never removed. See docs/METHODS.md sections 4 and 5.
"""

import zlib
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from myecho.evaluation.excerpt import cut

TARGET_LUFS = -23.0  # EBU R128
PEAK_CEILING_DBFS = -1.0

# Frames below this are silence. -50 leaves 3 of 70 stimuli without a
# measurable floor, -60 leaves 17. A floor at the target sits near -49 dBFS, so
# files at or below the target fall into the percentile fallback (see
# speech_and_pause_level); `pause_measured` reports that.
SILENCE_THRESHOLD_DBFS = -50.0

COMMON_SR = 24000  # lowest native rate across the models

# Speech level minus floor. Just under the quietest natural floor here
# (voxtral, 28.7 dB), so matching only ever adds noise.
TARGET_PAUSE_SNR_DB = 26.0

_FRAME_S = 0.02


def load_mono(path):
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return data.mean(axis=1), sr


def _db(x):
    return 20.0 * np.log10(max(float(x), 1e-12))


def _frame_db(x, sr):
    """Per-frame RMS in dBFS, 20 ms frames."""
    frame = max(int(_FRAME_S * sr), 1)
    frames = (x[: len(x) // frame * frame].reshape(-1, frame)
              if len(x) >= frame else x.reshape(1, -1))
    return np.array([_db(np.sqrt((f ** 2).mean())) for f in frames])


def speech_and_pause_level(x, sr, active=None):
    """(speech_dbfs, pause_dbfs) as medians over active and silent frames.

    Pass `active` from the signal before noise was added when measuring the
    result; re-deriving it from the noisy signal counts filled pauses as speech.
    """
    frame_db = _frame_db(x, sr)
    if active is None:
        active = frame_db > SILENCE_THRESHOLD_DBFS
    if not active.any():
        return None, None
    speech = float(np.median(frame_db[active]))
    if (~active).sum() >= 3:
        pause = float(np.median(frame_db[~active]))
    else:
        pause = float(np.percentile(frame_db, 10))
    return speech, pause


def _shaped_noise(n, rng):
    """1/f noise, unit RMS. Closer to room tone than white noise."""
    spectrum = rng.standard_normal(n // 2 + 1) + 1j * rng.standard_normal(n // 2 + 1)
    f = np.arange(len(spectrum))
    spectrum[1:] /= np.sqrt(f[1:])
    spectrum[0] = 0.0
    y = np.fft.irfft(spectrum, n).astype(np.float32)
    rms = float(np.sqrt((y ** 2).mean()))
    return y / rms if rms > 0 else y


def match_noise_floor(x, sr, target_snr_db=TARGET_PAUSE_SNR_DB, seed=0):
    """Raise the pause floor to speech - target_snr_db by adding noise. Only ever adds."""
    active = _frame_db(x, sr) > SILENCE_THRESHOLD_DBFS
    speech, pause = speech_and_pause_level(x, sr, active=active)
    n_silent = int((~active).sum())
    info = {"pause_snr_before_db": None, "pause_snr_after_db": None,
            "noise_added_dbfs": None, "raised": False,
            "n_silent_frames": n_silent,
            "pause_measured": "floor" if n_silent >= 3 else "fallback"}
    if speech is None or pause is None:
        return x, info

    target_floor_db = speech - target_snr_db
    info["pause_snr_before_db"] = round(speech - pause, 2)

    # Uncorrelated noise adds in power, not in level.
    have = 10.0 ** (pause / 10.0)
    want = 10.0 ** (target_floor_db / 10.0)
    if want <= have:
        info["pause_snr_after_db"] = info["pause_snr_before_db"]
        return x, info

    noise_rms = float(np.sqrt(want - have))
    rng = np.random.default_rng(seed)
    y = (x + noise_rms * _shaped_noise(len(x), rng)).astype(np.float32)

    speech_after, pause_after = speech_and_pause_level(y, sr, active=active)
    info.update({
        "noise_added_dbfs": round(_db(noise_rms), 2),
        "pause_snr_after_db": (round(speech_after - pause_after, 2)
                               if pause_after is not None else None),
        "raised": True,
    })
    return y, info


def measure(x, sr):
    """Level and QC measurements for one signal."""
    try:
        import pyloudnorm
    except ImportError:
        raise SystemExit("pyloudnorm missing - run: uv sync --extra eval")

    duration_s = len(x) / sr
    if duration_s >= 0.4:  # BS.1770 needs one 400 ms block
        lufs = pyloudnorm.Meter(sr).integrated_loudness(x)
    else:
        lufs = float("-inf")

    frame_db = _frame_db(x, sr)
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
    if sr == target_sr:
        return x, sr
    from math import gcd

    g = gcd(int(sr), int(target_sr))
    return resample_poly(x, int(target_sr // g), int(sr // g)).astype(np.float32), target_sr


def normalize(x, sr, target_lufs=TARGET_LUFS, peak_ceiling_dbfs=PEAK_CEILING_DBFS):
    """Scale to target loudness without exceeding the peak ceiling.

    info["reached_target"] is False when the ceiling forced a lower gain.
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


def prepare_stimulus(src_path, dest_path, target_lufs=TARGET_LUFS, target_sr=COMMON_SR,
                     span=None, target_pause_snr_db=TARGET_PAUSE_SNR_DB):
    """Resample, cut, normalize, match the pause floor, write.

    Cut before normalizing: BS.1770 integrates over the whole signal.
    """
    x, sr = load_mono(src_path)
    x, sr = to_common_sr(x, sr, target_sr)
    if span is not None:
        x = cut(x, sr, span["start"], span["end"])
    y, info = normalize(x, sr, target_lufs)

    seed = zlib.crc32(Path(dest_path).name.encode("utf-8"))  # same name, same bytes
    y, noise_info = match_noise_floor(y, sr, target_pause_snr_db, seed=seed)
    info.update(noise_info)

    sf.write(str(dest_path), y, sr, subtype="PCM_16")
    info["sample_rate"] = sr
    info["duration_s"] = round(len(y) / sr, 2)
    return info
