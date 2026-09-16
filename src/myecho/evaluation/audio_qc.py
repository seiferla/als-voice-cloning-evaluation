"""Loudness and pause-floor matching for listening-test stimuli.

Two things are equalized across every stimulus before a listening test, because
either of them would otherwise be rated instead of the voice:

    level        a quieter sample is rated less natural regardless of quality
    pause floor  the real recordings are denoised phone memos and the clones are
                 generated clean, which leaves the gaps between words as a
                 giveaway

Loudness is ITU-R BS.1770-4 integrated loudness (pyloudnorm), not RMS: its
gating ignores silence, so a gappy render is measured on its speech. Noise is
only ever added to a pause floor, never removed - lowering one would mean
denoising one condition and not the others.

Measured numbers and the reasoning: docs/METHODS.md sections 4 and 5.
"""

import zlib
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from myecho.evaluation.excerpt import cut

# EBU R128 target. Sits mid-range for this material, so most files need only
# a small gain change and clipping stays unlikely.
TARGET_LUFS = -23.0
PEAK_CEILING_DBFS = -1.0

# Frames below this count as silence for silence_ratio / active_rms.
#
# A frame counts as silence when it falls BELOW this, so a lower value makes
# silence harder to reach, not easier. Measured on this material: -50 dBFS
# leaves 3 of 70 stimuli without a measurable pause floor, -60 dBFS leaves 17.
#
# The value is entangled with TARGET_PAUSE_SNR_DB (26 dB) and the -23 LUFS
# programme level: a floor sitting at the target lands near -49 dBFS, i.e. right
# at this threshold, so files whose floor is at or worse than the target fall
# into the percentile fallback in speech_and_pause_level() and keep an unmatched
# floor. `pause_measured` in the returned info says when that happened.
#
# The principled fix is a threshold relative to each file's own speech level
# rather than an absolute one - not done here, because it also changes
# silence_ratio in measure(), which analyze_objective.py reports.
SILENCE_THRESHOLD_DBFS = -50.0

# Lowest native rate across the models (qwen3/voxtral 24k, s2pro 44.1k).
# Everything is resampled to this so audio bandwidth can't differ per system.
COMMON_SR = 24000

# Common pause floor for every stimulus, as speech level minus floor. Chosen
# just under the quietest natural floor in this material (voxtral, 28.7 dB) so
# matching only ever adds noise and never has to lift a stimulus much.
TARGET_PAUSE_SNR_DB = 26.0

_FRAME_S = 0.02

# Low anchor for the listening test, per ITU-T P.810 (Modulated Noise Reference
# Unit). Q is the speech-to-modulated-noise ratio in dB: lower is worse.
#
# The usable range is narrow and bounded on both sides. Below ~22 dB the
# impairment dominates the item and raters call the anchor a defective
# recording instead of a poor one (15 dB did exactly that). Above
# TARGET_PAUSE_SNR_DB (26 dB) the added noise sits under the pause floor every
# stimulus already carries, so it is masked and anchors nothing. 24 dB is inside
# that window - audible, 2 dB above the common floor, still judged as speech.
#
# Note the dimension mismatch this leaves: P.810 was written for codec and
# transmission quality, while the rating question here is naturalness. The
# anchor marks the bottom of the scale; it is not a bad voice.
#
# Why an anchor at all: with only good conditions in the set, raters compress
# their range and never award 1 or 2, which makes the absolute MOS values
# incomparable to any other study and hides scale-usage differences between
# people. The anchor is a fixed reference point, not a system under test.
MNRU_Q_DB = 24.0


def mnru(x, q_db=MNRU_Q_DB, seed=0):
    """Degrade a signal with signal-correlated noise (ITU-T P.810).

    y = x + 10^(-Q/20) * x * n,  n ~ N(0, 1)

    The noise is MULTIPLIED by the signal, so it scales with the waveform
    instead of sitting under it: the result sounds rough and grainy rather than
    hissy, and it stays put during silence. That is the point of MNRU over
    additive noise - it degrades the speech itself, which is what the scale's
    lower end is supposed to represent, and it needs no level calibration
    because the impairment is relative to the signal.
    """
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(len(x)).astype(np.float32)
    return (x + (10.0 ** (-q_db / 20.0)) * x * noise).astype(np.float32)


def load_mono(path):
    """Read any audio file as float32 mono."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return data.mean(axis=1), sr


def _db(x):
    return 20.0 * np.log10(max(float(x), 1e-12))


def _frame_db(x, sr):
    """Per-frame RMS in dBFS, on 20 ms frames."""
    frame = max(int(_FRAME_S * sr), 1)
    frames = (x[: len(x) // frame * frame].reshape(-1, frame)
              if len(x) >= frame else x.reshape(1, -1))
    return np.array([_db(np.sqrt((f ** 2).mean())) for f in frames])


def speech_and_pause_level(x, sr, active=None):
    """(speech_dbfs, pause_dbfs), as medians of the active and the silent frames.

    Medians so one loud breath in a gap does not decide the floor. A render with
    no frame below the threshold falls back to the quiet tail of the frame
    distribution - treating the floor as unknown would make the caller add noise
    to a stimulus that is already noisy.

    `active` overrides the frame classification. Pass the mask taken from the
    signal BEFORE noise was added when measuring the result of adding it:
    re-deriving the mask from the noisy signal counts the now-filled pauses as
    speech and drags the speech median down. With the absolute threshold that
    only stayed hidden because the added noise happened to land ~1 dB below it.
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
    """Pink-ish noise (1/f amplitude), unit RMS.

    White noise at this level reads as tape hiss and draws attention to itself;
    1/f shaping sits closer to the room tone of the original recordings, so the
    matched floor sounds like a recording rather than like an added defect.
    """
    spectrum = rng.standard_normal(n // 2 + 1) + 1j * rng.standard_normal(n // 2 + 1)
    f = np.arange(len(spectrum))
    spectrum[1:] /= np.sqrt(f[1:])
    spectrum[0] = 0.0  # no DC
    y = np.fft.irfft(spectrum, n).astype(np.float32)
    rms = float(np.sqrt((y ** 2).mean()))
    return y / rms if rms > 0 else y


def match_noise_floor(x, sr, target_snr_db=TARGET_PAUSE_SNR_DB, seed=0):
    """Raise the pause floor to a common speech-to-pause distance by adding noise.

    Only ever adds. A stimulus whose floor already sits above the target keeps
    it and is reported (`raised` False) - lowering a floor would mean denoising
    one condition and not the others, which is the confound this removes.

    Deterministic: the noise is drawn from a seed derived from the stimulus
    name, so rebuilding the test yields byte-identical files.
    """
    # Classified once, on the untouched signal, and reused for the measurement
    # after the noise is added - see speech_and_pause_level().
    active = _frame_db(x, sr) > SILENCE_THRESHOLD_DBFS
    speech, pause = speech_and_pause_level(x, sr, active=active)
    # Whether `pause` is a real noise floor or the percentile fallback. Without
    # this the two cases are indistinguishable downstream, and a fallback value
    # is not an SNR - it is the dynamic range of the speech.
    n_silent = int((~active).sum())
    info = {"pause_snr_before_db": None, "pause_snr_after_db": None,
            "noise_added_dbfs": None, "raised": False,
            "n_silent_frames": n_silent,
            "pause_measured": "floor" if n_silent >= 3 else "fallback"}
    if speech is None or pause is None:
        return x, info

    target_floor_db = speech - target_snr_db
    info["pause_snr_before_db"] = round(speech - pause, 2)

    # Noise and signal are uncorrelated, so their powers add: to reach the
    # target floor from an existing one, add the difference in power, not the
    # difference in level.
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


def prepare_stimulus(src_path, dest_path, target_lufs=TARGET_LUFS, target_sr=COMMON_SR,
                     span=None, target_pause_snr_db=TARGET_PAUSE_SNR_DB,
                     mnru_q_db=None):
    """Read, resample, cut, normalize the level, match the pause floor, write.

    The cut runs before normalization because BS.1770 loudness is integrated
    over the whole signal: normalizing a 36 s render and then cutting 10 s out
    of it would leave the excerpts at different levels again.

    span is {"start", "end"} in seconds, from select_excerpts.py.

    mnru_q_db turns the stimulus into the low anchor. Applied after the cut and
    before normalization, so the anchor lands at the same loudness as every
    other stimulus and differs only in the impairment.
    """
    x, sr = load_mono(src_path)
    x, sr = to_common_sr(x, sr, target_sr)
    if span is not None:
        x = cut(x, sr, span["start"], span["end"])
    if mnru_q_db is not None:
        x = mnru(x, mnru_q_db, seed=zlib.crc32(Path(dest_path).name.encode("utf-8")))
    y, info = normalize(x, sr, target_lufs)

    # Seed from the destination name: same stimulus set in, same bytes out.
    seed = zlib.crc32(Path(dest_path).name.encode("utf-8"))
    y, noise_info = match_noise_floor(y, sr, target_pause_snr_db, seed=seed)
    info.update(noise_info)

    sf.write(str(dest_path), y, sr, subtype="PCM_16")
    info["sample_rate"] = sr
    info["duration_s"] = round(len(y) / sr, 2)
    info["mnru_q_db"] = mnru_q_db
    return info
