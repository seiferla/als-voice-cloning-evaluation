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

Level is not the only channel confound, and it was not the worst one. The real
recordings are 2021 phone voice memos put through DeepFilterNet, the clones are
generated clean. Measured on the first stimulus set, that left the pauses
telling the rater the answer:

    source   pause fraction   speech - pause floor
    REAL     0.22             39.4 dB
    s2pro    0.08             33.0 dB
    voxtral  0.15             28.7 dB
    qwen3    0.10             29.6 dB

A single threshold on either column separates real from clone with 96 %
accuracy, so a rater could win the Turing test on "which one has clean gaps"
without ever judging the voice. And the cue is our own doing: denoising lifts
the real recordings from 0.132 / 30.6 dB to 0.219 / 36.8 dB, i.e. the denoiser
gives them digitally silent pauses that no TTS model produces.

So every stimulus also gets its pause floor matched to one common value by
adding shaped noise (never by removing any). After that the pause *structure* -
how often and how long the voice stops - still differs, which is right: that is
a property of the model that raters should hear. What is gone is the recording
chain that only the real side went through.
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
SILENCE_THRESHOLD_DBFS = -50.0

# Lowest native rate across the models (qwen3/voxtral 24k, s2pro 44.1k).
# Everything is resampled to this so audio bandwidth can't differ per system.
COMMON_SR = 24000

# Common pause floor for every stimulus, as speech level minus floor. Chosen
# just under the quietest natural floor in this material (voxtral, 28.7 dB) so
# matching only ever adds noise and never has to lift a stimulus much.
TARGET_PAUSE_SNR_DB = 26.0

_FRAME_S = 0.02


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


def speech_and_pause_level(x, sr):
    """(speech_dbfs, pause_dbfs) - the level of the speech and of the gaps.

    Medians, not means: one loud breath or one click in an otherwise silent gap
    should not decide what "the floor" of a stimulus is.

    Some renders have no frame below the silence threshold at all - either they
    never stop speaking, or their noise floor is itself above the threshold. In
    that case the quiet tail of the frame distribution is used instead. Falling
    back to "unknown" would be worse than a rough estimate: the caller would
    then treat the floor as silent and add noise to a stimulus that is already
    noisy, which is the opposite of matching it.

    pause_dbfs is None only for a signal with no measurable content at all.
    """
    frame_db = _frame_db(x, sr)
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
    speech, pause = speech_and_pause_level(x, sr)
    info = {"pause_snr_before_db": None, "pause_snr_after_db": None,
            "noise_added_dbfs": None, "raised": False}
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

    speech_after, pause_after = speech_and_pause_level(y, sr)
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
                     span=None, target_pause_snr_db=TARGET_PAUSE_SNR_DB):
    """Read, resample, cut a span, normalize the level, match the pause floor, write.

    The order is deliberate:

    1. resample - one audio bandwidth for every system
    2. cut - BEFORE normalization, because BS.1770 integrated loudness is
       measured over the whole signal; normalizing a 36 s render and then
       cutting 10 s out of it would leave the excerpts at different levels
       again, which is the confound this module exists to remove
    3. normalize - one loudness for every system
    4. match the pause floor - one recording chain for every system, so the
       Turing test cannot be won on "the real one has clean gaps"

    span is {"start", "end"} in seconds, from select_excerpts.py.
    """
    x, sr = load_mono(src_path)
    x, sr = to_common_sr(x, sr, target_sr)
    if span is not None:
        x = cut(x, sr, span["start"], span["end"])
    y, info = normalize(x, sr, target_lufs)

    # Seed from the destination name: same stimulus set in, same bytes out.
    seed = zlib.crc32(Path(dest_path).name.encode("utf-8"))
    y, noise_info = match_noise_floor(y, sr, target_pause_snr_db, seed=seed)
    info.update(noise_info)

    sf.write(str(dest_path), y, sr, subtype="PCM_16")
    info["sample_rate"] = sr
    info["duration_s"] = round(len(y) / sr, 2)
    return info
