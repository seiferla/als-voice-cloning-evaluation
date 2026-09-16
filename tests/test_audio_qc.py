"""Tests for level and pause-floor matching.

These guard the two confounds the module exists to remove: a level difference
between models, and a difference in the recording chain that only the real
recordings went through.
"""

import numpy as np
import pytest

from myecho.evaluation.audio_qc import (
    SILENCE_THRESHOLD_DBFS,
    TARGET_LUFS,
    _frame_db,
    match_noise_floor,
    mnru,
    measure,
    normalize,
    speech_and_pause_level,
)

SR = 24000


def speech_with_pauses(sr=SR, level=0.1, floor=0.0, seconds=6.0, seed=0):
    """Alternating 0.5 s of tone and 0.5 s of pause, plus an optional noise floor."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    tone = level * np.sin(2 * np.pi * 200 * t).astype(np.float32)
    gate = ((t * 2).astype(int) % 2 == 0).astype(np.float32)
    x = tone * gate
    if floor > 0:
        x = x + floor * rng.standard_normal(n).astype(np.float32)
    return x.astype(np.float32)


def test_normalize_reaches_the_target_loudness():
    x = speech_with_pauses(level=0.01)
    y, info = normalize(x, SR)
    assert info["reached_target"]
    assert measure(y, SR)["lufs"] == pytest.approx(TARGET_LUFS, abs=0.5)


def test_normalize_refuses_to_clip_and_says_so():
    # A file that cannot reach the target without clipping must be reported,
    # not silently rescued: it is still quieter than the others.
    x = speech_with_pauses(level=0.999)
    _, info = normalize(x, SR, target_lufs=0.0)
    assert not info["reached_target"]
    assert "would clip" in info["reason"]


def test_pause_floor_is_raised_to_the_common_target():
    quiet_pauses = speech_with_pauses(floor=1e-6)      # like a denoised recording
    noisy_pauses = speech_with_pauses(floor=1e-3, seed=1)

    a, _ = match_noise_floor(quiet_pauses, SR, target_snr_db=26.0)
    b, _ = match_noise_floor(noisy_pauses, SR, target_snr_db=26.0)

    def snr(x):
        speech, pause = speech_and_pause_level(x, SR)
        return speech - pause

    # Before: the two differ by a lot. After: they agree, which is the point -
    # a rater must not be able to tell the sources apart by their gaps.
    assert snr(quiet_pauses) - snr(noisy_pauses) > 10
    assert abs(snr(a) - snr(b)) < 3


def test_a_floor_already_above_target_is_left_alone():
    # Lowering a floor would mean denoising one condition and not the others,
    # which is exactly the confound this removes.
    noisy = speech_with_pauses(floor=1e-2)
    y, info = match_noise_floor(noisy, SR, target_snr_db=26.0)
    assert not info["raised"]
    assert np.array_equal(y, noisy)


def test_noise_matching_is_deterministic_for_a_given_seed():
    x = speech_with_pauses(floor=1e-6)
    a, _ = match_noise_floor(x, SR, seed=42)
    b, _ = match_noise_floor(x, SR, seed=42)
    c, _ = match_noise_floor(x, SR, seed=43)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_noise_matching_does_not_move_the_speech_level():
    x = speech_with_pauses(floor=1e-6)
    y, _ = match_noise_floor(x, SR)

    # Measured over the frames that were speech BEFORE the noise was added.
    # Re-deriving the mask from y instead counts the now-filled pauses as
    # speech and reports a ~10 dB drop that never happened to the speech - the
    # earlier version of this test did exactly that and passed only because the
    # added noise happened to land just under the old silence threshold.
    active = _frame_db(x, SR) > SILENCE_THRESHOLD_DBFS
    before, _ = speech_and_pause_level(x, SR, active=active)
    after, _ = speech_and_pause_level(y, SR, active=active)
    assert after == pytest.approx(before, abs=0.5)


def test_pause_floor_that_cannot_be_measured_is_reported_as_such():
    # No frame below the silence threshold means there is no pause floor to
    # measure, and the percentile fallback returns speech dynamics instead. That
    # must be visible, or a stimulus gets skipped on a number that is not an SNR.
    no_pauses = speech_with_pauses(floor=1e-2)
    _, info = match_noise_floor(no_pauses, SR, target_snr_db=26.0)
    assert info["pause_measured"] == "fallback"
    assert not info["raised"]


def test_mnru_noise_follows_the_signal():
    """The defining property of MNRU vs. simply adding noise.

    The impairment is multiplied by the waveform, so it degrades the speech and
    leaves the gaps alone. If it leaked into the pauses it would raise the noise
    floor, and the anchor would then differ from the other conditions in the one
    property audio_qc spends its effort equalizing.
    """
    x = speech_with_pauses(floor=0.0)
    y = mnru(x, q_db=15.0)
    added = y - x

    silent = x == 0.0
    assert silent.any() and (~silent).any()
    energy_in_pauses = float(np.abs(added[silent]).max())
    energy_in_speech = float(np.sqrt((added[~silent] ** 2).mean()))
    assert energy_in_pauses == 0.0
    assert energy_in_speech > 0.0


def test_mnru_q_controls_how_bad_it_is():
    x = speech_with_pauses(floor=0.0)
    mild = mnru(x, q_db=25.0)
    harsh = mnru(x, q_db=5.0)

    def added_rms(y):
        return float(np.sqrt(((y - x) ** 2).mean()))

    assert added_rms(harsh) > added_rms(mild)
    # Deterministic, so a rebuild yields byte-identical anchors.
    assert np.array_equal(mnru(x, q_db=15.0, seed=7), mnru(x, q_db=15.0, seed=7))
    assert not np.array_equal(mnru(x, q_db=15.0, seed=7), mnru(x, q_db=15.0, seed=8))


def test_silence_ratio_sees_the_pauses():
    m = measure(speech_with_pauses(floor=0.0), SR)
    assert 0.4 < m["silence_ratio"] < 0.6
