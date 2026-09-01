"""Tests for level and pause-floor matching.

These guard the two confounds the module exists to remove: a level difference
between models, and a difference in the recording chain that only the real
recordings went through.
"""

import numpy as np
import pytest

from myecho.evaluation.audio_qc import (
    TARGET_LUFS,
    match_noise_floor,
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
    before, _ = speech_and_pause_level(x, SR)
    after, _ = speech_and_pause_level(y, SR)
    assert after == pytest.approx(before, abs=0.5)


def test_silence_ratio_sees_the_pauses():
    m = measure(speech_with_pauses(floor=0.0), SR)
    assert 0.4 < m["silence_ratio"] < 0.6
