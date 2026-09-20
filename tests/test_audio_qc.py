"""Level and pause-floor matching."""

import numpy as np
import pytest

from myecho.evaluation.audio_qc import (
    SILENCE_THRESHOLD_DBFS,
    TARGET_LUFS,
    _frame_db,
    match_noise_floor,
    measure,
    normalize,
    speech_and_pause_level,
)

SR = 24000


def speech_with_pauses(sr=SR, level=0.1, floor=0.0, seconds=6.0, seed=0):
    """Alternating 0.5 s tone and 0.5 s pause, optional noise floor."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    tone = level * np.sin(2 * np.pi * 200 * t).astype(np.float32)
    gate = ((t * 2).astype(int) % 2 == 0).astype(np.float32)
    x = tone * gate
    if floor > 0:
        x = x + floor * rng.standard_normal(n).astype(np.float32)
    return x.astype(np.float32)


def test_normalize_hits_target():
    x = speech_with_pauses(level=0.01)
    y, info = normalize(x, SR)
    assert info["reached_target"]
    assert measure(y, SR)["lufs"] == pytest.approx(TARGET_LUFS, abs=0.5)


def test_normalize_reports_clipping():
    x = speech_with_pauses(level=0.999)
    _, info = normalize(x, SR, target_lufs=0.0)
    assert not info["reached_target"]
    assert "would clip" in info["reason"]


def test_floor_matching_equalizes_snr():
    quiet_pauses = speech_with_pauses(floor=1e-6)
    noisy_pauses = speech_with_pauses(floor=1e-3, seed=1)

    a, _ = match_noise_floor(quiet_pauses, SR, target_snr_db=26.0)
    b, _ = match_noise_floor(noisy_pauses, SR, target_snr_db=26.0)

    def snr(x):
        speech, pause = speech_and_pause_level(x, SR)
        return speech - pause

    assert snr(quiet_pauses) - snr(noisy_pauses) > 10
    assert abs(snr(a) - snr(b)) < 3


def test_floor_above_target_unchanged():
    noisy = speech_with_pauses(floor=1e-2)
    y, info = match_noise_floor(noisy, SR, target_snr_db=26.0)
    assert not info["raised"]
    assert np.array_equal(y, noisy)


def test_floor_matching_deterministic():
    x = speech_with_pauses(floor=1e-6)
    a, _ = match_noise_floor(x, SR, seed=42)
    b, _ = match_noise_floor(x, SR, seed=42)
    c, _ = match_noise_floor(x, SR, seed=43)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_floor_matching_keeps_speech_level():
    x = speech_with_pauses(floor=1e-6)
    y, _ = match_noise_floor(x, SR)

    # Mask from the clean signal; the noisy one would count filled pauses as speech.
    active = _frame_db(x, SR) > SILENCE_THRESHOLD_DBFS
    before, _ = speech_and_pause_level(x, SR, active=active)
    after, _ = speech_and_pause_level(y, SR, active=active)
    assert after == pytest.approx(before, abs=0.5)


def test_floor_fallback_flagged():
    no_pauses = speech_with_pauses(floor=1e-2)
    _, info = match_noise_floor(no_pauses, SR, target_snr_db=26.0)
    assert info["pause_measured"] == "fallback"
    assert not info["raised"]


def test_silence_ratio():
    m = measure(speech_with_pauses(floor=0.0), SR)
    assert 0.4 < m["silence_ratio"] < 0.6
