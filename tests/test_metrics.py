"""Tests for the text normalization that WER depends on.

Every one of these is a case where the raw strings differ but the TTS model did
nothing wrong, so a failure here shows up as a fake error rate.
"""

import pytest

from myecho.evaluation.metrics import normalize_text, wer


@pytest.mark.parametrize("a, b", [
    ("Hallo, Welt!", "hallo welt"),                      # punctuation and case
    ("schön", "schoen"),                                 # umlaut transliteration
    ("Straße", "strasse"),                               # sharp s
    ("Brief-Kasten", "brief kasten"),                    # compound hyphen
    ("Hallo   Welt", "hallo welt"),                      # whitespace
    ("Er sagt: „ja“", "er sagt ja"),           # German quotes
])
def test_normalize_text(a, b):
    assert normalize_text(a) == normalize_text(b)


def test_normalize_numbers():
    # Whisper writes numbers as digits or as words, unpredictably. The same
    # expansion on both sides is what keeps that from counting as an error.
    assert normalize_text("2021") == normalize_text("zweitausendeinundzwanzig")


def test_wer_zero():
    assert wer("Hallo, Welt!", "hallo welt") == 0.0


def test_wer_substitution():
    assert wer("hallo welt", "hallo wald") == pytest.approx(0.5)


def test_wer_empty_hypothesis():
    # A silent render must score 1.0. jiwer itself raises on an empty
    # hypothesis, which would abort a whole evaluation run.
    assert wer("hallo welt", "") == 1.0


def test_wer_empty_reference():
    with pytest.raises(ValueError):
        wer("...", "hallo")
