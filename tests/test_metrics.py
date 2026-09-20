"""Text normalization and WER."""

import pytest

from myecho.evaluation.metrics import normalize_text, wer


@pytest.mark.parametrize("a, b", [
    ("Hallo, Welt!", "hallo welt"),
    ("schön", "schoen"),
    ("Straße", "strasse"),
    ("Brief-Kasten", "brief kasten"),
    ("Hallo   Welt", "hallo welt"),
    ("Er sagt: „ja“", "er sagt ja"),
])
def test_normalize_text(a, b):
    assert normalize_text(a) == normalize_text(b)


def test_normalize_numbers():
    assert normalize_text("2021") == normalize_text("zweitausendeinundzwanzig")


def test_wer_zero():
    assert wer("Hallo, Welt!", "hallo welt") == 0.0


def test_wer_substitution():
    assert wer("hallo welt", "hallo wald") == pytest.approx(0.5)


def test_wer_empty_hypothesis():
    assert wer("hallo welt", "") == 1.0


def test_wer_empty_reference():
    with pytest.raises(ValueError):
        wer("...", "hallo")
