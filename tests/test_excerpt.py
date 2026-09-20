"""Excerpt selection and location."""

import numpy as np
import pytest

from myecho.evaluation.excerpt import cut, locate_span, select_excerpt

TEXT = (
    "Hi Lars, guten Morgen. Wollte mich mal melden. "
    "Ich hoffe bei euch ist alles gut. Das Wetter wird morgen noch trueb, "
    "aber die Sonne soll rauskommen. Wir wollen dann mit dem Fahrrad fahren. "
    "Vielleicht koennen wir heute Abend kurz telefonieren. "
    "Dann mach es gut und melde dich mal. Hab dich lieb. Ciao."
)


def test_select_deterministic():
    assert select_excerpt(TEXT) == select_excerpt(TEXT)


def test_select_length_and_boundary():
    excerpt = select_excerpt(TEXT)
    assert 20 <= excerpt["n_words"] <= 40
    assert excerpt["boundary"] in ("sentence", "comma")
    assert excerpt["text"] in TEXT or excerpt["boundary"] == "comma"


def test_select_skips_head():
    assert not select_excerpt(TEXT)["text"].startswith("Hi Lars")


def test_select_comma_fallback():
    run_on = "also " + ", ".join([f"teil {i} von dem satz" for i in range(12)])
    excerpt = select_excerpt(run_on)
    assert excerpt["boundary"] in ("comma", "word")
    assert excerpt["n_words"] > 0


def _words(text, start=0.0, step=0.5):
    return [{"word": w, "start": start + i * step, "end": start + (i + 1) * step}
            for i, w in enumerate(text.split())]


def test_locate_span():
    span = locate_span(_words("eins zwei drei vier fuenf sechs"), "drei vier fuenf")
    assert span is not None
    assert span["start"] == pytest.approx(1.0)
    assert span["end"] == pytest.approx(2.5)
    assert span["match_ratio"] == 1.0


def test_locate_span_fuzzy():
    span = locate_span(_words("eins zwei dreu vier fuenf sechs"), "drei vier fuenf")
    assert span is not None
    assert 0.6 <= span["match_ratio"] < 1.0


def test_locate_span_missing():
    assert locate_span(_words("ganz andere worte hier"), "drei vier fuenf") is None


def test_locate_span_empty():
    assert locate_span([], "drei vier") is None
    assert locate_span(_words("eins zwei"), "") is None


def test_cut_padding_and_fades():
    sr = 1000
    x = np.ones(5 * sr, dtype=np.float32)
    y = cut(x, sr, 1.0, 2.0, pad_before=0.2, pad_after=0.3, fade_s=0.02)
    assert len(y) == pytest.approx(1.5 * sr, abs=2)
    assert y[0] < 0.1 and y[-1] < 0.1
    assert y[len(y) // 2] == pytest.approx(1.0)
