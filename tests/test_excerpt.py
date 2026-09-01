"""Tests for excerpt selection and location."""

import numpy as np
import pytest

from myecho.evaluation.excerpt import cut, locate_span, select_excerpt, split_sentences

# Roughly the shape of the real material: spontaneous speech, short sentences.
TEXT = (
    "Hi Lars, guten Morgen. Wollte mich mal melden. "
    "Ich hoffe bei euch ist alles gut. Das Wetter wird morgen noch trueb, "
    "aber die Sonne soll rauskommen. Wir wollen dann mit dem Fahrrad fahren. "
    "Vielleicht koennen wir heute Abend kurz telefonieren. "
    "Dann mach es gut und melde dich mal. Hab dich lieb. Ciao."
)


def test_selection_is_deterministic():
    assert select_excerpt(TEXT) == select_excerpt(TEXT)


def test_selection_hits_the_target_length_and_a_clean_boundary():
    excerpt = select_excerpt(TEXT)
    assert 20 <= excerpt["n_words"] <= 40
    assert excerpt["boundary"] in ("sentence", "comma")
    assert excerpt["text"] in TEXT or excerpt["boundary"] == "comma"


def test_selection_skips_the_greeting_at_the_start():
    # The opening of a voice memo is a greeting, and the first seconds of a
    # render carry onset artifacts that differ per model.
    assert not select_excerpt(TEXT)["text"].startswith("Hi Lars")


def test_a_run_on_sentence_falls_back_to_commas():
    run_on = "also " + ", ".join([f"teil {i} von dem satz" for i in range(12)])
    excerpt = select_excerpt(run_on)
    assert excerpt["boundary"] in ("comma", "word")
    assert excerpt["n_words"] > 0


def test_split_sentences_ignores_empty_pieces():
    assert split_sentences("Eins. Zwei!  Drei?  ") == ["Eins.", "Zwei!", "Drei?"]


def _words(text, start=0.0, step=0.5):
    return [{"word": w, "start": start + i * step, "end": start + (i + 1) * step}
            for i, w in enumerate(text.split())]


def test_locate_span_finds_the_excerpt_and_its_time_range():
    span = locate_span(_words("eins zwei drei vier fuenf sechs"), "drei vier fuenf")
    assert span is not None
    assert span["start"] == pytest.approx(1.0)
    assert span["end"] == pytest.approx(2.5)
    assert span["match_ratio"] == 1.0


def test_locate_span_tolerates_asr_noise():
    # The ASR of a synthesized file never reproduces the text exactly, so an
    # exact search would fail on perfectly good audio.
    span = locate_span(_words("eins zwei dreu vier fuenf sechs"), "drei vier fuenf")
    assert span is not None
    assert 0.6 <= span["match_ratio"] < 1.0


def test_locate_span_returns_none_when_the_passage_was_not_spoken():
    # A finding, not something to paper over with a time window.
    assert locate_span(_words("ganz andere worte hier"), "drei vier fuenf") is None


def test_locate_span_handles_empty_input():
    assert locate_span([], "drei vier") is None
    assert locate_span(_words("eins zwei"), "") is None


def test_cut_respects_padding_and_fades_the_edges():
    sr = 1000
    x = np.ones(5 * sr, dtype=np.float32)
    y = cut(x, sr, 1.0, 2.0, pad_before=0.2, pad_after=0.3, fade_s=0.02)
    assert len(y) == pytest.approx(1.5 * sr, abs=2)
    # A hard cut clicks, and a click is heard - and rated - as an artifact.
    assert y[0] < 0.1 and y[-1] < 0.1
    assert y[len(y) // 2] == pytest.approx(1.0)
