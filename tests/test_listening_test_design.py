"""Tests for the listening-test design: rater plans, span rejection, scoring.

The properties checked here are the ones that cannot be fixed after the first
rater has answered.
"""

import pytest

import prepare_listening_test as P
import select_excerpts as S
from analyze_listening_test import real_side_displayed


MODELS = ["m1", "m2", "m3"]
TEXTS = [{"id": f"t{i}", "source_audio": f"t{i}.wav"} for i in range(9)]


def test_no_plan_uses_the_same_text_in_both_parts():
    # A rater who heard text T in part 1 already knows what the real voice
    # sounds like on T, which is the answer to T's pair in part 2.
    for plan in P.build_plans(TEXTS, MODELS):
        turing_texts = {text_id for text_id, _ in plan["turing"]}
        assert not turing_texts & set(plan["nmos_texts"])


def test_every_text_is_covered_by_exactly_one_role_per_plan():
    for plan in P.build_plans(TEXTS, MODELS):
        covered = {t for t, _ in plan["turing"]} | set(plan["nmos_texts"])
        assert covered == {entry["id"] for entry in TEXTS}


def test_the_rotation_pairs_every_model_with_every_text():
    # The old design assigned one model per text, so the Turing interval was
    # over 5 items no matter how many people answered.
    cells = {(text_id, model)
             for plan in P.build_plans(TEXTS, MODELS)
             for text_id, model in plan["turing"]}
    assert len(cells) == len(TEXTS) * len(MODELS)


def test_each_model_appears_equally_often_across_plans():
    counts = {model: 0 for model in MODELS}
    for plan in P.build_plans(TEXTS, MODELS):
        for _, model in plan["turing"]:
            counts[model] += 1
    assert len(set(counts.values())) == 1


def test_dropping_a_text_applies_to_every_model():
    excluded = {("m2", "t3"): "silence_heavy"}
    assert P.dropped_texts(excluded, keep_unbalanced=False) == {"t3"}
    assert P.dropped_texts(excluded, keep_unbalanced=True) == set()


def test_unlocated_renditions_become_exclusions():
    spans = {f"{m}/t{i}": {} for m in MODELS for i in range(9)}
    del spans["m2/t4"]
    missing = P.unlocated(spans, MODELS, TEXTS)
    assert list(missing) == [("m2", "t4")]


@pytest.mark.parametrize("real_side, swapped, expected", [
    ("a", False, "A"),
    ("a", True, "B"),
    ("b", False, "B"),
    ("b", True, "A"),
])
def test_the_per_rater_side_flip_is_undone_correctly(real_side, swapped, expected):
    # Getting this wrong inverts every Turing answer, which would look like a
    # plausible result rather than like a bug.
    assert real_side_displayed({"real_side": real_side}, swapped) == expected


def test_a_runaway_span_is_rejected_against_the_per_text_median():
    spans = {
        "m1/t0": {"duration": 9.0}, "m2/t0": {"duration": 10.0},
        "m3/t0": {"duration": 44.0},          # words all there, 4x the time
        "m1/t1": {"duration": 30.0}, "m2/t1": {"duration": 31.0},
        "m3/t1": {"duration": 32.0},          # a text that is long for everyone
    }
    kept, rejected = S.reject_runaway_spans(spans, max_ratio=2.0)
    assert [(s, t) for s, t, _ in rejected] == [("m3", "t0")]
    assert set(kept) == set(spans) - {"m3/t0"}
    assert "runaway" in rejected[0][2]
