"""Rater plans, span rejection, scoring."""

import pytest

import prepare_listening_test as P
import select_excerpts as S
from analyze_listening_test import real_side_displayed


MODELS = ["m1", "m2", "m3"]
TEXTS = [{"id": f"t{i}", "source_audio": f"t{i}.wav"} for i in range(9)]


def test_plans_split_texts_between_parts():
    for plan in P.build_plans(TEXTS, MODELS):
        turing_texts = {text_id for text_id, _ in plan["turing"]}
        assert not turing_texts & set(plan["nmos_texts"])


def test_plans_cover_every_text():
    for plan in P.build_plans(TEXTS, MODELS):
        covered = {t for t, _ in plan["turing"]} | set(plan["nmos_texts"])
        assert covered == {entry["id"] for entry in TEXTS}


def test_rotation_covers_all_cells():
    cells = {(text_id, model)
             for plan in P.build_plans(TEXTS, MODELS)
             for text_id, model in plan["turing"]}
    assert len(cells) == len(TEXTS) * len(MODELS)


def test_rotation_balanced():
    counts = {model: 0 for model in MODELS}
    for plan in P.build_plans(TEXTS, MODELS):
        for _, model in plan["turing"]:
            counts[model] += 1
    assert len(set(counts.values())) == 1


def test_dropped_texts():
    excluded = {("m2", "t3"): "silence_heavy"}
    assert P.dropped_texts(excluded, keep_unbalanced=False) == {"t3"}
    assert P.dropped_texts(excluded, keep_unbalanced=True) == set()


def test_unlocated():
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
def test_real_side_displayed(real_side, swapped, expected):
    assert real_side_displayed({"real_side": real_side}, swapped) == expected


def test_reject_runaway_spans():
    spans = {
        "m1/t0": {"duration": 9.0}, "m2/t0": {"duration": 10.0},
        "m3/t0": {"duration": 44.0},
        "m1/t1": {"duration": 30.0}, "m2/t1": {"duration": 31.0},
        "m3/t1": {"duration": 32.0},
    }
    kept, rejected = S.reject_runaway_spans(spans, max_ratio=2.0)
    assert [(s, t) for s, t, _ in rejected] == [("m3", "t0")]
    assert set(kept) == set(spans) - {"m3/t0"}
    assert "runaway" in rejected[0][2]
