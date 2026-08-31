"""Joins collected ratings with the hidden key and computes N-MOS + Turing
test results per model.

Usage:
    uv run python evaluation/analyze_listening_test.py

Reads every evaluation/results/listening_test/responses/*.json (one per
rater, downloaded from index.html) plus the hidden key.json, then prints:
    - N-MOS: mean/std naturalness rating per model
    - Turing test: fooling rate per model (share of raters who picked the
      clone as "the real voice" - higher means more convincing), split by
      whether the rater knows the cloned person (a known confound)

Also writes nmos_summary.csv and turing_summary.csv next to the responses,
for a spreadsheet with many raters.
"""

import csv
import json
import math
import statistics
from pathlib import Path

OUT_DIR = Path(__file__).parent / "results" / "listening_test"
KEY_PATH = OUT_DIR / "key.json"
RESPONSES_DIR = OUT_DIR / "responses"


def load_key():
    with open(KEY_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_responses():
    """Every rating, from both individually downloaded *.json files and the
    all_ratings.jsonl the local server appends to (evaluation/listening_test_server.py)."""
    responses = []
    for f in sorted(RESPONSES_DIR.glob("*.json")):
        with open(f, encoding="utf-8") as fh:
            responses.append(json.load(fh))

    jsonl_path = RESPONSES_DIR / "all_ratings.jsonl"
    if jsonl_path.exists():
        with open(jsonl_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    responses.append(json.loads(line))

    if not responses:
        raise SystemExit(f"No ratings found in {RESPONSES_DIR}")
    return responses


def nmos_by_model(key, responses):
    model_by_id = {e["anon_id"]: e["model"] for e in key["nmos"]}
    scores = {}
    for r in responses:
        for anon_id, value in r.get("nmos", {}).items():
            if value is None:
                continue
            model = model_by_id.get(anon_id)
            if model is None:
                continue
            scores.setdefault(model, []).append(value)
    return scores


def real_side_shown(info, response, pair_id):
    """Which side (A/B) actually held the real recording for THIS rater.

    The page flips the two slots per rater, so the key's real_position is only
    the unswapped baseline. Responses without the flag predate this and are
    read as unswapped.
    """
    side = info["real_position"]
    swapped = (response.get("presented") or {}).get("turing_swapped", {}).get(pair_id, False)
    if swapped:
        side = "B" if side == "A" else "A"
    return side


def turing_by_model(key, responses):
    """correct/total per model, counting one answered pair per rater."""
    pair_info = {e["pair_id"]: e for e in key["turing"]}
    correct = {}
    total = {}
    for r in responses:
        for pair_id, choice in r.get("turing", {}).items():
            if choice is None:
                continue
            info = pair_info.get(pair_id)
            if info is None:
                continue
            model = info["model"]
            total[model] = total.get(model, 0) + 1
            if choice == real_side_shown(info, r, pair_id):
                correct[model] = correct.get(model, 0) + 1
    return correct, total


def wilson_interval(k, n, z=1.96):
    """95 % Wilson score interval for a proportion.

    Treats every answered pair as independent, which it is not (each rater
    answers many pairs, each pair is answered by many raters), so this is
    slightly too narrow. Good enough to see whether a model is anywhere near
    the 0.50 chance level; for the write-up use a mixed-effects logistic model
    with random intercepts for rater and text.
    """
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def print_turing(key, responses, group_label, csv_rows):
    if not responses:
        return
    print(f"\n--- {group_label} (n={len(responses)} rater(s)) ---")
    correct, total = turing_by_model(key, responses)
    if not total:
        print("  no answers")
        return
    for model in sorted(total):
        n = total[model]
        fooled = n - correct.get(model, 0)
        rate = fooled / n
        lo, hi = wilson_interval(fooled, n)
        chance = "indistinguishable from chance" if lo <= 0.5 <= hi else (
            "above chance" if lo > 0.5 else "below chance")
        print(f"  {model}: fooling_rate={rate:.2f}  95% CI [{lo:.2f}, {hi:.2f}]  "
              f"({fooled}/{n} mistaken for real) - {chance}")
        csv_rows.append({"group": group_label, "model": model, "n": n, "fooled": fooled,
                         "fooling_rate": round(rate, 4), "ci_low": round(lo, 4),
                         "ci_high": round(hi, 4)})


def write_summaries(key, responses):
    """Recompute nmos_summary.csv / turing_summary.csv from scratch. Cheap
    enough (tens of raters) to call after every single submission, so the
    aggregate file is always current - no manual re-run needed."""
    nmos_rows = []
    for model, values in sorted(nmos_by_model(key, responses).items()):
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        nmos_rows.append({"model": model, "mean": round(mean, 4), "std": round(std, 4), "n": len(values)})

    turing_rows = []
    groups = {
        "all": responses,
        "knows_person": [r for r in responses if r.get("knows_person") == "yes"],
        "does_not_know_person": [r for r in responses if r.get("knows_person") == "no"],
        "familiarity_not_stated": [r for r in responses if r.get("knows_person") not in ("yes", "no")],
    }
    for label, group in groups.items():
        print_turing(key, group, label, turing_rows)

    with open(OUT_DIR / "nmos_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "mean", "std", "n"])
        writer.writeheader()
        writer.writerows(nmos_rows)

    with open(OUT_DIR / "turing_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "model", "n", "fooled", "fooling_rate", "ci_low", "ci_high"])
        writer.writeheader()
        writer.writerows(turing_rows)

    return nmos_rows, turing_rows


def main():
    key = load_key()
    responses = load_responses()
    print("Raters:")
    for r in responses:
        print(f"  {r.get('rater')}: age={r.get('age')}  knows_person={r.get('knows_person')}")

    print("\n=== N-MOS (naturalness, 1-5) ===")
    for model, values in sorted(nmos_by_model(key, responses).items()):
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        print(f"{model}: mean={mean:.2f}  std={std:.2f}  n={len(values)}")

    print("\n=== Turing test (fooling rate = share mistaken for the real voice) ===")
    print("Familiarity with the cloned voice is a known confound - split below.")
    write_summaries(key, responses)
    print(f"\nSaved {OUT_DIR / 'nmos_summary.csv'} and {OUT_DIR / 'turing_summary.csv'}")


if __name__ == "__main__":
    main()