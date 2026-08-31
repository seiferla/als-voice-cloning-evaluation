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
import statistics
from pathlib import Path

OUT_DIR = Path(__file__).parent / "results" / "listening_test"
KEY_PATH = OUT_DIR / "key.json"
RESPONSES_DIR = OUT_DIR / "responses"


def load_key():
    with open(KEY_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_responses():
    files = sorted(RESPONSES_DIR.glob("*.json"))
    if not files:
        raise SystemExit(f"No ratings found in {RESPONSES_DIR}")
    responses = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            responses.append(json.load(fh))
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


def turing_by_model(key, responses):
    """correct/total per model, optionally restricted to responses matching filter_fn(rater)."""
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
            if choice == info["real_position"]:
                correct[model] = correct.get(model, 0) + 1
    return correct, total


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
        print(f"  {model}: fooling_rate={rate:.2f}  ({fooled}/{n} mistaken for real)")
        csv_rows.append({"group": group_label, "model": model, "n": n, "fooled": fooled, "fooling_rate": round(rate, 4)})


def main():
    key = load_key()
    responses = load_responses()
    print("Raters:")
    for r in responses:
        print(f"  {r.get('rater')}: age={r.get('age')}  knows_person={r.get('knows_person')}")

    print("\n=== N-MOS (naturalness, 1-5) ===")
    nmos_rows = []
    for model, values in sorted(nmos_by_model(key, responses).items()):
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        print(f"{model}: mean={mean:.2f}  std={std:.2f}  n={len(values)}")
        nmos_rows.append({"model": model, "mean": round(mean, 4), "std": round(std, 4), "n": len(values)})

    print("\n=== Turing test (fooling rate = share mistaken for the real voice) ===")
    print("Familiarity with the cloned voice is a known confound - split below.")
    familiar = [r for r in responses if r.get("knows_person") == "yes"]
    unfamiliar = [r for r in responses if r.get("knows_person") == "no"]
    unknown = [r for r in responses if r.get("knows_person") not in ("yes", "no")]
    turing_rows = []
    print_turing(key, responses, "all", turing_rows)
    print_turing(key, familiar, "knows_person", turing_rows)
    print_turing(key, unfamiliar, "does_not_know_person", turing_rows)
    print_turing(key, unknown, "familiarity_not_stated", turing_rows)

    with open(OUT_DIR / "nmos_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "mean", "std", "n"])
        writer.writeheader()
        writer.writerows(nmos_rows)

    with open(OUT_DIR / "turing_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "model", "n", "fooled", "fooling_rate"])
        writer.writeheader()
        writer.writerows(turing_rows)

    print(f"\nSaved {OUT_DIR / 'nmos_summary.csv'} and {OUT_DIR / 'turing_summary.csv'}")


if __name__ == "__main__":
    main()