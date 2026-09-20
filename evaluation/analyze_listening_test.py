"""Joins the collected ratings with the hidden key and reports the results.

Usage:
    uv run python evaluation/analyze_listening_test.py
"""

import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

OUT_DIR = Path(__file__).parent / "results" / "listening_test"
KEY_PATH = OUT_DIR / "key.json"
RESPONSES_DIR = OUT_DIR / "responses"

# Below this many raters the bootstrap cannot estimate between-rater variance
MIN_RATERS_FOR_VERDICTS = 5

N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20260901
CI = 0.95


# --------------------------------------------------------------------------- #

def load_key():
    with open(KEY_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_responses():
    """Every rating, deduplicated."""
    raw = []
    for path in sorted(RESPONSES_DIR.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            raw.append((path.name, json.load(f)))

    jsonl_path = RESPONSES_DIR / "all_ratings.jsonl"
    if jsonl_path.exists():
        with open(jsonl_path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                # Two concurrent submissions can land on one line; decode every JSON object in it.
                decoder = json.JSONDecoder()
                text, pos = line, 0
                while pos < len(text) and text[pos:].strip():
                    try:
                        obj, end = decoder.raw_decode(text, pos)
                    except json.JSONDecodeError:
                        print(f"  WARNING: skipping unparseable data on "
                              f"{jsonl_path.name}:{lineno}: {text[pos:pos + 80]!r}")
                        break
                    raw.append((jsonl_path.name, obj))
                    pos = end

    seen, responses, duplicates, legacy = set(), [], 0, 0
    for source, response in raw:
        if response.get("schema") != 2:
            legacy += 1
            continue
        fingerprint = (response.get("rater"), response.get("timestamp"))
        if fingerprint in seen:
            duplicates += 1
            continue
        seen.add(fingerprint)
        response["_source"] = source
        responses.append(response)

    if duplicates:
        print(f"note: skipped {duplicates} duplicate submission(s) "
              "(same rater id and timestamp seen twice)")
    if legacy:
        print(f"note: skipped {legacy} rating file(s) from an older stimulus set "
              "(no schema 2). They cannot be scored against this key.json.")
    if not responses:
        raise SystemExit(f"No usable ratings found in {RESPONSES_DIR}")
    return responses


# --------------------------------------------------------------------------- #

def real_side_displayed(pair, swapped):
    """Which slot (A/B) the rater actually saw the real recording in."""
    return "A" if (pair["real_side"] == "a") != bool(swapped) else "B"


def nmos_rows(key, responses):
    """One row per naturalness answer. Repeats are marked, not dropped."""
    stimuli = key["stimuli"]
    rows = []
    for response in responses:
        for entry in response.get("nmos", []):
            info = stimuli.get(entry.get("stimulus"))
            if info is None or entry.get("rating") is None:
                continue
            rows.append({
                "rater": response.get("rater"),
                "plan_id": response.get("plan_id"),
                "knows_person": response.get("knows_person"),
                "headphones": response.get("headphones"),
                "age": response.get("age"),
                "stimulus": entry["stimulus"],
                "condition": info["condition"],
                "text_id": info["text_id"],
                "position": entry.get("position"),
                "repeat": bool(entry.get("repeat")),
                "rating": entry["rating"],
            })
    return rows


def turing_rows(key, responses):
    """One row per Turing answer, with correct/fooled already resolved."""
    pairs = key["pairs"]
    rows = []
    for response in responses:
        for entry in response.get("turing", []):
            pair = pairs.get(entry.get("pair_id"))
            if pair is None or entry.get("choice") is None:
                continue
            shown = real_side_displayed(pair, entry.get("swapped"))
            correct = entry["choice"] == shown
            rows.append({
                "rater": response.get("rater"),
                "plan_id": response.get("plan_id"),
                "knows_person": response.get("knows_person"),
                "headphones": response.get("headphones"),
                "age": response.get("age"),
                "pair_id": entry["pair_id"],
                "model": pair["model"],
                "text_id": pair["text_id"],
                "position": entry.get("position"),
                "correct": correct,
                "fooled": not correct,
            })
    return rows


# --------------------------------------------------------------------------- #

def _percentile(values, q):
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[index]


def cluster_bootstrap(rows, statistic, n=N_BOOTSTRAP, seed=BOOTSTRAP_SEED, ci=CI):
    """CI for `statistic(rows)` under resampling of raters AND texts."""
    if not rows:
        return float("nan"), float("nan")

    raters = sorted({r["rater"] for r in rows})
    texts = sorted({r["text_id"] for r in rows})
    by_cell = defaultdict(list)
    for row in rows:
        by_cell[(row["rater"], row["text_id"])].append(row)

    rng = random.Random(seed)
    estimates = []
    for _ in range(n):
        rater_draw = [rng.choice(raters) for _ in raters]
        text_draw = [rng.choice(texts) for _ in texts]
        resampled = []
        for rater in rater_draw:
            for text in text_draw:
                resampled.extend(by_cell.get((rater, text), ()))
        value = statistic(resampled)
        if value is not None:
            estimates.append(value)

    alpha = (1 - ci) / 2
    return _percentile(estimates, alpha), _percentile(estimates, 1 - alpha)


def mean_rating(rows):
    """Mean naturalness, averaged per rater first."""
    per_rater = defaultdict(list)
    for row in rows:
        per_rater[row["rater"]].append(row["rating"])
    means = [statistics.mean(v) for v in per_rater.values() if v]
    return statistics.mean(means) if means else None


def fooling_rate(rows):
    per_rater = defaultdict(list)
    for row in rows:
        per_rater[row["rater"]].append(1.0 if row["fooled"] else 0.0)
    means = [statistics.mean(v) for v in per_rater.values() if v]
    return statistics.mean(means) if means else None


# --------------------------------------------------------------------------- #

def nmos_report(key, rows):
    """Per-condition means with CI, plus every pairwise difference."""
    real = key.get("real_condition", "REAL")
    anchor = key.get("anchor_condition")
    scored = [r for r in rows if not r["repeat"]]
    conditions = sorted({r["condition"] for r in scored},
                        key=lambda c: (c == real, c))

    summary = []
    for condition in conditions:
        subset = [r for r in scored if r["condition"] == condition]
        mean = mean_rating(subset)
        low, high = cluster_bootstrap(subset, mean_rating)
        per_rater = defaultdict(list)
        for row in subset:
            per_rater[row["rater"]].append(row["rating"])
        summary.append({
            "condition": condition,
            "is_reference": condition == real,
            "is_anchor": condition == anchor,
            "mean": round(mean, 3) if mean is not None else None,
            "ci_low": round(low, 3), "ci_high": round(high, 3),
            "sd_of_answers": round(statistics.stdev([r["rating"] for r in subset]), 3)
                             if len(subset) > 1 else 0.0,
            "n_answers": len(subset),
            "n_raters": len(per_rater),
            "n_items": len({r["stimulus"] for r in subset}),
        })

    # 6 comparisons at 5 % each give ~26 % family-wise error; Bonferroni-adjusted verdict.
    compared = [c for c in conditions if c != anchor]
    n_comparisons = len(compared) * (len(compared) - 1) // 2
    adjusted_ci = 1 - (1 - CI) / max(n_comparisons, 1)

    pairwise = []
    for i, a in enumerate(compared):
        for b in compared[i + 1:]:
            subset = [r for r in scored if r["condition"] in (a, b)]

            def difference(rs, a=a, b=b):
                # Paired within rater to remove each rater's scale offset.
                per_rater = defaultdict(lambda: {a: [], b: []})
                for row in rs:
                    per_rater[row["rater"]][row["condition"]].append(row["rating"])
                diffs = [statistics.mean(v[a]) - statistics.mean(v[b])
                         for v in per_rater.values() if v[a] and v[b]]
                return statistics.mean(diffs) if diffs else None

            estimate = difference(subset)
            low, high = cluster_bootstrap(subset, difference)
            adj_low, adj_high = cluster_bootstrap(subset, difference, ci=adjusted_ci)
            pairwise.append({
                "condition_a": a, "condition_b": b,
                "difference": round(estimate, 3) if estimate is not None else None,
                "ci_low": round(low, 3), "ci_high": round(high, 3),
                "excludes_zero": bool(low > 0 or high < 0),
                "ci_low_adj": round(adj_low, 3), "ci_high_adj": round(adj_high, 3),
                "excludes_zero_adj": bool(adj_low > 0 or adj_high < 0),
                "n_comparisons": n_comparisons,
            })
    return summary, pairwise


def turing_report(rows):
    """Fooling rate per model, overall and split by familiarity."""
    groups = {
        "all": rows,
        "knows_person": [r for r in rows if r["knows_person"] == "yes"],
        "does_not_know_person": [r for r in rows if r["knows_person"] == "no"],
    }
    out = []
    for label, group in groups.items():
        if not group:
            continue
        for model in sorted({r["model"] for r in group}):
            subset = [r for r in group if r["model"] == model]
            rate = fooling_rate(subset)
            low, high = cluster_bootstrap(subset, fooling_rate)
            out.append({
                "group": label, "model": model,
                "fooling_rate": round(rate, 3) if rate is not None else None,
                "ci_low": round(low, 3), "ci_high": round(high, 3),
                "vs_chance": ("indistinguishable from chance" if low <= 0.5 <= high
                              else "above chance" if low > 0.5 else "below chance"),
                "n_answers": len(subset),
                "n_raters": len({r["rater"] for r in subset}),
                "n_items": len({r["pair_id"] for r in subset}),
            })
    return out


def familiarity_interaction(rows):
    """Does knowing the speaker change the fooling rate? One estimate, one CI."""
    out = []
    usable = [r for r in rows if r["knows_person"] in ("yes", "no")]
    for model in sorted({r["model"] for r in usable}):
        subset = [r for r in usable if r["model"] == model]

        def contrast(rs):
            knows = fooling_rate([r for r in rs if r["knows_person"] == "yes"])
            others = fooling_rate([r for r in rs if r["knows_person"] == "no"])
            return None if knows is None or others is None else knows - others

        estimate = contrast(subset)
        if estimate is None:
            continue
        low, high = cluster_bootstrap(subset, contrast)
        out.append({
            "model": model,
            "difference": round(estimate, 3),
            "ci_low": round(low, 3), "ci_high": round(high, 3),
            "excludes_zero": bool(low > 0 or high < 0),
            "n_knows": len({r["rater"] for r in subset if r["knows_person"] == "yes"}),
            "n_does_not": len({r["rater"] for r in subset if r["knows_person"] == "no"}),
        })
    return out


def reliability_report(rows):
    """Intra-rater agreement from the hidden repeats."""
    by_key = defaultdict(list)
    for row in rows:
        by_key[(row["rater"], row["stimulus"])].append(row["rating"])
    deltas = [max(v) - min(v) for v in by_key.values() if len(v) >= 2]
    if not deltas:
        return None
    return {
        "n_repeat_pairs": len(deltas),
        "mean_abs_difference": round(statistics.mean(deltas), 3),
        "identical_share": round(sum(d == 0 for d in deltas) / len(deltas), 3),
        "within_one_point_share": round(sum(d <= 1 for d in deltas) / len(deltas), 3),
    }


def mixed_models(nmos, turing, real_condition):
    """Confirmatory mixed models, if statsmodels is available."""
    try:
        import pandas as pd
        import statsmodels.formula.api as smf
    except ImportError:
        print("\n(statsmodels not installed - skipping the confirmatory mixed models.")
        print(" uv sync --extra eval installs it.)")
        return

    scored = [r for r in nmos if not r["repeat"]]
    if len({r["condition"] for r in scored}) > 1 and len({r["rater"] for r in scored}) > 2:
        frame = pd.DataFrame(scored)
        frame["condition"] = pd.Categorical(
            frame["condition"],
            categories=[real_condition] + sorted(set(frame["condition"]) - {real_condition}),
        )
        try:
            fit = smf.mixedlm("rating ~ condition", frame, groups=frame["rater"],
                              vc_formula={"text_id": "0 + C(text_id)"}).fit()
            print("\n=== Confirmatory: naturalness ~ condition, "
                  f"random rater + text (reference = {real_condition}) ===")
            print(fit.summary().tables[1])
        except Exception as e:
            print(f"\n(mixed model for N-MOS did not converge: {e})")

    if len({r["rater"] for r in turing}) > 2:
        frame = pd.DataFrame(turing)
        frame["y"] = frame["fooled"].astype(int)
        try:
            import statsmodels.api as sm
            fit = smf.gee("y ~ model", groups="rater", data=frame,
                          family=sm.families.Binomial()).fit()
            print("\n=== Confirmatory: fooled ~ model, GEE clustered by rater ===")
            print(fit.summary().tables[1])
        except Exception as e:
            print(f"\n(GEE for the Turing answers did not converge: {e})")


# --------------------------------------------------------------------------- #

def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def underpowered_warning(n_raters):
    """Text to print when there are too few raters for the intervals to mean much."""
    return (
        f"\n{'=' * 70}\n"
        f"PILOT DATA - {n_raters} rater(s). Do not read these numbers as results.\n"
        f"{'=' * 70}\n"
        f"The intervals resample raters and texts. With {n_raters} rater(s) the\n"
        f"rater dimension contributes nothing, so what is shown is a within-rater\n"
        f"interval over texts only: it does not say whether anyone else would\n"
        f"agree. Comparison verdicts are suppressed for that reason.\n"
        f"The design needs about 30 raters to detect a 0.30 MOS difference\n"
        f"(tools/check_ci_coverage.py). Use this run to check the procedure, not\n"
        f"to compare the systems.\n"
    )


def write_summaries(key, responses, quiet=False):
    """Recompute every summary from scratch."""
    nmos = nmos_rows(key, responses)
    turing = turing_rows(key, responses)
    real = key.get("real_condition", "REAL")

    n_raters = len({r["rater"] for r in nmos} | {r["rater"] for r in turing})
    underpowered = n_raters < MIN_RATERS_FOR_VERDICTS

    summary, pairwise = nmos_report(key, nmos)
    turing_summary = turing_report(turing)
    familiarity = familiarity_interaction(turing)
    reliability = reliability_report(nmos)

    write_csv(OUT_DIR / "nmos_summary.csv", summary,
              ["condition", "is_reference", "is_anchor", "mean", "ci_low", "ci_high",
               "sd_of_answers", "n_answers", "n_raters", "n_items"])
    write_csv(OUT_DIR / "nmos_pairwise.csv", pairwise,
              ["condition_a", "condition_b", "difference", "ci_low", "ci_high",
               "excludes_zero", "ci_low_adj", "ci_high_adj", "excludes_zero_adj",
               "n_comparisons"])
    if familiarity:
        write_csv(OUT_DIR / "turing_familiarity.csv", familiarity,
                  ["model", "difference", "ci_low", "ci_high", "excludes_zero",
                   "n_knows", "n_does_not"])
    write_csv(OUT_DIR / "turing_summary.csv", turing_summary,
              ["group", "model", "fooling_rate", "ci_low", "ci_high", "vs_chance",
               "n_answers", "n_raters", "n_items"])
    if reliability:
        write_csv(OUT_DIR / "reliability.csv", [reliability], list(reliability))
    write_csv(OUT_DIR / "ratings_long.csv",
              [{**r, "part": "nmos"} for r in nmos] + [{**r, "part": "turing"} for r in turing],
              ["part", "rater", "plan_id", "knows_person", "headphones", "age",
               "stimulus", "pair_id", "condition", "model", "text_id", "position",
               "repeat", "rating", "correct", "fooled"])

    if quiet:
        return summary, turing_summary

    if underpowered:
        print(underpowered_warning(n_raters))

    print(f"\n=== Naturalness (ACR 1-5), {len(responses)} rater(s) ===")
    print("Mean of per-rater means; CI from a two-way cluster bootstrap over "
          "raters and texts.")
    for row in summary:
        mark = "  <- human reference" if row["is_reference"] else ""
        print(f"  {row['condition']:8s} {row['mean']:.2f}  "
              f"95% CI [{row['ci_low']:.2f}, {row['ci_high']:.2f}]  "
              f"n={row['n_answers']:4d} answers / {row['n_items']} items / "
              f"{row['n_raters']} raters{mark}")

    reference = next((r for r in summary if r["is_reference"]), None)
    if reference and reference["sd_of_answers"] == 0.0 and reference["n_answers"] > 1:
        print(f"\n  NOTE {real} was given the identical score on all "
              f"{reference['n_answers']} items ({reference['mean']:.1f}).\n"
              "  A real phone recording scoring the top of the scale every time "
              "usually means\n  the rater recognized which items were genuine and "
              "scored that, not naturalness.\n  Check the scale wording and the "
              "training items before collecting more.")

    anchor = next((r for r in summary if r["is_reference"]), None)
    if anchor and anchor["mean"]:
        print(f"\n  Distance to the human reference ({real} = {anchor['mean']:.2f}):")
        for row in pairwise:
            if real not in (row["condition_a"], row["condition_b"]):
                continue
            other = row["condition_b"] if row["condition_a"] == real else row["condition_a"]
            sign = 1 if row["condition_a"] == real else -1
            gap = sign * row["difference"]
            print(f"    {other:8s} {gap:+.2f} MOS points vs {real}  "
                  f"(CI [{min(sign * row['ci_low'], sign * row['ci_high']):+.2f}, "
                  f"{max(sign * row['ci_low'], sign * row['ci_high']):+.2f}])")

    print("\n  Model vs model (verdict at the Bonferroni-adjusted level, "
          f"{pairwise[0]['n_comparisons'] if pairwise else 0} comparisons):")
    for row in pairwise:
        if real in (row["condition_a"], row["condition_b"]):
            continue
        verdict = ("not enough raters to judge" if underpowered
                   else "difference" if row["excludes_zero_adj"]
                   else "difference at 95% only, not after adjustment"
                   if row["excludes_zero"] else "no difference detected")
        print(f"    {row['condition_a']:8s} - {row['condition_b']:8s} "
              f"{row['difference']:+.2f}  CI [{row['ci_low']:+.2f}, {row['ci_high']:+.2f}]  "
              f"adj [{row['ci_low_adj']:+.2f}, {row['ci_high_adj']:+.2f}]  -> {verdict}")

    print("\n=== Turing test (fooling rate = share of answers that picked the clone) ===")
    print("0.50 = raters cannot tell. Familiarity with the voice is a known "
          "confound, so it is split out.")
    for label in ("all", "knows_person", "does_not_know_person"):
        group = [r for r in turing_summary if r["group"] == label]
        if not group:
            continue
        print(f"\n  --- {label} ---")
        for row in group:
            chance = ("not enough raters to judge" if underpowered
                      else row["vs_chance"])
            print(f"    {row['model']:8s} {row['fooling_rate']:.2f}  "
                  f"95% CI [{row['ci_low']:.2f}, {row['ci_high']:.2f}]  "
                  f"n={row['n_answers']:3d} / {row['n_items']} items / "
                  f"{row['n_raters']} raters  - {chance}")

    if familiarity:
        print("\n  Does knowing the speaker change the fooling rate?")
        print("  (the contrast with its own interval - two separate subgroup "
              "intervals are not a test)")
        for row in familiarity:
            verdict = ("not enough raters to judge" if underpowered
                       else "yes" if row["excludes_zero"] else "not detectable at this n")
            print(f"    {row['model']:8s} {row['difference']:+.2f}  "
                  f"CI [{row['ci_low']:+.2f}, {row['ci_high']:+.2f}]  "
                  f"({row['n_knows']} know / {row['n_does_not']} do not) -> {verdict}")

    if reliability:
        print("\n=== Rating noise (hidden repeats) ===")
        print(f"  {reliability['n_repeat_pairs']} repeated stimuli: mean absolute "
              f"difference {reliability['mean_abs_difference']:.2f} MOS points, "
              f"{reliability['identical_share']:.0%} identical, "
              f"{reliability['within_one_point_share']:.0%} within one point.")
        print("  A condition difference smaller than this is inside the noise of "
              "one rater's own scale.")

    if not underpowered:
        mixed_models(nmos, turing, real)
    print(f"\nSummaries written to {OUT_DIR}")
    if underpowered:
        print(underpowered_warning(n_raters))
    return summary, turing_summary


def main():
    key = load_key()
    responses = load_responses()
    print(f"{len(responses)} rater session(s):")
    for response in responses:
        print(f"  {response.get('rater')}: plan={response.get('plan_id')} "
              f"age={response.get('age')} knows_person={response.get('knows_person')} "
              f"headphones={response.get('headphones')}")
    write_summaries(key, responses)


if __name__ == "__main__":
    main()
