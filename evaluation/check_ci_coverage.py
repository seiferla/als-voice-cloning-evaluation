"""Does the reported confidence interval actually cover? And what can it detect?

Usage:
    uv run python evaluation/check_ci_coverage.py

analyze_listening_test.py reports 95 % intervals from a two-way cluster
bootstrap over raters and texts. With only ~13 texts, "cluster bootstrap" is not
self-evidently correct: cluster methods are known to under-cover when the number
of clusters is small, and an interval that is too narrow is worse than no
interval, because it turns noise into a claim.

So this script simulates the design that was actually built - the same number of
texts, conditions, plans and raters - draws data from a known model, and counts
how often the nominal 95 % interval contains the true value. It also reports the
smallest condition difference the design can detect, which is the number that
says whether 30 raters is enough.

Nothing here touches the collected ratings. It is a property of the design and
can be run before the first rater.

Reported for comparison:
    two-way          resample raters AND texts (what the analysis does)
    raters only      resample raters, treat the 13 texts as fixed
    naive pooled     treat every single answer as independent (the old behaviour)
"""

import argparse
import json
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).parent / "results" / "listening_test"

# Variance components, in MOS points. Defaults are deliberately pessimistic
# rather than fitted: this is a design check, run before any data exists.
SD_RATER = 0.35     # how much a rater's whole scale is shifted
SD_TEXT = 0.30      # how much a sentence is easier or harder for everyone
SD_NOISE = 0.70     # single-answer noise (matches typical MOS test-retest)


def design_from_key():
    """(n_texts, n_conditions, plan text sets) from the built test, if present."""
    key_path = OUT_DIR / "key.json"
    if not key_path.exists():
        return None
    key = json.loads(key_path.read_text(encoding="utf-8"))
    html = (OUT_DIR / "index.html").read_text(encoding="utf-8")
    import re
    manifest = json.loads(re.search(r"const MANIFEST = (\{.*?\});\n", html, re.S).group(1))
    stimuli = key["stimuli"]
    texts = sorted({i["text_id"] for i in stimuli.values() if i["role"] == "scored"})
    index = {t: i for i, t in enumerate(texts)}
    plans = [sorted({index[stimuli[s]["text_id"]] for s in plan["nmos"]})
             for plan in manifest["plans"]]
    return len(texts), len(key["conditions"]), plans


def simulate(n_texts, plans, n_raters, effect, n_boot, rng, kind):
    """One study. Returns (covered, width, detected) for a two-condition contrast.

    effect is the true difference between the two conditions, in MOS points.
    """
    rater_effect = rng.normal(0, SD_RATER, n_raters)
    text_effect = rng.normal(0, SD_TEXT, n_texts)
    plan_of = rng.integers(0, len(plans), n_raters)

    # cell[condition, rater, text]; nan where that rater did not rate that text
    cell = np.full((2, n_raters, n_texts), np.nan)
    for r in range(n_raters):
        ts = np.array(plans[plan_of[r]], dtype=int)
        if len(ts) == 0:
            continue
        for c, mu in enumerate((0.0, -effect)):
            cell[c, r, ts] = (mu + rater_effect[r] + text_effect[ts]
                              + rng.normal(0, SD_NOISE, len(ts)))

    def contrast(rows_r, rows_t):
        sub = cell[:, rows_r][:, :, rows_t]
        with np.errstate(invalid="ignore"):
            per_rater = np.nanmean(sub, axis=2)          # (2, raters)
            diff = per_rater[0] - per_rater[1]           # paired within rater
            return np.nanmean(diff)

    all_r, all_t = np.arange(n_raters), np.arange(n_texts)

    if kind == "naive pooled":
        # Every answer independent: the SE the old code assumed.
        values = (cell[0] - cell[1]).ravel()
        values = values[np.isfinite(values)]
        est = values.mean()
        half = 1.96 * values.std(ddof=1) / np.sqrt(len(values))
        lo, hi = est - half, est + half
    else:
        boots = []
        for _ in range(n_boot):
            dr = rng.integers(0, n_raters, n_raters)
            dt = (rng.integers(0, n_texts, n_texts) if kind == "two-way" else all_t)
            with np.errstate(invalid="ignore"):
                v = contrast(dr, dt)
            if np.isfinite(v):
                boots.append(v)
        lo, hi = np.percentile(boots, [2.5, 97.5])

    return (lo <= effect <= hi), (hi - lo), not (lo <= 0.0 <= hi)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--raters", type=int, default=30)
    parser.add_argument("--sims", type=int, default=400)
    parser.add_argument("--boot", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()

    built = design_from_key()
    if built is None:
        print("No built test found - using the nominal design (13 texts, 6 plans).\n")
        n_texts, n_cond = 13, 4
        plans = [[t for t in range(n_texts) if t % 2 != g] for g in range(2)] * 3
    else:
        n_texts, n_cond, plans = built
        print(f"Design read from the built test: {n_texts} texts, {n_cond} conditions, "
              f"{len(plans)} plans,\n{len(plans[0])}-{len(plans[-1])} texts per rater "
              f"in part 1.\n")

    rng = np.random.default_rng(args.seed)
    print(f"Assumed variance: rater SD {SD_RATER}, text SD {SD_TEXT}, "
          f"answer SD {SD_NOISE} MOS points")
    print(f"{args.sims} simulated studies of {args.raters} raters each.\n")

    print("1. Coverage of the nominal 95 % interval for a true 0.30 MOS difference")
    print("   (should be about 0.95; below ~0.90 means the interval overstates "
          "certainty)\n")
    for kind in ("two-way", "raters only", "naive pooled"):
        covered, widths = [], []
        for _ in range(args.sims):
            ok, width, _ = simulate(n_texts, plans, args.raters, 0.30,
                                    args.boot, rng, kind)
            covered.append(ok)
            widths.append(width)
        flag = ("ok" if np.mean(covered) >= 0.93 else
                "TOO NARROW - overstates certainty")
        print(f"   {kind:14s} coverage {np.mean(covered):.3f}   "
              f"mean width {np.mean(widths):.3f}   {flag}")

    print("\n2. Power: how often a true difference is detected (interval excludes 0)\n")
    print(f"   {'true difference':>16s}  {'power':>6s}")
    for effect in (0.0, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0):
        detected = [simulate(n_texts, plans, args.raters, effect,
                             args.boot, rng, "two-way")[2]
                    for _ in range(args.sims)]
        label = "  (false positive)" if effect == 0.0 else ""
        print(f"   {effect:16.2f}  {np.mean(detected):6.3f}{label}")

    print("\n3. Raters needed for 80 % power at a 0.30 MOS difference\n")
    for n in (10, 20, 30, 40, 60):
        detected = [simulate(n_texts, plans, n, 0.30, args.boot, rng, "two-way")[2]
                    for _ in range(max(args.sims // 2, 100))]
        print(f"   {n:3d} raters  power {np.mean(detected):.3f}")

    print("\nRead this together with the rating-noise number from the hidden "
          "repeats:\na difference the design can detect but that is smaller than "
          "one rater's own\ntest-retest spread is statistically real and "
          "practically meaningless.")


if __name__ == "__main__":
    main()
