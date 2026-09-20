"""Generate the LaTeX tables of the manuscript from the result files and check
the numbers quoted in the prose of every evaluation section.

Usage:
    uv run python tools/make_paper_tables.py

The manuscript must not carry hand-typed numbers. Tables are written to
docs/paper/tables/ and pulled into docs/paper/objective_metrics.tex with
\\input, so re-running the evaluation and re-running this script is enough to
bring the paper up to date.

A few values are also quoted in the running text, where a table cell cannot be
referenced. Those are listed in PROSE_CLAIMS and checked against the CSVs on
every run. A mismatch is printed with the sentence that has to be edited; the
script does not rewrite prose.
"""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "evaluation" / "results"
OUT_DIR = ROOT / "docs" / "paper" / "tables"

# Display names and the order they appear in every table. REAL is listed last
# and separated by a rule: it is a reference point, not a system under test.
LABELS = {
    "qwen3": "Qwen3",
    "s2pro": "Fish Speech S2 Pro",
    "voxtral": "Voxtral Mistral",
    "REAL": "Genuine recordings (reference)",
}
SYSTEMS = ["qwen3", "s2pro", "voxtral"]

# The comparison column needs names short enough to keep the table in one line.
SHORT = {"qwen3": "Qwen3", "s2pro": "S2 Pro", "voxtral": "Voxtral", "REAL": "Real"}

# Values quoted in the running text of the manuscript sections.
#   (name, result file, row selector, column, value as written, decimals)
OBJ, LT = "", "listening_test/"
PROSE_CLAIMS = [
    # objective_metrics.tex
    ("SIM Qwen3 vs REAL", OBJ + "objective_pairwise.csv",
     dict(metric="sim", model_a="qwen3", model_b="REAL"), "difference", -0.031, 3),
    ("SIM Qwen3 vs REAL, adj. low", OBJ + "objective_pairwise.csv",
     dict(metric="sim", model_a="qwen3", model_b="REAL"), "ci_low_adj", -0.068, 3),
    ("SIM Qwen3 vs REAL, adj. high", OBJ + "objective_pairwise.csv",
     dict(metric="sim", model_a="qwen3", model_b="REAL"), "ci_high_adj", 0.004, 3),
    ("WER Qwen3 vs S2 Pro", OBJ + "objective_pairwise.csv",
     dict(metric="wer", model_a="qwen3", model_b="s2pro"), "difference", -0.029, 3),
    ("WER Qwen3 vs S2 Pro, adj. low", OBJ + "objective_pairwise.csv",
     dict(metric="wer", model_a="qwen3", model_b="s2pro"), "ci_low_adj", -0.061, 3),
    ("WER Qwen3 vs S2 Pro, adj. high", OBJ + "objective_pairwise.csv",
     dict(metric="wer", model_a="qwen3", model_b="s2pro"), "ci_high_adj", 0.001, 3),
    ("WER reference", OBJ + "objective_summary.csv", dict(metric="wer", model="REAL"), "median", 0.190, 3),
    ("service RTF Voxtral", OBJ + "objective_summary.csv", dict(metric="service_rtf", model="voxtral"), "median", 0.143, 3),
    ("service RTF Qwen3", OBJ + "objective_summary.csv", dict(metric="service_rtf", model="qwen3"), "median", 0.392, 3),
    # model_selection.tex
    ("RTF S2 Pro (2 dp)", OBJ + "objective_summary.csv", dict(metric="service_rtf", model="s2pro"), "median", 0.23, 2),
    ("RTF Qwen3 (2 dp)", OBJ + "objective_summary.csv", dict(metric="service_rtf", model="qwen3"), "median", 0.39, 2),
    ("RTF Voxtral (2 dp)", OBJ + "objective_summary.csv", dict(metric="service_rtf", model="voxtral"), "median", 0.14, 2),
    ("Voxtral unusable renders", OBJ + "render_reliability.csv", dict(model="voxtral"), "n_unusable", 3, 0),
    # discussion.tex
    ("fooling S2 Pro (identified 3 of 4)", LT + "turing_summary.csv", dict(group="all", model="s2pro"), "fooling_rate", 0.25, 2),
    ("fooling Voxtral, strangers", LT + "turing_summary.csv", dict(group="does_not_know_person", model="voxtral"), "fooling_rate", 0.29, 2),
    ("latency Voxtral", OBJ + "objective_summary.csv", dict(metric="latency_s", model="voxtral"), "median", 5.7, 1),
    ("latency Qwen3", OBJ + "objective_summary.csv", dict(metric="latency_s", model="qwen3"), "median", 13.8, 1),
    # subjective_evaluation.tex, N-MOS
    ("N-MOS Qwen3", LT + "nmos_summary.csv", dict(condition="qwen3"), "mean", 3.04, 2),
    ("N-MOS Qwen3, low", LT + "nmos_summary.csv", dict(condition="qwen3"), "ci_low", 2.71, 2),
    ("N-MOS Qwen3, high", LT + "nmos_summary.csv", dict(condition="qwen3"), "ci_high", 3.37, 2),
    ("N-MOS S2 Pro", LT + "nmos_summary.csv", dict(condition="s2pro"), "mean", 3.92, 2),
    ("N-MOS S2 Pro, low", LT + "nmos_summary.csv", dict(condition="s2pro"), "ci_low", 3.64, 2),
    ("N-MOS S2 Pro, high", LT + "nmos_summary.csv", dict(condition="s2pro"), "ci_high", 4.22, 2),
    ("N-MOS Voxtral", LT + "nmos_summary.csv", dict(condition="voxtral"), "mean", 4.05, 2),
    ("N-MOS Voxtral, low", LT + "nmos_summary.csv", dict(condition="voxtral"), "ci_low", 3.79, 2),
    ("N-MOS Voxtral, high", LT + "nmos_summary.csv", dict(condition="voxtral"), "ci_high", 4.34, 2),
    ("N-MOS REAL", LT + "nmos_summary.csv", dict(condition="REAL"), "mean", 4.16, 2),
    ("N-MOS REAL, low", LT + "nmos_summary.csv", dict(condition="REAL"), "ci_low", 3.81, 2),
    ("N-MOS REAL, high", LT + "nmos_summary.csv", dict(condition="REAL"), "ci_high", 4.46, 2),
    ("repeat pairs", LT + "reliability.csv", {}, "n_repeat_pairs", 60, 0),
    ("rating noise", LT + "reliability.csv", {}, "mean_abs_difference", 0.55, 2),
    # subjective_evaluation.tex, Turing
    ("fooling Qwen3", LT + "turing_summary.csv", dict(group="all", model="qwen3"), "fooling_rate", 0.14, 2),
    ("fooling Qwen3, low", LT + "turing_summary.csv", dict(group="all", model="qwen3"), "ci_low", 0.00, 2),
    ("fooling Qwen3, high", LT + "turing_summary.csv", dict(group="all", model="qwen3"), "ci_high", 0.25, 2),
    ("fooling S2 Pro", LT + "turing_summary.csv", dict(group="all", model="s2pro"), "fooling_rate", 0.25, 2),
    ("fooling S2 Pro, low", LT + "turing_summary.csv", dict(group="all", model="s2pro"), "ci_low", 0.12, 2),
    ("fooling S2 Pro, high", LT + "turing_summary.csv", dict(group="all", model="s2pro"), "ci_high", 0.40, 2),
    ("fooling Voxtral", LT + "turing_summary.csv", dict(group="all", model="voxtral"), "fooling_rate", 0.41, 2),
    ("fooling Voxtral, low", LT + "turing_summary.csv", dict(group="all", model="voxtral"), "ci_low", 0.23, 2),
    ("fooling Voxtral, high", LT + "turing_summary.csv", dict(group="all", model="voxtral"), "ci_high", 0.62, 2),
    ("familiarity Qwen3", LT + "turing_familiarity.csv", dict(model="qwen3"), "difference", 0.13, 2),
    ("familiarity S2 Pro", LT + "turing_familiarity.csv", dict(model="s2pro"), "difference", 0.11, 2),
    ("familiarity Voxtral", LT + "turing_familiarity.csv", dict(model="voxtral"), "difference", 0.25, 2),
    ("fooling Voxtral knows", LT + "turing_summary.csv", dict(group="knows_person", model="voxtral"), "fooling_rate", 0.53, 2),
    ("fooling Voxtral knows, low", LT + "turing_summary.csv", dict(group="knows_person", model="voxtral"), "ci_low", 0.27, 2),
    ("fooling Voxtral knows, high", LT + "turing_summary.csv", dict(group="knows_person", model="voxtral"), "ci_high", 0.80, 2),
]


def read(name):
    with open(RESULTS / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def cell(row, decimals):
    """median with its interval, or an em dash when the metric was not logged."""
    if row is None:
        return "not recorded"
    return (f"{float(row['median']):.{decimals}f} "
            f"[{float(row['ci_low']):.{decimals}f}, {float(row['ci_high']):.{decimals}f}]")


def find(rows, **where):
    for row in rows:
        if all(row.get(k) == v for k, v in where.items()):
            return row
    return None


def table(body, header, caption, label, spec):
    # \small keeps the five-column table with intervals inside the text width.
    lines = ["\\begin{table}[H]", "    \\centering", "    \\small",
             f"    \\begin{{tabular}}{{{spec}}}", "        \\toprule",
             "        " + " & ".join(header) + r" \\", "        \\midrule"]
    lines += body
    lines += ["        \\bottomrule", "    \\end{tabular}",
              f"    \\caption{{{caption}}}", f"    \\label{{{label}}}",
              "\\end{table}", ""]
    return "\n".join(lines)


def render_reliability_table(rows):
    body = []
    for model in SYSTEMS:
        row = find(rows, model=model)
        usable = int(row["n_texts"]) - int(row["n_unusable"])
        body.append(f"        {LABELS[model]} & {usable} / {row['n_texts']} "
                    f"& {float(row['usable_rate']):.2f} \\\\")
    return table(
        body,
        [r"\textbf{System}", r"\textbf{Usable renders}", r"\textbf{Rate}"],
        "Proportion of the 16 test texts rendered usably by each system.",
        "tab:render_reliability", "lcc")


def quality_table(summary):
    body = []
    for model in SYSTEMS + ["REAL"]:
        if model == "REAL":
            body.append("        \\midrule")
        sim = find(summary, metric="sim", model=model)
        wer = find(summary, metric="wer", model=model)
        body.append(f"        {LABELS[model]} & {cell(sim, 3)} & {cell(wer, 3)} \\\\")
    n_sys = find(summary, metric="sim", model="qwen3")["n_texts"]
    n_ref = find(summary, metric="sim", model="REAL")["n_texts"]
    return table(
        body,
        [r"\textbf{Condition}", r"\textbf{SIM} $\uparrow$", r"\textbf{WER} $\downarrow$"],
        "Speaker similarity and word error rate. Median across texts with "
        f"bootstrap 95\\,\\% confidence intervals; $n = {n_sys}$ texts per system "
        f"and $n = {n_ref}$ for the reference condition, which excludes the "
        "recording used for enrolment.",
        "tab:objective_quality", "lcc")


def speed_table(summary):
    body = []
    for model in SYSTEMS:
        rtf = find(summary, metric="service_rtf", model=model)
        ttfa = find(summary, metric="ttfa_s", model=model)
        lat = find(summary, metric="latency_s", model=model)
        body.append(f"        {LABELS[model]} & {cell(rtf, 3)} & {cell(ttfa, 2)} "
                    f"& {cell(lat, 1)} \\\\")
    return table(
        body,
        [r"\textbf{Model}", r"\textbf{Service RTF} $\downarrow$",
         r"\textbf{TTFA (s)} $\downarrow$", r"\textbf{End-to-end (s)} $\downarrow$"],
        "Response time of the three services, measured end-to-end over the "
        "internet and including network transfer. Median across texts with "
        "bootstrap 95\\,\\% confidence intervals. No first-chunk timestamp was "
        "logged for Voxtral in this run.",
        "tab:objective_speed", "lccc")


def results_table(summary):
    """One table for the main text. TTFA is left out on purpose: it is missing
    for Voxtral (no first-chunk timestamp logged) and is already given in the
    running text, where a gap in the data does not read as a hole in a table.
    Service RTF is kept instead - it is complete for all three systems and is
    the one figure the vendors themselves quote in Section 'Model choice',
    which is what makes the measured value worth a table cell."""
    body = []
    for model in SYSTEMS + ["REAL"]:
        if model == "REAL":
            body.append("        \\midrule")
        sim = find(summary, metric="sim", model=model)
        wer = find(summary, metric="wer", model=model)
        rtf = find(summary, metric="service_rtf", model=model)
        speed = "---" if model == "REAL" else cell(rtf, 3)
        body.append(f"        {SHORT[model]} & {cell(sim, 3)} & {cell(wer, 3)} & {speed} \\\\")
    n_sys = find(summary, metric="sim", model="qwen3")["n_texts"]
    n_ref = find(summary, metric="sim", model="REAL")["n_texts"]
    return table(
        body,
        [r"\textbf{Condition}", r"\textbf{SIM} $\uparrow$", r"\textbf{WER} $\downarrow$",
         r"\textbf{Service RTF} $\downarrow$"],
        "Objective results, reported as median [95\\,\\% CI] across texts; intervals "
        "from 4,000 bootstrap replicates over texts. "
        f"$n = {n_sys}$ texts per system, $n = {n_ref}$ for the reference condition "
        "(Real), which excludes the recording used as the similarity reference. "
        "Service RTF applies to the three services only.",
        "tab:objective_results", "lccc")


def pairwise_table(pairwise):
    body = []
    for i, (metric, name) in enumerate([("sim", "SIM"), ("wer", "WER")]):
        if i:
            body.append("        \\midrule")
        rows = [r for r in pairwise if r["metric"] == metric]
        for j, row in enumerate(rows):
            a, b = SHORT[row["model_a"]], SHORT[row["model_b"]]
            body.append(
                f"        {name if j == 0 else ''} & {a} $-$ {b} "
                f"& ${float(row['difference']):+.3f}$ "
                f"& $[{float(row['ci_low_adj']):+.3f}, {float(row['ci_high_adj']):+.3f}]$ \\\\")
    return table(
        body,
        [r"\textbf{Metric}", r"\textbf{Comparison}", r"\textbf{Difference}",
         r"\textbf{Adjusted 95\,\% CI}"],
        "Differences between conditions, computed per text and bootstrapped "
        "over texts. Intervals are Bonferroni-adjusted over the six comparisons "
        "of each metric. An interval containing zero indicates that the "
        "difference is not detectable on this material.",
        "tab:objective_pairwise", "llcc")


def check_prose():
    problems = []
    cache = {}
    for name, file, where, field, written, decimals in PROSE_CLAIMS:
        rows = cache.setdefault(file, read(file))
        row = find(rows, **where)
        if row is None:
            problems.append(f"  {name}: no longer present in the results")
            continue
        actual = round(float(row[field]), decimals)
        if abs(actual - written) > 10 ** -decimals / 2:
            problems.append(f"  {name}: text says {written}, results say {actual}")
    if problems:
        print("\nThe running text of the manuscript sections is out of date:")
        print("\n".join(problems))
        print("Edit those sentences by hand, then re-run this script.")
    else:
        print("\nRunning text matches the result files.")
    return not problems


def main():
    summary = read("objective_summary.csv")
    pairwise = read("objective_pairwise.csv")
    reliability = read("render_reliability.csv")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # objective_results.tex is the one the manuscript uses; the others are kept
    # for an appendix or a longer version.
    written = {
        "objective_results.tex": results_table(summary),
        "render_reliability.tex": render_reliability_table(reliability),
        "objective_quality.tex": quality_table(summary),
        "objective_speed.tex": speed_table(summary),
        "objective_pairwise.tex": pairwise_table(pairwise),
    }
    for name, content in written.items():
        (OUT_DIR / name).write_text(content, encoding="utf-8")
        print(f"wrote {(OUT_DIR / name).relative_to(ROOT)}")

    ok = check_prose()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
