# One entry point for the evaluation pipeline. Each target is a thin wrapper
# around a script in evaluation/ - see README.md for what each step does and
# docs/METHODS.md for why.

PY := uv run python
CASE ?=
SOURCE ?=
REPEATS ?= 3

.PHONY: help enrollment objective stimuli check sessions results coverage report test clean-test

help:
	@echo "myEcho evaluation pipeline"
	@echo
	@echo "  make enrollment SOURCE=data/wav/x.wav   one canonical voice sample for all 3 services"
	@echo "  make objective [REPEATS=3]              synthesize + SIM/WER/latency   (uses the APIs)"
	@echo "  make stimuli                            build the blinded listening test"
	@echo "  make check                              verify it - run before the first rater"
	@echo "  make sessions                           serve the test at localhost:8765"
	@echo "  make results                            naturalness, 2AFC and objective comparisons"
	@echo "  make clean-test                         archive this round and start the test over"
	@echo
	@echo "Supporting analyses (not part of the pipeline):"
	@echo "  make coverage                           does the reported interval cover?"
	@echo "  make report                             rebuild the report tables and figures"
	@echo "  make test                               unit tests"
	@echo
	@echo "Only 'objective' calls the APIs. Everything else runs on its output."

# Step 0 - one reference file for all three services, so a similarity difference
# can be attributed to the model instead of to the input it was given.
enrollment:
	@test -n "$(SOURCE)" || (echo "usage: make enrollment SOURCE=data/wav/<file>.wav"; exit 1)
	$(PY) evaluation/prepare_enrollment.py $(SOURCE)

# Step 1 - the only step that costs money.
objective:
	$(PY) evaluation/run_eval.py --repeats $(REPEATS)

# Step 2 - excerpts, screening, then the frozen stimulus set.
stimuli:
	$(PY) evaluation/select_excerpts.py $(CASE)
	$(PY) evaluation/qc_samples.py $(CASE)
	$(PY) evaluation/prepare_listening_test.py $(CASE)

# Step 3 - blinding and design invariants. Exits non-zero on any failure,
# because none of them can be repaired after the first rating.
check:
	$(PY) evaluation/check_listening_test.py

# Step 4 - one browser per rater; the rater enters their own scores.
sessions:
	$(PY) evaluation/listening_test_server.py

# Step 5 - everything the report needs.
results:
	$(PY) evaluation/analyze_listening_test.py
	$(PY) evaluation/analyze_objective.py

coverage:
	$(PY) tools/check_ci_coverage.py

# Rebuild the report tables and figures from the result CSVs, so no number in
# the manuscript is hand-typed. Also re-checks the values quoted in the prose.
report:
	uv run --extra notebook python tools/make_paper_tables.py
	uv run --extra notebook python tools/make_paper_figures.py

test:
	uv run --extra dev pytest -q

# Step 5b - start the listening test over. Archives the current round into
# results/archive/<stamp>/ instead of deleting it, so a reset is reversible.
# Never touches case_*/ (the rendered audio, which cost API money) or
# alignments/ (the cached ASR passes).
clean-test:
	@stamp=$$(date +%Y%m%d_%H%M%S); \
	dir=evaluation/results/archive/$$stamp; \
	mkdir -p $$dir; \
	for f in responses key.json index.html run_meta.json stimuli.csv \
	         cue_audit.csv share.zip nmos_summary.csv nmos_pairwise.csv \
	         turing_summary.csv turing_familiarity.csv reliability.csv \
	         ratings_long.csv samples; do \
	  test -e evaluation/results/listening_test/$$f && \
	    mv evaluation/results/listening_test/$$f $$dir/ || true; \
	done; \
	echo "archived the previous round to $$dir"; \
	echo "rendered audio and the ASR cache were not touched."; \
	echo "next: make stimuli && make check"
