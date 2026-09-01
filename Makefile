# One entry point for the evaluation pipeline. Each target is a thin wrapper
# around a script in evaluation/ - see README.md for what each step does and
# docs/METHODS.md for why.

PY := uv run python
CASE ?=
SOURCE ?=
REPEATS ?= 3

.PHONY: help enrollment objective stimuli check sessions results long-form coverage test clean-test

help:
	@echo "myEcho evaluation pipeline"
	@echo
	@echo "  make enrollment SOURCE=data/wav/x.wav   one canonical voice sample for all 3 services"
	@echo "  make objective [REPEATS=3]              synthesize + SIM/WER/latency   (uses the APIs)"
	@echo "  make stimuli                            build the blinded listening test"
	@echo "  make check                              verify it - run before the first rater"
	@echo "  make sessions                           serve the test at localhost:8765"
	@echo "  make results                            naturalness, 2AFC and objective comparisons"
	@echo
	@echo "  make long-form                          stability of the full renders"
	@echo "  make coverage                           does the reported interval cover?"
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

long-form:
	$(PY) evaluation/analyze_objective.py --long-form

coverage:
	$(PY) tools/check_ci_coverage.py

test:
	uv run pytest -q
