# One command regenerates the submission from the data bundle and this repository: `make submit`.
# See docs/REPRODUCE.md. TEAM names the file, as the submission form asks.

PY   := uv run python
TEAM ?= katenaball
OUT  := submissions/$(TEAM).csv
WORKERS ?= 3

.PHONY: submit bundle external model predict validate test check-target check-public

submit: bundle external model predict validate

# Are the bundle's files where predict will look for them? Existence only. First, so a misplaced bundle stops
# the run before the download and the training.
bundle:
	$(PY) scripts/make_submission.py bundle

external:
	$(PY) scripts/build_external.py

model:
	$(PY) scripts/make_submission.py corpus --workers $(WORKERS)
	$(PY) scripts/make_submission.py worlds --workers $(WORKERS)
	$(PY) scripts/make_submission.py train

predict:
	$(PY) scripts/make_submission.py predict --out "$(OUT)"

validate:
	$(PY) evaluation/validate_submission.py "$(OUT)"
	$(PY) -c "import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], 'rb').read()).hexdigest(), sys.argv[1])" "$(OUT)"

test:
	uv run pytest -q

# Human-run, once the data bundle is in data/ (J1). Every script prints aggregates or pass/fail only.
check-target:
	$(PY) scripts/probe_schema.py
	$(PY) scripts/check_target.py
	$(PY) scripts/check_target_layout.py
	$(PY) scripts/check_touch_target.py
	$(PY) scripts/check_kickoff_tags.py

# Booleans only; the public window is a pipeline check, never a selector. Log each run in the decision log.
check-public:
	$(PY) scripts/check_public_window.py "$(OUT)"
