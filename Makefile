# One command regenerates the submission from the data bundle and this repository: `make submit`.
# See docs/REPRODUCE.md. TEAM names the file, as the submission form asks.

PY   := uv run python
TEAM ?= katenaball
OUT  := submissions/$(TEAM).csv
WORKERS ?= 3

.PHONY: submit external model predict validate test check-target check-public

submit: external model predict validate

external:
	$(PY) scripts/build_external.py

model:
	$(PY) scripts/make_submission.py corpus --workers $(WORKERS)
	$(PY) scripts/make_submission.py worlds --workers $(WORKERS)
	$(PY) scripts/make_submission.py train

predict:
	$(PY) scripts/make_submission.py predict --out $(OUT)

validate:
	$(PY) evaluation/validate_submission.py $(OUT)
	shasum -a 256 $(OUT)

test:
	uv run pytest -q

# Human-run, once the data bundle is in data/ (J1). Every script prints aggregates or pass/fail only.
check-target:
	$(PY) scripts/probe_schema.py
	$(PY) scripts/check_target.py
	$(PY) scripts/check_target_layout.py
	$(PY) scripts/check_touch_target.py

# Booleans only; the public window is a pipeline check, never a selector. Log each run in the decision log.
check-public:
	$(PY) scripts/check_public_window.py $(OUT)
