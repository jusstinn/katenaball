# One command regenerates the submission from the data bundle and this repository: `make submit`.
# See docs/REPRODUCE.md. TEAM names the file, as the submission form asks.

PY   := uv run python
TEAM ?= katenaball
OUT  := submissions/$(TEAM).csv

.PHONY: submit external model predict validate test

submit: external model predict validate

external:
	$(PY) scripts/build_external.py

model:
	$(PY) scripts/make_submission.py corpus
	$(PY) scripts/make_submission.py worlds --workers 3
	$(PY) scripts/make_submission.py train

predict:
	$(PY) scripts/make_submission.py predict --out $(OUT)

validate:
	$(PY) evaluation/validate_submission.py $(OUT)
	shasum -a 256 $(OUT)

test:
	uv run pytest -q
