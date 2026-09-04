PY ?= python3

.PHONY: setup lint typecheck test smoke pilot full figures report clean-smoke

setup:
	$(PY) -m pip install -e ".[dev]"

lint:
	$(PY) -m ruff check src tests

typecheck:
	$(PY) -m mypy src/belief_feedback

test:
	$(PY) -m pytest tests

smoke:
	bash scripts/run_all.sh configs/smoke.yaml

pilot:
	bash scripts/run_all.sh configs/pilot.yaml

full:
	bash scripts/run_all.sh configs/full.yaml

figures:
	$(PY) -m belief_feedback.cli make-figures --config $(or $(CONFIG),configs/smoke.yaml)

report:
	$(PY) -m belief_feedback.cli make-report --config $(or $(CONFIG),configs/smoke.yaml)

clean-smoke:
	rm -rf artifacts/runs/smoke artifacts/data/smoke artifacts/vectors/smoke artifacts/models/smoke \
	  artifacts/figures/smoke artifacts/figure_data/smoke artifacts/tables/smoke artifacts/reports/smoke \
	  artifacts/manifests/smoke artifacts/activations/smoke
