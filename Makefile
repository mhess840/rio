.PHONY: format
format:
	uv run ruff check --fix
	uv run ruff format

.PHONY: type
type:
	uv run ty check

.PHONY: check
check:
	uv run ruff check
	uv run ruff format --check
	uv run ty check

.PHONY: test
test:
	uv run pytest -m "not hardware and not gpu and not benchmark"

.PHONY: test-unit
test-unit:
	uv run pytest -m "unit and not hardware and not gpu"

.PHONY: test-integration
test-integration:
	uv run pytest -m "integration and not hardware and not gpu and not benchmark"

.PHONY: test-gpu
test-gpu:
	uv run pytest -m gpu

# Same selection as `test`, with LeRobot/Torch installed so the π₀.₅, conversion and merge tests run.
.PHONY: test-pi
test-pi:
	uv run --group pi pytest -m "not hardware and not gpu and not benchmark"

.PHONY: test-hardware
test-hardware:
	uv run pytest -m hardware

.PHONY: build
build:
	uv build
