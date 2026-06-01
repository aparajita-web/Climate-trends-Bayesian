install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest

format:
	black src tests

lint:
	ruff check src tests

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete

notebook:
	jupyter lab