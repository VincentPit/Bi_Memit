# Bi-MEMIT Makefile
# 
# Common development tasks for the Bi-MEMIT project

.PHONY: help install install-dev test lint format docs clean build upload

# Default target
help:
	@echo "Bi-MEMIT Development Commands:"
	@echo ""
	@echo "  install      Install the package"
	@echo "  install-dev  Install development dependencies"  
	@echo "  test         Run the test suite"
	@echo "  lint         Run code linting"
	@echo "  format       Format code with black and isort"
	@echo "  docs         Build documentation"
	@echo "  clean        Clean build artifacts"
	@echo "  build        Build distribution packages"
	@echo "  upload       Upload to PyPI (maintainers only)"
	@echo "  setup        Complete development setup"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=src --cov-report=html --cov-report=term

test-fast:
	pytest tests/ -m "not slow" -v

# Code Quality
lint:
	flake8 src tests
	mypy src

format:
	black src tests
	isort src tests

check:
	black --check src tests
	isort --check-only src tests
	flake8 src tests
	mypy src

# Documentation
docs:
	cd docs && make html

docs-serve:
	cd docs/_build/html && python -m http.server 8000

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Building and Distribution
build: clean
	python -m build

upload: build
	python -m twine upload dist/*

upload-test: build
	python -m twine upload --repository testpypi dist/*

# Development Setup
setup: install-dev
	@echo "Development environment setup complete!"
	@echo "Run 'make test' to verify everything is working."

# CI/CD Commands
ci-install:
	pip install -e ".[dev]"

ci-test: test-cov lint

# Docker (if applicable)
docker-build:
	docker build -t bi-memit .

docker-run:
	docker run -it bi-memit

# Jupyter
notebook:
	jupyter notebook examples/notebooks/

lab:
	jupyter lab examples/notebooks/