# Makefile for GeoAI-STGNN

# Variables
PYTHON := python
PIP := pip
TEST_DIR := tests
SRC_DIR := .

# Help target
.PHONY: help
help:
	@echo "GeoAI-STGNN Makefile"
	@echo "===================="
	@echo "Available targets:"
	@echo "  help          - Show this help message"
	@echo "  install       - Install dependencies"
	@echo "  install-dev   - Install development dependencies"
	@echo "  test          - Run all tests"
	@echo "  test-cov      - Run tests with coverage"
	@echo "  lint          - Run code linting"
	@echo "  format        - Format code with black and isort"
	@echo "  clean         - Clean up generated files"
	@echo "  run           - Run the main pipeline"
	@echo "  streamlit     - Run the Streamlit app"
	@echo "  docs          - Generate documentation"

# Install dependencies
.PHONY: install
install:
	$(PIP) install -r requirements.txt

# Install development dependencies
.PHONY: install-dev
install-dev:
	$(PIP) install -r requirements.txt
	$(PIP) install pytest black isort flake8 mypy pre-commit

# Run tests
.PHONY: test
test:
	$(PYTHON) -m pytest $(TEST_DIR) -v

# Run tests with coverage
.PHONY: test-cov
test-cov:
	$(PYTHON) -m pytest $(TEST_DIR) --cov=$(SRC_DIR) --cov-report=html --cov-report=term

# Run code linting
.PHONY: lint
lint:
	$(PYTHON) -m flake8 $(SRC_DIR)
	$(PYTHON) -m mypy $(SRC_DIR)

# Format code
.PHONY: format
format:
	$(PYTHON) -m isort $(SRC_DIR)
	$(PYTHON) -m black $(SRC_DIR)

# Clean up generated files
.PHONY: clean
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf .mypy_cache

# Run the main pipeline
.PHONY: run
run:
	$(PYTHON) main.py

# Run the Streamlit app
.PHONY: streamlit
streamlit:
	$(PYTHON) -m streamlit run app/main_app.py

# Generate documentation
.PHONY: docs
docs:
	@echo "Documentation is available in the docs/ directory"
	@echo "API documentation: docs/api.md"
	@echo "Architecture: docs/architecture.md"
	@echo "User guide: docs/user_guide.md"
	@echo "Contributing guide: docs/contributing.md"

# Setup pre-commit hooks
.PHONY: setup-hooks
setup-hooks:
	$(PYTHON) -m pre_commit install

# Run pre-commit hooks
.PHONY: pre-commit
pre-commit:
	$(PYTHON) -m pre_commit run --all-files
