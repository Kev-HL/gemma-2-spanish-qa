# Makefile for Gemma-2 Spanish QA Finetuning Project

# Run tests
test:
	@echo "Running tests (placeholder: No tests defined yet)"
	# python -m pytest tests/    # Uncomment when tests are available

# Lint code
lint:
	@echo "Linting source code"
	# black src/ tests/ scripts/ # Uncomment when code available
	# flake8 src/ tests/ scripts/ # Uncomment when code available

# Clean temporary/model files
clean:
	@echo "Cleaning up..."
	rm -rf __pycache__ .pytest_cache *.pyc .mypy_cache output/ models/*

# Download datasets
download_data:
	@echo "Downloading datasets (placeholder)"
	# Add commands to download datasets here

# Help: show available commands
help:
	@echo "Available make targets:"
	@grep -E '^[a-zA-Z_-]+:' Makefile | cut -d':' -f1 | grep -v '^_' | sort

.PHONY: test lint clean download_data help