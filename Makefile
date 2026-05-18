# Makefile for Gemma-2 Spanish QA Finetuning Project

# Install dependencies
setup:
	pip install -r requirements.txt
	pip install -e .

# Run all tests
test:
	@echo "Running tests..."
	python -m pytest tests/ -v --cov=src

# Run only CI tests (not dependant on external resources like tokenizers)
test_ci:
	@echo "Running CI tests..."
	python -m pytest tests/ -m "not integration" -v --cov=src

# Run only integration tests (local, dependant on external resources like tokenizers)
test_integration:
	@echo "Running integration tests"
	python -m pytest tests/ -m "integration" -v --cov=src

# Lint code (format with black and check with flake8))
lint:
	@echo "Linting source code"
	python -m black src/ tests/
	# .flake8 file is configured with max-line-length = 88 
	# and ignore = E203, W503 for compatibility with black formatting
	python -m flake8 src/ tests/

# Clean temporary files
clean:
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
	find . -type f -name "*.pyc" -exec rm -f {} +
	@echo "Clean complete!"

# Download datasets
download_data:
	@apt-get update && apt-get install -y curl unzip
	@echo "Downloading datasets..."
	# Create directories
	@mkdir -p data/squad_es data/mlqa data/xquad

	# Download SQuAD-es
	@curl -L -o data/squad_es/train-v1.1-es_small.json \
		https://raw.githubusercontent.com/ccasimiro88/TranslateAlignRetrieve/master/SQuAD-es-v1.1/train-v1.1-es_small.json
	@curl -L -o data/squad_es/dev-v1.1-es_small.json \
		https://raw.githubusercontent.com/ccasimiro88/TranslateAlignRetrieve/master/SQuAD-es-v1.1/dev-v1.1-es_small.json
	@curl -L -o data/squad_es/LICENSE \
		https://raw.githubusercontent.com/ccasimiro88/TranslateAlignRetrieve/master/SQuAD-es-v1.1/LICENSE
	@echo "Translated SQuAD available at https://github.com/ccasimiro88/TranslateAlignRetrieve" \
		> data/squad_es/source.txt

	# Download MLQA
	@curl -L -o data/mlqa/MLQA_V1.zip https://dl.fbaipublicfiles.com/MLQA/MLQA_V1.zip
	@unzip data/mlqa/MLQA_V1.zip -d data/mlqa/
	@cp -f data/mlqa/MLQA_V1/LICENSE data/mlqa/
	@cp -f data/mlqa/MLQA_V1/dev/dev-context-es-question-es.json data/mlqa/
	@cp -f data/mlqa/MLQA_V1/test/test-context-es-question-es.json data/mlqa/
	@rm -rf data/mlqa/MLQA_V1
	@rm data/mlqa/MLQA_V1.zip
	@echo "MLQA dataset is available at https://github.com/facebookresearch/MLQA" \
		> data/mlqa/source.txt

	# Download XQuAD License (dataset directly loaded from Hugging Face)
	@curl -L -o data/xquad/LICENSE.txt \
		https://creativecommons.org/licenses/by-sa/4.0/legalcode.txt
	@echo "XQuAD dataset is available at https://github.com/google-deepmind/xquad and https://huggingface.co/datasets/google/xquad" \
		> data/xquad/source.txt

# Help: show available commands
help:
	@echo "Available make targets:"
	@grep -E '^[a-zA-Z_-]+:' Makefile | cut -d':' -f1 | grep -v '^_' | sort

.PHONY: setup test lint clean download_data help