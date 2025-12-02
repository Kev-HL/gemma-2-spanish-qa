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
	@apt-get update && apt-get install -y curl unzip
	@echo "Downloading datasets..."
	# Create directories
	@mkdir -p data/squad_es data/mlqa data/xquad
	# Download SQuAD-es
	@curl -L -o data/squad_es/train-v1.1-es_small.json https://raw.githubusercontent.com/ccasimiro88/TranslateAlignRetrieve/master/SQuAD-es-v1.1/train-v1.1-es_small.json
	@curl -L -o data/squad_es/dev-v1.1-es_small.json https://raw.githubusercontent.com/ccasimiro88/TranslateAlignRetrieve/master/SQuAD-es-v1.1/dev-v1.1-es_small.json
	@curl -L -o data/squad_es/LICENSE https://raw.githubusercontent.com/ccasimiro88/TranslateAlignRetrieve/master/SQuAD-es-v1.1/LICENSE
	@echo "Translated SQuAD available at https://github.com/ccasimiro88/TranslateAlignRetrieve" > data/squad_es/source.txt
	# Download MLQA
	@curl -L -o data/mlqa/MLQA_V1.zip https://dl.fbaipublicfiles.com/MLQA/MLQA_V1.zip
	@unzip data/mlqa/MLQA_V1.zip -d data/mlqa/
	@cp -f data/mlqa/MLQA_V1/LICENSE data/mlqa/
	@cp -f data/mlqa/MLQA_V1/dev/dev-context-es-question-es.json data/mlqa/
	@cp -f data/mlqa/MLQA_V1/test/test-context-es-question-es.json data/mlqa/
	@rm -rf data/mlqa/MLQA_V1
	@rm data/mlqa/MLQA_V1.zip
	@echo "MLQA dataset is available at https://github.com/facebookresearch/MLQA" > data/mlqa/source.txt
	# Download XQuAD License (dataset directly loaded from Hugging Face)
	@curl -L -o data/xquad/LICENSE.txt https://creativecommons.org/licenses/by-sa/4.0/legalcode.txt
	@echo "XQuAD dataset is available at https://github.com/google-deepmind/xquad and https://huggingface.co/datasets/google/xquad" > data/xquad/source.txt



# Help: show available commands
help:
	@echo "Available make targets:"
	@grep -E '^[a-zA-Z_-]+:' Makefile | cut -d':' -f1 | grep -v '^_' | sort

.PHONY: test lint clean download_data help