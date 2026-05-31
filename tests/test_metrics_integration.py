"""
Unit tests for some of the metrics.py functions, located in src/metrics.py
These are integration tests that use real tokenizers (external resources).
This is intended to run locally and not in CI.
"""

# Standard imports
import random
from unittest.mock import Mock

# Third party imports
import numpy as np
import pytest
from datasets import Dataset
from transformers import AutoTokenizer

# Local imports
from metrics import factory_compute_metrics_mbert, factory_compute_metrics_gemma2
from preprocessing import factory_preprocess_mbert, factory_preprocess_gemma2_train


# Fixtures for tokenizers
@pytest.fixture(scope="module")
def tkn_mbert(hf_token: str) -> AutoTokenizer:
    """Load mBERT tokenizer once per module"""
    return AutoTokenizer.from_pretrained(
        "bert-base-multilingual-cased",
        token=hf_token,  # Public access, but passing token to increase rate limits
    )


@pytest.fixture(scope="module")
def tkn_gemma2(hf_token: str) -> AutoTokenizer:
    """Load Gemma2 tokenizer once per module"""
    return AutoTokenizer.from_pretrained(
        "google/gemma-2-2b", token=hf_token  # Gated access, token required
    )


# Fixtures for data (raw and preprocessed)
@pytest.fixture(scope="module")
def sample_batch() -> Dataset:
    """Load and preprocess a small batch of SQuAD data (train split)"""
    # Load a small batch of training data
    data_path = "data/squad_es/clean_train-v1.1-es_small.json"
    squad_train = Dataset.from_json(data_path)
    sample_batch = squad_train.select(range(100))

    return sample_batch


@pytest.fixture(scope="module")
def mbert_preprocessed_data(
    tkn_mbert: AutoTokenizer, sample_batch: Dataset
) -> tuple[int, Dataset]:
    """Load and preprocess a small batch of SQuAD data for mBERT metrics tests"""
    # Set max sequence length
    max_seq_length = 512

    # Create the preprocessing function using the factory
    preprocess_fn = factory_preprocess_mbert(
        tokenizer=tkn_mbert, max_seq_length=max_seq_length
    )

    # Call the preprocessing function on the test batch
    processed_batch = sample_batch.map(
        preprocess_fn, batched=True, remove_columns=sample_batch.column_names
    )

    return max_seq_length, processed_batch


@pytest.fixture(scope="module")
def gemma2_preprocessed_data(
    tkn_gemma2: AutoTokenizer, sample_batch: Dataset
) -> Dataset:
    """Load and preprocess a small batch of SQuAD data for Gemma2 metrics tests"""
    # Set max sequence length
    max_seq_length = 512

    # Create the preprocessing function using the factory
    preprocess_fn = factory_preprocess_gemma2_train(
        tokenizer=tkn_gemma2, max_seq_length=max_seq_length
    )

    # Call the preprocessing function on the test batch
    processed_batch = sample_batch.map(
        preprocess_fn, batched=True, remove_columns=sample_batch.column_names
    )

    return processed_batch


@pytest.mark.integration
class TestFactoryComputeMetricsMbert:
    """Integration tests for factory_compute_metrics_mbert"""

    def test_integration_perfect_scores(
        self, sample_batch: Dataset, mbert_preprocessed_data: tuple[int, Dataset]
    ) -> None:
        """
        Test that the inner function computes metrics without errors.
        Uses a subset of the real training data, preprocessed with the real tokenizer,
        and made up perfect predictions.
        Logits are created to match the true start and end positions, so that the
        output metrics are perfect (100% F1 and EM).
        """
        # Get the data fixture for mBERT metrics tests
        max_seq_length, processed_batch = mbert_preprocessed_data

        # Create perfect predictions from the processed batch
        start_logits_list = []
        end_logits_list = []
        for sample in range(len(processed_batch)):
            start_position = processed_batch[sample]["start_positions"]
            end_position = processed_batch[sample]["end_positions"]
            start_logit = [
                max_seq_length - abs(i - start_position) for i in range(max_seq_length)
            ]
            end_logit = [
                max_seq_length - abs(i - end_position) for i in range(max_seq_length)
            ]
            start_logits_list.append(start_logit)
            end_logits_list.append(end_logit)
        eval_preds = Mock()
        eval_preds.predictions = [start_logits_list, end_logits_list]

        # Create the compute_metrics function
        compute_metrics = factory_compute_metrics_mbert(
            raw_dataset=sample_batch,
            preprocessed_dataset=processed_batch,
        )

        # Call the compute_metrics function
        metrics = compute_metrics(eval_preds)

        # Check that the output is a dictionary with expected keys and metric scores
        assert isinstance(metrics, dict)
        assert "f1" in metrics
        assert "exact_match" in metrics
        assert metrics["f1"] == 100.0, "Expected perfect F1 score"
        assert metrics["exact_match"] == 100.0, "Expected perfect EM score"

    def test_integration_random_scores(
        self, sample_batch: Dataset, mbert_preprocessed_data: tuple[int, Dataset]
    ) -> None:
        """
        Test that the inner function computes metrics without errors.
        Uses a subset of the real training data, preprocessed with the real tokenizer,
        and made up random predictions.
        Logits are created randomly, to check that the metrics output is in range.
        """
        # Get the data fixture for mBERT metrics tests
        max_seq_length, processed_batch = mbert_preprocessed_data

        # Create random predictions from the processed batch
        start_logits_list = []
        end_logits_list = []
        for _ in range(len(processed_batch)):
            start_logit = [random.random() for _ in range(max_seq_length)]
            end_logit = [random.random() for _ in range(max_seq_length)]
            start_logits_list.append(start_logit)
            end_logits_list.append(end_logit)
        eval_preds = Mock()
        eval_preds.predictions = [start_logits_list, end_logits_list]

        # Create the compute_metrics function
        compute_metrics = factory_compute_metrics_mbert(
            raw_dataset=sample_batch,
            preprocessed_dataset=processed_batch,
        )

        # Call the compute_metrics function
        metrics = compute_metrics(eval_preds)

        # Check that the output metrics are in valid ranges (0-100)
        assert 0.0 <= metrics["f1"] <= 100.0
        assert 0.0 <= metrics["exact_match"] <= 100.0


@pytest.mark.integration
class TestFactoryComputeMetricsGemma2:
    """Integration tests for factory_compute_metrics_gemma2"""

    def test_integration_perfect_scores(
        self,
        tkn_gemma2: AutoTokenizer,
        sample_batch: Dataset,
        gemma2_preprocessed_data: Dataset,
    ) -> None:
        """
        Test that the inner function computes metrics without errors.
        Uses a subset of the real training data, preprocessed with the real tokenizer,
        and made up perfect predictions.
        Predictions are created using the tokenized inputs, so that the
        output metrics are perfect (100% F1 and EM).
        """
        # Create perfect predictions from the processed batch
        encoded_preds = np.array(gemma2_preprocessed_data["input_ids"])
        labels = np.array(gemma2_preprocessed_data["labels"])
        eval_preds = [encoded_preds, labels]

        # Create the compute_metrics function
        compute_metrics = factory_compute_metrics_gemma2(
            raw_dataset=sample_batch,
            tokenizer=tkn_gemma2,
        )

        # Call the compute_metrics function
        metrics = compute_metrics(eval_preds)

        # Check that the output is a dictionary with expected keys and metric scores
        assert isinstance(metrics, dict)
        assert "f1" in metrics
        assert "exact_match" in metrics
        assert metrics["f1"] == 100.0, "Expected perfect F1 score"
        assert metrics["exact_match"] == 100.0, "Expected perfect EM score"

    def test_integration_random_scores(
        self,
        tkn_gemma2: AutoTokenizer,
        sample_batch: Dataset,
        gemma2_preprocessed_data: Dataset,
    ) -> None:
        """
        Test that the inner function computes metrics without errors.
        Uses a subset of the real training data, preprocessed with the real tokenizer,
        and made up not-perfect predictions.
        Predictions are created using the tokenized inputs, which then get randomly
        shuffled, so that the output metrics are not perfect (less than 100% F1 and EM).
        """
        # Create not-perfect predictions from the processed batch
        SHUFFLE_PERCENT = 0.25  # shuffle 25% of the samples for not-perfect predictions
        rng = np.random.default_rng()
        encoded_preds = np.array(gemma2_preprocessed_data["input_ids"])
        n_rows = encoded_preds.shape[0]
        n_shuffle = int(SHUFFLE_PERCENT * n_rows)
        idx = rng.choice(n_rows, size=n_shuffle, replace=False)
        encoded_preds[idx] = encoded_preds[rng.permutation(idx)]
        labels = np.array(gemma2_preprocessed_data["labels"])
        eval_preds = [encoded_preds, labels]

        # Create the compute_metrics function
        compute_metrics = factory_compute_metrics_gemma2(
            raw_dataset=sample_batch,
            tokenizer=tkn_gemma2,
        )

        # Call the compute_metrics function
        metrics = compute_metrics(eval_preds)

        # Check that the output metrics are not perfect
        assert metrics["f1"] < 100.0
        assert metrics["exact_match"] < 100.0
