"""
Unit tests for some of the preprocessing.py functions, located in src/preprocessing.py
These are integration tests that use real tokenizers (external resources).
This is intended to run locally and not in CI.
"""

# Third party imports
import pytest
from transformers import AutoTokenizer
from datasets import Dataset

# Local imports
from preprocessing import factory_preprocess_mbert, factory_preprocess_gemma2_train


# Fixtures for tokenizers
@pytest.fixture(scope="module")
def tkn_mbert(hf_token):
    """Load mBERT tokenizer once per module"""
    return AutoTokenizer.from_pretrained(
        "bert-base-multilingual-cased",
        token=hf_token,  # Public access, but passing token to increase rate limits
    )


@pytest.fixture(scope="module")
def tkn_gemma2(hf_token):
    """Load Gemma2 tokenizer once per module"""
    return AutoTokenizer.from_pretrained(
        "google/gemma-2-2b", token=hf_token  # Gated access, token required
    )


@pytest.mark.integration
class TestFactoryPreprocessMbert:
    """Integration tests for factory_preprocess_mbert"""

    def test_integration_mbert(self, tkn_mbert: AutoTokenizer) -> None:
        """Test that the preprocessing function works end-to-end with mBERT tokenizer"""
        # Settings for factory function
        max_seq_length = 200  # small to test truncation

        # Create the preprocessing function using the factory
        preprocess_fn = factory_preprocess_mbert(
            tokenizer=tkn_mbert, max_seq_length=max_seq_length
        )

        # Load a small batch of training data
        data_path = "data/squad_es/clean_train-v1.1-es_small.json"
        squad_train = Dataset.from_json(data_path)
        sample_batch = squad_train.select(range(100))

        # Call the preprocessing function on the test batch
        processed_batch = sample_batch.map(
            preprocess_fn, batched=True, remove_columns=sample_batch.column_names
        )

        # Verify structure
        expected_fields = {
            "input_ids",
            "token_type_ids",
            "attention_mask",
            "offset_mapping",
            "example_id",
            "start_positions",
            "end_positions",
        }
        assert expected_fields.issubset(set(processed_batch.column_names))

        # Verify data validity
        for sample in processed_batch:
            assert len(sample["input_ids"]) <= max_seq_length
            assert len(sample["attention_mask"]) == len(sample["input_ids"])
            assert len(sample["token_type_ids"]) == len(sample["input_ids"])
            assert 0 <= sample["start_positions"]
            assert sample["start_positions"] <= sample["end_positions"]
            assert sample["end_positions"] < len(sample["input_ids"])

            none_count = sum(1 for o in sample["offset_mapping"] if o is None)
            assert none_count > 0, "Expected None values for non-context tokens"
            assert (
                none_count < max_seq_length
            ), "Expected some valid values for context tokens"

        assert len(processed_batch) > len(sample_batch), (
            f"Expected overflow samples with max_seq_length=200. "
            f"Got {len(processed_batch)} from {len(sample_batch)}"
        )


@pytest.mark.integration
class TestFactoryPreprocessGemma2Train:
    """Integration tests for factory_preprocess_gemma2_train"""

    def test_integration_gemma2_train(self, tkn_gemma2: AutoTokenizer) -> None:
        """
        Test that the preprocessing function works end-to-end with Gemma2 tokenizer
        """
        # Settings for factory function
        max_seq_length = 512  # training data pruned to fit 512, no truncation expected

        # Create the preprocessing function using the factory
        preprocess_fn = factory_preprocess_gemma2_train(
            tokenizer=tkn_gemma2, max_seq_length=max_seq_length
        )

        # Load a small batch of training data
        data_path = "data/squad_es/clean_train-v1.1-es_small.json"
        squad_train = Dataset.from_json(data_path)
        sample_batch = squad_train.select(range(100))

        # Call the preprocessing function on the test batch
        processed_batch = sample_batch.map(
            preprocess_fn, batched=True, remove_columns=sample_batch.column_names
        )

        # Verify structure
        expected_fields = {
            "input_ids",
            "attention_mask",
            "labels",
        }
        assert expected_fields.issubset(set(processed_batch.column_names))

        # Verify data validity
        for sample in processed_batch:
            assert len(sample["input_ids"]) <= max_seq_length
            assert len(sample["attention_mask"]) == len(sample["input_ids"])
            assert len(sample["labels"]) == len(sample["input_ids"])

            non_answer_count = sum(1 for o in sample["labels"] if o == -100)
            assert non_answer_count > 0, "Expected -100 values for non-answer tokens"
            assert (
                non_answer_count < max_seq_length
            ), "Expected some non-answer tokens (prompt, padding)"

    def test_truncation_gemma2_train(self, tkn_gemma2: AutoTokenizer) -> None:
        """
        Test that the preprocessing function truncates without overflow.
        """
        # Settings for factory function
        max_seq_length = 200  # small to test truncation

        # Create the preprocessing function using the factory
        preprocess_fn = factory_preprocess_gemma2_train(
            tokenizer=tkn_gemma2, max_seq_length=max_seq_length
        )

        # Load a small batch of training data
        data_path = "data/squad_es/clean_train-v1.1-es_small.json"
        squad_train = Dataset.from_json(data_path)
        sample_batch = squad_train.select(range(100))

        # Call the preprocessing function on the test batch
        processed_batch = sample_batch.map(
            preprocess_fn, batched=True, remove_columns=sample_batch.column_names
        )

        # Verify truncation occurred without overflow
        assert len(processed_batch) == len(
            sample_batch
        ), "Truncation enabled without overflow, expected same input and output size."
