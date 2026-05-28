"""
Unit tests for some of the preprocessing.py functions, located in src/preprocessing.py
"""

# Standard imports
import logging
from unittest.mock import Mock

# Third party imports
import pytest

# Local imports
from preprocessing import factory_preprocess_mbert, factory_preprocess_gemma2_train


class TestFactoryPreprocessMbert:
    """Unit tests for factory_preprocess_mbert"""

    # ========= FACTORY FUNCTION TESTS =========
    def test_function_creation(self, caplog: pytest.LogCaptureFixture) -> None:
        """
        Test that the factory function returns a callable with valid config
        and logs the expected info message.
        """
        # Set up mock tokenizer and settings
        tokenizer_mock = Mock()
        tokenizer_mock.model_max_length = 512
        max_seq_length = 384

        # Call the factory function
        with caplog.at_level(logging.INFO):
            preprocess_fn = factory_preprocess_mbert(
                tokenizer_mock,
                max_seq_length,
            )
        assert callable(preprocess_fn)
        assert "function set with max_length=384 and doc_stride=128." in caplog.text

    def test_invalid_max_seq_length(self) -> None:
        """Test that invalid max_seq_length raises ValueError"""
        # Set up mock tokenizer
        tokenizer_mock = Mock()

        # Call function under test
        with pytest.raises(ValueError):
            factory_preprocess_mbert(tokenizer_mock, max_seq_length=-1)

    def test_invalid_doc_stride(self) -> None:
        """Test that invalid doc_stride raises ValueError"""
        # Set up mock tokenizer and settings
        tokenizer_mock = Mock()
        tokenizer_mock.model_max_length = 512
        max_seq_length = 384

        # Call function under test
        with pytest.raises(ValueError):
            factory_preprocess_mbert(tokenizer_mock, max_seq_length, doc_stride=-1)
        with pytest.raises(ValueError):
            factory_preprocess_mbert(tokenizer_mock, max_seq_length, doc_stride=400)

    # ========= INNER FUNCTION TESTS =========

    def test_preprocess_fn_basic_case(self) -> None:
        """Test that the inner function processes a batch with expected results"""

        # Set up mock tokenizer and settings
        # mBERT does [CLS] question [SEP] context [SEP] [PAD]...
        class TokenizerBehavior(dict):
            """Mock .sequence_ids() method"""

            def sequence_ids(self, i: int) -> list[int | None]:
                """Return sequence IDs for batch index i."""
                # Customize per your test needs
                return [None, 0, None, 1, None, None, None, None, None, None]

        tokenizer_mock = Mock()
        tokenizer_mock.model_max_length = 512
        tokenizer_mock.return_value = TokenizerBehavior(
            {  # Doesn't match test_samples
                "input_ids": [
                    [101, 200, 102, 201, 102, 0, 0, 0, 0, 0],
                    [101, 300, 102, 301, 102, 0, 0, 0, 0, 0],
                ],
                "token_type_ids": [
                    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                ],
                "attention_mask": [
                    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                ],
                "offset_mapping": [
                    [
                        (0, 0),
                        (0, 1),
                        (0, 0),
                        (0, 1),
                        (0, 0),
                        (0, 0),
                        (0, 0),
                        (0, 0),
                        (0, 0),
                        (0, 0),
                    ],
                    [
                        (0, 0),
                        (0, 1),
                        (0, 0),
                        (0, 1),
                        (0, 0),
                        (0, 0),
                        (0, 0),
                        (0, 0),
                        (0, 0),
                        (0, 0),
                    ],
                ],
                "overflow_to_sample_mapping": [0, 1],
            }
        )

        # Settings for factory function
        max_seq_length = 10
        doc_stride = 5

        # Create batch of test data (dict of lists)
        test_samples = {
            "id": ["1", "2"],
            "context": ["context 1", "context 2"],
            "question": ["question 1", "question 2"],
            "answers": [
                {"answer_start": [0], "text": ["answer 1"]},
                {"answer_start": [100], "text": ["answer 2"]},
            ],
        }

        # Call the factory function
        preprocess_fn = factory_preprocess_mbert(
            tokenizer_mock,
            max_seq_length,
            doc_stride,
        )

        # Call the inner function with test samples
        result = preprocess_fn(test_samples)  # type: ignore (using dict for simplicity)

        # Assertions on the result
        # Real case, result would be transformers.tokenization_utils_base.BatchEncoding
        assert isinstance(result, dict)
        assert "input_ids" in result
        assert "token_type_ids" in result
        assert "attention_mask" in result
        assert "offset_mapping" in result
        assert "example_id" in result
        assert "start_positions" in result
        assert "end_positions" in result
        assert result["example_id"] == ["1", "2"]
        assert result["start_positions"] == [3, 0]  # 2nd answer out of context
        assert result["end_positions"] == [4, 0]
        assert len(result["input_ids"]) == 2  # Batch size 2


class TestFactoryPreprocessGemma2Train:
    """Unit tests for factory_preprocess_gemma2_train"""

    # ========= FACTORY FUNCTION TESTS =========
    def test_function_creation(self, caplog: pytest.LogCaptureFixture) -> None:
        """
        Test that the factory function returns a callable with valid config
        and logs the expected info message.
        """
        # Set up mock tokenizer and settings
        tokenizer_mock = Mock()
        max_seq_length = 384

        # Call the factory function
        with caplog.at_level(logging.INFO):
            preprocess_fn = factory_preprocess_gemma2_train(
                tokenizer_mock,
                max_seq_length,
            )
        assert callable(preprocess_fn)
        assert "Gemma 2 preprocessing function set with max_length=384" in caplog.text

    def test_invalid_max_seq_length(self) -> None:
        """Test that invalid max_seq_length raises ValueError"""
        # Set up mock tokenizer
        tokenizer_mock = Mock()

        # Call function under test
        with pytest.raises(ValueError):
            factory_preprocess_gemma2_train(tokenizer_mock, max_seq_length=-1)

    # ========= INNER FUNCTION TESTS =========

    def test_preprocess_fn_basic_case(self) -> None:
        """Test that the inner function processes a batch with expected results"""
        # Set up mock tokenizer and settings
        # Gemma 2 does <bos> input... or if padding, <pad> <pad>... <bos> input
        # Usage:
        tokenizer_mock = Mock()
        tokenizer_mock.return_value = {  # Doesn't match test_samples
            "input_ids": [[2, 101, 102, 103, 104, 105], [2, 206, 207, 208, 209, 210]],
            "attention_mask": [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]],
        }  # Note: Padding will be tested in integration test

        # Settings for factory function
        max_seq_length = 20

        # Create batch of test data (dict of lists)
        test_samples = {
            "id": ["1", "2"],
            "context": ["context 1", "context 2"],
            "question": ["question 1", "question 2"],
            "answers": [
                {"answer_start": [0], "text": ["answer 1"]},
                {"answer_start": [100], "text": ["answer 2"]},
            ],
        }

        # Call the factory function
        preprocess_fn = factory_preprocess_gemma2_train(
            tokenizer_mock,
            max_seq_length,
        )

        # Call the inner function with test samples
        result = preprocess_fn(test_samples)  # type: ignore (using dict for simplicity)

        # Assertions on the result
        # Real case, result would be transformers.tokenization_utils_base.BatchEncoding
        assert isinstance(result, dict)
        assert "input_ids" in result
        assert "attention_mask" in result
        assert "labels" in result
        assert len(result["labels"][0]) == 6  # Length of mocked input_ids
        assert len(result["input_ids"]) == 2  # Batch size 2
