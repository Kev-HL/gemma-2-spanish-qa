"""Unit tests for some of the metrics.py functions, located in src/metrics.py"""

# Standard imports
import logging
from unittest.mock import Mock, patch, MagicMock

# Third party imports
import pytest

# Local imports
from metrics import (
    factory_compute_metrics_mbert,
    factory_compute_metrics_gemma2,
)


class TestFactoryComputeMetricsMbert:
    """Unit tests for factory_compute_metrics_mbert"""

    # ========= FACTORY FUNCTION TESTS =========
    @patch("metrics.evaluate")
    def test_function_creation(
        self, mock_evaluate: Mock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """
        Test that the factory function returns a callable with valid config
        and logs the expected info message.
        """
        # Create the compute_metrics function
        with caplog.at_level(logging.INFO):
            compute_metrics = factory_compute_metrics_mbert(
                raw_dataset=Mock(),
                preprocessed_dataset=Mock(),
            )
        # Assertions
        assert callable(compute_metrics)
        assert "SQuAD metric loaded for mBERT" in caplog.text
        assert "mBERT compute_metrics function set up with n_best=20" in caplog.text
        assert "and max_answer_length=50." in caplog.text

    def test_invalid_n_best(self) -> None:
        """Test that invalid n_best raises ValueError"""
        with pytest.raises(ValueError):
            factory_compute_metrics_mbert(
                raw_dataset=Mock(),
                preprocessed_dataset=Mock(),
                n_best=0,
            )

    def test_invalid_max_answer_length(self) -> None:
        """Test that invalid max_answer_length raises ValueError"""
        with pytest.raises(ValueError):
            factory_compute_metrics_mbert(
                raw_dataset=Mock(),
                preprocessed_dataset=Mock(),
                max_answer_length=0,
            )

    # ========= INNER FUNCTION TESTS =========
    @patch("metrics.evaluate")
    def test_compute_metrics_basic_case(self, mock_evaluate: Mock) -> None:
        """
        Test that the inner function computes metrics without errors.
        Mocks the evaluate library, and the predictions with perfect logits.

        Uses a sample with two examples, with the real tokenizer (preprocessed) data.
        """

        # Mock the evaluate.load function to return a mock metric
        mock_metric = Mock()
        mock_evaluate.load.return_value = mock_metric
        mock_metric.compute.return_value = {"f1": 70.1, "exact_match": 55.5}

        # Mock datasets with actual structure and tokenized content
        raw_dataset = [
            {
                "id": "A1",
                "context": "A is hello extra",
                "question": "What is A?",
                "answers": {"answer_start": [5], "text": ["hello"]},
            },
            {
                "id": "B2",
                "context": "B is a world extra",
                "question": "What is B?",
                "answers": {"answer_start": [7], "text": ["world"]},
            },
        ]

        preprocessed_dataset = [
            {
                "input_ids": [
                    101,
                    12489,
                    10124,
                    138,
                    136,
                    102,
                    138,
                    10124,
                    61694,
                    10133,
                    19868,
                    102,
                    0,
                    0,
                    0,
                ],
                "token_type_ids": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                "offset_mapping": [
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    [0, 1],
                    [2, 4],
                    [5, 9],
                    [9, 10],
                    [11, 16],
                    None,
                    None,
                    None,
                    None,
                ],
                "example_id": "A1",
                "start_positions": 8,
                "end_positions": 9,
            },
            {
                "input_ids": [
                    101,
                    12489,
                    10124,
                    139,
                    136,
                    102,
                    139,
                    10124,
                    169,
                    11356,
                    19868,
                    102,
                    0,
                    0,
                    0,
                ],
                "token_type_ids": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                "offset_mapping": [
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    [0, 1],
                    [2, 4],
                    [5, 6],
                    [7, 12],
                    [13, 18],
                    None,
                    None,
                    None,
                    None,
                ],
                "example_id": "B2",
                "start_positions": 9,
                "end_positions": 9,
            },
        ]

        # Mock EvalPrediction with dummy predictions and labels
        mock_eval_preds = Mock()
        mock_eval_preds.predictions = (
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            ],
        )

        # Create the compute_metrics function
        compute_metrics = factory_compute_metrics_mbert(
            raw_dataset=raw_dataset,
            preprocessed_dataset=preprocessed_dataset,
        )

        # Call the compute_metrics function
        metrics = compute_metrics(mock_eval_preds)

        # Check that the output is a dictionary with expected keys
        assert isinstance(metrics, dict)
        assert mock_metric.compute.called_once()
        assert "f1" in metrics
        assert "exact_match" in metrics

    @patch("metrics.evaluate")
    def test_compute_metrics_squad_compute_fails(
        self, mock_evaluate: Mock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """
        Test that if the SQuAD metric's compute method raises an error, it propagates.
        """
        # Mock the evaluate.load function to return a mock metric
        mock_metric = Mock()
        mock_evaluate.load.return_value = mock_metric
        mock_metric.compute.side_effect = Exception("SQuAD compute failed")

        # Mock EvalPrediction with dummy predictions and labels
        mock_eval_preds = Mock()
        mock_eval_preds.predictions = [[0, 1, 2], [2, 1, 0]]

        # Create the compute_metrics function
        compute_metrics = factory_compute_metrics_mbert(
            raw_dataset=MagicMock(),
            preprocessed_dataset=MagicMock(),
        )

        # Call the compute_metrics function and check that it raises the error
        with caplog.at_level(logging.ERROR):
            with pytest.raises(Exception, match="SQuAD compute failed"):
                compute_metrics(mock_eval_preds)

        assert mock_metric.compute.called_once()
        assert "Failed to compute metrics: SQuAD compute failed" in caplog.text


class TestFactoryComputeMetricsGemma2:
    """Unit tests for factory_compute_metrics_gemma2"""

    # ========= FACTORY FUNCTION TESTS =========
    @patch("metrics.evaluate")
    def test_function_creation(
        self, mock_evaluate: Mock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """
        Test that the factory function returns a callable with valid config
        and logs the expected info message.
        """
        # Create the compute_metrics function
        with caplog.at_level(logging.INFO):
            compute_metrics = factory_compute_metrics_gemma2(
                raw_dataset=Mock(),
                tokenizer=Mock(),
            )
        # Assertions
        assert callable(compute_metrics)
        assert "SQuAD metric loaded for Gemma 2" in caplog.text

    # ========= INNER FUNCTION TESTS =========
    @patch("metrics.evaluate")
    def test_compute_metrics_basic_case(self, mock_evaluate: Mock) -> None:
        """
        Test that the inner function computes metrics without errors.
        Mocks the evaluate library and the predictions.

        Uses a sample with two examples, with fake predictions.
        """

        # Mock the evaluate.load function to return a mock metric
        mock_metric = Mock()
        mock_evaluate.load.return_value = mock_metric
        mock_metric.compute.return_value = {"f1": 70.1, "exact_match": 55.5}

        # Mock datasets with actual structure and tokenized content
        raw_dataset = [
            {
                "id": "A1",
                "context": "A is hello extra",
                "question": "What is A?",
                "answers": {"answer_start": [5], "text": ["hello"]},
            },
            {
                "id": "B2",
                "context": "B is a world extra",
                "question": "What is B?",
                "answers": {"answer_start": [7], "text": ["world"]},
            },
        ]

        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.batch_decode.return_value = ["hello", "world"]

        # EvalPrediction with dummy predictions and labels ()
        # Without mocking the tokenizer, for the metrics code to work predictions would
        # have to be Numpy arrays or encoded_preds[labels == -100] would not work.
        # eval_preds = [np.array([[...], [...]]), np.array([[...], [...]])
        mock_eval_preds = [
            [
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    2,
                    235292,
                    25612,
                    1,
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    2,
                    137339,
                    235292,
                    235292,
                    2134,
                    1,
                ],
            ],
            [
                [
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    25612,
                    1,
                ],
                [
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    2134,
                    1,
                ],
            ],
        ]

        # Create the compute_metrics function
        compute_metrics = factory_compute_metrics_gemma2(
            raw_dataset=raw_dataset,
            tokenizer=mock_tokenizer,
        )

        # Call the compute_metrics function
        metrics = compute_metrics(mock_eval_preds)

        # Check that the output is a dictionary with expected keys
        assert isinstance(metrics, dict)
        assert mock_metric.compute.called_once()
        assert "f1" in metrics
        assert "exact_match" in metrics

    @patch("metrics.evaluate")
    def test_compute_metrics_squad_compute_fails(
        self, mock_evaluate: Mock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """
        Test that if the SQuAD metric's compute method raises an error, it propagates.
        """
        # Mock the evaluate.load function to return a mock metric
        mock_metric = Mock()
        mock_evaluate.load.return_value = mock_metric
        mock_metric.compute.side_effect = Exception("SQuAD compute failed")

        # Mock EvalPrediction with dummy predictions and labels
        mock_eval_preds = [[0, 1, 2], [0, 1, 2]]

        # Create the compute_metrics function
        compute_metrics = factory_compute_metrics_gemma2(
            raw_dataset=MagicMock(),
            tokenizer=MagicMock(),
        )

        # Call the compute_metrics function and check that it raises the error
        with caplog.at_level(logging.ERROR):
            with pytest.raises(Exception, match="SQuAD compute failed"):
                compute_metrics(mock_eval_preds)

        assert mock_metric.compute.called_once()
        assert "Failed to compute metrics: SQuAD compute failed" in caplog.text
