"""Unit tests for some of the callbacks.py classes, located in src/callbacks.py"""

# Standard imports
import logging
from unittest.mock import Mock, patch

# Local imports
from callbacks import ClearCacheCallback, LogMetricsCallback


@patch("callbacks.torch")
class TestClearCacheCallback:
    """Unit tests for ClearCacheCallback"""

    def test_on_evaluate_with_cuda_available(self, mock_torch, caplog):
        """
        Test that on_evaluate with CUDA available calls torch.cuda.empty_cache and
        torch.cuda.reset_peak_memory_stats.
        """
        # Mock torch.cuda.is_available to return True
        mock_torch.cuda.is_available.return_value = True

        # Create instance of ClearCacheCallback and call on_evaluate
        callback = ClearCacheCallback()
        with caplog.at_level(logging.INFO, logger="callbacks"):
            callback.on_evaluate(args=Mock(), state=Mock(), control=Mock())

        # Assert actions
        mock_torch.cuda.is_available.assert_called_once()
        mock_torch.cuda.empty_cache.assert_called_once()
        mock_torch.cuda.reset_peak_memory_stats.assert_called_once()
        assert "GPU cache cleared and peak memory stats reset" in caplog.text

    def test_on_evaluate_with_cuda_not_available(self, mock_torch, caplog):
        """
        Test that on_evaluate with CUDA not available does not call
        torch.cuda.empty_cache or torch.cuda.reset_peak_memory_stats.
        """
        # Mock torch.cuda.is_available to return False
        mock_torch.cuda.is_available.return_value = False

        # Create instance of ClearCacheCallback and call on_evaluate
        callback = ClearCacheCallback()
        with caplog.at_level(logging.INFO, logger="callbacks"):
            callback.on_evaluate(args=Mock(), state=Mock(), control=Mock())

        # Assert actions
        mock_torch.cuda.is_available.assert_called_once()
        mock_torch.cuda.empty_cache.assert_not_called()
        mock_torch.cuda.reset_peak_memory_stats.assert_not_called()
        assert "GPU cache cleared and peak memory stats reset" not in caplog.text
        assert "CUDA not available, skipping GPU cache clearing" in caplog.text


@patch("callbacks.wandb")
class TestLogMetricsCallback:
    """Unit tests for LogMetricsCallback"""

    def test_on_evaluate_with_wandb(self, mock_wandb, caplog):
        """
        Test that on_evaluate logs custom metrics to W&B, if W&B is initialized.
        """
        # Mock wandb.run to simulate W&B being initialized
        mock_wandb.run.return_value = True  # not None

        # Create instance of LogMetricsCallback
        callback = LogMetricsCallback()

        # Mock evaluation metrics
        mock_metrics = {"eval_f1": 50.51, "eval_em": 50.52}

        # Mock state.epoch
        mock_state = Mock()
        mock_state.epoch = 2

        # Mock expected metrics to be logged by W&B
        called_with_metrics = {
            "eval_f1": 50.51,
            "eval_em": 50.52,
            "epoch": 2,
        }

        # Call on_evaluate with mocked metrics
        with caplog.at_level(logging.DEBUG, logger="callbacks"):
            callback.on_evaluate(
                args=Mock(),
                state=mock_state,
                control=Mock(),
                metrics=mock_metrics,
            )

        # Assert actions
        mock_wandb.log.assert_called_once_with(called_with_metrics)
        assert "Logged metrics to W&B at epoch 2" in caplog.text

    def test_on_evaluate_without_wandb(self, mock_wandb, caplog):
        """
        Test that on_evaluate does not log metrics to W&B if W&B is not initialized.
        """
        # Mock wandb.run to simulate W&B not being initialized
        mock_wandb.run = None

        # Create instance of LogMetricsCallback
        callback = LogMetricsCallback()

        # Call on_evaluate with mocked metrics
        with caplog.at_level(logging.WARNING, logger="callbacks"):
            callback.on_evaluate(
                args=Mock(),
                state=Mock(),
                control=Mock(),
                metrics=Mock(),
            )

        # Assert actions
        assert "skipping custom metrics logging" in caplog.text
        mock_wandb.log.assert_not_called()
