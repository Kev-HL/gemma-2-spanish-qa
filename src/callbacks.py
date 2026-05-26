"""
Callbacks for Hugging Face Trainer to be used during training and evaluation.
"""

# Standard imports
import logging

# Third party imports
import torch
import wandb
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

# Set up logger
logger = logging.getLogger(__name__)


# After epoch cleanup callback
class ClearCacheCallback(TrainerCallback):
    """Callback to clear GPU cache after evaluation to free up memory."""

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        """Called after evaluation ends.

        Args:
            args: Training arguments containing configuration
            state: Current state of training (epoch, step, etc.)
            control: Control object to modify training flow
            **kwargs: Additional arguments passed by Trainer
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            logger.info(
                "GPU cache cleared and peak memory stats reset after evaluation."
            )
        else:
            logger.info("CUDA not available, skipping GPU cache clearing.")


# Logging custom metrics to W&B callback
class LogMetricsCallback(TrainerCallback):
    """
    Log custom evaluation metrics to Weights & Biases.
    W&B by default logs only built-in metrics.

    Note: Requires W&B to be initialized before training starts.
    """

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, float],
        **kwargs,
    ) -> None:
        """Called after evaluation ends.

        Args:
            args: Training arguments containing configuration
            state: Current state of training (epoch, step, etc.)
            control: Control object to modify training flow
            metrics: Dictionary of computed metrics from evaluation
            **kwargs: Additional arguments passed by Trainer
        """
        # Metrics dict contains eval_loss, eval_f1, eval_em, etc.
        if wandb.run is None:
            logger.warning("W&B not initialized, skipping custom metrics logging.")
            return
        wandb.log(
            {
                "eval_f1": metrics.get("eval_f1"),
                "eval_em": metrics.get("eval_em"),
                "epoch": state.epoch,
            }
        )
        logger.debug(f"Logged metrics to W&B at epoch {state.epoch}")
