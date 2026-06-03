"""
Script for training an extractive QA model, with optional W&B integration.
Configurable via a JSON file that adheres to the schema in
configs/schemas/train_mbert.json.

Intended to be used with mBERT models, could be adaptable to others.

Process:
1. Load config and credentials
2. Initialize W&B (if enabled)
3. Load model and tokenizer
4. Load and preprocess datasets and set up data collator
5. Set up metrics and trainer
6. Run training

Example usage:
    python scripts/training_mbert.py configs/training/test_mbert.json
"""

# Standard imports
import gc
import logging
import sys
from datetime import datetime
from pathlib import Path

# Third party imports
import torch
import wandb
from transformers import DefaultDataCollator, Trainer, TrainerCallback
from datasets import Dataset

# Local imports
from callbacks import ClearCacheCallback, LogMetricsCallback
from config import load_training_config
from credentials import load_wandb_credentials, load_hf_credentials
from metrics import factory_compute_metrics_mbert
from models import load_qa_model
from preprocessing import factory_preprocess_mbert
from training import load_TrainingArguments

# Set up logging to logs folder in repo root
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
log_dir = PROJECT_ROOT / "logs"
log_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_dir / f"training_mbert_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ],
)

logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
logger.info(f"Logging to: {log_file}")


def main(config_path: str) -> None:
    logger.info("Starting mBERT training script")
    # ============================ SETUP =======================================
    # Load and validate config
    cfg = load_training_config(config_path, cfg_type="mbert")

    if cfg["trainer"]["report_to"] == "wandb":
        # Load WandB API key and login
        wandb_api_key = load_wandb_credentials()
        wandb.login(key=wandb_api_key)

        # Initialize W&B
        experiment_cfg = cfg.get("experiment", {})
        wandb.init(
            entity=experiment_cfg.get("team_name", "test"),
            project=experiment_cfg.get("project_name", "test"),
            name=experiment_cfg.get("experiment_name", "test"),
            tags=experiment_cfg.get("tags", []),
            config=cfg,  # Log config to W&B
        )
        logger.info("W&B initialized")

    # Check CUDA availability and log GPU info
    try:
        num_gpus = torch.cuda.device_count()
    except Exception as e:
        logger.debug(f"Error while checking CUDA availability: {e}")
        num_gpus = 0
    if torch.cuda.is_available() and num_gpus > 0:
        logger.info(f"CUDA available with {num_gpus} GPU(s) detected")
        for i in range(num_gpus):
            logger.debug(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.warning("CUDA and/or GPU not available, training will run on CPU")

    # =================== MODEL LOADING & PREPARATION ==========================
    # Load HF access token
    hf_token = load_hf_credentials()

    # Load model and tokenizer
    model, tokenizer = load_qa_model(cfg, hf_token)

    # =================== DATA LOADING & PREPARATION ===========================
    # Load datasets
    try:
        train_dataset = Dataset.from_json(cfg["data"]["train_data_path"])
        logger.info(f"Loaded train dataset: {len(train_dataset)} examples")

        eval_dataset = Dataset.from_json(cfg["data"]["eval_data_path"])
        logger.info(f"Loaded eval dataset: {len(eval_dataset)} examples")
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        raise

    # Preprocess datasets
    preprocessing_cfg = cfg.get("preprocessing", {})
    preprocessing_max_seq_length = preprocessing_cfg.get("max_seq_length", 512)
    preprocessing_batched = preprocessing_cfg.get("batched", True)
    logger.info(
        f"Preprocessing config: max_seq_length={preprocessing_max_seq_length}, "
        f"batched={preprocessing_batched}"
    )
    preprocess_fn = factory_preprocess_mbert(tokenizer, preprocessing_max_seq_length)
    logger.info("Preprocessing train dataset...")
    tokenized_train_dataset = train_dataset.map(
        preprocess_fn,
        batched=preprocessing_batched,
        remove_columns=train_dataset.column_names,
    )
    logger.info("Preprocessing eval dataset...")
    tokenized_eval_dataset = eval_dataset.map(
        preprocess_fn,
        batched=preprocessing_batched,
        remove_columns=eval_dataset.column_names,
    )

    # Set up data collator
    data_collator = DefaultDataCollator()

    # =================== METRICS & TRAINING SETUP ===========================
    # Set up metrics
    compute_metrics = factory_compute_metrics_mbert(
        eval_dataset, tokenized_eval_dataset
    )

    # Create training args
    training_args = load_TrainingArguments(cfg)

    # Set up callbacks
    callbacks_list: list[TrainerCallback] = [ClearCacheCallback()]
    if training_args.report_to == "wandb":
        logger.info("W&B reporting enabled, adding custom callback for metrics logging")
        callbacks_list.append(LogMetricsCallback())

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks_list,
    )
    logger.info("Trainer initialized")

    # =================== TRAINING ===========================
    logger.info("Starting training...")
    try:
        # Train the model
        torch.cuda.empty_cache()
        trainer.train()
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        # Finish W&B run
        if cfg["trainer"]["report_to"] == "wandb":
            try:
                wandb.finish()
                logger.info("W&B run finished")
            except Exception as e:
                # Catches if wandb.finish() fails for any reason
                logger.debug(f"W&B finish note: {e}")
                # Don't log as error, because it might be intentional

        # =================== CLEANUP ===========================
        logger.info("Cleaning up resources...")
        try:
            # Clean up trainer
            if "trainer" in locals():
                del trainer

            # Clean up model
            if "model" in locals():
                del model

            # Clean up datasets
            if "train_dataset" in locals():
                del train_dataset
            if "eval_dataset" in locals():
                del eval_dataset
            if "tokenized_train_dataset" in locals():
                del tokenized_train_dataset
            if "tokenized_eval_dataset" in locals():
                del tokenized_eval_dataset
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            logger.info("CUDA memory cleared")

        # Call garbage collector
        gc.collect()
        logger.info("Cleanup completed")


if __name__ == "__main__":
    main(sys.argv[1])
