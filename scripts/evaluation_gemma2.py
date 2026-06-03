"""
Script for evaluating an generative QA model, with optional W&B integration.
Configurable via a JSON file that adheres to the schema in
configs/schemas/eval_gemma2.json.

Intended to be used with Gemma 2 models, could be adaptable to others.

Process:
1. Load config and credentials
2. Initialize W&B (if enabled)
3. Load model and tokenizer
4. Load and preprocess the evaluation dataset and set up data collator
5. Set up metrics and trainer
6. Run evaluation

Example usage:
    python scripts/evaluation_gemma2.py configs/eval/test_gemma2.json
"""

# Standard imports
import gc
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Third party imports
import torch
import wandb
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments
from datasets import Dataset

# Local imports
from config import load_eval_config
from credentials import load_wandb_credentials, load_hf_credentials
from metrics import factory_compute_metrics_gemma2, preprocess_logits_for_metrics
from models import load_causal_lm_model
from preprocessing import factory_preprocess_gemma2_train

# Set up logging to logs folder in repo root
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
log_dir = PROJECT_ROOT / "logs"
log_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_dir / f"evaluation_gemma2_{timestamp}.log"

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
    logger.info("Starting Gemma 2 evaluation script")
    # ============================ SETUP =======================================

    # Load and validate config
    cfg = load_eval_config(config_path)

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
        logger.warning("CUDA and/or GPU not available, evaluation will run on CPU")

    # ============================ MODEL LOADING ================================
    # Load HF access token
    hf_token = load_hf_credentials()

    logger.info(f"Loading Gemma 2 from directory: {cfg['model']['model_name_or_path']}")
    model, tokenizer = load_causal_lm_model(cfg, hf_token)

    # ================== DATA LOADING AND PREPARATION ==========================
    # Load dataset
    try:
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
    preprocess_fn = factory_preprocess_gemma2_train(
        tokenizer, preprocessing_max_seq_length
    )
    logger.info("Preprocessing eval dataset...")
    tokenized_eval_dataset = eval_dataset.map(
        preprocess_fn,
        batched=preprocessing_batched,
        remove_columns=eval_dataset.column_names,
    )

    # Set up data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )
    # Preprocessing already handles padding and label masking. These collator settings
    # are safe defaults that could become active if preprocessing settings are modified.

    # ============================ EVALUATION SETUP ================================
    # Set up metrics
    compute_metrics = factory_compute_metrics_gemma2(eval_dataset, tokenizer)

    # Set up Trainer for evaluation
    model_dtype = cfg["model"].get("model_dtype", "")
    if "bfloat16" in model_dtype:
        bf16_full_eval = True
        fp16_full_eval = False
    elif "float16" in model_dtype:
        bf16_full_eval = False
        fp16_full_eval = True
    else:
        bf16_full_eval = False
        fp16_full_eval = False
    logger.info(f"bf16_full_eval={bf16_full_eval}, fp16_full_eval={fp16_full_eval}")
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=cfg["trainer"]["output_dir"],
            per_device_eval_batch_size=cfg["trainer"]["per_device_eval_batch_size"],
            bf16_full_eval=bf16_full_eval,
            fp16_full_eval=fp16_full_eval,
            report_to=cfg["trainer"]["report_to"],
        ),
        eval_dataset=tokenized_eval_dataset,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        data_collator=data_collator,
    )

    # ============================ RUN EVALUATION ================================
    logger.info("Starting evaluation...")
    try:
        # Evaluate the model
        torch.cuda.empty_cache()
        results = trainer.evaluate()
        logger.info("Evaluation completed successfully.")
        logger.info(f"Dataset: {cfg['data']['eval_data_path']}")
        logger.info(f"Results: {results}")

        # Set path for saving results
        results_dir = Path(cfg["trainer"]["output_dir"])
        results_dir.mkdir(parents=True, exist_ok=True)

        # Timestamp filename
        results_file = results_dir / f"eval_results_{timestamp}.json"

        # Save
        results["dataset"] = cfg["data"]["eval_data_path"]
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {results_file}")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
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
            if "eval_dataset" in locals():
                del eval_dataset
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
