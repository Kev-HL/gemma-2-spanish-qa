"""
Config loading and validation for the Gemma 2 Spanish QA project.
"""

# Standard imports
import json
import logging
from pathlib import Path
from typing import Dict, Any

# Set up logger
logger = logging.getLogger(__name__)


def validate_training_config(cfg: Dict[str, Any], cfg_type: str) -> None:
    """
    Validate critical aspects of the training configuration to catch common issues.
    Validates:
    - Basic required sections existence
    - Valid paths for data and output
    - Logical consistency (no simultaneous bf16 and fp16, LoRA exists when required...)
    - Reasonable values for critical parameters (learning rate, epochs...)

    Supported cfg_types:
    - 'gemma2' (for fine-tuning Gemma 2)
    - 'mbert' (for fine-tuning multilingual BERT).

    Valid schemas can be found in:
    - configs/schemas/train_gemma2.json
    - configs/schemas/train_mbert.json

    Args:
        cfg: The configuration dictionary to validate
        cfg_type: The type of config, determines which schema to validate against
    """
    # ======== REQUIRED SECTIONS ========
    required_sections = ["data", "model", "preprocessing", "trainer"]
    for section in required_sections:
        if section not in cfg:
            raise ValueError(f"Missing required section: {section}")

    # ======== DATA AND OUTPUT PATHS ========
    # Data paths must exist and files must be JSON
    train_path = Path(cfg["data"]["train_data_path"])
    assert train_path.exists(), f"Train data not found: {train_path}"
    if not train_path.suffix == ".json":
        raise ValueError("Train data file must be a JSON file")
    eval_path = Path(cfg["data"]["eval_data_path"])
    assert eval_path.exists(), f"Eval data not found: {eval_path}"
    if not eval_path.suffix == ".json":
        raise ValueError("Eval data file must be a JSON file")
    # Output dir must be accessible (will be created if it doesn't exist)
    output_dir = Path(cfg["trainer"]["output_dir"])
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Data paths validated: {train_path}, {eval_path}")

    # ======== LOGIC AND INTERDEPENDENCIES ========
    # Validate logic and interdependencies.
    trainer_cfg = cfg["trainer"]
    # Can't use both precision formats
    if trainer_cfg.get("bf16") and trainer_cfg.get("fp16"):
        raise ValueError("Cannot use both bf16 and fp16")
    if trainer_cfg.get("bf16_full_eval") and trainer_cfg.get("fp16_full_eval"):
        raise ValueError("Cannot use both bf16_full_eval and fp16_full_eval")
    model_dtype = cfg["model"].get("model_dtype", "")
    # Check bf16 settings
    if trainer_cfg.get("bf16_full_eval") or trainer_cfg.get("bf16"):
        if model_dtype != "bfloat16":
            logger.warning(
                f"bf16 enabled but model_dtype is '{model_dtype}', not 'bfloat16'. "
                f"Consider matching them for consistency."
            )
    # Check fp16 settings
    if trainer_cfg.get("fp16_full_eval") or trainer_cfg.get("fp16"):
        if model_dtype != "float16":
            logger.warning(
                f"fp16 enabled but model_dtype is '{model_dtype}', not 'float16'. "
                f"Consider matching them for consistency."
            )
    # load_best_model_at_end requires compatible strategies
    if trainer_cfg.get("load_best_model_at_end"):
        save_strat = trainer_cfg.get("save_strategy", "epoch")
        eval_strat = trainer_cfg.get("eval_strategy", "epoch")
        if save_strat != eval_strat and save_strat != "best":
            raise ValueError(
                f"load_best_model_at_end requires save_strategy='best' or "
                f"save_strategy=eval_strategy. Got save_strategy={save_strat}, "
                f"eval_strategy={eval_strat}"
            )
    # LoRA specific validation
    if cfg_type == "gemma2":
        assert "lora" in cfg, "LoRA config section is missing"
        lora_cfg = cfg["lora"]
        if lora_cfg.get("r") not in [8, 16, 32, 64, 128]:
            raise ValueError(
                f"LoRA rank must be in [8, 16, 32, 64, 128], got {lora_cfg.get('r')}"
            )
    # W&B experiment config validation
    if trainer_cfg.get("report_to") == "wandb":
        if "experiment" not in cfg:
            raise ValueError(
                "Experiment config required in cfg when report_to='wandb'. "
                "Add 'experiment' section with team_name, project_name, experiment_name"
            )
    logger.info("Config logic and interdependencies validated")

    # ======== REASONABLE VALUE CHECKS ========
    # Learning rate should be reasonable
    lr = trainer_cfg.get("learning_rate", 2e-5)
    if not (1e-6 < lr < 1.0):
        logger.warning(
            f"Learning rate {lr} is unusual. "
            f"Typical range: 1e-6 to 1.0. Continuing anyway."
        )
    # Batch size shouldn't be absurdly large
    batch_size = trainer_cfg.get("per_device_train_batch_size", 8)
    if batch_size > 512:
        logger.warning(f"Batch size {batch_size} is very large, may cause OOM")
    # Number of epochs seems reasonable
    epochs = trainer_cfg.get("num_train_epochs", 3)
    if epochs > 5:
        logger.warning(
            f"Number of epochs {epochs} is unusually high for the task faced in this "
            f"project (SQuAD type QA fine-tuning). Regardless of the extractive or "
            f"generative approach, 1-3 epochs as typical. Continuing anyway."
        )
    logger.info("Critical config values are in reasonable ranges")


def load_training_config(path: str, cfg_type: str) -> Dict[str, Any]:
    """
    Loads and validates training configuration.

    Supported cfg_types:
    - 'gemma2' (for fine-tuning Gemma 2)
    - 'mbert' (for fine-tuning multilingual BERT).

    Valid schemas can be found in:
    - configs/schemas/train_gemma2.json
    - configs/schemas/train_mbert.json

    Args:
        path: The path to the configuration file to validate
        cfg_type: The type of config, determines which schema to validate against
    Returns:
        A validated configuration dictionary ready for use in training
    """
    if cfg_type not in ["gemma2", "mbert"]:
        raise ValueError(f"Unsupported cfg_type: {cfg_type}. Use 'gemma2' or 'mbert'")
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if not config_path.suffix == ".json":
        raise ValueError("Config file must be a JSON file")

    try:
        with open(config_path) as f:
            cfg = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file: {e}")
    logger.info(f"Loaded config from {path}")

    validate_training_config(cfg, cfg_type)
    logger.info(f"Config validation passed for {cfg_type}")

    return cfg
