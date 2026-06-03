"""
Training utilities for the Gemma 2 Spanish QA project.
"""

# Standard imports
import logging

# Third party imports
from transformers import TrainingArguments, PreTrainedModel
from trl import SFTConfig
from peft import LoraConfig, get_peft_model, PeftModel

# Set up logger
logger = logging.getLogger(__name__)


def load_SFTConfig(cfg: dict) -> SFTConfig:
    """
    Set up SFTConfig for supervised fine-tuning with TRL.

    Args:
    - cfg: Configuration dictionary containing trainer settings

    Returns:
    - SFTConfig object with specified training parameters for use with SFTTrainer
    """
    trainer_cfg = cfg.get("trainer", {})
    kwargs = {}
    if trainer_cfg.get("eval_strategy") == "steps":
        kwargs["eval_steps"] = trainer_cfg.get("eval_steps", 100)
    training_args = SFTConfig(
        # Output dir
        output_dir=trainer_cfg.get("output_dir", "./artifacts"),
        # Training hyperparameters
        num_train_epochs=trainer_cfg.get("num_train_epochs", 1),
        per_device_train_batch_size=trainer_cfg.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=trainer_cfg.get("gradient_accumulation_steps", 1),
        per_device_eval_batch_size=trainer_cfg.get("per_device_eval_batch_size", 4),
        learning_rate=trainer_cfg.get("learning_rate", 3e-5),
        optim=trainer_cfg.get("optim", "adamw_torch_fused"),
        lr_scheduler_type=trainer_cfg.get("lr_scheduler_type", "cosine"),
        warmup_steps=trainer_cfg.get("warmup_steps", 0),
        weight_decay=trainer_cfg.get("weight_decay", 0.01),
        max_grad_norm=trainer_cfg.get("max_grad_norm", 1.0),
        # Mixed precision settings
        bf16=trainer_cfg.get("bf16", True),
        fp16=trainer_cfg.get("fp16", False),
        bf16_full_eval=trainer_cfg.get("bf16_full_eval", True),
        fp16_full_eval=trainer_cfg.get("fp16_full_eval", False),
        # Compile settings
        torch_compile=trainer_cfg.get("torch_compile", True),
        # Evaluation and checkpointing
        eval_strategy=trainer_cfg.get("eval_strategy", "epoch"),
        save_strategy=trainer_cfg.get("save_strategy", "epoch"),
        save_total_limit=trainer_cfg.get("save_total_limit", 2),
        load_best_model_at_end=trainer_cfg.get("load_best_model_at_end", True),
        metric_for_best_model=trainer_cfg.get("metric_for_best_model", "f1"),
        # W&B Integration
        report_to=trainer_cfg.get("report_to", "none"),
        logging_steps=trainer_cfg.get("logging_steps", 10),
        logging_first_step=trainer_cfg.get("logging_first_step", True),
        # SFT-specific settings
        dataset_kwargs={"skip_prepare_dataset": True},  # tokenization handled manually
        # Conditional kwargs for eval strategy
        **kwargs
    )
    logger.info("SFTConfig created successfully.")
    return training_args


def load_TrainingArguments(cfg: dict) -> TrainingArguments:
    """
    Set up standard TrainingArguments for use with Hugging Face Trainer.

    Args:
    - cfg: Configuration dictionary containing trainer settings

    Returns:
    - TrainingArguments object with specified training parameters for use with Trainer
    """
    trainer_cfg = cfg.get("trainer", {})
    kwargs = {}
    if trainer_cfg.get("eval_strategy") == "steps":
        kwargs["eval_steps"] = trainer_cfg.get("eval_steps", 100)
    training_args = TrainingArguments(
        # Output dir
        output_dir=trainer_cfg.get("output_dir", "./artifacts"),
        # Training hyperparameters
        num_train_epochs=trainer_cfg.get("num_train_epochs", 1),
        per_device_train_batch_size=trainer_cfg.get("per_device_train_batch_size", 32),
        gradient_accumulation_steps=trainer_cfg.get("gradient_accumulation_steps", 1),
        per_device_eval_batch_size=trainer_cfg.get("per_device_eval_batch_size", 32),
        learning_rate=trainer_cfg.get("learning_rate", 3e-5),
        optim=trainer_cfg.get("optim", "adamw_torch_fused"),
        lr_scheduler_type=trainer_cfg.get("lr_scheduler_type", "cosine"),
        warmup_steps=trainer_cfg.get("warmup_steps", 0),
        weight_decay=trainer_cfg.get("weight_decay", 0.01),
        max_grad_norm=trainer_cfg.get("max_grad_norm", 1.0),
        # Mixed precision settings
        bf16=trainer_cfg.get("bf16", True),
        fp16=trainer_cfg.get("fp16", False),
        bf16_full_eval=trainer_cfg.get("bf16_full_eval", True),
        fp16_full_eval=trainer_cfg.get("fp16_full_eval", False),
        # Compile settings
        torch_compile=trainer_cfg.get("torch_compile", True),
        # Evaluation and checkpointing
        eval_strategy=trainer_cfg.get("eval_strategy", "epoch"),
        save_strategy=trainer_cfg.get("save_strategy", "epoch"),
        save_total_limit=trainer_cfg.get("save_total_limit", 2),
        load_best_model_at_end=trainer_cfg.get("load_best_model_at_end", True),
        metric_for_best_model=trainer_cfg.get("metric_for_best_model", "f1"),
        # W&B Integration
        report_to=trainer_cfg.get("report_to", "none"),
        logging_steps=trainer_cfg.get("logging_steps", 10),
        logging_first_step=trainer_cfg.get("logging_first_step", True),
        # Conditional kwargs for eval strategy
        **kwargs
    )
    logger.info("TrainingArguments created successfully.")
    return training_args


def wrap_model_with_lora(model: PreTrainedModel, cfg: dict) -> PeftModel:
    """
    Set up LoRA configuration and wrap model using PEFT library.

    Args:
    - model: Model to wrap with LoRA
    - cfg: Configuration dictionary containing LoRA settings

    Returns:
    - Model wrapped with LoRA
    """
    lora_cfg = cfg.get("lora", {})
    lora_config = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("alpha", 32),
        target_modules=lora_cfg.get(
            "target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]
        ),
        lora_dropout=lora_cfg.get("dropout", 0.1),
        task_type="CAUSAL_LM",
        use_rslora=lora_cfg.get("use_rslora", False),
    )
    try:
        model = get_peft_model(model, lora_config)
    except Exception as e:
        logger.error(f"Failed to wrap model with LoRA: {e}")
        raise
    logger.info("Model wrapped with LoRA successfully.")
    return model
