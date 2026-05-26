"""
Functions related to loading and managing models.
"""

# Standard imports
import logging
from typing import Dict, Any

# Third-party imports
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForQuestionAnswering,
)

# Set up logger
logger = logging.getLogger(__name__)


# Load HF causal LM and tokenizer
def load_causal_lm_model(
    cfg: Dict[str, Any], hf_token: str
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a causal language model (decoder-only) and its tokenizer.
    Supports gradient checkpointing and mixed precision training.

    Args:
        cfg: Configuration dictionary containing model settings
        hf_token: Hugging Face API token for authentication

    Returns:
        Tuple of (model, tokenizer)

    Raises:
        ValueError: If model_name_or_path not specified in config
                    or if tokenizer has no pad_token or pads on the wrong side
        Exception: If model loading fails (invalid model, network error, etc.)
    """

    model_cfg = cfg.get("model", {})
    model_name = model_cfg.get("model_name_or_path", None)
    if not model_name:
        raise ValueError("model_name_or_path must be specified in config")
    dtype_str = model_cfg.get("model_dtype", "bfloat16")
    dtype = getattr(torch, dtype_str) if hasattr(torch, dtype_str) else torch.float16

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    except Exception as e:
        logger.error(f"Failed to load tokenizer for {model_name}: {e}")
        raise
    if tokenizer.pad_token is None:
        raise ValueError(
            f"Tokenizer for {model_name} has no pad_token set. "
            f"Add it manually or use a different model."
        )
    if tokenizer.padding_side != "left":
        raise ValueError(
            f"Model {model_name} pads {tokenizer.padding_side}, "
            f"but this code expects left padding for causal LM."
        )
    try:
        use_cache = not model_cfg.get("gradient_checkpointing_enabled", False)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=model_cfg.get("device_map", "auto"),
            use_cache=use_cache,  # cache disabled if gradient checkpointing enabled
            token=hf_token,
        )
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        raise
    logger.info(f"Loaded '{model_name}' model and tokenizer successfully")

    if model_cfg.get("gradient_checkpointing_enabled", False):
        # Enable gradient checkpointing for memory efficiency.
        model.gradient_checkpointing_enable()
        logger.info(f"Enabled gradient checkpointing for {model_name}")

    return model, tokenizer


# Load HF QA model and tokenizer
def load_qa_model(
    cfg: Dict[str, Any], hf_token: str
) -> tuple[AutoModelForQuestionAnswering, AutoTokenizer]:
    """
    Load a question-answering model (encoder-only) and its tokenizer.

    Args:
        cfg: Configuration dictionary containing model settings
        hf_token: Hugging Face API token for authentication

    Returns:
        Tuple of (model, tokenizer)

    Raises:
        ValueError: If model_name_or_path not specified in config
                    or if tokenizer has no pad_token
        Exception: If model loading fails (invalid model, network error, etc.)
    """
    model_cfg = cfg.get("model", {})
    model_name = model_cfg.get("model_name_or_path", None)
    if not model_name:
        raise ValueError("model_name_or_path must be specified in config")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    except Exception as e:
        logger.error(f"Failed to load tokenizer for {model_name}: {e}")
        raise
    try:
        use_cache = not model_cfg.get("gradient_checkpointing_enabled", False)
        model = AutoModelForQuestionAnswering.from_pretrained(
            model_name,
            use_cache=use_cache,  # cache disabled if gradient checkpointing enabled
            token=hf_token,
        )
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        raise
    logger.info(f"Loaded '{model_name}' model and tokenizer successfully")

    if model_cfg.get("gradient_checkpointing_enabled", False):
        # Enable gradient checkpointing for memory efficiency.
        model.gradient_checkpointing_enable()
        logger.info(f"Enabled gradient checkpointing for {model_name}")

    return model, tokenizer
