"""
Auxiliary functions for loading credentials from environment variables.
"""

# Standard imports
import logging
import os

# Third party imports
from dotenv import load_dotenv

# Set up logger
logger = logging.getLogger(__name__)

# Load .env file once at module level
load_dotenv()


def load_wandb_credentials() -> str:
    """Load WandB API key from environment variables"""
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if not wandb_api_key:
        raise ValueError(
            "WandB API key not found in environment variables. "
            "Please set WANDB_API_KEY in your .env file."
        )
    logger.info("WandB API key loaded successfully.")

    return wandb_api_key


def load_hf_credentials() -> str:
    """Load Hugging Face access token from environment variables"""
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError(
            "Hugging Face access token not found in environment variables. "
            "Please set HF_TOKEN in your .env file."
        )
    logger.info("Hugging Face access token loaded successfully.")

    return hf_token
