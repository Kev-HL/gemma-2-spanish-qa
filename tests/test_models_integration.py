"""
Unit tests for some of the models.py functions, located in src/models.py
These are integration tests that use real tokenizers and models (external resources).
This is intended to run locally and not in CI.
"""

# Standard imports
import logging

# Third party imports
import pytest
import torch
from transformers import PreTrainedTokenizerBase, PreTrainedModel

# Local imports
from models import load_causal_lm_model, load_qa_model


@pytest.mark.integration
class TestLoadCausalLMModel:
    def test_load_gemma2_2b_success(
        self, hf_token: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        """
        Test that the Gemma 2 2B model loads succesfully with a standard config.
        """
        # Define a sample config
        gemma2_2b_name = "google/gemma-2-2b"
        cfg = {
            "model": {
                "model_name_or_path": gemma2_2b_name,
                "model_dtype": "bfloat16",
                "gradient_checkpointing_enabled": True,
            }
        }

        # Call the function
        with caplog.at_level(logging.INFO):
            model, tokenizer = load_causal_lm_model(cfg, hf_token)

        # Assertions
        assert tokenizer.pad_token is not None
        assert tokenizer.padding_side == "left"
        assert isinstance(tokenizer, PreTrainedTokenizerBase)
        assert isinstance(model, PreTrainedModel)
        assert model.dtype == torch.bfloat16
        assert f"'{gemma2_2b_name}' model and tokenizer successfully" in caplog.text
        assert f"Enabled gradient checkpointing for {gemma2_2b_name}" in caplog.text


@pytest.mark.integration
class TestLoadQAModel:
    def test_load_mbert_success(
        self, hf_token: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        """
        Test that the mBERT model loads succesfully with a standard config.
        """
        # Define a sample config
        mbert_name = "bert-base-multilingual-cased"
        cfg = {
            "model": {
                "model_name_or_path": mbert_name,
                "gradient_checkpointing_enabled": True,
            }
        }

        # Call the function
        with caplog.at_level(logging.INFO):
            model, tokenizer = load_qa_model(cfg, hf_token)

        # Assertions
        assert isinstance(tokenizer, PreTrainedTokenizerBase)
        assert isinstance(model, PreTrainedModel)
        assert f"'{mbert_name}' model and tokenizer successfully" in caplog.text
        assert f"Enabled gradient checkpointing for {mbert_name}" in caplog.text
