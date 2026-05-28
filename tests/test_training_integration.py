"""
Unit tests for some of the training.py functions, located in src/training.py
These are integration tests that use real tokenizers and models (external resources).
This is intended to run locally and not in CI.
"""

# Standard imports
import logging

# Third party imports
import pytest
from peft import PeftModel

# Local imports
from config import load_training_config
from models import load_causal_lm_model
from training import wrap_model_with_lora


@pytest.mark.integration
class TestWrapModelWithLora:
    """Unit tests for the wrap_model_with_lora function"""

    def test_basic_case(self, hf_token: str, caplog: pytest.LogCaptureFixture) -> None:
        """
        Test that wrap_model_with_lora returns a model wrapped with LoRA when the
        config has the necessary LoRA settings.

        Use Gemma 2 2B (target model) for the test, along with its training schema
        config, which contains LoRA settings and can be found in:
        configs/schemas/train_gemma2.json
        """
        # Load config and model
        cfg = load_training_config("configs/schemas/train_gemma2.json", "gemma2")
        model, _ = load_causal_lm_model(cfg, hf_token=hf_token)

        # Call the function under test
        with caplog.at_level(logging.INFO):
            model = wrap_model_with_lora(model, cfg)

        # Assertions
        assert isinstance(model, PeftModel)
        assert "Model wrapped with LoRA successfully" in caplog.text
        lora_cfg = model.peft_config[model.active_adapter]
        assert lora_cfg.r == cfg["lora"]["r"]
        assert lora_cfg.lora_alpha == cfg["lora"]["alpha"]
        assert lora_cfg.lora_dropout == cfg["lora"]["dropout"]
        assert lora_cfg.task_type == "CAUSAL_LM"
        assert lora_cfg.target_modules == set(cfg["lora"]["target_modules"])
        assert lora_cfg.base_model_name_or_path == cfg["model"]["model_name_or_path"]
