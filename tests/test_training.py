"""Unit tests for some of the training.py functions, located in src/training.py"""

# Standard imports
import json
import logging
from pathlib import Path
from unittest.mock import Mock, patch

# Third-party imports
import pytest
import torch
from peft import LoraConfig
from transformers import TrainingArguments
from trl import SFTConfig

# Local imports
from training import load_SFTConfig, load_TrainingArguments, wrap_model_with_lora


class TestLoadSFTConfig:
    """Unit tests for the load_SFTConfig function"""

    def test_basic_case(self, caplog: pytest.LogCaptureFixture) -> None:
        """
        Test that load_SFTConfig returns a SFTConfig object with custom values when
        a trainer config is provided.

        We use the training schema as input for a valid config
        Gemma 2 training schema can be found in configs/schemas/train_gemma2.json
        """
        # Load config (absolute path to avoid issues with CI)
        repo_root = Path(__file__).parent.parent
        cfg_path = repo_root / "configs" / "schemas" / "train_gemma2.json"
        with open(cfg_path) as f:
            cfg = json.load(f)

        # Override precision settings if CUDA not available to avoid issues with CI
        if not torch.cuda.is_available():
            cfg["trainer"]["bf16"] = False
            cfg["trainer"]["fp16"] = False
            cfg["trainer"]["bf16_full_eval"] = False
            cfg["trainer"]["fp16_full_eval"] = False

        # Call the function under test
        with caplog.at_level(logging.INFO):
            sft_config = load_SFTConfig(cfg)

        # Assertions
        assert isinstance(sft_config, SFTConfig)
        assert "SFTConfig created successfully." in caplog.text
        assert sft_config.output_dir == "./artifacts"
        assert sft_config.num_train_epochs == 2
        assert sft_config.per_device_train_batch_size == 4
        assert sft_config.gradient_accumulation_steps == 12
        assert sft_config.per_device_eval_batch_size == 1
        assert sft_config.learning_rate == 3e-5
        assert sft_config.optim == "adamw_torch_fused"
        assert sft_config.lr_scheduler_type == "linear"
        assert sft_config.warmup_steps == 96
        assert sft_config.weight_decay == 0.01
        assert sft_config.max_grad_norm == 1.0
        assert sft_config.bf16 is True
        assert sft_config.fp16 is False
        assert sft_config.bf16_full_eval is True
        assert sft_config.fp16_full_eval is False
        assert sft_config.torch_compile is True
        assert sft_config.eval_strategy == "epoch"
        assert sft_config.save_strategy == "epoch"
        assert sft_config.save_total_limit == 2
        assert sft_config.load_best_model_at_end is True
        assert sft_config.metric_for_best_model == "f1"
        assert sft_config.report_to == ["wandb"]
        assert sft_config.logging_steps == 10
        assert sft_config.logging_first_step is True

    def test_default_values(self) -> None:
        """
        Test that load_SFTConfig returns a SFTConfig object with default values when
        no trainer config is provided.
        """
        # Create empty config
        cfg = {}

        # Override precision settings if CUDA not available to avoid issues with CI
        if not torch.cuda.is_available():
            cfg["trainer"]["bf16"] = False
            cfg["trainer"]["fp16"] = False
            cfg["trainer"]["bf16_full_eval"] = False
            cfg["trainer"]["fp16_full_eval"] = False

        # Call the function under test
        sft_config = load_SFTConfig(cfg)

        # Assertions
        assert sft_config.output_dir == "./artifacts"
        assert sft_config.num_train_epochs == 1
        assert sft_config.per_device_train_batch_size == 4
        assert sft_config.gradient_accumulation_steps == 1
        assert sft_config.per_device_eval_batch_size == 4
        assert sft_config.learning_rate == 3e-5
        assert sft_config.optim == "adamw_torch_fused"
        assert sft_config.lr_scheduler_type == "linear"
        assert sft_config.warmup_steps == 0
        assert sft_config.weight_decay == 0.01
        assert sft_config.max_grad_norm == 1.0
        assert sft_config.bf16 is True
        assert sft_config.fp16 is False
        assert sft_config.bf16_full_eval is True
        assert sft_config.fp16_full_eval is False
        assert sft_config.torch_compile is True
        assert sft_config.eval_strategy == "epoch"
        assert sft_config.save_strategy == "epoch"
        assert sft_config.save_total_limit == 2
        assert sft_config.load_best_model_at_end is True
        assert sft_config.metric_for_best_model == "f1"
        assert sft_config.report_to == []  # "none" gets converted to an empty list
        assert sft_config.logging_steps == 10
        assert sft_config.logging_first_step is True


class TestLoadTrainingArguments:
    """Unit tests for the load_TrainingArguments function"""

    def test_basic_case(self, caplog: pytest.LogCaptureFixture) -> None:
        """
        Test that load_TrainingArguments returns a TrainingArguments object with
        custom values when a trainer config is provided.

        The config file is loaded using the function provided in the config.py module.

        We use the training schema as input for a valid config
        mBERT training schema can be found in configs/schemas/train_mbert.json
        """
        # Load config (absolute path to avoid issues with CI)
        repo_root = Path(__file__).parent.parent
        cfg_path = repo_root / "configs" / "schemas" / "train_mbert.json"
        with open(cfg_path) as f:
            cfg = json.load(f)

        # Override precision settings if CUDA not available to avoid issues with CI
        if not torch.cuda.is_available():
            cfg["trainer"]["bf16"] = False
            cfg["trainer"]["fp16"] = False
            cfg["trainer"]["bf16_full_eval"] = False
            cfg["trainer"]["fp16_full_eval"] = False

        # Call the function under test
        with caplog.at_level(logging.INFO):
            training_args = load_TrainingArguments(cfg)

        # Assertions
        assert isinstance(training_args, TrainingArguments)
        assert "TrainingArguments created successfully." in caplog.text
        assert training_args.output_dir == "./artifacts"
        assert training_args.num_train_epochs == 3
        assert training_args.per_device_train_batch_size == 128
        assert training_args.gradient_accumulation_steps == 1
        assert training_args.per_device_eval_batch_size == 64
        assert training_args.learning_rate == 3e-5
        assert training_args.optim == "adamw_torch_fused"
        assert training_args.lr_scheduler_type == "linear"
        assert training_args.warmup_steps == 36
        assert training_args.weight_decay == 0.01
        assert training_args.max_grad_norm == 1.0
        assert training_args.bf16 is False
        assert training_args.fp16 is False
        assert training_args.bf16_full_eval is False
        assert training_args.fp16_full_eval is False
        assert training_args.torch_compile is True
        assert training_args.eval_strategy == "epoch"
        assert training_args.save_strategy == "epoch"
        assert training_args.save_total_limit == 2
        assert training_args.load_best_model_at_end is True
        assert training_args.metric_for_best_model == "f1"
        assert training_args.report_to == []
        assert training_args.logging_steps == 10
        assert training_args.logging_first_step is True

    def test_default_values(self) -> None:
        """
        Test that load_TrainingArguments returns a TrainingArguments object with
        default values when no trainer config is provided.
        """
        # Create empty config
        cfg = {}

        # Override precision settings if CUDA not available to avoid issues with CI
        if not torch.cuda.is_available():
            cfg["trainer"]["bf16"] = False
            cfg["trainer"]["fp16"] = False
            cfg["trainer"]["bf16_full_eval"] = False
            cfg["trainer"]["fp16_full_eval"] = False

        # Call the function under test
        training_args = load_TrainingArguments(cfg)

        # Assertions
        assert training_args.output_dir == "./artifacts"
        assert training_args.num_train_epochs == 1
        assert training_args.per_device_train_batch_size == 32
        assert training_args.gradient_accumulation_steps == 1
        assert training_args.per_device_eval_batch_size == 32
        assert training_args.learning_rate == 3e-5
        assert training_args.optim == "adamw_torch_fused"
        assert training_args.lr_scheduler_type == "linear"
        assert training_args.warmup_steps == 0
        assert training_args.weight_decay == 0.01
        assert training_args.max_grad_norm == 1.0
        assert training_args.bf16 is True
        assert training_args.fp16 is False
        assert training_args.bf16_full_eval is True
        assert training_args.fp16_full_eval is False
        assert training_args.torch_compile is True
        assert training_args.eval_strategy == "epoch"
        assert training_args.save_strategy == "epoch"
        assert training_args.save_total_limit == 2
        assert training_args.load_best_model_at_end is True
        assert training_args.metric_for_best_model == "f1"
        assert training_args.report_to == []  # "none" gets converted to an empty list
        assert training_args.logging_steps == 10
        assert training_args.logging_first_step is True


@patch("training.get_peft_model")
class TestWrapModelWithLora:
    """Unit tests for the wrap_model_with_lora function"""

    def test_basic_case(
        self, mock_get_peft_model: Mock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """
        Test that wrap_model_with_lora returns a model wrapped with LoRA when the
        config has the necessary LoRA settings.
        """
        # Set up mocks for get_peft_model to return a Mock object
        mock_PreTrainedModel = Mock()
        mock_peft_model = Mock()
        mock_get_peft_model.return_value = mock_peft_model
        # Set up config with LoRA settings
        cfg = {
            "lora": {
                "r": 16,
                "alpha": 32,
                "dropout": 0.1,
                "target_modules": ["q_proj", "v_proj"],
            }
        }
        # Expected LoRA config that should be passed to get_peft_model
        lora_config = LoraConfig(
            r=cfg["lora"]["r"],
            lora_alpha=cfg["lora"]["alpha"],
            target_modules=cfg["lora"]["target_modules"],
            lora_dropout=cfg["lora"]["dropout"],
            task_type="CAUSAL_LM",
        )
        # Call the function under test
        with caplog.at_level(logging.INFO):
            model = wrap_model_with_lora(mock_PreTrainedModel, cfg)
        # Assertions
        assert model == mock_peft_model
        mock_get_peft_model.assert_called_with(mock_PreTrainedModel, lora_config)
        assert "Model wrapped with LoRA successfully" in caplog.text

    def test_default_values(self, mock_get_peft_model: Mock) -> None:
        """
        Test that wrap_model_with_lora returns a model wrapped with LoRA when the
        config has the necessary LoRA settings.
        """
        # Set up mocks for get_peft_model to return a Mock object
        mock_PreTrainedModel = Mock()
        mock_peft_model = Mock()
        mock_get_peft_model.return_value = mock_peft_model

        # Create empty config
        cfg = {}

        # Expected LoRA config that should be passed to get_peft_model
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.1,
            task_type="CAUSAL_LM",
        )
        # Call the function under test
        _ = wrap_model_with_lora(mock_PreTrainedModel, cfg)

        # Assertions
        mock_get_peft_model.assert_called_with(mock_PreTrainedModel, lora_config)

    def test_error_wrapping_model(
        self, mock_get_peft_model: Mock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """
        Test that an exception during model wrapping is properly logged and raised.
        """
        # Set up mock to raise an exception
        mock_get_peft_model.side_effect = Exception("test error")

        # Create empty config
        cfg = {}

        # Call the function under test and assert that it raises the exception
        with pytest.raises(Exception, match="test error"):
            _ = wrap_model_with_lora(Mock(), cfg)

        # Assert that the error was logged
        assert "Failed to wrap model with LoRA: test error" in caplog.text
