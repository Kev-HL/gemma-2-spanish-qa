"""Unit tests for some of the models.py functions, located in src/models.py"""

# Standard imports
import logging
from unittest.mock import Mock, patch

# Third party imports
import torch
import pytest

# Local imports
from models import load_causal_lm_model, load_qa_model


class TestLoadCausalLMModel:
    # ===== SUCCESS CASES =====
    @patch("models.AutoTokenizer.from_pretrained")
    @patch("models.AutoModelForCausalLM.from_pretrained")
    def test_load_success(
        self,
        mock_model_from_pretrained: Mock,
        mock_tokenizer_from_pretrained: Mock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """
        Test that the model and tokenizer are loaded successfully with correct
        parameters.
        """
        # Mock the tokenizer and model
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "<PAD>"
        mock_tokenizer.padding_side = "left"
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer

        mock_model = Mock()
        mock_model_from_pretrained.return_value = mock_model

        # Define a sample config
        cfg = {
            "model": {
                "model_name_or_path": "MODEL_NAME",
                "model_dtype": "bfloat16",
            }
        }
        hf_token = "fake_hf_token"

        # Call the function
        with caplog.at_level(logging.INFO):
            model, tokenizer = load_causal_lm_model(cfg, hf_token)

        # Assertions
        mock_tokenizer_from_pretrained.assert_called_once_with(
            "MODEL_NAME", token=hf_token
        )
        mock_model_from_pretrained.assert_called_once_with(
            "MODEL_NAME",
            dtype=torch.bfloat16,
            device_map="auto",  # default value in function
            use_cache=True,  # default value in function
            token=hf_token,
        )
        # Assert that gradient checkpointing is not enabled by default
        mock_model_from_pretrained.gradient_checkpointing_enable.assert_not_called()
        assert model == mock_model
        assert tokenizer == mock_tokenizer
        assert "Loaded 'MODEL_NAME' model and tokenizer successfully" in caplog.text

    @patch("models.AutoTokenizer.from_pretrained")
    @patch("models.AutoModelForCausalLM.from_pretrained")
    def test_gradient_checkpointing(
        self,
        mock_model_from_pretrained: Mock,
        mock_tokenizer_from_pretrained: Mock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """
        Test that gradient checkpointing is enabled when specified in the config.
        """
        # Mock the tokenizer and model
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "<PAD>"
        mock_tokenizer.padding_side = "left"
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer

        mock_model = Mock()
        mock_model_from_pretrained.return_value = mock_model

        # Define a sample config
        cfg = {
            "model": {
                "model_name_or_path": "MODEL_NAME",
                "model_dtype": "bfloat16",
                "gradient_checkpointing_enabled": True,
            }
        }
        hf_token = "fake_hf_token"

        # Call the function
        with caplog.at_level(logging.INFO):
            model, tokenizer = load_causal_lm_model(cfg, hf_token)

        # Assertions
        mock_model_from_pretrained.assert_called_once_with(
            "MODEL_NAME",
            dtype=torch.bfloat16,
            device_map="auto",  # default value in function
            use_cache=False,  # forced if gradient checkpointing enabled
            token=hf_token,
        )
        mock_model.gradient_checkpointing_enable.assert_called_once()
        assert "Enabled gradient checkpointing for MODEL_NAME" in caplog.text

    # ===== CONFIG VALIDATION =====
    def test_no_model_name(self) -> None:
        """Test that value error is raised if model_name_or_path is not in config"""
        # Define minimal config without model_name_or_path
        cfg = {
            "model": {
                "model_name_or_path": None,
            }
        }
        hf_token = "fake_hf_token"

        # Call the function and assert that it raises a ValueError
        with pytest.raises(ValueError):
            _, _ = load_causal_lm_model(cfg, hf_token)

    def test_missing_model_dtype(self) -> None:
        """Test that value error is raised if model_dtype is not in config"""
        # Define minimal config without model_dtype
        cfg = {
            "model": {
                "model_name_or_path": "MODEL_NAME",
            }
        }
        hf_token = "fake_hf_token"

        # Call the function and assert that it raises a ValueError
        with pytest.raises(ValueError):
            _, _ = load_causal_lm_model(cfg, hf_token)

    def test_invalid_model_dtype(self) -> None:
        """Test that value error is raised if model_dtype is invalid"""
        # Define minimal config with invalid model_dtype
        cfg = {
            "model": {
                "model_name_or_path": "MODEL_NAME",
                "model_dtype": "invalid_dtype",
            }
        }
        hf_token = "fake_hf_token"

        # Call the function and assert that it raises a ValueError
        with pytest.raises(ValueError):
            _, _ = load_causal_lm_model(cfg, hf_token)

    # ===== TOKENIZER VALIDATION =====
    @patch("models.AutoTokenizer.from_pretrained")
    def test_no_pad_token_defined(self, mock_tokenizer_from_pretrained: Mock) -> None:
        """
        Test that error is raised if the tokenizer has no pad_token defined.
        """
        # Mock the tokenizer and model
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer

        # Define a sample config
        cfg = {
            "model": {
                "model_name_or_path": "MODEL_NAME",
                "model_dtype": "bfloat16",
            }
        }
        hf_token = "fake_hf_token"

        # Call the function
        with pytest.raises(ValueError):
            _, _ = load_causal_lm_model(cfg, hf_token)

    @patch("models.AutoTokenizer.from_pretrained")
    def test_wrong_padding_side(self, mock_tokenizer_from_pretrained: Mock) -> None:
        """
        Test that error is raised if the tokenizer has the wrong padding side.
        """
        # Mock the tokenizer and model
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "<PAD>"
        mock_tokenizer.padding_side = "right"
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer

        # Define a sample config
        cfg = {
            "model": {
                "model_name_or_path": "MODEL_NAME",
                "model_dtype": "bfloat16",
            }
        }
        hf_token = "fake_hf_token"

        # Call the function
        with pytest.raises(ValueError):
            _, _ = load_causal_lm_model(cfg, hf_token)

    # ===== LOADING ERRORS =====
    @patch("models.AutoTokenizer.from_pretrained")
    def test_error_loading_tokenizer(
        self, mock_tokenizer_from_pretrained: Mock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """
        Test that an exception during tokenizer loading is properly logged and raised.
        """
        # Mock the tokenizer and model
        mock_tokenizer_from_pretrained.side_effect = Exception("test error")

        # Define a sample config
        cfg = {
            "model": {
                "model_name_or_path": "MODEL_NAME",
                "model_dtype": "bfloat16",
            }
        }
        hf_token = "fake_hf_token"

        # Call the function
        with caplog.at_level(logging.ERROR):
            with pytest.raises(Exception):
                _, _ = load_causal_lm_model(cfg, hf_token)
        assert "Failed to load tokenizer for MODEL_NAME: test error" in caplog.text

    @patch("models.AutoTokenizer.from_pretrained")
    @patch("models.AutoModelForCausalLM.from_pretrained")
    def test_error_loading_model(
        self,
        mock_model_from_pretrained: Mock,
        mock_tokenizer_from_pretrained: Mock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """
        Test that an exception during model loading is properly logged and raised.
        """
        # Mock the tokenizer and model
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "<PAD>"
        mock_tokenizer.padding_side = "left"
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer

        mock_model_from_pretrained.side_effect = Exception("test error")

        # Define a sample config
        cfg = {
            "model": {
                "model_name_or_path": "MODEL_NAME",
                "model_dtype": "bfloat16",
            }
        }
        hf_token = "fake_hf_token"

        # Call the function
        with caplog.at_level(logging.ERROR):
            with pytest.raises(Exception):
                _, _ = load_causal_lm_model(cfg, hf_token)
        assert "Failed to load model MODEL_NAME: test error" in caplog.text


class TestLoadQAModel:
    # ===== SUCCESS CASES =====
    @patch("models.AutoTokenizer.from_pretrained")
    @patch("models.AutoModelForQuestionAnswering.from_pretrained")
    def test_load_success(
        self,
        mock_model_from_pretrained: Mock,
        mock_tokenizer_from_pretrained: Mock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """
        Test that the model and tokenizer are loaded successfully with correct
        parameters.
        """
        # Mock the tokenizer and model
        mock_tokenizer = Mock()
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer

        mock_model = Mock()
        mock_model_from_pretrained.return_value = mock_model

        # Define a sample config
        cfg = {
            "model": {
                "model_name_or_path": "MODEL_NAME",
            }
        }
        hf_token = "fake_hf_token"

        # Call the function
        with caplog.at_level(logging.INFO):
            model, tokenizer = load_qa_model(cfg, hf_token)

        # Assertions
        mock_tokenizer_from_pretrained.assert_called_once_with(
            "MODEL_NAME", token=hf_token
        )
        mock_model_from_pretrained.assert_called_once_with(
            "MODEL_NAME",
            use_cache=True,  # default value in function
            token=hf_token,
        )
        # Assert that gradient checkpointing is not enabled by default
        mock_model_from_pretrained.gradient_checkpointing_enable.assert_not_called()
        assert model == mock_model
        assert tokenizer == mock_tokenizer
        assert "Loaded 'MODEL_NAME' model and tokenizer successfully" in caplog.text

    @patch("models.AutoTokenizer.from_pretrained")
    @patch("models.AutoModelForQuestionAnswering.from_pretrained")
    def test_gradient_checkpointing(
        self,
        mock_model_from_pretrained: Mock,
        mock_tokenizer_from_pretrained: Mock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """
        Test that gradient checkpointing is enabled when specified in the config.
        """
        # Mock the tokenizer and model
        mock_tokenizer = Mock()
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer

        mock_model = Mock()
        mock_model_from_pretrained.return_value = mock_model

        # Define a sample config
        cfg = {
            "model": {
                "model_name_or_path": "MODEL_NAME",
                "gradient_checkpointing_enabled": True,
            }
        }
        hf_token = "fake_hf_token"

        # Call the function
        with caplog.at_level(logging.INFO):
            model, tokenizer = load_qa_model(cfg, hf_token)

        # Assertions
        mock_model_from_pretrained.assert_called_once_with(
            "MODEL_NAME",
            use_cache=False,  # forced if gradient checkpointing enabled
            token=hf_token,
        )
        mock_model.gradient_checkpointing_enable.assert_called_once()
        assert "Enabled gradient checkpointing for MODEL_NAME" in caplog.text

    # ===== CONFIG VALIDATION =====
    def test_no_model_name(self) -> None:
        """Test that value error is raised if model_name_or_path is not in config"""
        # Define minimal config without model_name_or_path
        cfg = {
            "model": {
                "model_name_or_path": None,
            }
        }
        hf_token = "fake_hf_token"

        # Call the function and assert that it raises a ValueError
        with pytest.raises(ValueError):
            _, _ = load_qa_model(cfg, hf_token)

    # ===== LOADING ERRORS =====
    @patch("models.AutoTokenizer.from_pretrained")
    def test_error_loading_tokenizer(
        self, mock_tokenizer_from_pretrained: Mock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """
        Test that an exception during tokenizer loading is properly logged and raised.
        """
        # Mock the tokenizer and model
        mock_tokenizer_from_pretrained.side_effect = Exception("test error")

        # Define a sample config
        cfg = {
            "model": {
                "model_name_or_path": "MODEL_NAME",
            }
        }
        hf_token = "fake_hf_token"

        # Call the function
        with caplog.at_level(logging.ERROR):
            with pytest.raises(Exception):
                _, _ = load_qa_model(cfg, hf_token)
        assert "Failed to load tokenizer for MODEL_NAME: test error" in caplog.text

    @patch("models.AutoTokenizer.from_pretrained")
    @patch("models.AutoModelForQuestionAnswering.from_pretrained")
    def test_error_loading_model(
        self,
        mock_model_from_pretrained: Mock,
        mock_tokenizer_from_pretrained: Mock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """
        Test that an exception during model loading is properly logged and raised.
        """
        # Mock the tokenizer and model
        mock_tokenizer = Mock()
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer

        mock_model_from_pretrained.side_effect = Exception("test error")

        # Define a sample config
        cfg = {
            "model": {
                "model_name_or_path": "MODEL_NAME",
            }
        }
        hf_token = "fake_hf_token"

        # Call the function
        with caplog.at_level(logging.ERROR):
            with pytest.raises(Exception):
                _, _ = load_qa_model(cfg, hf_token)
        assert "Failed to load model MODEL_NAME: test error" in caplog.text
