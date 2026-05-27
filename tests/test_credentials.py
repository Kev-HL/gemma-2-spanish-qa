"""Unit tests for some of the credentials.py functions, located in src/credentials.py"""

# Standard imports
import logging
import os
from unittest.mock import patch

# Third-party imports
import pytest

# Local imports
from credentials import load_hf_credentials, load_wandb_credentials


class TestLoadWandbCredentials:
    """Unit tests for load_wandb_credentials"""

    def test_credentials_available(self, caplog):
        """Test that load_wandb_credentials successfully loads the API key."""
        with caplog.at_level(logging.INFO):
            with patch.dict(os.environ, {"WANDB_API_KEY": "test-token"}):
                token = load_wandb_credentials()
        assert token == "test-token"
        assert "WandB API key loaded successfully." in caplog.text

    def test_credentials_missing(self):
        """Test that load_wandb_credentials raises ValueError when key is missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="WandB API key not found in env"):
                load_wandb_credentials()


class TestLoadHfCredentials:
    """Unit tests for load_hf_credentials"""

    def test_credentials_available(self, caplog):
        """Test that load_hf_credentials successfully loads the HF token."""
        with caplog.at_level(logging.INFO):
            with patch.dict(os.environ, {"HF_TOKEN": "test-hf-token"}):
                token = load_hf_credentials()
        assert token == "test-hf-token"
        assert "Hugging Face access token loaded successfully." in caplog.text

    def test_credentials_missing(self):
        """Test that load_hf_credentials raises ValueError when token is missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Hugging Face access token not found"):
                load_hf_credentials()
