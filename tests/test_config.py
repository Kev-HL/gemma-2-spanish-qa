"""Unit tests for some of the config.py functions, located in src/config.py"""

# Third-party imports
import pytest

# Local imports
from config import load_training_config


class TestLoadTrainingConfig:
    """Unit tests for load_training_config"""

    def test_valid_mbert_cfg(self):
        """
        Test that load_training_config correctly loads a valid mBERT config file.
        """
        # Use the training schema as input for a valid config
        # mBERT training schema can be found in configs/schemas/train_mbert.json
        cfg_path = "configs/schemas/train_mbert.json"
        _ = load_training_config(cfg_path, "mbert")

    def test_valid_gemma2_cfg(self):
        """
        Test that load_training_config correctly loads a valid Gemma 2 config file.
        """
        # Use the training schema as input for a valid config
        # Gemma 2 training schema can be found in configs/schemas/train_gemma2.json
        cfg_path = "configs/schemas/train_gemma2.json"
        _ = load_training_config(cfg_path, "gemma2")

    def test_invalid_cfg_type(self):
        """
        Test that load_training_config raises a ValueError with an unsupported cfg_type.
        """
        cfg_path = "configs/schemas/train_mbert.json"
        with pytest.raises(ValueError, match="Unsupported cfg_type"):
            load_training_config(cfg_path, "invalid")

    def test_nonexistent_file(self):
        """
        Test that load_training_config raises a FileNotFoundError when the config file
        does not exist.
        """
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_training_config("nonexistent_config.json", "mbert")

    def test_invalid_json(self, tmp_path):
        """
        Test that load_training_config raises a ValueError when the config file contains
        invalid JSON.
        """
        bad_json = tmp_path / "bad.json"
        bad_json.write_text("{invalid json}")
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_training_config(str(bad_json), "mbert")
