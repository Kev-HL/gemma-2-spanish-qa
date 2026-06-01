"""Unit tests for some of the config.py functions, located in src/config.py"""

# Standard imports
from pathlib import Path

# Third-party imports
import pytest

# Local imports
from config import load_training_config, load_eval_config


class TestLoadTrainingConfig:
    """Unit tests for load_training_config"""

    def test_valid_mbert_cfg(self, tmp_path: Path, monkeypatch) -> None:
        """
        Test that load_training_config correctly loads a valid mBERT config file.
        """
        # Use the training schema as input for a valid config
        # mBERT training schema can be found in configs/schemas/train_mbert.json

        # Save absolute path to config file
        repo_root = Path(__file__).parent.parent
        cfg_path = str(repo_root / "configs/schemas/train_mbert.json")

        # Change cwd to tmp_path
        monkeypatch.chdir(tmp_path)

        # Create valid data files that the config expects
        tmp_data_dir = tmp_path / "data" / "squad_es"
        tmp_data_dir.mkdir(parents=True)
        (tmp_data_dir / "clean_train-v1.1-es_small.json").write_text("{}")
        (tmp_data_dir / "clean_dev-v1.1-es_small.json").write_text("{}")

        _ = load_training_config(cfg_path, "mbert")

    def test_valid_gemma2_cfg(self, tmp_path: Path, monkeypatch) -> None:
        """
        Test that load_training_config correctly loads a valid Gemma 2 config file.
        """
        # Use the training schema as input for a valid config
        # Gemma 2 training schema can be found in configs/schemas/train_gemma2.json

        # Save absolute path to config file
        repo_root = Path(__file__).parent.parent
        cfg_path = str(repo_root / "configs/schemas/train_gemma2.json")

        # Change cwd to tmp_path
        monkeypatch.chdir(tmp_path)

        # Create valid data files that the config expects
        tmp_data_dir = tmp_path / "data" / "squad_es"
        tmp_data_dir.mkdir(parents=True)
        (tmp_data_dir / "clean_train-v1.1-es_small.json").write_text("{}")
        (tmp_data_dir / "clean_dev-v1.1-es_small.json").write_text("{}")
        _ = load_training_config(cfg_path, "gemma2")

    def test_invalid_cfg_type(self) -> None:
        """
        Test that load_training_config raises a ValueError with an unsupported cfg_type.
        """
        cfg_path = "config.json"
        with pytest.raises(ValueError, match="Unsupported cfg_type"):
            load_training_config(cfg_path, "invalid")

    def test_nonexistent_file(self) -> None:
        """
        Test that load_training_config raises a FileNotFoundError when the config file
        does not exist.
        """
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_training_config("nonexistent_config.json", "mbert")

    def test_invalid_json(self, tmp_path: Path) -> None:
        """
        Test that load_training_config raises a ValueError when the config file contains
        invalid JSON.
        """
        bad_json = tmp_path / "bad.json"
        bad_json.write_text("{invalid json}")
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_training_config(str(bad_json), "mbert")


class TestLoadEvalConfig:
    """Unit tests for load_eval_config"""

    def test_valid_cfg(self, tmp_path: Path, monkeypatch) -> None:
        """
        Test that load_eval_config correctly loads a valid config file.
        """
        # Use one of the evaluation schemas as input for a valid config
        # Gemma 2 evaluation schema can be found in configs/schemas/eval_gemma2.json

        # Save absolute path to config file
        repo_root = Path(__file__).parent.parent
        cfg_path = str(repo_root / "configs/schemas/eval_gemma2.json")

        # Change cwd to tmp_path
        monkeypatch.chdir(tmp_path)

        # Create valid data files that the config expects
        tmp_data_dir = tmp_path / "data" / "squad_es"
        tmp_data_dir.mkdir(parents=True)
        (tmp_data_dir / "clean_dev-v1.1-es_small.json").write_text("{}")
        tmp_model_dir = tmp_path / "artifacts" / "test_experiment" / "checkpoint-123"
        tmp_model_dir.mkdir(parents=True)
        (tmp_model_dir / "model.safetensors").write_text("")
        _ = load_eval_config(cfg_path)

    def test_nonexistent_cfg_file(self) -> None:
        """
        Test that load_eval_config raises a FileNotFoundError when the config file
        does not exist.
        """
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_eval_config("nonexistent_config.json")

    def test_invalid_cfg_json(self, tmp_path: Path) -> None:
        """
        Test that load_eval_config raises a ValueError when the config file contains
        invalid JSON.
        """
        bad_json = tmp_path / "bad.json"
        bad_json.write_text("{invalid json}")
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_eval_config(str(bad_json))

    def test_missing_model_file(self, tmp_path: Path, monkeypatch) -> None:
        """
        Test that load_eval_config raises a FileNotFoundError when the specified model
        file does not exist.
        """
        # Use one of the evaluation schemas as input for a valid config
        # Gemma 2 evaluation schema can be found in configs/schemas/eval_gemma2.json

        # Save absolute path to config file
        repo_root = Path(__file__).parent.parent
        cfg_path = str(repo_root / "configs/schemas/eval_gemma2.json")

        # Change cwd to tmp_path
        monkeypatch.chdir(tmp_path)

        # Create valid data files that the config expects
        tmp_data_dir = tmp_path / "data" / "squad_es"
        tmp_data_dir.mkdir(parents=True)
        (tmp_data_dir / "clean_dev-v1.1-es_small.json").write_text("{}")
        tmp_model_dir = tmp_path / "artifacts" / "test_experiment" / "checkpoint-123"
        tmp_model_dir.mkdir(parents=True)
        with pytest.raises(
            FileNotFoundError, match="Model or adapter .safetensors file not found"
        ):
            _ = load_eval_config(cfg_path)
