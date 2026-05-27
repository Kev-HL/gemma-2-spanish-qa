"""Fixtures for integration tests requiring Hugging Face credentials."""

# Third party imports
import pytest

# Local imports
from credentials import load_hf_credentials


@pytest.fixture(scope="module")
def hf_token():
    """Load HF token for integration tests.

    Skips module if HF_TOKEN environment variable not configured.
    """
    try:
        return load_hf_credentials()
    except ValueError:
        pytest.skip("HF_TOKEN not configured", allow_module_level=True)
