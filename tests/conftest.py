import pytest
from credentials import load_hf_credentials


@pytest.fixture(scope="module")
def hf_token():
    """Load HF credentials for integration tests."""
    try:
        return load_hf_credentials()
    except ValueError:  # Skips module if HF_TOKEN is not configured
        pytest.skip("HF_TOKEN not configured", allow_module_level=True)
