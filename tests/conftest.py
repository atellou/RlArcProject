import os
import pytest
import torch


@pytest.fixture(scope="session")
def device():
    # Force CPU for stable tests across environments during migration
    return torch.device("cpu")


@pytest.fixture(autouse=True)
def _set_log_level_env(monkeypatch):
    # Ensure consistent logging level during tests
    monkeypatch.setenv("LOG_LEVEL", os.environ.get("LOG_LEVEL", "INFO"))
    yield
