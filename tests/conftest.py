"""
Test configuration and fixtures
"""

import pytest
import os
import sys

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))


@pytest.fixture(scope="session")
def test_config():
    """Global test configuration"""
    return {
        "model_name": "gpt2",
        "device": "cpu",  # Use CPU for tests to avoid GPU dependencies
        "batch_size": 1,
        "max_length": 50
    }


@pytest.fixture
def temp_dir(tmp_path):
    """Create temporary directory for test outputs"""
    return tmp_path


def pytest_configure(config):
    """Pytest configuration hook"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests as requiring GPU"
    )