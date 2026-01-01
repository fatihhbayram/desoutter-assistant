"""
Pytest Configuration for Desoutter Assistant Tests
==================================================
Shared fixtures and configuration for all test modules.
"""

import os
import sys
from pathlib import Path

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def api_base_url():
    """Get API base URL from environment or default"""
    return os.getenv("RAG_TEST_API_URL", "http://localhost:8000")


@pytest.fixture(scope="session")
def request_timeout():
    """Get request timeout from environment or default"""
    return int(os.getenv("RAG_TEST_TIMEOUT", "60"))


@pytest.fixture(scope="session")
def test_queries():
    """Get standard test queries"""
    from tests.fixtures.standard_queries import STANDARD_TEST_QUERIES
    return STANDARD_TEST_QUERIES


@pytest.fixture(scope="session")
def idk_queries():
    """Get queries expected to return 'I don't know'"""
    from tests.fixtures.standard_queries import get_idk_queries
    return get_idk_queries()


@pytest.fixture(scope="module")
def rag_engine():
    """Get RAG engine instance (for unit tests)"""
    try:
        from src.llm.rag_engine import RAGEngine
        return RAGEngine()
    except Exception as e:
        pytest.skip(f"Could not initialize RAGEngine: {e}")


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers"""
    # Add integration marker to tests that use API
    for item in items:
        if "api" in item.fixturenames or "api_base_url" in item.fixturenames:
            item.add_marker(pytest.mark.integration)


# =============================================================================
# HOOKS
# =============================================================================

@pytest.hookimpl(tryfirst=True)
def pytest_runtest_makereport(item, call):
    """Custom reporting for test results"""
    if call.when == "call":
        if call.excinfo is not None:
            # Test failed
            pass  # Can add custom logging here
