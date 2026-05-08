"""Pytest conftest template — injected into work dir to provide project_dir fixture."""
import pytest
from pathlib import Path


@pytest.fixture
def project_dir():
    """Fixture pointing to the directory containing generated code."""
    return Path(__file__).parent
