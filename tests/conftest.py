import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hqprompt.paths import get_paths  # noqa: E402


@pytest.fixture
def hqp_root(tmp_path, monkeypatch):
    """Paths whose out_root (the only writable area) is a fresh tmp dir."""
    monkeypatch.setenv("HQP_ROOT", str(tmp_path))
    monkeypatch.delenv("HQP_MODEL_CACHE", raising=False)
    return get_paths()
