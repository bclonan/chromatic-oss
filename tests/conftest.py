"""Keep test scratch files in a unique, writable workspace directory."""
from pathlib import Path
import uuid
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def pytest_configure(config):
    if config.option.basetemp is None:
        parent = Path(__file__).resolve().parents[1] / ".pytest-work"
        parent.mkdir(parents=True, exist_ok=True)
        config.option.basetemp = str(parent / uuid.uuid4().hex)

