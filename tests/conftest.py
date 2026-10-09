"""Records the git commit in pytest output (queue steps run pytest -q; see training/provenance.py)."""
from training.provenance import log_git_commit


def pytest_configure(config):
    log_git_commit()
