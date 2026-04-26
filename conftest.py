"""Pytest conftest: ensures repo-root-relative packages (eval/, diagnostics/) resolve."""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.resolve()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
