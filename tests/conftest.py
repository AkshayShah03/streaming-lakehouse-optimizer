import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lakehouse.optimizer.cost_model import CostModel  # noqa: E402
from lakehouse.optimizer.train import generate_dataset  # noqa: E402


@pytest.fixture(scope="session")
def trained_model():
    """Train once per test session (cost model fit is the slow part)."""
    X, y_lat, y_cost, y_wamp = generate_dataset(n_workloads=25, seed=3)
    return CostModel().fit(X, y_lat, y_cost, y_wamp), (X, y_lat)
