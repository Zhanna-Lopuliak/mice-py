"""
Pytest configuration and shared fixtures for testing.

The fixtures draw their data from their own ``np.random.RandomState(42)``
(the same numbers as the former ``np.random.seed(42)``), so they neither
depend on nor change NumPy's global random state.
"""

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # headless; set before pyplot is imported
import matplotlib.pyplot as plt  # noqa: E402


@pytest.fixture(autouse=True)
def _mute_matplotlib_show(monkeypatch):
    # plt.show() on the Agg backend warns that it cannot show anything
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)


def _patients(rs, n):
    return {
        "age": rs.normal(50, 15, n),
        "bmi": rs.normal(25, 5, n),
        "cholesterol": rs.normal(200, 40, n),
        "gender": rs.choice(["M", "F"], n),
        "smoker": rs.choice([0, 1], n),
    }


@pytest.fixture
def sample_data_complete():
    """Create a complete dataset without missing values for testing."""
    return pd.DataFrame(_patients(np.random.RandomState(42), 100))


@pytest.fixture
def sample_data_missing():
    """Create a dataset with missing values for testing imputation."""
    rs = np.random.RandomState(42)
    n = 100
    df = pd.DataFrame(_patients(rs, n))

    # 20% of bmi, 15% of cholesterol and 10% of smoker missing
    df.loc[rs.choice(n, size=int(0.2 * n), replace=False), "bmi"] = np.nan
    df.loc[rs.choice(n, size=int(0.15 * n), replace=False), "cholesterol"] = np.nan
    df.loc[rs.choice(n, size=int(0.1 * n), replace=False), "smoker"] = np.nan
    return df


@pytest.fixture
def small_data_missing():
    """Create a small dataset with missing values for quick tests."""
    data = {"x1": [1, 2, np.nan, 4, 5], "x2": [10, np.nan, 30, 40, 50], "x3": ["A", "B", "C", np.nan, "E"]}
    return pd.DataFrame(data)


@pytest.fixture
def temp_dir(tmp_path):
    """A temporary directory for test outputs."""
    return tmp_path


@pytest.fixture
def predictor_matrix():
    """bmi predicted by cholesterol, and cholesterol by bmi."""
    columns = ["bmi", "cholesterol"]
    return pd.DataFrame([[0, 1], [1, 0]], index=columns, columns=columns)


@pytest.fixture
def imputed_datasets():
    """Create mock imputed datasets for plotting tests."""
    rs = np.random.RandomState(42)
    n = 50
    return [
        pd.DataFrame(
            {
                "age": rs.normal(50, 15, n),
                "bmi": rs.normal(25, 5, n),
                "cholesterol": rs.normal(200, 40, n),
            }
        )
        for _ in range(3)  # 3 imputations
    ]


@pytest.fixture
def missing_pattern():
    """Create a missing pattern matrix for plotting tests."""
    rs = np.random.RandomState(42)
    n = 50
    return pd.DataFrame(
        {
            "age": np.ones(n),  # No missing values
            "bmi": rs.choice([0, 1], n, p=[0.2, 0.8]),  # 20% missing
            "cholesterol": rs.choice([0, 1], n, p=[0.15, 0.85]),  # 15% missing
        }
    )
