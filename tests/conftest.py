"""
Pytest configuration and shared fixtures for testing.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
import matplotlib
matplotlib.use("Agg")  # ensure headless
import matplotlib.pyplot as plt

# Set random seed for reproducible tests
np.random.seed(42)

@pytest.fixture(autouse=True)
def _mute_matplotlib_show(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

@pytest.fixture
def sample_data_complete():
    """Create a complete dataset without missing values for testing."""
    np.random.seed(42)
    n = 100
    data = {
        'age': np.random.normal(50, 15, n),
        'bmi': np.random.normal(25, 5, n),
        'cholesterol': np.random.normal(200, 40, n),
        'gender': np.random.choice(['M', 'F'], n),
        'smoker': np.random.choice([0, 1], n)
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_data_missing():
    """Create a dataset with missing values for testing imputation."""
    np.random.seed(42)
    n = 100
    data = {
        'age': np.random.normal(50, 15, n),
        'bmi': np.random.normal(25, 5, n),
        'cholesterol': np.random.normal(200, 40, n),
        'gender': np.random.choice(['M', 'F'], n),
        'smoker': np.random.choice([0, 1], n)
    }
    df = pd.DataFrame(data)
    
    # Introduce missing values
    # Make 20% of BMI values missing
    missing_bmi = np.random.choice(n, size=int(0.2 * n), replace=False)
    df.loc[missing_bmi, 'bmi'] = np.nan
    
    # Make 15% of cholesterol values missing
    missing_chol = np.random.choice(n, size=int(0.15 * n), replace=False)
    df.loc[missing_chol, 'cholesterol'] = np.nan
    
    # Make 10% of smoker values missing
    missing_smoker = np.random.choice(n, size=int(0.1 * n), replace=False)
    df.loc[missing_smoker, 'smoker'] = np.nan
    
    return df

@pytest.fixture
def small_data_missing():
    """Create a small dataset with missing values for quick tests."""
    data = {
        'x1': [1, 2, np.nan, 4, 5],
        'x2': [10, np.nan, 30, 40, 50],
        'x3': ['A', 'B', 'C', np.nan, 'E']
    }
    return pd.DataFrame(data)

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def predictor_matrix():
    """Create a simple predictor matrix for testing."""
    columns = ['bmi', 'cholesterol']
    matrix = pd.DataFrame(
        [[0, 1],   # bmi predicted by cholesterol
         [1, 0]],  # cholesterol predicted by bmi
        index=columns,
        columns=columns
    )
    return matrix

@pytest.fixture
def imputed_datasets():
    """Create mock imputed datasets for plotting tests."""
    np.random.seed(42)
    n = 50
    datasets = []
    
    for i in range(3):  # 3 imputations
        data = {
            'age': np.random.normal(50, 15, n),
            'bmi': np.random.normal(25, 5, n),
            'cholesterol': np.random.normal(200, 40, n)
        }
        datasets.append(pd.DataFrame(data))
    
    return datasets

@pytest.fixture
def missing_pattern():
    """Create a missing pattern matrix for plotting tests."""
    np.random.seed(42)
    n = 50
    pattern = pd.DataFrame({
        'age': np.ones(n),  # No missing values
        'bmi': np.random.choice([0, 1], n, p=[0.2, 0.8]),  # 20% missing
        'cholesterol': np.random.choice([0, 1], n, p=[0.15, 0.85])  # 15% missing
    })
    return pattern

# Disable matplotlib GUI for testing
@pytest.fixture(autouse=True)
def setup_matplotlib():
    """Setup matplotlib for testing (non-interactive backend)."""
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    yield
