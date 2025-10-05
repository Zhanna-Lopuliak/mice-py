"""
Tests for the core MICE functionality.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
import warnings

from imputation import MICE, configure_logging


class TestMICEInitialization:
    """Test MICE object initialization."""
    
    def test_init_with_valid_data(self, sample_data_missing):
        """Test MICE initialization with valid data."""
        mice = MICE(sample_data_missing)
        
        assert isinstance(mice.data, pd.DataFrame)
        assert mice.data.shape == sample_data_missing.dropna(how='all').shape
        assert hasattr(mice, 'id_obs')
        assert hasattr(mice, 'id_mis')
        
    def test_init_with_invalid_data(self):
        """Test MICE initialization with invalid data."""
        with pytest.raises(ValueError, match="Input data cannot be converted to DataFrame"):
            MICE("invalid_data")
            
    def test_init_with_duplicate_columns(self):
        """Test MICE initialization with duplicate column names."""
        data = pd.DataFrame([[1, 2], [3, 4]], columns=['A', 'A'])
        with pytest.raises(ValueError, match="DataFrame contains duplicate column names"):
            MICE(data)
            
    def test_missing_value_tracking(self, sample_data_missing):
        """Test that missing values are correctly tracked."""
        mice = MICE(sample_data_missing)
        
        for col in mice.data.columns:
            # Check that id_obs and id_mis are complementary
            assert (mice.id_obs[col] == ~mice.id_mis[col]).all()
            
            # Check that counts match
            expected_missing = sample_data_missing[col].isna().sum()
            actual_missing = mice.id_mis[col].sum()
            assert actual_missing == expected_missing


class TestMICEImputation:
    """Test MICE imputation functionality."""
    
    def test_basic_imputation(self, sample_data_missing):
        """Test basic MICE imputation."""
        mice = MICE(sample_data_missing)
        
        mice.impute(n_imputations=2, maxit=2)
        
        assert mice.imputed_datasets is not None
        assert hasattr(mice, 'imputed_datasets')
        assert len(mice.imputed_datasets) == 2
        
        for dataset in mice.imputed_datasets:
            assert not dataset.isna().any().any()
            
    def test_imputation_parameters(self, sample_data_missing):
        """Test imputation with different parameters."""
        mice = MICE(sample_data_missing)
        
        mice.impute(
            n_imputations=3,
            maxit=5,
            method='sample',
            initial='sample'
        )
        
        assert len(mice.imputed_datasets) == 3
        
    def test_imputation_with_predictor_matrix(self, sample_data_missing, predictor_matrix):
        """Test imputation with custom predictor matrix."""
        data_subset = sample_data_missing[['age', 'bmi', 'cholesterol']].copy()
        mice = MICE(data_subset)
        
        mice.impute(
            n_imputations=2,
            maxit=2,
            predictor_matrix=predictor_matrix
        )
        
        assert mice.imputed_datasets is not None
        assert len(mice.imputed_datasets) == 2


class TestMICEMethods:
    """Test different imputation methods."""
    
    @pytest.mark.parametrize("method", ["sample", "cart"])
    def test_methods_with_categorical_data(self, small_data_missing, method):
        """Test imputation methods that can handle categorical data."""
        mice = MICE(small_data_missing)
        
        mice.impute(n_imputations=2, maxit=2, method=method)
        assert mice is not None
        assert len(mice.imputed_datasets) == 2
        
    @pytest.mark.parametrize("method", ["pmm", "midas"])
    def test_methods_numeric_only(self, sample_data_missing, method):
        """Test imputation methods that require numeric-only data."""
        # Use only numeric columns from sample_data_missing
        numeric_data = sample_data_missing[['age', 'bmi', 'cholesterol']].copy()
        mice = MICE(numeric_data)
        
        mice.impute(n_imputations=2, maxit=2, method=method)
        assert mice.imputed_datasets is not None
        assert len(mice.imputed_datasets) == 2
            
    def test_method_dictionary(self, sample_data_missing):
        """Test using method dictionary for different columns."""
        mice = MICE(sample_data_missing)
        
        method_dict = {
            'bmi': 'sample',
            'cholesterol': 'sample'
        }
        
        mice.impute(
            n_imputations=2,
            maxit=2,
            method=method_dict
        )
        
        assert mice.imputed_datasets is not None


class TestMICELogging:
    """Test MICE logging functionality."""
    
    def test_logging_configuration(self, sample_data_missing):
        """Test that logging can be configured."""
        logger = configure_logging(level='DEBUG', file_logging=False)
        
        mice = MICE(sample_data_missing)
        mice.impute(n_imputations=1, maxit=1)
        
        assert logger is not None
        
    def test_logging_warning_without_config(self, sample_data_missing):
        """Test that warning is shown when logging not configured."""
        from imputation.logging_config import reset_logging
        reset_logging()
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mice = MICE(sample_data_missing)
