"""
Tests for individual imputation methods.
"""

import pytest
import pandas as pd
import numpy as np

from imputation.utils import get_imputer_func
from imputation.PMM import pmm
from imputation.sample import sample
from imputation.cart import cart
from imputation.rf import rf
from imputation.midas import midas


class TestImputerUtils:
    """Test imputer utility functions."""
    
    def test_get_imputer_func_valid(self):
        """Test getting valid imputer functions."""
        pmm_func = get_imputer_func('pmm')
        assert callable(pmm_func)
        
        sample_func = get_imputer_func('sample')
        assert callable(sample_func)
        
        cart_func = get_imputer_func('cart')
        assert callable(cart_func)
        
        rf_func = get_imputer_func('rf')
        assert callable(rf_func)
        
        midas_func = get_imputer_func('midas')
        assert callable(midas_func)
        
    def test_get_imputer_func_invalid(self):
        """Test error with invalid imputer method."""
        with pytest.raises(ValueError, match="Unsupported or unimplemented imputation method"):
            get_imputer_func('invalid_method')


class TestSampleImputation:
    """Test sample imputation method."""
    
    def test_sample_basic(self):
        """Test basic sample imputation."""
        y = np.array([1, 2, np.nan, 4, 5])
        id_obs = ~np.isnan(y)
        x = np.array([[1], [2], [3], [4], [5]])  # Dummy predictors
        
        result = sample(y, id_obs, x)
        
        assert len(result) == 1
        assert result[0] in [1, 2, 4, 5]
        
    def test_sample_multiple_missing(self):
        """Test sample imputation with multiple missing values."""
        y = np.array([1, np.nan, 3, np.nan, 5])
        id_obs = ~np.isnan(y)
        x = np.array([[1], [2], [3], [4], [5]])
        
        result = sample(y, id_obs, x)

        assert len(result) == 2
        for val in result:
            assert val in [1, 3, 5]
            
    def test_sample_with_random_state(self):
        """Test sample imputation with random state for reproducibility."""
        y = np.array([1, 2, np.nan, 4, 5])
        id_obs = ~np.isnan(y)
        x = np.array([[1], [2], [3], [4], [5]])
        
        result1 = sample(y, id_obs, x, random_state=42)
        result2 = sample(y, id_obs, x, random_state=42)
        
        assert result1[0] == result2[0]


class TestPMMImputation:
    """Test PMM imputation method."""
    
    def test_pmm_basic(self):
        """Test basic PMM imputation."""
        np.random.seed(42)
        # Use larger dataset with more variation to avoid numerical issues
        n = 20
        y = np.random.normal(10, 3, n)
        y[5] = np.nan
        y[12] = np.nan
        id_obs = ~np.isnan(y)
        x = np.random.randn(n, 2)
        
        result = pmm(y, id_obs, x, donors=3)
        
        # Test outcomes, not internal logic
        assert len(result) == 2  # Should impute 2 missing values
        assert not np.isnan(result).any()  # No NaN in results
        # Key test: imputed values should be from observed donor pool
        observed_values = y[id_obs]
        for imputed_val in result:
            assert imputed_val in observed_values, f"Imputed value {imputed_val} not in donor pool"
            
    def test_pmm_parameters(self):
        """Test PMM with different parameters."""
        np.random.seed(42)
        # Use robust dataset
        n = 15
        y = np.random.normal(5, 2, n)
        y[3] = np.nan
        id_obs = ~np.isnan(y)
        x = np.random.randn(n, 2)
        
        result = pmm(y, id_obs, x, donors=2, matchtype=0)
        assert len(result) == 1
        assert not np.isnan(result[0])
        observed_values = y[id_obs]
        assert result[0] in observed_values


class TestCARTImputation:
    """Test CART imputation method."""
    
    def test_cart_basic(self):
        """Test basic CART imputation."""
        np.random.seed(42)
        y = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0])
        id_obs = ~np.isnan(y)
        x = np.random.randn(8, 2)
        
        result = cart(y, id_obs, x, min_samples_leaf=2)
        assert len(result) == 1
        assert not np.isnan(result[0])
            
    def test_cart_parameters(self):
        """Test CART with different parameters."""
        np.random.seed(42)
        y = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        id_obs = ~np.isnan(y)
        x = np.random.randn(10, 3)
        

        result = cart(y, id_obs, x, min_samples_leaf=1, ccp_alpha=0.0)
        assert len(result) == 1
        assert not np.isnan(result[0])


class TestRFImputation:
    """Test Random Forest imputation method."""
    
    def test_rf_basic(self):
        """Test basic RF imputation."""
        np.random.seed(42)
        y = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0])
        id_obs = ~np.isnan(y)
        x = np.random.randn(8, 2)
        
        result = rf(y, id_obs, x, n_estimators=5)
        assert len(result) == 1
        assert not np.isnan(result[0])
            
    def test_rf_parameters(self):
        """Test RF with different parameters."""
        np.random.seed(42)
        y = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        id_obs = ~np.isnan(y)
        x = np.random.randn(10, 3)
        
        result = rf(y, id_obs, x, n_estimators=3, random_state=42)
        assert len(result) == 1
        assert not np.isnan(result[0])


class TestMIDASImputation:
    """Test MIDAS imputation method."""
    
    def test_midas_basic(self):
        """Test basic MIDAS imputation."""
        np.random.seed(42)
        y = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0])
        id_obs = ~np.isnan(y)
        x = np.random.randn(8, 2)
        
        result = midas(y, id_obs, x)
        assert len(result) == 1
        assert not np.isnan(result[0])
        # Should be one of the observed values (MIDAS samples from donors)
        assert result[0] in y[id_obs]
            
    def test_midas_parameters(self):
        """Test MIDAS with different parameters."""
        np.random.seed(42)
        y = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        id_obs = ~np.isnan(y)
        x = np.random.randn(10, 3)
        
        result = midas(y, id_obs, x, ridge=1e-4, midas_kappa=3.0)
        assert len(result) == 1
        assert not np.isnan(result[0])
        assert result[0] in y[id_obs]
            
    def test_midas_outout_parameter(self):
        """Test MIDAS with outout parameter."""
        np.random.seed(42)
        y = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        id_obs = ~np.isnan(y)
        x = np.random.randn(10, 2)
        
        result = midas(y, id_obs, x, outout=False)
        assert len(result) == 1
        assert not np.isnan(result[0])
        assert result[0] in y[id_obs]
            
    def test_midas_numeric_validation(self):
        """Test MIDAS validation for numeric data."""
        y = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        id_obs = ~np.isnan(y)
        
        # Test with non-numeric predictors (should fail)
        x_non_numeric = np.array([['a', 'b'], ['c', 'd'], ['e', 'f'], ['g', 'h'], ['i', 'j']])
        
        with pytest.raises(ValueError, match="Predictors must be numeric for midas"):
            midas(y, id_obs, x_non_numeric)
            
    def test_midas_missing_predictors(self):
        """Test MIDAS validation for missing values in predictors."""
        y = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        id_obs = ~np.isnan(y)
        
        # Test with missing values in predictors (should fail)
        x_with_missing = np.array([[1.0, 2.0], [3.0, np.nan], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
        
        with pytest.raises(ValueError, match="Predictors must not contain missing values for midas"):
            midas(y, id_obs, x_with_missing)
            
    def test_midas_non_numeric_target(self):
        """Test MIDAS validation for non-numeric target."""
        y = np.array(['a', 'b', np.nan, 'd', 'e'])
        id_obs = pd.notna(y)
        x = np.random.randn(5, 2)
        
        with pytest.raises(ValueError, match="Target y must be numeric for midas"):
            midas(y, id_obs, x)
            
    def test_midas_multiple_missing(self):
        """Test MIDAS with multiple missing values."""
        np.random.seed(42)
        y = np.array([1.0, np.nan, 3.0, np.nan, 5.0, 6.0, 7.0, 8.0])
        id_obs = ~np.isnan(y)
        x = np.random.randn(8, 2)
        
        result = midas(y, id_obs, x)
        assert len(result) == 2
        assert not np.isnan(result).any()
        for val in result:
            assert val in y[id_obs]


class TestMethodConsistency:
    """Test consistency across imputation methods."""
    
    def test_all_methods_same_interface(self):
        """Test that all methods follow the same interface."""
        np.random.seed(42)
        y = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0])
        id_obs = ~np.isnan(y)
        x = np.random.randn(8, 2)
        
        methods = ['sample', 'pmm', 'cart', 'rf', 'midas']
        
        for method_name in methods:
            method_func = get_imputer_func(method_name)
            result = method_func(y, id_obs, x)

            assert isinstance(result, np.ndarray)
            assert len(result) == (~id_obs).sum()
            assert not np.isnan(result).any()
