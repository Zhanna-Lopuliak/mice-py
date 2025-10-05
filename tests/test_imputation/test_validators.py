"""
Tests for validation functions in the imputation module.
"""

import pytest
import pandas as pd
import numpy as np
import warnings

from imputation.validators import (
    validate_dataframe,
    validate_columns,
    check_n_imputations,
    check_maxit,
    check_method,
    check_initial_method,
    validate_predictor_matrix,
    check_visit_sequence,
    validate_formula
)


class TestValidateDataframe:
    """Test dataframe validation."""
    
    def test_valid_dataframe(self, sample_data_complete):
        """Test validation of valid dataframe."""
        result = validate_dataframe(sample_data_complete)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == sample_data_complete.shape
        
    def test_convert_array_to_dataframe(self):
        """Test conversion of array to dataframe."""
        data = [[1, 2], [3, 4]]
        result = validate_dataframe(data)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (2, 2)
        
    def test_duplicate_columns_error(self):
        """Test error on duplicate column names."""
        data = pd.DataFrame([[1, 2], [3, 4]], columns=['A', 'A'])
        with pytest.raises(ValueError, match="DataFrame contains duplicate column names"):
            validate_dataframe(data)
            
    def test_drop_empty_rows(self):
        """Test dropping of fully empty rows."""
        data = pd.DataFrame({
            'A': [1, np.nan, 3],
            'B': [2, np.nan, 4]
        })
        result = validate_dataframe(data)
        assert result.shape[0] == 2  # One row should be dropped
        
    def test_drop_empty_columns(self):
        """Test dropping of fully empty columns."""
        data = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [np.nan, np.nan, np.nan],
            'C': [4, 5, 6]
        })
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = validate_dataframe(data)
        assert 'B' not in result.columns
        assert result.shape[1] == 2


class TestValidateColumns:
    """Test column validation."""
    
    def test_valid_columns(self, sample_data_missing):
        """Test validation of valid columns."""
        result = validate_columns(sample_data_missing)
        assert isinstance(result, pd.DataFrame)
        
    def test_drop_nan_only_columns(self):
        """Test dropping of columns with only NaN values."""
        data = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [np.nan, np.nan, np.nan],
            'C': [4, 5, 6]
        })
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = validate_columns(data)
        assert 'B' not in result.columns


class TestCheckParameters:
    """Test parameter validation functions."""
    
    def test_check_n_imputations_valid(self):
        """Test valid n_imputations values."""
        check_n_imputations(5)  # Should not raise
        check_n_imputations(1)  # Should not raise
        
    def test_check_n_imputations_invalid(self):
        """Test invalid n_imputations values."""
        with pytest.raises(ValueError, match="n_imputations must be a positive integer"):
            check_n_imputations(0)
        with pytest.raises(ValueError, match="n_imputations must be a positive integer"):
            check_n_imputations(-1)
        with pytest.raises(ValueError, match="n_imputations must be a positive integer"):
            check_n_imputations(1.5)
            
    def test_check_maxit_valid(self):
        """Test valid maxit values."""
        check_maxit(10)  # Should not raise
        check_maxit(1)   # Should not raise
        
    def test_check_maxit_invalid(self):
        """Test invalid maxit values."""
        with pytest.raises(ValueError, match="maxit must be a positive integer"):
            check_maxit(0)
        with pytest.raises(ValueError, match="maxit must be a positive integer"):
            check_maxit(-1)
            
    def test_check_maxit_warning(self):
        """Test warning for large maxit values."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_maxit(100)
            assert len(w) == 1
            assert "maxit is greater than 50" in str(w[0].message)


class TestCheckMethod:
    """Test method validation."""
    
    def test_check_method_string(self):
        """Test method validation with string input."""
        columns = ['A', 'B', 'C']
        result = check_method('sample', columns)
        expected = {'A': 'sample', 'B': 'sample', 'C': 'sample'}
        assert result == expected
        
    def test_check_method_invalid_string(self):
        """Test method validation with invalid string."""
        columns = ['A', 'B']
        with pytest.raises(ValueError, match="Unsupported method"):
            check_method('invalid_method', columns)
            
    def test_check_method_dictionary(self):
        """Test method validation with dictionary input."""
        columns = ['A', 'B', 'C']
        method_dict = {'A': 'pmm', 'B': 'sample'}
        result = check_method(method_dict, columns)
        
        assert result['A'] == 'pmm'
        assert result['B'] == 'sample'
        
    def test_check_method_invalid_column(self):
        """Test method validation with invalid column in dictionary."""
        columns = ['A', 'B']
        method_dict = {'A': 'pmm', 'C': 'sample'}  # C doesn't exist
        with pytest.raises(ValueError, match="Columns not found in data"):
            check_method(method_dict, columns)


class TestCheckInitialMethod:
    """Test initial method validation."""
    
    def test_check_initial_method_valid(self):
        """Test valid initial methods."""
        check_initial_method('sample')  
        check_initial_method('meanobs')  
        
    def test_check_initial_method_invalid(self):
        """Test invalid initial methods."""
        with pytest.raises(ValueError, match="Unsupported initial method"):
            check_initial_method('invalid_method')
        with pytest.raises(ValueError, match="initial_method must be a string"):
            check_initial_method(123)


class TestValidatePredictorMatrix:
    """Test predictor matrix validation."""
    
    def test_valid_predictor_matrix(self, predictor_matrix, sample_data_missing):
        """Test validation of valid predictor matrix."""
        data_subset = sample_data_missing[['bmi', 'cholesterol']]
        result = validate_predictor_matrix(
            predictor_matrix, 
            list(data_subset.columns), 
            data_subset
        )
        assert isinstance(result, pd.DataFrame)
        assert result.shape == predictor_matrix.shape
        
    def test_predictor_matrix_invalid_type(self, sample_data_complete):
        """Test error with invalid predictor matrix type."""
        with pytest.raises(ValueError, match="predictor_matrix must be a pandas DataFrame"):
            validate_predictor_matrix([[1, 0], [0, 1]], ['A', 'B'], sample_data_complete)
            
    def test_predictor_matrix_invalid_columns(self, sample_data_complete):
        """Test error with invalid columns in predictor matrix."""
        matrix = pd.DataFrame([[1, 0], [0, 1]], 
                            index=['A', 'B'], 
                            columns=['A', 'C'])  # C doesn't exist
        with pytest.raises(ValueError, match="predictor_matrix contains columns not in data"):
            validate_predictor_matrix(matrix, ['A', 'B'], sample_data_complete)
            
    def test_predictor_matrix_invalid_values(self, sample_data_complete):
        """Test error with invalid values in predictor matrix."""
        matrix = pd.DataFrame([[1, 0.5], [0, 1]], 
                            index=['A', 'B'], 
                            columns=['A', 'B'])
        with pytest.raises(ValueError, match="predictor_matrix must contain only 0s and 1s"):
            validate_predictor_matrix(matrix, ['A', 'B'], sample_data_complete)


class TestCheckVisitSequence:
    """Test visit sequence validation."""
    
    def test_check_visit_sequence_string(self):
        """Test visit sequence validation with string."""
        columns = ['A', 'B', 'C']
        check_visit_sequence('monotone', columns)
        
    def test_check_visit_sequence_list(self):
        """Test visit sequence validation with list."""
        columns = ['A', 'B', 'C']
        check_visit_sequence(['A', 'B', 'C'], columns)  
        
    def test_check_visit_sequence_invalid_column(self):
        """Test visit sequence validation with invalid column."""
        columns = ['A', 'B']
        with pytest.raises(ValueError, match="Visit sequence contains columns not in data"):
            check_visit_sequence(['A', 'B', 'C'], columns)


class TestValidateFormula:
    """Test formula validation."""
    
    def test_valid_formula(self):
        """Test validation of valid formula."""
        columns = ['age', 'bmi', 'cholesterol']
        formula = 'age ~ bmi + cholesterol'
        validate_formula(formula, columns) 
        
    def test_formula_missing_variables(self):
        """Test error with missing variables in formula."""
        columns = ['age', 'bmi']
        formula = 'age ~ bmi + cholesterol'  # cholesterol not in columns
        with pytest.raises(ValueError, match="variables in the formula are not present"):
            validate_formula(formula, columns)
            
    def test_formula_invalid_type(self):
        """Test error with invalid formula type."""
        columns = ['age', 'bmi']
        with pytest.raises(ValueError, match="formula must be a string"):
            validate_formula(123, columns)
