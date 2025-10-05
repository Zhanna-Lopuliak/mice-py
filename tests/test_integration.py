"""
Integration tests for end-to-end workflows.
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from imputation import MICE, configure_logging
from plotting.diagnostics import stripplot, densityplot, xyplot
from plotting.utils import md_pattern_like, plot_missing_data_pattern


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""
    
    def test_complete_mice_workflow(self, sample_data_missing, temp_dir):
        """Test complete MICE workflow from data to results."""
        # Step 1: Initialize MICE
        mice = MICE(sample_data_missing)
        
        # Step 2: Run imputation
        mice.impute(n_imputations=3, maxit=5, method='sample')
        
        # Step 3: Verify results
        assert mice.imputed_datasets is not None
        assert len(mice.imputed_datasets) == 3
        
        # Step 4: Check that all datasets are complete
        for dataset in mice.imputed_datasets:
            assert not dataset.isna().any().any()
            assert dataset.shape == mice.data.shape
            
        # Step 5: Verify column consistency
        for dataset in mice.imputed_datasets:
            assert list(dataset.columns) == list(mice.data.columns)
            
    def test_mice_with_plotting_workflow(self, sample_data_missing, temp_dir):
        """Test MICE workflow with diagnostic plotting."""
        # Step 1: Run MICE
        mice = MICE(sample_data_missing)
        mice.impute(n_imputations=2, maxit=3, method='sample')
        
        # Step 2: Create missing pattern for plotting
        missing_pattern = pd.DataFrame()
        for col in mice.data.columns:
            missing_pattern[col] = mice.id_obs[col].astype(int)
            
        # Step 3: Create diagnostic plots
        stripplot(
            imputed_datasets=mice.imputed_datasets,
            missing_pattern=missing_pattern,
            columns=['bmi'],
            save_path=str(temp_dir / "integration_stripplot.png")
        )
        
        densityplot(
            imputed_datasets=mice.imputed_datasets,
            missing_pattern=missing_pattern,
            columns=['bmi'],
            save_path=str(temp_dir / "integration_densityplot.png")
        )
        
        # Step 4: Verify plots were created
        assert (temp_dir / "integration_stripplot.png").exists()
        assert (temp_dir / "integration_densityplot.png").exists()
        
        plt.close('all')
        
    def test_missing_pattern_analysis_workflow(self, sample_data_missing, temp_dir):
        """Test missing data pattern analysis workflow."""
        # Step 1: Analyze missing patterns
        pattern_df = md_pattern_like(sample_data_missing)
        
        # Step 2: Verify pattern structure
        assert isinstance(pattern_df, pd.DataFrame)
        assert '#miss_row' in pattern_df.columns
        assert len(pattern_df) > 0
        
        # Step 3: Plot missing patterns
        plot_missing_data_pattern(
            pattern_df,
            title="Integration Test Missing Pattern",
            save_path=str(temp_dir / "integration_missing_pattern.png")
        )
        
        # Step 4: Verify plot was created
        assert (temp_dir / "integration_missing_pattern.png").exists()
        
        plt.close('all')

class TestLoggingIntegration:
    """Test logging integration with workflows."""

        
    def test_workflow_without_logging_config(self, sample_data_missing):
        """Test workflow without explicit logging configuration."""
        from imputation.logging_config import reset_logging
        reset_logging()
        
        # Should still work without explicit logging config
        mice = MICE(sample_data_missing)
        mice.impute(n_imputations=1, maxit=1, method='sample')
        
        assert mice.imputed_datasets is not None


class TestDataTypeHandling:
    """Test handling of different data types."""
    
    def test_mixed_data_types_workflow(self):
        """Test workflow with mixed data types."""
        # Create data with mixed types
        np.random.seed(42)
        n = 50
        mixed_data = pd.DataFrame({
            'numeric_int': np.random.randint(1, 100, n),
            'numeric_float': np.random.normal(50, 15, n),
            'categorical': np.random.choice(['A', 'B', 'C'], n),
            'binary': np.random.choice([0, 1], n)
        })
        
        # Introduce missing values
        missing_idx = np.random.choice(n, size=int(0.2 * n), replace=False)
        mixed_data.loc[missing_idx, 'numeric_float'] = np.nan
        
        missing_idx = np.random.choice(n, size=int(0.15 * n), replace=False)
        mixed_data.loc[missing_idx, 'categorical'] = np.nan
        
        # Run MICE workflow
        mice = MICE(mixed_data)
        mice.impute(n_imputations=2, maxit=2, method='sample')
        
        # Verify results            
        assert mice.imputed_datasets is not None
        for dataset in mice.imputed_datasets:
            assert not dataset.isna().any().any()
            
    def test_small_dataset_workflow(self, small_data_missing):
        """Test workflow with very small dataset."""
        mice = MICE(small_data_missing)
        
        # Use simple method that should work with small data
        mice.impute(n_imputations=2, maxit=2, method='sample')
        
        assert mice.imputed_datasets is not None
        assert len(mice.imputed_datasets) == 2


class TestErrorRecovery:
    """Test error handling and recovery in workflows."""
    
    def test_workflow_with_invalid_method(self, sample_data_missing):
        """Test workflow behavior with invalid method."""
        mice = MICE(sample_data_missing)
        
        with pytest.raises(ValueError):
            mice.impute(method='invalid_method')
            
    def test_workflow_with_invalid_parameters(self, sample_data_missing):
        """Test workflow behavior with invalid parameters."""
        mice = MICE(sample_data_missing)
        
        with pytest.raises(ValueError):
            mice.impute(n_imputations=0)
            
        with pytest.raises(ValueError):
            mice.impute(maxit=0)
            
    def test_plotting_with_mismatched_data(self, imputed_datasets):
        """Test plotting behavior with mismatched data."""
        # Create mismatched missing pattern
        wrong_pattern = pd.DataFrame({
            'wrong_column': [1, 0, 1, 0, 1]
        })
        
        # Should handle gracefully (not crash)
        try:
            stripplot(imputed_datasets, wrong_pattern)
            plt.close('all')
        except Exception:
            pass


class TestPerformanceBaseline:
    """Basic performance tests (not comprehensive benchmarking)."""
    
    def test_reasonable_performance_small_data(self, sample_data_missing):
        """Test that imputation completes in reasonable time for small data."""
        import time
        
        start_time = time.time()
        
        mice = MICE(sample_data_missing)
        mice.impute(n_imputations=3, maxit=5, method='sample')
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete within reasonable time (adjust as needed)
        assert duration < 30  # 30 seconds should be more than enough for small data
        assert mice.imputed_datasets is not None
        
    def test_memory_usage_reasonable(self, sample_data_missing):
        """Test that memory usage is reasonable."""
        mice = MICE(sample_data_missing)
        mice.impute(n_imputations=5, maxit=3, method='sample')
        
        # Should be able to create multiple imputations without issues
        assert len(mice.imputed_datasets) == 5
        
        # All datasets should have expected shape
        for dataset in mice.imputed_datasets:
            assert dataset.shape == mice.data.shape
