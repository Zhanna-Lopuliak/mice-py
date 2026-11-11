"""
Integration tests for end-to-end workflows.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from imputation import MICE
from plotting.diagnostics import densityplot, stripplot


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""

    def test_complete_mice_workflow(self, sample_data_missing, temp_dir):
        """Test complete MICE workflow from data to results."""
        # Step 1: Initialize MICE
        mice = MICE(sample_data_missing)

        # Step 2: Run imputation
        mice.impute(n_imputations=3, maxit=5, method="sample")

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
        mice.impute(n_imputations=2, maxit=3, method="sample")

        # Step 2: Create missing pattern for plotting
        missing_pattern = pd.DataFrame()
        for col in mice.data.columns:
            missing_pattern[col] = mice.id_obs[col].astype(int)

        # Step 3: Create diagnostic plots
        stripplot(
            imputed_datasets=mice.imputed_datasets,
            missing_pattern=missing_pattern,
            columns=["bmi"],
            save_path=str(temp_dir / "integration_stripplot.png"),
        )

        densityplot(
            imputed_datasets=mice.imputed_datasets,
            missing_pattern=missing_pattern,
            columns=["bmi"],
            save_path=str(temp_dir / "integration_densityplot.png"),
        )

        # Step 4: Verify plots were created
        assert (temp_dir / "integration_stripplot.png").exists()
        assert (temp_dir / "integration_densityplot.png").exists()

        plt.close("all")


class TestDataTypeHandling:
    """Test handling of different data types."""

    def test_mixed_data_types_workflow(self):
        """Test workflow with mixed data types."""
        # Create data with mixed types
        rs = np.random.RandomState(42)
        n = 50
        mixed_data = pd.DataFrame(
            {
                "numeric_int": rs.randint(1, 100, n),
                "numeric_float": rs.normal(50, 15, n),
                "categorical": rs.choice(["A", "B", "C"], n),
                "binary": rs.choice([0, 1], n),
            }
        )

        # Introduce missing values
        missing_idx = rs.choice(n, size=int(0.2 * n), replace=False)
        mixed_data.loc[missing_idx, "numeric_float"] = np.nan

        missing_idx = rs.choice(n, size=int(0.15 * n), replace=False)
        mixed_data.loc[missing_idx, "categorical"] = np.nan

        # Run MICE workflow
        mice = MICE(mixed_data)
        mice.impute(n_imputations=2, maxit=2, method="sample")

        # Verify results
        assert mice.imputed_datasets is not None
        for dataset in mice.imputed_datasets:
            assert not dataset.isna().any().any()


class TestErrorRecovery:
    """Test error handling and recovery in workflows."""

    def test_workflow_with_invalid_method(self, sample_data_missing):
        """Test workflow behavior with invalid method."""
        mice = MICE(sample_data_missing)

        with pytest.raises(ValueError):
            mice.impute(method="invalid_method")

    def test_workflow_with_invalid_parameters(self, sample_data_missing):
        """Test workflow behavior with invalid parameters."""
        mice = MICE(sample_data_missing)

        with pytest.raises(ValueError):
            mice.impute(n_imputations=0)

        with pytest.raises(ValueError):
            mice.impute(maxit=0)

    def test_plotting_with_mismatched_data(self, imputed_datasets):
        """A pattern column that the datasets lack fails loudly instead of plotting nonsense."""
        wrong_pattern = pd.DataFrame({"wrong_column": [1, 0, 1, 0, 1]})
        with pytest.raises(KeyError, match="wrong_column"):
            stripplot(imputed_datasets, wrong_pattern)


class TestPerformanceBaseline:
    """Basic performance tests (not comprehensive benchmarking)."""

    def test_reasonable_performance_small_data(self, sample_data_missing):
        """Test that imputation completes in reasonable time for small data."""
        import time

        start_time = time.time()

        mice = MICE(sample_data_missing)
        mice.impute(n_imputations=3, maxit=5, method="sample")

        end_time = time.time()
        duration = end_time - start_time

        # Should complete within reasonable time (adjust as needed)
        assert duration < 30  # 30 seconds should be more than enough for small data
        assert mice.imputed_datasets is not None
