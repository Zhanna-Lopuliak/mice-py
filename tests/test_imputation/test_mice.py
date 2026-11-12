"""
Tests for the core MICE functionality.
"""

import re

import numpy as np
import pandas as pd
import pytest

from imputation import MICE

from .._logged_events import impute_checking_log_warning


class TestMICEInitialization:
    """Test MICE object initialization."""

    def test_init_with_valid_data(self, sample_data_missing):
        """Test MICE initialization with valid data."""
        mice = MICE(sample_data_missing)

        assert isinstance(mice.data, pd.DataFrame)
        assert mice.data.shape == sample_data_missing.shape
        assert hasattr(mice, "id_obs")
        assert hasattr(mice, "id_mis")

    def test_init_with_invalid_data(self):
        """Test MICE initialization with invalid data."""
        with pytest.raises(ValueError, match="Input data cannot be converted to DataFrame"):
            MICE("invalid_data")

    def test_init_with_duplicate_columns(self):
        """Test MICE initialization with duplicate column names."""
        data = pd.DataFrame([[1, 2], [3, 4]], columns=["A", "A"])
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
        assert hasattr(mice, "imputed_datasets")
        assert len(mice.imputed_datasets) == 2

        for dataset in mice.imputed_datasets:
            assert not dataset.isna().any().any()

    def test_imputation_parameters(self, sample_data_missing):
        """Test imputation with different parameters."""
        mice = MICE(sample_data_missing)

        mice.impute(n_imputations=3, maxit=5, method="sample", initial="sample")

        assert len(mice.imputed_datasets) == 3

    def test_imputation_with_predictor_matrix(self, sample_data_missing, predictor_matrix):
        """Test imputation with custom predictor matrix."""
        data_subset = sample_data_missing[["age", "bmi", "cholesterol"]].copy()
        mice = MICE(data_subset)

        mice.impute(n_imputations=2, maxit=2, predictor_matrix=predictor_matrix)

        assert mice.imputed_datasets is not None
        assert len(mice.imputed_datasets) == 2


class TestMICEMethods:
    """Test different imputation methods."""

    @pytest.mark.parametrize("method", ["sample", "cart"])
    def test_methods_with_categorical_data(self, small_data_missing, method):
        """Test imputation methods that can handle categorical data."""
        mice = MICE(small_data_missing)

        # five rows: x2 is ten times x1, so it is not imputed (a logged event)
        impute_checking_log_warning(mice, n_imputations=2, maxit=2, method=method)
        assert len(mice.imputed_datasets) == 2
        assert "x2" in mice.logged_events["out"].tolist()

    @pytest.mark.parametrize("method", ["pmm", "midas", "norm", "norm.nob", "norm.boot", "norm.predict", "mean"])
    def test_methods_numeric_only(self, sample_data_missing, method):
        """Test imputation methods that require numeric-only data."""
        # Use only numeric columns from sample_data_missing
        numeric_data = sample_data_missing[["age", "bmi", "cholesterol"]].copy()
        mice = MICE(numeric_data)

        mice.impute(n_imputations=2, maxit=2, method=method)
        assert mice.imputed_datasets is not None
        assert len(mice.imputed_datasets) == 2

    @pytest.mark.parametrize("method", ["logreg", "logreg.boot", "polyreg", "polr"])
    def test_methods_for_two_categories(self, sample_data_missing, method):
        """smoker holds 0 and 1: logistic and multinomial regression impute 0 or 1
        (polr through its fallback to the multinomial model), with R's warning
        about a categorical method on a numeric column."""
        mice = MICE(sample_data_missing)

        mismatch = rf"Type mismatch for variable\(s\): smoker\nImputation method {re.escape(method)} is for categorical"
        with pytest.warns(UserWarning, match=mismatch):
            mice.impute(n_imputations=2, maxit=2, method={"smoker": method}, seed=1)
        assert mice.method["smoker"] == method
        missing = sample_data_missing["smoker"].isna()
        for dataset in mice.imputed_datasets:
            assert set(dataset.loc[missing, "smoker"]) <= {0.0, 1.0}

    @staticmethod
    def _two_level_ordered_frame():
        """An ordered column with two levels, which polr fits by its fallback, and a complete predictor."""
        rng = np.random.default_rng(3)
        x = rng.normal(size=40)
        grade = np.where(x + rng.normal(size=40) > 0, "hi", "lo")
        df = pd.DataFrame({"grade": pd.Categorical(grade, categories=["lo", "hi"], ordered=True), "x": x})
        df.loc[[1, 5, 9, 13, 20], "grade"] = np.nan
        return df

    def test_polr_fallback_is_logged_with_polr_log_fallback(self):
        """With ``polr_log_fallback=True`` every fallback of polr to multinom is
        a row of logged_events (R's polr.to.loggedEvents = TRUE), one per
        iteration and imputation; without it nothing is logged."""
        df = self._two_level_ordered_frame()
        mice = MICE(df)
        impute_checking_log_warning(
            mice, n_imputations=2, maxit=3, method={"grade": "polr"}, seed=1, polr_log_fallback=True
        )
        events = mice.logged_events
        assert sorted(zip(events["it"], events["im"])) == [(it, im) for it in (1, 2, 3) for im in (1, 2)]
        assert set(events["dep"]) == {"grade"}
        assert set(events["meth"]) == {"polr"}
        assert set(events["out"]) == {"polr falls back to multinom"}

        quiet = MICE(df)
        impute_checking_log_warning(quiet, n_imputations=2, maxit=3, method={"grade": "polr"}, seed=1)
        assert quiet.logged_events.empty
        for logged, unlogged in zip(mice.imputed_datasets, quiet.imputed_datasets):
            pd.testing.assert_frame_equal(logged, unlogged)

    def test_method_dictionary(self, sample_data_missing):
        """Test using method dictionary for different columns."""
        mice = MICE(sample_data_missing)

        method_dict = {"bmi": "sample", "cholesterol": "sample"}

        mice.impute(n_imputations=2, maxit=2, method=method_dict)

        assert mice.imputed_datasets is not None
