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
        pmm_func = get_imputer_func("pmm")
        assert callable(pmm_func)

        sample_func = get_imputer_func("sample")
        assert callable(sample_func)

        cart_func = get_imputer_func("cart")
        assert callable(cart_func)

        rf_func = get_imputer_func("rf")
        assert callable(rf_func)

        midas_func = get_imputer_func("midas")
        assert callable(midas_func)

    def test_get_imputer_func_invalid(self):
        """Test error with invalid imputer method."""
        with pytest.raises(ValueError, match="Unsupported or unimplemented imputation method"):
            get_imputer_func("invalid_method")


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

    def test_sample_with_rng(self):
        """Test sample imputation with a seeded generator for reproducibility."""
        y = np.array([1, 2, np.nan, 4, 5])
        id_obs = ~np.isnan(y)
        x = np.array([[1], [2], [3], [4], [5]])

        result1 = sample(y, id_obs, x, rng=np.random.default_rng(42))
        result2 = sample(y, id_obs, x, rng=np.random.default_rng(42))

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

    @staticmethod
    def _recording(monkeypatch, name):
        """Replace a tree class of imputation.cart by one that records its
        parameters and the data it is fitted on."""
        import imputation.cart

        seen = {}
        base = getattr(imputation.cart, name)

        class Recording(base):
            def fit(self, X, y, *args, **kwargs):
                seen.update(params=self.get_params(), X=np.array(X), y=np.array(y))
                fitted = super().fit(X, y, *args, **kwargs)
                seen["tree"] = self
                return fitted

        monkeypatch.setattr(imputation.cart, name, Recording)
        return seen

    @staticmethod
    def _problem(n=60, seed=0):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n, 2))
        y = x[:, 0] + rng.normal(scale=0.5, size=n)
        id_obs = np.ones(n, dtype=bool)
        id_obs[::5] = False
        y[~id_obs] = np.nan
        return y, id_obs, x

    def test_cart_numeric_tree_follows_rpart_defaults(self, monkeypatch):
        """minsplit = 3 * minbucket, maxdepth 30, cp relative to the population
        variance of y (rpart's root risk per observation), no bootstrap."""
        seen = self._recording(monkeypatch, "DecisionTreeRegressor")
        y, id_obs, x = self._problem()
        cart(y, id_obs, x, rng=np.random.default_rng(1))
        params = seen["params"]
        assert (params["min_samples_leaf"], params["min_samples_split"], params["max_depth"]) == (5, 15, 30)
        assert params["ccp_alpha"] == pytest.approx(1e-4 * np.var(y[id_obs]), rel=1e-12)
        np.testing.assert_array_equal(seen["X"], x[id_obs])
        np.testing.assert_array_equal(seen["y"], y[id_obs])

    def test_cart_settings_can_be_changed(self, monkeypatch):
        seen = self._recording(monkeypatch, "DecisionTreeRegressor")
        y, id_obs, x = self._problem()
        cart(y, id_obs, x, min_samples_leaf=2, cp=0.01, max_depth=4, rng=np.random.default_rng(1))
        params = seen["params"]
        assert (params["min_samples_leaf"], params["min_samples_split"], params["max_depth"]) == (2, 6, 4)
        assert params["ccp_alpha"] == pytest.approx(0.01 * np.var(y[id_obs]), rel=1e-12)
        cart(y, id_obs, x, min_samples_split=9, ccp_alpha=0.25, rng=np.random.default_rng(1))
        assert (seen["params"]["min_samples_split"], seen["params"]["ccp_alpha"]) == (9, 0.25)

    def test_cart_categorical_tree_is_not_pruned_unless_asked(self, monkeypatch):
        seen = self._recording(monkeypatch, "DecisionTreeClassifier")
        y, id_obs, x = self._problem()
        labels = np.where(np.isnan(y), None, np.where(y > 0, "hi", "lo")).astype(object)
        cart(labels, id_obs, x, rng=np.random.default_rng(1))
        assert seen["params"]["ccp_alpha"] == 0.0
        assert (seen["params"]["min_samples_split"], seen["params"]["max_depth"]) == (15, 30)
        cart(labels, id_obs, x, ccp_alpha=0.01, rng=np.random.default_rng(1))
        assert seen["params"]["ccp_alpha"] == 0.01

    def test_cart_donors_are_the_observed_values_in_the_leaf(self, monkeypatch):
        seen = self._recording(monkeypatch, "DecisionTreeRegressor")
        y, id_obs, x = self._problem()
        rng = np.random.default_rng(3)
        for _ in range(20):
            out = cart(y, id_obs, x, rng=rng)
            tree = seen["tree"]
            leaf_obs, leaf_mis = tree.apply(x[id_obs]), tree.apply(x[~id_obs])
            for value, leaf in zip(out, leaf_mis):
                assert value in y[id_obs][leaf_obs == leaf]

    def test_cart_bootstrap_resamples_the_observed_rows(self, monkeypatch):
        """bootstrap=True (mice-py 0.1): the tree is fitted on a bootstrap
        sample; the donors are still observed values."""
        seen = self._recording(monkeypatch, "DecisionTreeRegressor")
        y, id_obs, x = self._problem()
        out = cart(y, id_obs, x, bootstrap=True, rng=np.random.default_rng(1))
        assert seen["X"].shape == x[id_obs].shape
        assert len(np.unique(seen["y"])) < id_obs.sum()
        assert set(seen["y"]) <= set(y[id_obs])
        assert np.isin(out, y[id_obs]).all()


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

        result = rf(y, id_obs, x, n_estimators=3, rng=np.random.default_rng(42))
        assert len(result) == 1
        assert not np.isnan(result[0])
        again = rf(y, id_obs, x, n_estimators=3, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(result, again)

    @staticmethod
    def _recording(monkeypatch, name):
        """Replace a forest class of imputation.rf by one that records itself."""
        import imputation.rf

        seen = {}
        base = getattr(imputation.rf, name)

        class Recording(base):
            def fit(self, *args, **kwargs):
                fitted = super().fit(*args, **kwargs)
                seen.update(params=self.get_params(), forest=self)
                return fitted

        monkeypatch.setattr(imputation.rf, name, Recording)
        return seen

    @staticmethod
    def _problem(n=60, p=5, seed=0):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n, p))
        y = x[:, 0] + rng.normal(scale=0.5, size=n)
        id_obs = np.ones(n, dtype=bool)
        id_obs[::5] = False
        y[~id_obs] = np.nan
        return y, id_obs, x

    def test_rf_regression_follows_ranger_defaults(self, monkeypatch):
        """10 trees on bootstrap samples, mtry floor(sqrt(p)), no process pool."""
        seen = self._recording(monkeypatch, "RandomForestRegressor")
        y, id_obs, x = self._problem()
        rf(y, id_obs, x, rng=np.random.default_rng(1))
        params = seen["params"]
        assert params["n_estimators"] == 10 and params["bootstrap"] is True
        assert (params["min_samples_split"], params["min_samples_leaf"]) == (5, 1)
        assert params["max_depth"] is None and params["max_features"] == "sqrt"
        assert params["n_jobs"] is None
        assert all(tree.max_features_ == 2 for tree in seen["forest"].estimators_)  # floor(sqrt(5))

    def test_rf_classification_follows_ranger_defaults(self, monkeypatch):
        seen = self._recording(monkeypatch, "RandomForestClassifier")
        y, id_obs, x = self._problem()
        labels = np.where(np.isnan(y), None, np.where(y > 0, "hi", "lo")).astype(object)
        out = rf(labels, id_obs, x, rng=np.random.default_rng(1))
        params = seen["params"]
        assert (params["min_samples_split"], params["min_samples_leaf"]) == (2, 1)
        assert params["max_features"] == "sqrt" and params["n_jobs"] is None
        assert set(out) <= {"hi", "lo"}

    def test_rf_forest_params_reach_the_forest(self, monkeypatch):
        seen = self._recording(monkeypatch, "RandomForestRegressor")
        y, id_obs, x = self._problem()
        rf(
            y,
            id_obs,
            x,
            n_estimators=4,
            min_samples_split=7,
            forest_params={"max_features": 1.0, "n_jobs": 2},
            rng=np.random.default_rng(1),
        )
        params = seen["params"]
        assert (params["n_estimators"], params["min_samples_split"]) == (4, 7)
        assert (params["max_features"], params["n_jobs"]) == (1.0, 2)

    def test_rf_donors_are_pooled_over_all_trees(self, monkeypatch):
        """Every imputation is an observed value that shares the missing row's
        leaf in at least one tree."""
        seen = self._recording(monkeypatch, "RandomForestRegressor")
        y, id_obs, x = self._problem()
        rng = np.random.default_rng(4)
        for _ in range(10):
            out = rf(y, id_obs, x, rng=rng)
            forest = seen["forest"]
            leaves_obs, leaves_mis = forest.apply(x[id_obs]), forest.apply(x[~id_obs])
            for value, leaves in zip(out, leaves_mis):
                pool = y[id_obs][(leaves_obs == leaves).any(axis=1)]
                assert value in pool


class TestMIDASImputation:
    """Test MIDAS imputation method."""

    def test_midas_basic(self):
        """Test basic MIDAS imputation."""
        np.random.seed(42)
        y = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0])
        id_obs = ~np.isnan(y)
        x = np.random.randn(8, 2)

        result = midas(y, id_obs, x, rng=np.random.default_rng(1))
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

        result = midas(y, id_obs, x, ridge=1e-4, kappa=3.0, rng=np.random.default_rng(2))
        assert len(result) == 1
        assert not np.isnan(result[0])
        assert result[0] in y[id_obs]

    def test_midas_outout_parameter(self):
        """Test MIDAS with outout parameter."""
        np.random.seed(42)
        y = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        id_obs = ~np.isnan(y)
        x = np.random.randn(10, 2)

        result = midas(y, id_obs, x, outout=False, rng=np.random.default_rng(3))
        assert len(result) == 1
        assert not np.isnan(result[0])
        assert result[0] in y[id_obs]

    def test_midas_numeric_validation(self):
        """Test MIDAS validation for numeric data."""
        y = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        id_obs = ~np.isnan(y)

        # Test with non-numeric predictors (should fail)
        x_non_numeric = np.array([["a", "b"], ["c", "d"], ["e", "f"], ["g", "h"], ["i", "j"]])

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
        y = np.array(["a", "b", np.nan, "d", "e"])
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

        result = midas(y, id_obs, x, rng=np.random.default_rng(4))
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

        methods = ["sample", "pmm", "cart", "rf", "midas"]

        for method_name in methods:
            method_func = get_imputer_func(method_name)
            result = method_func(y, id_obs, x)

            assert isinstance(result, np.ndarray)
            assert len(result) == (~id_obs).sum()
            assert not np.isnan(result).any()
