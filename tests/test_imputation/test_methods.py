"""
Tests for individual imputation methods.
"""

import numpy as np
import pandas as pd
import pytest

from imputation.cart import cart
from imputation.midas import midas
from imputation.PMM import pmm
from imputation.rf import rf
from imputation.sample import sample
from imputation.utils import get_imputer_func


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

        result = sample(y, id_obs, x, rng=np.random.default_rng(7))

        assert len(result) == 1
        assert result[0] in [1, 2, 4, 5]

    def test_sample_multiple_missing(self):
        """Test sample imputation with multiple missing values."""
        y = np.array([1, np.nan, 3, np.nan, 5])
        id_obs = ~np.isnan(y)
        x = np.array([[1], [2], [3], [4], [5]])

        result = sample(y, id_obs, x, rng=np.random.default_rng(8))

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
        rs = np.random.RandomState(42)
        # Use larger dataset with more variation to avoid numerical issues
        n = 20
        y = rs.normal(10, 3, n)
        y[5] = np.nan
        y[12] = np.nan
        id_obs = ~np.isnan(y)
        x = rs.randn(n, 2)

        result = pmm(y, id_obs, x, donors=3, rng=np.random.default_rng(1))

        # Test outcomes, not internal logic
        assert len(result) == 2  # Should impute 2 missing values
        assert not np.isnan(result).any()  # No NaN in results
        # Key test: imputed values should be from observed donor pool
        observed_values = y[id_obs]
        for imputed_val in result:
            assert imputed_val in observed_values, f"Imputed value {imputed_val} not in donor pool"

    def test_pmm_parameters(self):
        """Test PMM with different parameters."""
        rs = np.random.RandomState(42)
        # Use robust dataset
        n = 15
        y = rs.normal(5, 2, n)
        y[3] = np.nan
        id_obs = ~np.isnan(y)
        x = rs.randn(n, 2)

        result = pmm(y, id_obs, x, donors=2, matchtype=0, rng=np.random.default_rng(2))
        assert len(result) == 1
        assert not np.isnan(result[0])
        observed_values = y[id_obs]
        assert result[0] in observed_values


class TestCARTImputation:
    """Test CART imputation method."""

    def test_cart_basic(self):
        """Test basic CART imputation."""
        rs = np.random.RandomState(42)
        y = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0])
        id_obs = ~np.isnan(y)
        x = rs.randn(8, 2)

        result = cart(y, id_obs, x, min_samples_leaf=2, rng=np.random.default_rng(3))
        assert len(result) == 1
        assert not np.isnan(result[0])

    def test_cart_parameters(self):
        """Test CART with different parameters."""
        rs = np.random.RandomState(42)
        y = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        id_obs = ~np.isnan(y)
        x = rs.randn(10, 3)

        result = cart(y, id_obs, x, min_samples_leaf=1, ccp_alpha=0.0, rng=np.random.default_rng(4))
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
        rs = np.random.RandomState(42)
        y = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0])
        id_obs = ~np.isnan(y)
        x = rs.randn(8, 2)

        result = rf(y, id_obs, x, n_estimators=5, rng=np.random.default_rng(5))
        assert len(result) == 1
        assert not np.isnan(result[0])

    def test_rf_parameters(self):
        """Test RF with different parameters."""
        rs = np.random.RandomState(42)
        y = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        id_obs = ~np.isnan(y)
        x = rs.randn(10, 3)

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
        rs = np.random.RandomState(42)
        y = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0])
        id_obs = ~np.isnan(y)
        x = rs.randn(8, 2)

        result = midas(y, id_obs, x, rng=np.random.default_rng(1))
        assert len(result) == 1
        assert not np.isnan(result[0])
        # Should be one of the observed values (MIDAS samples from donors)
        assert result[0] in y[id_obs]

    def test_midas_parameters(self):
        """Test MIDAS with different parameters."""
        rs = np.random.RandomState(42)
        y = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        id_obs = ~np.isnan(y)
        x = rs.randn(10, 3)

        result = midas(y, id_obs, x, ridge=1e-4, kappa=3.0, rng=np.random.default_rng(2))
        assert len(result) == 1
        assert not np.isnan(result[0])
        assert result[0] in y[id_obs]

    def test_midas_outout_parameter(self):
        """Test MIDAS with outout parameter."""
        rs = np.random.RandomState(42)
        y = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        id_obs = ~np.isnan(y)
        x = rs.randn(10, 2)

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
        x = np.random.default_rng(0).normal(size=(5, 2))

        with pytest.raises(ValueError, match="Target y must be numeric for midas"):
            midas(y, id_obs, x)

    def test_midas_multiple_missing(self):
        """Test MIDAS with multiple missing values."""
        rs = np.random.RandomState(42)
        y = np.array([1.0, np.nan, 3.0, np.nan, 5.0, 6.0, 7.0, 8.0])
        id_obs = ~np.isnan(y)
        x = rs.randn(8, 2)

        result = midas(y, id_obs, x, rng=np.random.default_rng(4))
        assert len(result) == 2
        assert not np.isnan(result).any()
        for val in result:
            assert val in y[id_obs]


class TestNormImputation:
    """Input handling of norm, norm.nob, norm.boot, norm.predict and mean."""

    REGRESSIONS = ["norm", "norm.nob", "norm.boot", "norm.predict"]

    @staticmethod
    def _problem(n=12):
        x = np.random.default_rng(0).normal(size=(n, 2))
        y = 1.0 + x @ np.array([2.0, -1.0]) + 0.1 * np.cos(np.arange(n))
        id_obs = np.arange(n) % 4 != 0
        y[~id_obs] = np.nan
        return y, id_obs, x

    @pytest.mark.parametrize("method", REGRESSIONS)
    def test_invalid_predictors(self, method):
        imputer = get_imputer_func(method)
        y, id_obs, x = self._problem()
        with pytest.raises(ValueError, match=f"Predictors must be numeric for {method}"):
            imputer(y, id_obs, np.full((12, 2), "a"))
        x_nan = x.copy()
        x_nan[3, 1] = np.nan
        with pytest.raises(ValueError, match=f"Predictors must not contain missing values for {method}"):
            imputer(y, id_obs, x_nan)
        frame = pd.DataFrame(x_nan, columns=["u", "v"])
        with pytest.raises(ValueError, match=r"Columns with missing values: \['v'\]"):
            imputer(y, id_obs, frame)
        with pytest.raises(ValueError, match="x has 11 rows, but y has 12 entries"):
            imputer(y, id_obs, x[:11])

    @pytest.mark.parametrize("method", [*REGRESSIONS, "mean"])
    def test_nullable_and_complex_targets(self, method):
        """A nullable numeric target works as its float version; complex values are refused."""
        imputer = get_imputer_func(method)
        y, id_obs, x = self._problem()
        nullable = pd.array(np.where(id_obs, y, np.nan), dtype="Float64")
        expected = imputer(y, id_obs, x, rng=np.random.default_rng(1))
        np.testing.assert_array_equal(imputer(nullable, id_obs, x, rng=np.random.default_rng(1)), expected)
        with pytest.raises(ValueError, match=f"numeric for {method}, got complex128 values"):
            imputer(y.astype(complex), id_obs, x, rng=np.random.default_rng(1))

    def test_norm_predict_is_the_least_squares_prediction(self):
        y, id_obs, x = self._problem()
        design = np.c_[np.ones(12), x]
        coef = np.linalg.lstsq(design[id_obs], y[id_obs], rcond=None)[0]
        predict = get_imputer_func("norm.predict")
        np.testing.assert_allclose(predict(y, id_obs, x), design[~id_obs] @ coef)
        # a 1-D x is one predictor
        np.testing.assert_array_equal(predict(y, id_obs, x[:, 0]), predict(y, id_obs, x[:, :1]))

    def test_aliased_predictor_gets_coefficient_zero(self):
        """A duplicated column is aliased in the QR fit; R imputes NA, the port
        sets its coefficient to 0, which leaves the prediction unchanged.
        (norm.nob and norm.boot still count it in their residual df, as R.)"""
        y, id_obs, x = self._problem()
        with_copy = np.c_[x, x[:, 0]]
        predict = get_imputer_func("norm.predict")
        np.testing.assert_allclose(predict(y, id_obs, with_copy), predict(y, id_obs, x), rtol=1e-10)
        for method in ("norm.nob", "norm.boot"):
            out = get_imputer_func(method)(y, id_obs, with_copy, rng=np.random.default_rng(3))
            assert np.isfinite(out).all()


class TestLogregImputation:
    """Input handling and building blocks of logreg and logreg.boot."""

    METHODS = ["logreg", "logreg.boot"]

    @staticmethod
    def _problem(n=16):
        x = np.random.default_rng(0).normal(size=(n, 2))
        y = np.where(x[:, 0] + np.cos(np.arange(n)) > 0, "b", "a").astype(object)
        id_obs = np.arange(n) % 4 != 0
        y[~id_obs] = None
        return y, id_obs, x

    def test_uniform_equal_to_the_probability_gives_the_second_category(self):
        """R's ``runif(n) <= p``: equality draws the second category."""
        from imputation.logreg import _draw_codes

        p = 0.3
        u = np.array([p, p, p, 0.0, 1.0])
        prob = np.array([p, np.nextafter(p, 0), np.nextafter(p, 1), 0.0, 1.0])
        np.testing.assert_array_equal(_draw_codes(u, prob), [1, 0, 1, 1, 1])

    @pytest.mark.parametrize(
        "y, levels, codes",
        [
            (np.array([True, None, False, True], dtype=object), [False, True], [1, -1, 0, 1]),
            (np.array([2.0, np.nan, 0.5, 2.0]), [0.5, 2.0], [1, -1, 0, 1]),
            (np.array(["yes", None, "no", "yes"], dtype=object), ["no", "yes"], [1, -1, 0, 1]),
            (pd.array([True, None, False, True], dtype="boolean"), [False, True], [1, -1, 0, 1]),
            (pd.array(["yes", None, "no", "yes"], dtype="string"), ["no", "yes"], [1, -1, 0, 1]),
            # unordered: sorted labels present, the unused "c" dropped
            (pd.Categorical(["b", None, "a", "b"], categories=["c", "b", "a"]), ["a", "b"], [1, -1, 0, 1]),
            # ordered: declared order of the levels present
            (
                pd.Categorical(["b", None, "a", "b"], categories=["c", "b", "a"], ordered=True),
                ["b", "a"],
                [0, -1, 1, 0],
            ),
        ],
    )
    def test_target_categories(self, y, levels, codes):
        from imputation.logreg import target_categories

        categories = target_categories(y)
        assert categories.levels == levels
        np.testing.assert_array_equal(categories.codes, codes)
        assert categories.labels.dtype == np.asarray(y).dtype
        assert categories.labels.tolist() == levels

    def test_target_categories_keeps_unused_ordered_levels_on_request(self):
        from imputation.logreg import target_categories

        y = pd.Categorical(["b", "a"], categories=["c", "b", "a"], ordered=True)
        categories = target_categories(y, keep_unused=True)
        assert categories.levels == ["c", "b", "a"]
        np.testing.assert_array_equal(categories.codes, [1, 2])
        unordered = pd.Categorical(["b", "a"], categories=["c", "b", "a"])
        assert target_categories(unordered, keep_unused=True).levels == ["a", "b"]

    def test_values_of_mixed_types_raise(self):
        from imputation.logreg import target_categories

        with pytest.raises(TypeError, match="mixed types"):
            target_categories(np.array(["a", 1, None], dtype=object))

    @pytest.mark.parametrize("method", METHODS)
    def test_invalid_inputs(self, method):
        imputer = get_imputer_func(method)
        y, id_obs, x = self._problem()
        with pytest.raises(ValueError, match=f"Predictors must be numeric for {method}"):
            imputer(y, id_obs, np.full((16, 2), "a"))
        x_nan = x.copy()
        x_nan[3, 1] = np.nan
        with pytest.raises(ValueError, match=f"Predictors must not contain missing values for {method}"):
            imputer(y, id_obs, x_nan)
        with pytest.raises(ValueError, match=f"{method}: y is missing in entries that id_obs marks as observed"):
            imputer(y, np.ones(16, dtype=bool), x)
        with pytest.raises(ValueError, match=r"two categories, but y has 16 \(0\.0, 1\.0, 2\.0, 3\.0, 4\.0, \.\.\.\)"):
            imputer(np.arange(16.0), id_obs, x)

    @pytest.mark.parametrize("method", METHODS)
    def test_x_as_data_frame_with_a_categorical_column(self, method):
        """A DataFrame predictor is dummy-coded as MICE does."""
        y, id_obs, x = self._problem()
        frame = pd.DataFrame({"u": x[:, 0], "g": np.where(x[:, 1] > 0, "p", "q")})
        coded = np.c_[x[:, 0], (x[:, 1] <= 0).astype(float)]
        first = get_imputer_func(method)(y, id_obs, frame, rng=np.random.default_rng(4))
        second = get_imputer_func(method)(y, id_obs, coded, rng=np.random.default_rng(4))
        np.testing.assert_array_equal(first, second)

    def test_logit_fit_rejects_invalid_input(self):
        from imputation.fitting import logit_fit

        x = np.c_[np.ones(4), [0.0, 1.0, 2.0, 3.0]]
        with pytest.raises(ValueError, match=r"y values must be 0 <= y <= 1"):
            logit_fit(x, [0, 1, 2, 1])
        with pytest.raises(ValueError, match="non-negative"):
            logit_fit(x, [0, 1, 0, 1], weights=[1, -1, 1, 1])
        with pytest.raises(ValueError, match="x has 4 rows, y 3"):
            logit_fit(x, [0, 1, 0])
        with pytest.raises(ValueError, match="maxit must be at least 1"):
            logit_fit(x, [0, 1, 0, 1], maxit=0)

    def test_logit_fit_leaves_out_rows_with_weight_zero(self):
        from imputation.fitting import logit_fit

        x = np.c_[np.ones(7), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]
        y = np.array([0, 0, 1, 0, 1, 1, 1.0])
        weighted = logit_fit(x, y, weights=[1, 1, 1, 1, 1, 1, 0])
        plain = logit_fit(x[:6], y[:6])
        np.testing.assert_allclose(weighted.coef, plain.coef, rtol=1e-12)
        np.testing.assert_allclose(weighted.cov_unscaled, plain.cov_unscaled, rtol=1e-12)

    def test_one_dimensional_x_and_default_generator(self):
        """A 1-D x is one column; without `rng` a fresh generator is used."""
        from imputation.fitting import logit_fit
        from imputation.logreg import augment

        fit = logit_fit(np.ones(4), [0, 1, 1, 0])  # intercept only: logit(1/2) = 0
        assert fit.coef.shape == (1,) and abs(fit.coef[0]) < 1e-12
        y, id_obs, x = self._problem()
        codes = np.where(y == "b", 1, np.where(y == "a", 0, -1))
        np.testing.assert_array_equal(
            augment(codes, id_obs, x[:, 0], ~id_obs).x, augment(codes, id_obs, x[:, :1], ~id_obs).x
        )
        for method in self.METHODS:
            assert set(get_imputer_func(method)(y, id_obs, x)) <= {"a", "b"}

    def test_logit_fit_of_separated_data_stops_at_maxit(self):
        """As R's glm.fit: no error, ``converged`` False, large coefficients."""
        from imputation.fitting import logit_fit

        x = np.c_[np.ones(6), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]]
        fit = logit_fit(x, [0, 0, 0, 1, 1, 1], maxit=10)
        assert (fit.iterations, fit.converged) == (10, False)
        assert fit.coef[1] > 5
        assert not fit.aliased.any()


class TestPolyregImputation:
    """Input handling and building blocks of polyreg and multinom_fit."""

    @staticmethod
    def _problem(n=24):
        x = np.random.default_rng(0).normal(size=(n, 2))
        score = x[:, 0] + np.cos(np.arange(n))
        y = np.where(score > 0.6, "c", np.where(score > -0.6, "b", "a")).astype(object)
        id_obs = np.arange(n) % 4 != 0
        y[~id_obs] = None
        return y, id_obs, x

    def test_uniform_equal_to_a_cumulative_probability_stays_in_the_lower_category(self):
        """R's ``un > cumsum(p)``: a number equal to a cumulative probability
        does not exceed it; one just above it does."""
        from imputation.polyreg import _draw_categories

        prob = np.tile([0.25, 0.5, 0.25], (6, 1))
        u = np.array([0.25, np.nextafter(0.25, 1), 0.75, np.nextafter(0.75, 1), 0.0, np.nextafter(0.25, 0)])
        np.testing.assert_array_equal(_draw_categories(u, prob), [0, 1, 1, 2, 0, 0])

    def test_uniform_above_the_last_cumulative_probability_gives_the_last_category(self):
        """The probabilities may sum to slightly less than 1; R imputes NA
        above the sum, the port the last category with a positive probability."""
        from imputation.polyreg import _draw_categories

        prob = np.array([[0.1, 0.2, 0.7 - 1e-12], [0.5, 0.5 - 1e-12, 0.0]])
        assert np.cumsum(prob, axis=1)[:, -1].max() < 1 - 1e-13
        np.testing.assert_array_equal(_draw_categories([1 - 1e-13, 1 - 1e-13], prob), [2, 1])

    def test_zero_probability_category_is_never_drawn(self):
        from imputation.polyreg import _draw_categories

        prob = np.array([[0.4, 0.0, 0.6]] * 4)
        np.testing.assert_array_equal(_draw_categories([0.1, 0.4, np.nextafter(0.4, 1), 0.99], prob), [0, 0, 2, 2])

    def test_invalid_inputs(self):
        from imputation.polyreg import polyreg

        y, id_obs, x = self._problem()
        with pytest.raises(ValueError, match="Predictors must be numeric for polyreg"):
            polyreg(y, id_obs, np.full((24, 2), "a"))
        x_nan = x.copy()
        x_nan[3, 1] = np.nan
        with pytest.raises(ValueError, match="Predictors must not contain missing values for polyreg"):
            polyreg(y, id_obs, x_nan)
        with pytest.raises(ValueError, match="polyreg: y is missing in entries that id_obs marks as observed"):
            polyreg(y, np.ones(24, dtype=bool), x)

    def test_x_as_data_frame_and_default_generator(self):
        """A DataFrame predictor is dummy-coded as MICE does; without `rng` a fresh generator is used."""
        from imputation.polyreg import polyreg

        y, id_obs, x = self._problem()
        frame = pd.DataFrame({"u": x[:, 0], "g": np.where(x[:, 1] > 0, "p", "q")})
        coded = np.c_[x[:, 0], (x[:, 1] <= 0).astype(float)]
        first = polyreg(y, id_obs, frame, rng=np.random.default_rng(4))
        np.testing.assert_array_equal(first, polyreg(y, id_obs, coded, rng=np.random.default_rng(4)))
        assert set(polyreg(y, id_obs, x)) <= {"a", "b", "c"}

    def test_parameters_reach_the_fit(self, monkeypatch):
        import imputation.polyreg

        seen = []
        original = imputation.polyreg.multinom_fit

        def recording(*args, **kwargs):
            seen.append((kwargs["maxit"], kwargs["max_weights"]))
            return original(*args, **kwargs)

        monkeypatch.setattr(imputation.polyreg, "multinom_fit", recording)
        y, id_obs, x = self._problem()
        imputation.polyreg.polyreg(y, id_obs, x, maxit=7, max_weights=50, rng=np.random.default_rng(1))
        assert seen == [(7, 50)]

    def test_multinom_fit_rejects_invalid_input(self):
        from imputation.fitting import multinom_fit

        x = np.c_[np.ones(6), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]]
        y = [0, 1, 2, 0, 1, 2]
        with pytest.raises(ValueError, match="x has 6 rows, y 5"):
            multinom_fit(x, y[:5])
        with pytest.raises(ValueError, match="missing values in 'x'"):
            multinom_fit(np.where(x == 3.0, np.nan, x), y)
        with pytest.raises(ValueError, match="invalid weights vector"):
            multinom_fit(x, y, weights=[1, 1, -1, 1, 1, 1])
        with pytest.raises(ValueError, match="integers from 0 to 1"):
            multinom_fit(x, y, n_classes=2)
        with pytest.raises(ValueError, match="integers from 0 to 2"):
            multinom_fit(x, [0, 1, 2, 0, 1.5, 2])
        with pytest.raises(ValueError, match="need two or more classes"):
            multinom_fit(x, [1] * 6, n_classes=3)
        # 3 classes and 2 columns: 3 * (2 + 1) = 9 weights; 2 classes: 2 + 1
        with pytest.raises(ValueError, match=r"too many \(9\) weights.*max_weights=8"):
            multinom_fit(x, y, max_weights=8)
        assert multinom_fit(x, [0, 1, 1, 0, 1, 0], max_weights=3).logistic

    def test_multinom_fit_with_one_dimensional_x_and_an_empty_class(self):
        """A 1-D x is one column; a class without rows is not fitted and has probability 0."""
        from imputation.fitting import multinom_fit

        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        fit = multinom_fit(x, [0, 0, 3, 0, 3, 3], n_classes=4)
        assert fit.classes.tolist() == [0, 3] and fit.logistic and fit.coef.shape == (1, 1)
        prob = fit.predict(x)
        assert prob.shape == (6, 4) and np.all(prob[:, 1:3] == 0)
        np.testing.assert_allclose(prob.sum(axis=1), 1.0, rtol=1e-15)

    def test_softmax_probability_of_zero_costs_1000_times_the_weight(self):
        """nnet adds ``1000 w`` where the probability of the observed class underflows to 0."""
        from imputation.fitting import _softmax_objective

        x = np.c_[np.ones(3), [-1.0, 0.0, 1.0]]
        fn, gr = _softmax_objective(x, np.array([0, 1, 2]), np.array([2.0, 1.0, 0.5]), 3)
        beta = np.array([0.0, -800.0, 0.0, 800.0])  # eta = (0, -800 x, 800 x)
        # row 1 (class 1): p = exp(-800) / (...) = 0; row 2 (class 2): p = 1/3; row 3: p = 1
        assert fn(beta) == pytest.approx(2.0 * 1000 + 1.0 * np.log(3) + 0.0, rel=1e-15)
        assert np.isfinite(gr(beta)).all()

    def test_nnet_sigmoid_is_cut_off_beyond_15(self):
        from imputation.fitting import _nnet_sigmoid

        out = _nnet_sigmoid(np.array([-800.0, -15.000001, -15.0, 0.0, 15.0, 15.000001]))
        assert out[0] == 0.0 and out[1] == 0.0 and out[-1] == 1.0
        np.testing.assert_allclose(out[2:5], [1 / (1 + np.exp(15)), 0.5, 1 / (1 + np.exp(-15))], rtol=1e-15)
        assert 0 < out[2] and out[4] < 1


class TestPolrImputation:
    """Input handling and building blocks of polr and polr_fit."""

    @staticmethod
    def _problem(n=30):
        x = np.random.default_rng(0).normal(size=(n, 2))
        score = x[:, 0] - 0.5 * x[:, 1] + np.cos(np.arange(n))
        labels = np.array(["low", "mid", "high", "top"])[np.digitize(score, [-0.8, 0.0, 0.8])]
        id_obs = np.arange(n) % 4 != 0
        y = pd.Categorical(np.where(id_obs, labels, None), categories=["low", "mid", "high", "top"], ordered=True)
        return y, id_obs, x

    def test_invalid_inputs(self):
        from imputation.polr import polr

        y, id_obs, x = self._problem()
        with pytest.raises(ValueError, match="Predictors must be numeric for polr"):
            polr(y, id_obs, np.full((30, 2), "a"))
        x_nan = x.copy()
        x_nan[3, 1] = np.nan
        with pytest.raises(ValueError, match="Predictors must not contain missing values for polr"):
            polr(y, id_obs, x_nan)
        with pytest.raises(ValueError, match="polr: y is missing in entries that id_obs marks as observed"):
            polr(y, np.ones(30, dtype=bool), x)

    def test_without_predictors_the_cut_points_alone_are_fitted(self):
        """R stops (both of its fits see no response); the port fits the
        model without predictors, and multinom on the intercept where polr
        does not apply."""
        from imputation.fitting import MultinomFit, PolrFit
        from imputation.polr import _polr

        y, id_obs, _ = self._problem()
        for x in (None, np.empty((30, 0)), pd.DataFrame(index=range(30))):
            steps = _polr(y, id_obs, x, None, 100, 1500, False, np.random.default_rng(1), None)
            assert isinstance(steps.fit, PolrFit) and steps.fit.coef.shape == (0,)
            assert steps.prob.shape == (8, 4) and np.all(steps.prob == steps.prob[0])
            assert set(steps.labels) <= {"low", "mid", "high", "top"}
        two = np.where(id_obs, np.where(np.asarray(y) == "low", "low", "high"), None)
        steps = _polr(two, id_obs, None, None, 100, 1500, False, np.random.default_rng(1), None)
        assert isinstance(steps.fit, MultinomFit) and steps.fit.coef.shape == (1, 1)
        assert set(steps.labels) <= {"low", "high"}

    def test_x_as_data_frame_and_default_generator(self):
        """A DataFrame predictor is dummy-coded as MICE does; without `rng` a fresh generator is used."""
        from imputation.polr import polr

        y, id_obs, x = self._problem()
        frame = pd.DataFrame({"u": x[:, 0], "g": np.where(x[:, 1] > 0, "p", "q")})
        coded = np.c_[x[:, 0], (x[:, 1] <= 0).astype(float)]
        first = polr(y, id_obs, frame, rng=np.random.default_rng(4))
        np.testing.assert_array_equal(first, polr(y, id_obs, coded, rng=np.random.default_rng(4)))
        assert set(polr(y, id_obs, x)) <= {"low", "mid", "high", "top"}

    def test_nothing_observed_and_nothing_to_impute(self):
        from imputation.polr import polr

        y, _, x = self._problem()
        none = np.zeros(30, dtype=bool)
        assert polr(y, none, x, id_mis=none).shape == (0,)
        with pytest.raises(ValueError, match="polr: the target has no observed value"):
            polr(y, none, x)

    def test_fallback_parameters_reach_multinom_fit_only(self, monkeypatch):
        """``maxit`` and ``max_weights`` are R's nnet.maxit and nnet.MaxNWts: they
        reach the fallback; the proportional-odds fit keeps R's 100 iterations."""
        import imputation.polr

        seen = []
        original_multinom, original_polr = imputation.polr.multinom_fit, imputation.polr.polr_fit

        def recording_multinom(*args, **kwargs):
            seen.append(("multinom", kwargs["maxit"], kwargs["max_weights"]))
            return original_multinom(*args, **kwargs)

        def recording_polr(*args, **kwargs):
            seen.append(("polr", sorted(kwargs)))
            return original_polr(*args, **kwargs)

        monkeypatch.setattr(imputation.polr, "multinom_fit", recording_multinom)
        monkeypatch.setattr(imputation.polr, "polr_fit", recording_polr)
        y, id_obs, x = self._problem()
        imputation.polr.polr(y, id_obs, x, maxit=7, max_weights=50, rng=np.random.default_rng(1))
        numeric = np.where(id_obs, y.codes.astype(float), np.nan)
        imputation.polr.polr(numeric, id_obs, x, maxit=7, max_weights=50, rng=np.random.default_rng(1))
        assert seen == [("polr", ["n_levels", "weights"]), ("multinom", 7, 50)]

    def test_targets_that_are_no_factor_in_r(self):
        """Booleans and numbers go to the fallback (R: not a factor); text and Categoricals do not."""
        from imputation.polr import _is_factor

        assert _is_factor(pd.Categorical(["a", "b"]))
        assert _is_factor(np.array(["a", None, "b"], dtype=object))
        assert not _is_factor(np.array([True, None, False], dtype=object))
        assert not _is_factor(np.array([1.0, np.nan, 2.0]))
        assert not _is_factor(np.array([True, False]))

    def test_polr_fit_rejects_invalid_input(self):
        from imputation.fitting import PolrError, polr_fit

        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        y = [0, 1, 2, 0, 1, 2]
        with pytest.raises(ValueError, match="x has 6 rows, y 5"):
            polr_fit(x, y[:5])
        with pytest.raises(ValueError, match="missing values in 'x'"):
            polr_fit(np.where(x == 3.0, np.nan, x), y)
        with pytest.raises(ValueError, match="weights must be non-negative"):
            polr_fit(x, y, weights=[1, 1, -1, 1, 1, 1])
        with pytest.raises(ValueError, match="integers from 0 to 1"):
            polr_fit(x, y, n_levels=2)
        with pytest.raises(ValueError, match="integers from 0 to 2"):
            polr_fit(x, [0, 1, 2, 0, 1.5, 2])
        with pytest.raises(PolrError, match="response must have 3 or more levels") as info:
            polr_fit(x, [0, 1, 1, 0, 1, 0])
        assert info.value.condition == PolrError.FEW_LEVELS

    def test_polr_gradient_is_that_of_the_objective(self):
        """The analytic gradient of polr's objective agrees with central differences."""
        from imputation.fitting import _polr_objective

        rng = np.random.default_rng(2)
        x = rng.normal(size=(25, 2))
        codes = np.digitize(x @ [1.0, -0.7] + rng.logistic(size=25), [-1.0, 0.0, 1.0])
        fn, gr = _polr_objective(x, codes, rng.uniform(0.5, 2.0, size=25), 4)
        par = np.array([0.8, -0.5, -0.9, -0.1, 0.2])
        h = 1e-6
        numeric = [(fn(par + h * e) - fn(par - h * e)) / (2 * h) for e in np.eye(5)]
        np.testing.assert_allclose(gr(par), numeric, rtol=1e-6, atol=1e-7)

    def test_polr_objective_is_infinite_where_a_probability_is_zero(self):
        """R's ``Inf`` when any probability is not positive: a middle level far
        from its cut points has probability exactly 0."""
        from imputation.fitting import _polr_objective

        fn, _ = _polr_objective(np.array([[-60.0], [0.0], [1.0]]), np.array([1, 0, 2]), np.ones(3), 3)
        assert fn(np.array([1.0, 0.0, 0.0])) == np.inf
        assert np.isfinite(fn(np.array([0.0, 0.0, 0.0])))

    def test_plogis_and_dlogis_are_rs(self):
        from imputation.fitting import _dlogis, _plogis

        z = np.array([-np.inf, -800.0, -100.0, 0.0, 2.0, 100.0, np.inf])
        np.testing.assert_array_equal(_plogis(z)[[0, 1, 3, 5, 6]], [0.0, 0.0, 0.5, 1.0, 1.0])
        np.testing.assert_allclose(_plogis(z[[2, 4]]), [np.exp(-100.0), 1 / (1 + np.exp(-2.0))], rtol=1e-15)
        density = np.exp(-2) / (1 + np.exp(-2)) ** 2
        np.testing.assert_allclose(_dlogis(np.array([0.0, 2.0, -2.0])), [0.25, density, density])


class TestMethodConsistency:
    """Test consistency across imputation methods."""

    def test_all_methods_same_interface(self):
        """Test that all methods follow the same interface."""
        rs = np.random.RandomState(42)
        y = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0])
        id_obs = ~np.isnan(y)
        x = rs.randn(8, 2)

        methods = ["sample", "pmm", "cart", "rf", "midas", "norm", "norm.nob", "norm.boot", "norm.predict", "mean"]

        for method_name in methods:
            method_func = get_imputer_func(method_name)
            result = method_func(y, id_obs, x, rng=np.random.default_rng(6))

            assert isinstance(result, np.ndarray)
            assert len(result) == (~id_obs).sum()
            assert not np.isnan(result).any()

        # the categorical methods impute a target with two categories
        y01 = np.array([0.0, 1.0, np.nan, 1.0, 0.0, 1.0, 0.0, 1.0])
        for method_name in ["logreg", "logreg.boot", "polyreg", "polr"]:
            result = get_imputer_func(method_name)(y01, id_obs, x, rng=np.random.default_rng(6))

            assert isinstance(result, np.ndarray)
            assert len(result) == (~id_obs).sum()
            assert set(result) <= {0.0, 1.0}
