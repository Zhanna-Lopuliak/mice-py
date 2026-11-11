"""Item 3: argument handling (C2, C4).

The imputers have no ``**kwargs``, so an unknown argument raises
``TypeError``; ``impute`` validates ``<method>_<param>`` names against the
imputer signatures and raises on unknown names. Planned: NumPy integers are
accepted for ``n_imputations`` and ``maxit``.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from ._helpers import METHODS, full_predictor_matrix, impute, numeric_frame, pmm_reference_frame


def _problem():
    rng = np.random.default_rng(0)
    n = 30
    x = rng.normal(size=(n, 2))
    y = x.sum(axis=1) + rng.normal(size=n)
    ry = np.ones(n, dtype=bool)
    ry[::6] = False
    y[~ry] = np.nan
    return y, ry, x


@pytest.mark.parametrize("method", METHODS)
def test_imputer_rejects_unknown_keyword(method):
    from imputation.utils import get_imputer_func

    y, ry, x = _problem()
    with pytest.raises(TypeError):
        get_imputer_func(method)(y, ry, x, not_a_parameter=1, rng=np.random.default_rng(0))


@pytest.mark.parametrize("method", METHODS)
def test_impute_rejects_misspelled_method_parameter(method):
    df = numeric_frame(n=30)
    with pytest.raises((TypeError, ValueError)):
        impute(df, method, n_imputations=1, maxit=1, **{f"{method}_not_a_parameter": 3})


def test_impute_rejects_unknown_method_prefix():
    df = numeric_frame(n=30)
    with pytest.raises((TypeError, ValueError)):
        impute(df, "pmm", n_imputations=1, maxit=1, nosuchmethod_donors=3)


def test_impute_rejects_unknown_plain_keyword():
    df = numeric_frame(n=30)
    with pytest.raises((TypeError, ValueError)):
        impute(df, "pmm", n_imputations=1, maxit=1, n_imputation=3)


def test_unknown_argument_message_names_it_and_lists_valid_ones():
    df = numeric_frame(n=30)
    with pytest.raises((TypeError, ValueError)) as info:
        impute(df, "pmm", n_imputations=1, maxit=1, pmm_donor=3)
    message = str(info.value)
    assert "pmm_donor" in message
    assert "pmm_donors" in message and "pmm_matchtype" in message


def test_valid_parameter_of_another_method_is_accepted():
    """``cart_min_samples_leaf`` names a real parameter; with pmm it is unused, not wrong."""
    df = numeric_frame(n=30)
    _, out = impute(df, "pmm", n_imputations=1, maxit=1, cart_min_samples_leaf=3)
    assert len(out) == 1


def test_quickpred_arguments_still_accepted():
    """Deprecated for one version; they build the matrix with imputation.quickpred,
    whose rows of complete columns (d, k) are 0, as in R."""
    df = numeric_frame(n=30)
    with pytest.warns(DeprecationWarning, match="imputation.quickpred"):
        mice, _ = impute(
            df, "pmm", n_imputations=1, maxit=1, predictor_matrix=None,
            min_cor=0.0, min_puc=0.0, include=["d"], exclude=["k"], correlation_method="spearman",
        )
    assert mice.predictor_matrix.loc[["a", "b", "c"], "d"].eq(1).all()
    assert mice.predictor_matrix["k"].eq(0).all()
    assert mice.predictor_matrix.loc[["d", "k"]].eq(0).all().all()


def test_quickpred_arguments_with_predictor_matrix_are_ignored_and_deprecated():
    df = numeric_frame(n=30)
    with pytest.warns(DeprecationWarning, match="min_cor"):
        mice, _ = impute(df, "pmm", n_imputations=1, maxit=1, min_cor=0.9)
    pd.testing.assert_frame_equal(mice.predictor_matrix, full_predictor_matrix(df), check_dtype=False)


def test_without_quickpred_arguments_the_full_matrix_is_used_without_warning():
    df = numeric_frame(n=30)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        mice, _ = impute(df, "pmm", n_imputations=1, maxit=1, predictor_matrix=None)
    pd.testing.assert_frame_equal(mice.predictor_matrix, full_predictor_matrix(df), check_dtype=False)


def test_output_dir_is_deprecated_and_ignored(tmp_path):
    df = numeric_frame(n=30)
    target = tmp_path / "out"
    with pytest.warns(DeprecationWarning, match="output_dir"):
        _, out = impute(df, "sample", n_imputations=1, maxit=1, output_dir=str(target))
    assert len(out) == 1
    assert not target.exists()


def test_formerly_implicit_parameters_reach_the_backend(monkeypatch):
    """``ls_meth`` (pmm), ``tree_params`` (cart) and ``forest_params`` (rf) are
    explicit parameters now and arrive where they belong."""
    import imputation.cart
    import imputation.PMM
    import imputation.rf

    seen = {}
    norm_draw = imputation.PMM.norm_draw

    def recording_norm_draw(*args, **kwargs):
        seen["ls_meth"] = kwargs.get("ls_meth")
        return norm_draw(*args, **kwargs)

    class RecordingTree(imputation.cart.DecisionTreeRegressor):
        def fit(self, *args, **kwargs):
            seen["tree"] = self.get_params()
            return super().fit(*args, **kwargs)

    class RecordingForest(imputation.rf.RandomForestRegressor):
        def fit(self, *args, **kwargs):
            seen["forest"] = self.get_params()
            return super().fit(*args, **kwargs)

    monkeypatch.setattr(imputation.PMM, "norm_draw", recording_norm_draw)
    monkeypatch.setattr(imputation.cart, "DecisionTreeRegressor", RecordingTree)
    monkeypatch.setattr(imputation.rf, "RandomForestRegressor", RecordingForest)
    df = numeric_frame(n=30)
    impute(df, "pmm", n_imputations=1, maxit=1, pmm_ls_meth="ridge")
    # max_depth is an argument of cart now, so it cannot go through tree_params
    impute(
        df, "cart", n_imputations=1, maxit=1,
        cart_max_depth=3, cart_tree_params={"max_leaf_nodes": 7},
    )
    impute(
        df, "rf", n_imputations=1, maxit=1,
        rf_max_depth=4, rf_min_samples_leaf=2, rf_forest_params={"max_features": 0.5},
    )
    assert seen["ls_meth"] == "ridge"
    assert seen["tree"]["max_depth"] == 3
    assert seen["tree"]["max_leaf_nodes"] == 7
    assert (seen["forest"]["max_depth"], seen["forest"]["min_samples_leaf"]) == (4, 2)
    assert seen["forest"]["max_features"] == 0.5


@pytest.mark.parametrize(
    "method, argument, key",
    [
        ("cart", "tree_params", "random_state"),
        ("cart", "tree_params", "ccp_alpha"),
        ("cart", "tree_params", "max_depth"),
        ("cart", "tree_params", "min_samples_split"),
        ("rf", "forest_params", "random_state"),
        ("rf", "forest_params", "n_estimators"),
    ],
)
def test_estimator_params_cannot_override_explicit_arguments(method, argument, key):
    from imputation.utils import get_imputer_func

    y, ry, x = _problem()
    with pytest.raises(TypeError, match=key):
        get_imputer_func(method)(y, ry, x, rng=np.random.default_rng(0), **{argument: {key: 1}})


def _nearest_fitted_donor(df, mis):
    """PMM with one donor and matchtype 0: the observed y whose OLS fit is closest."""
    X = np.c_[np.ones(len(df)), df[["x1", "x2"]].to_numpy()]
    y = df["y"].to_numpy()
    coef, *_ = np.linalg.lstsq(X[~mis], y[~mis], rcond=None)
    yhat = X @ coef
    dist = np.abs(yhat[mis][:, None] - yhat[~mis][None, :])
    gaps = np.sort(dist, axis=1)
    assert (gaps[:, 1] - gaps[:, 0] > 1e-3).all(), "test data must not have near ties"
    return y[~mis][dist.argmin(axis=1)]


def test_prefixed_parameters_reach_the_imputer():
    """``pmm_donors=1`` with ``pmm_matchtype=0`` makes PMM deterministic: every
    imputation is the observed value whose fitted value is closest."""
    df, mis = pmm_reference_frame()
    expected = _nearest_fitted_donor(df, mis)
    _, out = impute(df, "pmm", n_imputations=3, maxit=1, pmm_donors=1, pmm_matchtype=0)
    for completed in out:
        np.testing.assert_allclose(completed.loc[mis, "y"].to_numpy(), expected, rtol=0, atol=0)


def test_default_donors_are_not_deterministic():
    """Control for the test above: with the default 5 donors the result varies."""
    df, mis = pmm_reference_frame()
    expected = _nearest_fitted_donor(df, mis)
    _, out = impute(df, "pmm", n_imputations=3, maxit=1)
    assert not all(np.array_equal(c.loc[mis, "y"].to_numpy(), expected) for c in out)


@pytest.mark.parametrize("dtype", [np.int64, np.int32], ids=["int64", "int32"])
def test_numpy_integer_n_imputations(dtype):
    df = numeric_frame(n=30)
    mice, out = impute(df, "sample", n_imputations=dtype(3), maxit=2)
    assert len(out) == 3
    assert all(v.shape == (2, 3) for v in mice.chain_mean.values())


@pytest.mark.parametrize("dtype", [np.int64, np.int32], ids=["int64", "int32"])
def test_numpy_integer_maxit(dtype):
    df = numeric_frame(n=30)
    mice, out = impute(df, "sample", n_imputations=2, maxit=dtype(3))
    assert len(out) == 2
    assert all(v.shape == (3, 2) for v in mice.chain_mean.values())
