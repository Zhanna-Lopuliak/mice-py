"""Direct calls of the imputers with a predictor DataFrame (B2, B7 to B9).

``pmm``, ``midas``, ``cart`` and ``rf`` accept ``x`` as an array or a
DataFrame; a DataFrame with non-numeric columns is dummy-coded by
``imputation.design.obtain_design``, as MICE does.
"""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# direct calls of the imputers
# ---------------------------------------------------------------------------

def _direct_problem():
    rng = np.random.default_rng(9)
    n = 50
    x = pd.DataFrame({
        "num": rng.normal(size=n),
        "g": pd.Series(rng.choice(["b", "a", "c"], n), dtype=object),
        "s": pd.Series(rng.choice(["u", "v"], n), dtype="string"),
        "size": pd.Categorical(rng.choice(["S", "M", "L"], n), categories=["S", "M", "L"], ordered=True),
        "flag": rng.random(n) > 0.5,
    })
    y = x["num"] + (x["g"] == "c") + rng.normal(scale=0.5, size=n)
    y = y.to_numpy()
    ry = np.ones(n, dtype=bool)
    ry[::7] = False
    y[~ry] = np.nan
    return y, ry, x


@pytest.mark.parametrize("method", ["pmm", "midas", "cart", "rf"])
def test_direct_call_with_categorical_dataframe_equals_design_matrix(method):
    """A DataFrame with non-numeric predictors goes through obtain_design, so a
    direct call gives what a call with the design matrix gives."""
    from imputation.design import obtain_design
    from imputation.utils import get_imputer_func

    imputer = get_imputer_func(method)
    y, ry, x = _direct_problem()
    design = obtain_design(x, list(x.columns))
    assert list(design.columns) == ["num", "gb", "gc", "sv", "size.L", "size.Q", "flagTRUE"]
    from_frame = imputer(y, ry, x, rng=np.random.default_rng(4))
    from_design = imputer(y, ry, design.to_numpy(), rng=np.random.default_rng(4))
    np.testing.assert_array_equal(from_frame, from_design)
    assert np.isin(from_frame, y[ry]).all()


@pytest.mark.parametrize("method", ["cart", "rf"])
def test_tree_methods_without_predictors_draw_observed_values(method):
    """B7: rf without predictors behaves like cart: a draw from the observed values."""
    from imputation.utils import get_imputer_func

    imputer = get_imputer_func(method)
    rng = np.random.default_rng(0)
    y = rng.normal(size=40)
    ry = np.ones(40, dtype=bool)
    ry[::5] = False
    for x in (np.empty((40, 0)), pd.DataFrame(index=range(40))):
        out = imputer(y, ry, x, rng=np.random.default_rng(2))
        assert out.shape == (8,)
        assert np.isin(out, y[ry]).all()
    draws = np.concatenate([imputer(y, ry, np.empty((40, 0)), rng=np.random.default_rng(s)) for s in range(40)])
    assert len(np.unique(draws)) > 16  # not a single donor


@pytest.mark.parametrize("method", ["cart", "rf"])
@pytest.mark.parametrize("kind", ["object", "numpy"])
def test_tree_methods_impute_booleans_with_booleans(method, kind):
    """B8: an incomplete boolean target is classified and imputed with booleans."""
    from imputation.utils import get_imputer_func

    imputer = get_imputer_func(method)
    rng = np.random.default_rng(1)
    n = 40
    x = rng.normal(size=(n, 2))
    flag = x[:, 0] + rng.normal(scale=0.5, size=n) > 0
    ry = np.ones(n, dtype=bool)
    ry[[2, 6, 11, 30]] = False
    y = pd.Series(flag, dtype=object).where(ry, np.nan).to_numpy() if kind == "object" else flag
    out = imputer(y, ry, x, rng=np.random.default_rng(0))
    assert len(out) == 4
    assert all(isinstance(v, (bool, np.bool_)) for v in out), list(out)


@pytest.mark.parametrize("method", ["cart", "rf"])
def test_tree_methods_accept_string_dtype_predictors_directly(method):
    """B9: a pandas ``string`` predictor is dummy-coded, not cast to float."""
    from imputation.utils import get_imputer_func

    imputer = get_imputer_func(method)
    rng = np.random.default_rng(6)
    n = 40
    s = pd.Series(rng.choice(["u", "v", "w"], n), dtype="string")
    y = (s == "w").to_numpy() * 2.0 + rng.normal(size=n)
    ry = np.ones(n, dtype=bool)
    ry[::8] = False
    out = imputer(y, ry, pd.DataFrame({"s": s, "x": rng.normal(size=n)}), rng=np.random.default_rng(0))
    assert len(out) == 5 and np.isin(out, y[ry]).all()
