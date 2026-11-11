"""Utility helpers.

This module offers :func:`get_imputer_func`, which maps a string identifying
an imputation method to the concrete callable that implements that method, and
:func:`get_imputer_params`, which lists the parameters a user may set for it.
"""

from __future__ import annotations

import inspect

from .cart import cart
from .constants import ImputationMethod
from .midas import midas

# Import concrete imputation back-ends
from .PMM import pmm
from .rf import rf
from .sample import sample

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

# Map of method-name -> imputer function
_IMPUTER_MAP = {
    ImputationMethod.PMM.value: pmm,
    ImputationMethod.MIDAS.value: midas,
    ImputationMethod.CART.value: cart,
    ImputationMethod.SAMPLE.value: sample,
    ImputationMethod.RF.value: rf,
}

# Arguments that MICE supplies to every imputer itself
_SUPPLIED_BY_MICE = ("y", "id_obs", "x", "id_mis", "rng")


def get_imputer_func(method_name: str):
    """Return the imputer callable for *method_name*.

    Parameters
    ----------
    method_name : str
        Name of the imputation method. Must be one of the values defined in
        :class:`imputation.constants.ImputationMethod`.

    Returns
    -------
    Callable
        The function implementing the requested imputation strategy.

    Raises
    ------
    ValueError
        If *method_name* is unknown or not yet implemented.
    """
    if method_name not in _IMPUTER_MAP:
        raise ValueError(
            "Unsupported or unimplemented imputation method: "
            f"'{method_name}'. Supported methods are: {list(_IMPUTER_MAP.keys())}"
        )

    return _IMPUTER_MAP[method_name]


def get_imputer_params(method_name: str):
    """Return the names of the parameters a user may set for *method_name*.

    These are the parameters of the imputer's signature except the ones MICE
    supplies itself (``y``, ``id_obs``, ``x``, ``id_mis`` and ``rng``). In
    :meth:`imputation.MICE.impute` they are passed with the method name as
    prefix, e.g. ``pmm_donors``.

    Parameters
    ----------
    method_name : str
        Name of the imputation method.

    Returns
    -------
    list of str
        Parameter names in signature order.

    Raises
    ------
    ValueError
        If *method_name* is unknown or not yet implemented.
    """
    signature = inspect.signature(get_imputer_func(method_name))
    return [
        name
        for name, param in signature.parameters.items()
        if name not in _SUPPLIED_BY_MICE and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
    ]
