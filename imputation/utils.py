"""Utility helpers.

This module offers :func:`get_imputer_func`, which maps a string identifying
an imputation method to the concrete callable that implements that method,
:func:`get_imputer_params`, which lists the parameters a user may set for it,
and :func:`method_prefix` and :func:`split_method_param`, which relate a
keyword argument ``<prefix>_<param>`` of ``MICE.impute`` to its method.
"""

from __future__ import annotations

import inspect
from collections.abc import Collection, Mapping
from typing import Callable

import numpy as np

from .cart import cart
from .constants import ImputationMethod
from .logreg import logreg, logreg_boot
from .mean import mean
from .midas import midas
from .norm import norm, norm_boot, norm_nob, norm_predict

# Import concrete imputation back-ends
from .PMM import pmm
from .polr import polr
from .polyreg import polyreg
from .rf import rf
from .sample import sample

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

# Map of method-name -> imputer function
_IMPUTER_MAP: dict[str, Callable[..., np.ndarray]] = {
    ImputationMethod.PMM.value: pmm,
    ImputationMethod.MIDAS.value: midas,
    ImputationMethod.CART.value: cart,
    ImputationMethod.SAMPLE.value: sample,
    ImputationMethod.RF.value: rf,
    ImputationMethod.NORM.value: norm,
    ImputationMethod.NORM_NOB.value: norm_nob,
    ImputationMethod.NORM_BOOT.value: norm_boot,
    ImputationMethod.NORM_PREDICT.value: norm_predict,
    ImputationMethod.MEAN.value: mean,
    ImputationMethod.LOGREG.value: logreg,
    ImputationMethod.LOGREG_BOOT.value: logreg_boot,
    ImputationMethod.POLYREG.value: polyreg,
    ImputationMethod.POLR.value: polr,
}

# Arguments that MICE supplies to the imputers itself (`log_event` only to
# an imputer whose signature has it)
_SUPPLIED_BY_MICE = ("y", "id_obs", "x", "id_mis", "rng", "log_event")


def get_imputer_func(method_name: str) -> Callable[..., np.ndarray]:
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


def get_imputer_params(method_name: str) -> list[str]:
    """Return the names of the parameters a user may set for *method_name*.

    These are the parameters of the imputer's signature except the ones MICE
    supplies itself (``y``, ``id_obs``, ``x``, ``id_mis``, ``rng`` and
    ``log_event``). In :meth:`imputation.MICE.impute` they are passed with
    the method's prefix (see :func:`method_prefix`), e.g. ``pmm_donors``.

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


def method_prefix(method_name: str) -> str:
    """Return the keyword prefix of a method: its name with every dot replaced by an underscore.

    Parameter ``param`` of the method is passed to
    :meth:`imputation.MICE.impute` as ``<prefix>_<param>``, e.g.
    ``norm_boot_ridge`` for parameter ``ridge`` of method ``norm.boot``.

    Parameters
    ----------
    method_name : str
        Name of the imputation method.

    Returns
    -------
    str
        The prefix.
    """
    return method_name.replace(".", "_")


def split_method_param(key: str, method_params: Mapping[str, Collection[str]]) -> tuple[str, str] | None:
    """Split a keyword argument ``<prefix>_<param>`` into method and parameter.

    Several methods may match a keyword: ``norm_boot_ridge`` starts with the
    prefixes of ``norm`` and of ``norm.boot``. The methods whose prefix (see
    :func:`method_prefix`) and an underscore start *key* are tried from the
    longest prefix to the shortest; the first of which the rest of *key* is
    a parameter wins.

    Parameters
    ----------
    key : str
        The keyword argument, e.g. ``"pmm_donors"``.
    method_params : mapping of str to collection of str
        The parameters a user may set for each method, as
        :func:`get_imputer_params` lists them.

    Returns
    -------
    tuple of (str, str) or None
        ``(method, param)``. If no matching method has that parameter,
        ``method`` is the one with the longest matching prefix and ``param``
        is not in ``method_params[method]`` (so that an error message can
        name the method). None if no method matches.

    Examples
    --------
    >>> params = {"norm": ["ridge"], "norm.boot": ["ridge"]}
    >>> split_method_param("norm_boot_ridge", params)
    ('norm.boot', 'ridge')
    >>> split_method_param("norm_ridge", params)
    ('norm', 'ridge')
    >>> split_method_param("norm_boot_donors", params)
    ('norm.boot', 'donors')
    >>> split_method_param("pmm_donors", params) is None
    True
    """
    matches = sorted(
        (method for method in method_params if key.startswith(f"{method_prefix(method)}_")),
        key=lambda method: len(method_prefix(method)),
        reverse=True,
    )
    for method in matches:
        param = key[len(method_prefix(method)) + 1 :]
        if param in method_params[method]:
            return method, param
    if not matches:
        return None
    return matches[0], key[len(method_prefix(matches[0])) + 1 :]
