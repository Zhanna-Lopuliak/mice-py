from __future__ import annotations

"""Utility helpers.

Currently this module offers a single public function, :func:`get_imputer_func`,
which maps a string identifying an imputation method to the concrete callable
that implements that method.
"""

from .constants import ImputationMethod

# Import concrete imputation back-ends
from .PMM import pmm
from .midas import midas
from .cart import cart
from .sample import sample
from .rf import rf

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