"""
Multiple Imputation by Chained Equations (MICE), a Python port of R's ``mice``.

The main entry point is :class:`MICE`; :func:`quickpred` builds a predictor
matrix as R's ``quickpred`` does.

Logging
-------
Every module logs to a logger named after it (``imputation.MICE``,
``imputation.pooling``, ...). As a library, the package only adds a
``logging.NullHandler`` to the ``imputation`` logger, so nothing is shown
unless the application configures logging, for example with
``logging.basicConfig(level=logging.INFO)`` or :func:`configure_logging`.
"""

import logging
from importlib.metadata import PackageNotFoundError, version

from .design import quickpred
from .logging_config import disable_logging, get_logger, reset_logging, setup_logging
from .MICE import MICE

try:
    __version__ = version("mice-py")
except PackageNotFoundError:  # not installed, e.g. imported from a source tree
    __version__ = "0+unknown"

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ["MICE", "quickpred", "setup_logging", "get_logger", "disable_logging", "reset_logging", "configure_logging"]


def configure_logging(**kwargs) -> logging.Logger:
    """
    Configure logging for the imputation package.

    An optional convenience wrapper around :func:`setup_logging`, with the
    same arguments. It attaches handlers to the ``imputation`` logger only
    and writes no log file unless ``file_logging=True`` is passed.

    Parameters
    ----------
    **kwargs
        Passed to :func:`setup_logging`; see there for the parameters.

    Returns
    -------
    logging.Logger
        The configured ``imputation`` logger.

    Examples
    --------
    Messages at INFO and above on the console:

    >>> import imputation
    >>> imputation.configure_logging()  # doctest: +SKIP

    Debug messages:

    >>> imputation.configure_logging(level="DEBUG", console_level="DEBUG")  # doctest: +SKIP

    Also write a rotating log file into ``my_project/logs``:

    >>> imputation.configure_logging(file_logging=True, log_dir="my_project/logs")  # doctest: +SKIP

    Warnings and errors only:

    >>> imputation.configure_logging(console_level="WARNING")  # doctest: +SKIP
    """
    return setup_logging(**kwargs)
