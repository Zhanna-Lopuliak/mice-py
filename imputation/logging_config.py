"""
Optional logging helpers for the imputation package.

The package itself only adds a ``logging.NullHandler`` to the ``imputation``
logger (see :mod:`imputation`). These helpers are a convenience for scripts
and notebooks that want to see the package's messages without configuring
:mod:`logging` themselves. They only touch the ``imputation`` logger, never
the root logger; its records still propagate to the root logger as usual.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

_PACKAGE = "imputation"


def _as_level(level: str | int) -> int:
    """A level name such as ``"DEBUG"`` or a level number, as a number."""
    if isinstance(level, str):
        return getattr(logging, level.upper())
    return level


def _reset_handlers(logger: logging.Logger) -> None:
    """Remove every handler of `logger` and put back a single NullHandler."""
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.addHandler(logging.NullHandler())


def setup_logging(
    level: str | int = "INFO",
    log_dir: str | None = None,
    console: bool = True,
    file_logging: bool = False,
    format_string: str | None = None,
    console_level: str | int | None = None,
    file_level: str | int | None = None,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 5,
) -> logging.Logger:
    """
    Configure the ``imputation`` logger.

    Replaces the handlers of the ``imputation`` logger (not of the root
    logger) by a console handler and, if asked, a rotating log file. Safe to
    call several times. Records still propagate to the root logger, so an
    application that also configures the root logger sees them there too.

    Parameters
    ----------
    level : str or int, default "INFO"
        Level of the ``imputation`` logger: a name ("DEBUG", "INFO",
        "WARNING", "ERROR") or a ``logging`` constant.
    log_dir : str, optional
        Directory of the log file when `file_logging` is True; ``"logs"``
        (relative to the working directory) if None.
    console : bool, default True
        Whether to log to the console (standard error).
    file_logging : bool, default False
        Whether to also write a rotating log file
        ``<log_dir>/mice_<date>.log``. No file is written unless this is True.
    format_string : str, optional
        Format of the messages of both handlers. If None, the console shows
        ``"<level> - <message>"`` and the file adds the time and logger name.
    console_level : str or int, optional
        Level of the console handler; INFO if None.
    file_level : str or int, optional
        Level of the file handler; DEBUG if None.
    max_bytes : int, default 5 MB
        Size at which the log file is rotated.
    backup_count : int, default 5
        Number of rotated log files to keep.

    Returns
    -------
    logging.Logger
        The ``imputation`` logger.

    Examples
    --------
    >>> import imputation
    >>> logger = imputation.setup_logging()  # doctest: +SKIP
    >>> logger = imputation.setup_logging(level="DEBUG", console_level="WARNING",
    ...                                   file_logging=True, log_dir="my_logs")  # doctest: +SKIP
    """
    logger = logging.getLogger(_PACKAGE)
    logger.setLevel(_as_level(level))
    _reset_handlers(logger)

    if format_string is None:
        file_format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        console_format_string = "%(levelname)s - %(message)s"
    else:
        file_format_string = format_string
        console_format_string = format_string

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(_as_level(console_level) if console_level is not None else logging.INFO)
        console_handler.setFormatter(logging.Formatter(console_format_string))
        logger.addHandler(console_handler)

    if file_logging:
        if log_dir is None:
            log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, f"mice_{datetime.now().strftime('%Y-%m-%d')}.log")
        file_handler = RotatingFileHandler(log_file_path, maxBytes=max_bytes, backupCount=backup_count)
        file_handler.setLevel(_as_level(file_level) if file_level is not None else logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(file_format_string))
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger below the ``imputation`` logger.

    Parameters
    ----------
    name : str
        A logger name. Names that do not start with ``"imputation"`` are put
        below it: ``"__main__"`` becomes ``"imputation.main"`` and any other
        name ``"imputation.<name>"``.

    Returns
    -------
    logging.Logger
        The logger, which inherits the configuration of the ``imputation``
        logger.

    Examples
    --------
    >>> from imputation.logging_config import get_logger
    >>> get_logger("my_script").name
    'imputation.my_script'
    """
    if not name.startswith(_PACKAGE):
        name = f"{_PACKAGE}.main" if name == "__main__" else f"{_PACKAGE}.{name}"
    return logging.getLogger(name)


def disable_logging() -> None:
    """
    Silence the ``imputation`` logger.

    Removes its handlers (a NullHandler remains) and sets its level above
    CRITICAL, so that the package emits no log records at all, also not to
    handlers of the root logger. :func:`reset_logging` undoes this.
    """
    logger = logging.getLogger(_PACKAGE)
    _reset_handlers(logger)
    logger.setLevel(logging.CRITICAL + 1)


def reset_logging() -> None:
    """
    Restore the state the ``imputation`` logger has after import.

    A single NullHandler, no level of its own (``NOTSET``, so the level of
    the root logger applies) and propagation to the root logger.
    """
    logger = logging.getLogger(_PACKAGE)
    _reset_handlers(logger)
    logger.setLevel(logging.NOTSET)
    logger.propagate = True
