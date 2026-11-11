"""Logging of the packages: silent by default, standard logger names, optional helper."""

import logging
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from imputation import MICE, configure_logging, disable_logging, reset_logging


@pytest.fixture
def small_frame():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"a": rng.normal(size=30), "b": rng.normal(size=30)})
    df.loc[::4, "a"] = np.nan
    return df


@pytest.fixture(autouse=True)
def _restore_logging():
    yield
    reset_logging()


def test_import_adds_only_a_null_handler():
    """A fresh import configures nothing but a NullHandler per package."""
    code = (
        "import logging, imputation, plotting\n"
        "for name in ('imputation', 'plotting'):\n"
        "    lg = logging.getLogger(name)\n"
        "    print(name, [type(h).__name__ for h in lg.handlers], lg.propagate, lg.level)\n"
        "print('root', logging.getLogger().handlers)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.splitlines() == [
        "imputation ['NullHandler'] True 0",
        "plotting ['NullHandler'] True 0",
        "root []",
    ]
    assert out.stderr == ""


def test_records_propagate_under_module_names(small_frame, caplog):
    with caplog.at_level(logging.DEBUG):
        MICE(small_frame).impute(n_imputations=1, maxit=1, seed=1)
    names = {record.name for record in caplog.records}
    assert "imputation.MICE" in names
    assert all(name.startswith("imputation.") for name in names)


def test_no_warning_without_configuration(small_frame, caplog, recwarn):
    """No 'logging is not configured' message, neither as log record nor as warning."""
    with caplog.at_level(logging.WARNING):
        MICE(small_frame).impute(n_imputations=1, maxit=1, seed=1)
    assert caplog.records == []
    assert len(recwarn) == 0


def test_configure_logging_writes_no_file_unless_asked(small_frame, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    logger = configure_logging(level="DEBUG")
    assert logger.name == "imputation"
    assert [type(h) for h in logger.handlers] == [logging.NullHandler, logging.StreamHandler]
    MICE(small_frame).impute(n_imputations=1, maxit=1, seed=1)
    assert list(tmp_path.iterdir()) == []
    assert "INFO - Initializing MICE object" in capsys.readouterr().err


def test_configure_logging_file_on_request(small_frame, tmp_path):
    log_dir = tmp_path / "logs"
    configure_logging(console=False, file_logging=True, log_dir=str(log_dir))
    MICE(small_frame).impute(n_imputations=1, maxit=1, seed=1)
    reset_logging()  # closes the file
    (log_file,) = log_dir.iterdir()
    assert "imputation.MICE - INFO - Initializing MICE object" in log_file.read_text()


def test_disable_and_reset_logging(small_frame, caplog):
    disable_logging()
    with caplog.at_level(logging.DEBUG):
        MICE(small_frame).impute(n_imputations=1, maxit=1, seed=1)
    assert caplog.records == []
    reset_logging()
    lg = logging.getLogger("imputation")
    assert [type(h) for h in lg.handlers] == [logging.NullHandler]
    assert lg.level == logging.NOTSET and lg.propagate
    with caplog.at_level(logging.DEBUG):
        MICE(small_frame).impute(n_imputations=1, maxit=1, seed=1)
    assert caplog.records
