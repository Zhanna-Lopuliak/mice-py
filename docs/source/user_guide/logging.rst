Logging
=======

mice-py uses Python's standard :mod:`logging`. It prints nothing by itself;
anything that needs attention is a warning (``warnings.warn``), and progress
and details are log records that you see only if you configure logging.

Loggers
-------

Every module logs to a logger named after it: ``imputation.MICE``,
``imputation.pooling``, ``imputation.cart``, ``plotting.diagnostics``, and so
on. The packages add only a ``NullHandler`` to their top loggers
``imputation`` and ``plotting``, and the records propagate to the root
logger. So the usual configuration of an application works:

.. code-block:: python

   import logging

   logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")

   from imputation import MICE

   MICE(df).impute(seed=1)   # INFO records: start, one per chain, total time

(``df`` is the :ref:`example data <example-data>`.) Use the logger names to
choose what you see, for example only warnings from the package:

.. code-block:: python

   logging.getLogger("imputation").setLevel(logging.WARNING)

Levels used: ``DEBUG`` for details of every step (parameters, predictors per
column, chain statistics), ``INFO`` for the main steps and timings,
``WARNING`` for unusual situations (for example parameters given for a
method that is not used), ``ERROR`` just before an exception is raised.

The helpers
-----------

For scripts and notebooks, ``imputation`` has helpers that configure the
``imputation`` logger only (not the root logger and not ``plotting``):

.. code-block:: python

   import imputation

   imputation.configure_logging()                                    # INFO and above on the console
   imputation.configure_logging(level="DEBUG", console_level="DEBUG") # everything
   imputation.configure_logging(console_level="WARNING")             # warnings and errors only

   imputation.disable_logging()   # no records at all from imputation
   imputation.reset_logging()     # back to the state after import

``configure_logging(**kwargs)`` is ``setup_logging`` under a friendlier
name. Calling it again replaces the handlers it added before. Its arguments:

.. list-table::
   :header-rows: 1
   :widths: 22 22 56

   * - Argument
     - Default
     - Meaning
   * - ``level``
     - ``"INFO"``
     - level of the ``imputation`` logger
   * - ``console``
     - ``True``
     - log to standard error
   * - ``console_level``
     - ``None`` (INFO)
     - level of the console handler
   * - ``file_logging``
     - ``False``
     - also write a rotating file ``<log_dir>/mice_<date>.log``
   * - ``log_dir``
     - ``None`` (``"logs"``)
     - directory of the log file
   * - ``file_level``
     - ``None`` (DEBUG)
     - level of the file handler
   * - ``format_string``
     - ``None``
     - format of both handlers; by default ``"<level> - <message>"`` on the
       console and time, logger name, level and message in the file
   * - ``max_bytes``, ``backup_count``
     - 5 MB, 5
     - rotation of the log file

A log file is written only with ``file_logging=True``:

.. code-block:: python

   imputation.configure_logging(file_logging=True, log_dir="my_logs", console=False)
   imputation.reset_logging()

``imputation.get_logger(name)`` returns a logger below ``imputation`` (for
example ``get_logger("my_script")`` is ``imputation.my_script``), so that a
script's own messages follow the same configuration.

Warnings
--------

Independent of logging, ``impute`` issues a ``UserWarning`` when the setup
step or a method changed the model ("Number of logged events"), for rows
without any observed value, for a categorical method on a numeric column
("Type mismatch"), and for more than 100 imputations or 50 iterations;
``pool`` warns for a single imputation; the plot functions warn when there
is nothing to plot. Control them with the :mod:`warnings` module, for
example ``warnings.simplefilter("ignore", UserWarning)`` inside a
``warnings.catch_warnings()`` block.
