"""Check MICE's "Number of logged events" warning where tests impute in bulk.

Many tests impute frames that change the imputation model (constant or
collinear columns, too few observed rows), so ``MICE.impute`` warns, as R's
``mice`` prints "Number of logged events". Instead of letting those warnings
pile up in the test report, the helpers check them: the warning must appear
exactly when ``MICE.logged_events`` is not empty, once, with its number of
rows. Every other warning is passed on unchanged.
"""

import re
import warnings

_LOG_WARNING = re.compile(r"Number of logged events: (\d+)\.")


def impute_checking_log_warning(mice, **kwargs):
    """``mice.impute(**kwargs)``, with the logged-events warning checked."""
    caught = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = mice.impute(**kwargs)
    finally:
        for warning in caught:
            if not _LOG_WARNING.match(str(warning.message)):
                warnings.warn_explicit(warning.message, warning.category, warning.filename, warning.lineno)
    counts = [
        int(match.group(1)) for match in (_LOG_WARNING.match(str(w.message)) for w in caught) if match is not None
    ]
    n_events = len(mice.logged_events)
    assert counts == ([n_events] if n_events else []), (counts, n_events)
    return out
