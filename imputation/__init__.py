"""
Imputation package for Multiple Imputation by Chained Equations (MICE).

This package provides robust imputation methods including MICE, PMM, CART, and Random Forest.
It follows Python logging best practices and allows users to configure logging as needed.
"""

import logging

# Add a null handler to prevent "No handlers" warnings when the package is used
# This follows Python logging best practices for libraries
package_logger = logging.getLogger(__name__)
package_logger.addHandler(logging.NullHandler())
package_logger.propagate = False  # Don't propagate to root logger by default

# Import main classes and functions
from .MICE import MICE
from .logging_config import setup_logging, get_logger, disable_logging, reset_logging

# Package logger for internal use
logger = logging.getLogger(__name__)

# Public API
__all__ = [
    'MICE', 
    'setup_logging', 
    'get_logger',
    'disable_logging',
    'reset_logging',
    'configure_logging'
]


def configure_logging(**kwargs):
    """
    Configure logging for the imputation package.
    
    This is a user-friendly wrapper around setup_logging() that provides
    the same functionality with a more intuitive name.
    
    Parameters
    ----------
    **kwargs
        All arguments are passed to setup_logging(). See setup_logging()
        documentation for full list of parameters.
        
    Returns
    -------
    logging.Logger
        The configured package logger
        
    Examples
    --------
    Basic usage:
    >>> import imputation
    >>> imputation.configure_logging()
    
    Enable debug logging:
    >>> imputation.configure_logging(level='DEBUG')
    
    Custom log directory:
    >>> imputation.configure_logging(log_dir='my_project/logs')
    
    Console only (no file logging):
    >>> imputation.configure_logging(file_logging=False)
    
    Quiet mode (warnings and errors only):
    >>> imputation.configure_logging(console_level='WARNING')
    """
    return setup_logging(**kwargs)


