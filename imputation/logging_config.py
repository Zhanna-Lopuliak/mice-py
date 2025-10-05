"""
Logging configuration module for the imputation package.

This module provides proper logging setup following Python best practices,
allowing users to configure logging behavior without affecting the global
logging state.
"""

import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional, Union


def setup_logging(
    level: Union[str, int] = "INFO",
    log_dir: Optional[str] = None,
    console: bool = True,
    file_logging: bool = True,
    format_string: Optional[str] = None,
    console_level: Optional[Union[str, int]] = None,
    file_level: Optional[Union[str, int]] = None,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 5
) -> logging.Logger:
    """
    Configure logging for the imputation package.
    
    This function sets up a package-specific logger without affecting the
    root logger or other packages. It's safe to call multiple times.
    
    Parameters
    ----------
    level : str or int, default="INFO"
        Base logging level for the package ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        or logging level constant (e.g., logging.DEBUG)
    log_dir : str, optional
        Directory for log files. If None, uses './logs'
    console : bool, default=True
        Whether to enable console logging
    file_logging : bool, default=True
        Whether to enable file logging
    format_string : str, optional
        Custom format string for log messages. If None, uses default format
    console_level : str or int, optional
        Logging level for console handler. If None, uses 'INFO'
    file_level : str or int, optional
        Logging level for file handler. If None, uses 'DEBUG'
    max_bytes : int, default=5MB
        Maximum size of log file before rotation
    backup_count : int, default=5
        Number of backup log files to keep
        
    Returns
    -------
    logging.Logger
        The configured package logger
        
    Examples
    --------
    Basic usage:
    >>> import imputation
    >>> logger = imputation.setup_logging()
    
    Custom configuration:
    >>> logger = imputation.setup_logging(
    ...     level='DEBUG',
    ...     log_dir='my_logs',
    ...     console_level='WARNING'
    ... )
    
    Disable file logging:
    >>> logger = imputation.setup_logging(file_logging=False)
    """
    # Convert string levels to logging constants
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    if isinstance(console_level, str):
        console_level = getattr(logging, console_level.upper())
    if isinstance(file_level, str):
        file_level = getattr(logging, file_level.upper())
    
    # Set defaults for handler levels
    if console_level is None:
        console_level = logging.INFO
    if file_level is None:
        file_level = logging.DEBUG
    
    # Get the package logger (not root logger!)
    logger = logging.getLogger('imputation')
    logger.setLevel(level)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    logger.propagate = False  # Don't propagate to root logger to avoid interference
    
    # Set up formatters
    if format_string is None:
        file_format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        console_format_string = '%(levelname)s - %(message)s'
    else:
        file_format_string = format_string
        console_format_string = format_string
    
    file_formatter = logging.Formatter(file_format_string)
    console_formatter = logging.Formatter(console_format_string)
    
    # Add console handler
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # Add file handler
    if file_logging:
        if log_dir is None:
            log_dir = 'logs'
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Generate log filename with current date
        log_filename = f"mice_{datetime.now().strftime('%Y-%m-%d')}.log"
        log_file_path = os.path.join(log_dir, log_filename)
        
        # Create rotating file handler
        file_handler = RotatingFileHandler(
            log_file_path, 
            maxBytes=max_bytes, 
            backupCount=backup_count
        )
        file_handler.setLevel(file_level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module within the imputation package.
    
    This function returns a child logger of the main package logger,
    ensuring proper hierarchy and inheritance of configuration.
    
    Parameters
    ----------
    name : str
        Module name, typically __name__ or a descriptive string
        
    Returns
    -------
    logging.Logger
        A logger instance for the specified module
        
    Examples
    --------
    In a module file:
    >>> from imputation.logging_config import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("This is a log message")
    
    For simulation scripts:
    >>> logger = get_logger('imputation.simulation.fdgs')
    """
    # Ensure the name is under the imputation package
    if not name.startswith('imputation'):
        if name == '__main__':
            name = 'imputation.main'
        elif '.' in name and name.split('.')[-1] in ['simulate_fdgs', 'simulate_mcar', 'simulate_next_data']:
            # Handle simulation scripts
            script_name = name.split('.')[-1]
            name = f'imputation.simulation.{script_name}'
        else:
            name = f'imputation.{name}'
    
    return logging.getLogger(name)


def disable_logging():
    """
    Disable logging for the imputation package.
    
    This is useful for testing or when logging output is not desired.
    """
    logger = logging.getLogger('imputation')
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.propagate = False


def reset_logging():
    """
    Reset logging configuration to default state.
    
    This removes all handlers and sets the logger back to default
    configuration with only a NullHandler.
    """
    logger = logging.getLogger('imputation')
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.WARNING)
    logger.propagate = False  # Silent by default


# Module-level logger for this configuration module
_config_logger = logging.getLogger('imputation.config')
