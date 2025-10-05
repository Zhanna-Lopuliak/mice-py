# Logging Guide for Imputation Package

This document explains how to configure and use logging with the imputation package.

## Quick Start

```python
import imputation

# Package is silent by default
mice = imputation.MICE(data)  # No logging output
mice.impute(n_imputations=5, maxit=10)  # No logging output

# Enable logging when you want it
imputation.configure_logging()
mice2 = imputation.MICE(data)  # Shows logging messages
mice2.impute(n_imputations=5, maxit=10)  # Shows logging messages
```

## Configuration Options

### Basic Configuration

```python
import imputation

# Enable logging with default settings
imputation.configure_logging()

# Custom log level
imputation.configure_logging(level='DEBUG')

# Custom log directory
imputation.configure_logging(log_dir='my_project/logs')
```

### Advanced Configuration

```python
# Full configuration example
imputation.configure_logging(
    level='INFO',                    # Base logging level
    log_dir='custom_logs',          # Directory for log files
    console=True,                   # Enable console output
    file_logging=True,              # Enable file logging
    console_level='WARNING',        # Console shows warnings+ only
    file_level='DEBUG',             # File captures everything
    max_bytes=10*1024*1024,        # 10MB max file size
    backup_count=3                  # Keep 3 backup files
)
```

## Logging Levels

- **DEBUG**: Detailed information for diagnosing problems
- **INFO**: General information about program execution
- **WARNING**: Something unexpected happened, but the program continues
- **ERROR**: A serious problem occurred

## Usage Scenarios

### 1. Development and Debugging

```python
# Enable verbose logging for development
imputation.configure_logging(
    level='DEBUG',
    console_level='DEBUG'
)
```

### 2. Production Use

```python
# Quiet console, detailed file logging for production
imputation.configure_logging(
    level='INFO',
    console_level='ERROR',    # Only show errors on console
    file_level='DEBUG'        # But log everything to file
)
```

### 3. Batch Processing/Scripts

```python
# Console disabled, file logging only
imputation.configure_logging(
    console=False,
    file_logging=True,
    level='INFO'
)
```

### 4. Testing/No Logging

```python
# Completely disable logging
imputation.disable_logging()
```

### 5. Integration with Existing Logging

```python
import logging

# Set up your project logger
project_logger = logging.getLogger('my_project')
# ... configure your project logger ...

# Configure imputation package separately
imputation.configure_logging(
    level='INFO',
    console_level='ERROR',  # Minimize console noise
    log_dir='my_project/logs'
)
```

## Module-Specific Loggers

You can get loggers for specific parts of the package:

```python
# Get logger for specific modules
cart_logger = imputation.get_logger('imputation.cart')
mice_logger = imputation.get_logger('imputation.mice')
pooling_logger = imputation.get_logger('imputation.pooling')

# Use them in your code
cart_logger.info("Custom message from CART module")
```

## Log File Management

- **Default location**: `./logs/mice_YYYY-MM-DD.log`
- **Rotation**: Files rotate when they exceed 5MB (configurable)
- **Backup count**: 5 backup files kept by default (configurable)
- **Format**: `TIMESTAMP - MODULE - LEVEL - MESSAGE`

Example log output:
```
2025-09-25 10:30:15,123 - imputation.mice - INFO - Starting imputation process
2025-09-25 10:30:15,124 - imputation.mice - DEBUG - Parameters: n_imputations=5, maxit=10
2025-09-25 10:30:15,456 - imputation.cart - DEBUG - Starting CART imputation
```

## Control Functions

### `configure_logging(**kwargs)`
Main function to set up logging. See parameters above.

### `disable_logging()`
Completely disable all logging from the imputation package.

### `reset_logging()`
Reset logging to default state (only NullHandler).

### `setup_logging(**kwargs)`
Lower-level function (same as `configure_logging`).

### `get_logger(name)`
Get a logger for a specific module or component.

## Best Practices

### 1. Configure Early
Set up logging at the start of your script:

```python
import imputation
imputation.configure_logging(level='INFO')

# Rest of your code...
```

### 2. Use Appropriate Levels
- Use `DEBUG` during development
- Use `INFO` for production monitoring
- Use `WARNING`/`ERROR` for production alerts

### 3. Separate File and Console Levels
```python
# Show only important messages on console, log everything to file
imputation.configure_logging(
    console_level='WARNING',
    file_level='DEBUG'
)
```

### 4. Custom Log Directory
Organize logs with your project structure:

```python
imputation.configure_logging(
    log_dir='my_project/logs/imputation'
)
```

### 5. Integration Testing
For tests, disable logging to avoid noise:

```python
# In test setup
imputation.disable_logging()

# Or use quiet file logging only
imputation.configure_logging(
    console=False,
    file_logging=True,
    log_dir='test_logs'
)
```

## Migration from Previous Version

If you were using the package before this logging update, no changes are required. The package is now **completely silent by default** - no logging messages or warnings will appear unless you explicitly enable logging.

**Before (old version):** Package would show logging messages automatically
**After (new version):** Package is silent by default, you must call `imputation.configure_logging()` to see logging messages

This follows Python package best practices - libraries should be silent by default.

## Troubleshooting

### No log messages appearing
1. Check if logging is configured: `imputation.configure_logging()`
2. Verify log level: `imputation.configure_logging(level='DEBUG')`
3. Check console level: `imputation.configure_logging(console_level='DEBUG')`

### Log files not created
1. Check directory permissions
2. Verify log directory exists or can be created
3. Check disk space

### Too many log messages
1. Increase log level: `imputation.configure_logging(level='WARNING')`
2. Reduce console output: `imputation.configure_logging(console_level='ERROR')`

### Integration issues
If the package logging interferes with your application:

```python
# Use quiet file-only logging
imputation.configure_logging(
    console=False,
    file_logging=True,
    log_dir='separate_logs'
)
```

## Examples

See `example_logging_usage.py` for complete working examples of all logging configurations.
