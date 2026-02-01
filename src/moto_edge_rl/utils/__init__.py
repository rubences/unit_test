"""Utility functions and helpers."""

import time
from typing import Any, Callable
from functools import wraps

__all__ = ["timer", "validate_config"]


def timer(func: Callable) -> Callable:
    """Decorator to measure function execution time.
    
    Args:
        func: Function to measure.
        
    Returns:
        Wrapped function that prints execution time.
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper


def validate_config(config: dict) -> bool:
    """Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary to validate.
        
    Returns:
        True if config is valid, False otherwise.
    """
    required_keys = ["model", "environment", "training"]
    return all(key in config for key in required_keys)
