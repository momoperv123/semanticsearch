"""
Timing utilities for performance measurement.
"""

import time
from contextlib import contextmanager
from typing import Generator


@contextmanager
def timer(description: str = "Operation") -> Generator[None, None, None]:
    """
    Context manager for timing operations.

    Usage:
        with timer("Processing images"):
            process_images()

    Args:
        description: Description of the operation being timed
    """
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        print(f"{description} took {elapsed:.2f} seconds")


def time_function(func):
    """
    Decorator to time a function execution.

    Usage:
        @time_function
        def my_function():
            ...
    """

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__} took {elapsed:.2f} seconds")
        return result

    return wrapper
