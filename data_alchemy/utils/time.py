from datetime import datetime
from functools import wraps
from time import time
from typing import Callable

from data_alchemy.utils.io import rich_print


def timeit(func: Callable) -> Callable:
    """
    Decorator to time a function.

    Args:
        func (Callable): Function to time.

    Returns:
        Callable: Wrapped function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        tic = time()
        result = func(*args, **kwargs)
        toc = time()
        rich_print(f"function {func.__name__} took {toc - tic:.3f}s")
        return result

    return wrapper


def get_timestamp() -> str:
    """
    Returns the current timestamp in the format YYYYMMDDHHMMSS.

    Returns:
        str: Current timestamp.
    """
    return datetime.now().strftime("%Y%m%d%H%M%S")


def str2time(time_str: str) -> datetime:
    """
    Converts a string to a timestamp.

    Args:
        time_str (str): String to convert.

    Returns:
        float: Timestamp.
    """
    return datetime.strptime(time_str, "%Y%m%d%H%M%S")
