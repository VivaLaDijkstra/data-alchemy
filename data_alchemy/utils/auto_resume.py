import functools
import hashlib
import inspect
import json
import pickle
from inspect import iscoroutinefunction
from pathlib import Path
from typing import Any, Callable, Optional

from data_alchemy.datasets_.base import BaseMessage, BaseSample, RootSample
from data_alchemy.utils.io import rich_print
from data_alchemy.utils.logging import logger


def to_cache_key(obj: Any) -> str:
    if isinstance(obj, (str, int, float, bool)):
        return str(obj)
    elif isinstance(obj, list):
        return f"[{', '.join(to_cache_key(item) for item in obj)}]"
    elif isinstance(obj, tuple):
        return f"({', '.join(to_cache_key(item) for item in obj)})"
    elif isinstance(obj, set):
        return f"{{{', '.join(sorted(to_cache_key(item) for item in obj))}}}"
    elif isinstance(obj, dict):
        return f"{{{', '.join(sorted(f'{k}: {to_cache_key(v)}' for k, v in obj.items()))}}}"
    elif isinstance(obj, (BaseSample, RootSample, BaseMessage)):
        return json.dumps(obj.to_raw())
    elif isinstance(obj, Cacheable):
        return f"{obj.__class__}"
    else:
        logger.warning(
            f"type `{type(obj)}` is neither a basic type nor a Cacheable."
            " This may lead to consistently different cache keys."
        )
        return str(obj)


class Cacheable:
    def __str__(self) -> str:
        return to_cache_key(self)

    def __repr__(self) -> str:
        return to_cache_key(self)


# Helper function to create a cache key from the function arguments
def _generate_cache_key(
    func: Callable,
    args: tuple[Any, ],
    kwargs: dict[str, Any],
    include_source: bool = True,
) -> tuple[str, str]:
    func_name = func.__name__
    func_source = inspect.getsource(func) if include_source else None
    func_args = to_cache_key(args)
    func_kwargs = to_cache_key(kwargs)

    key_string = json.dumps(
        dict(
            func_name=func_name,
            func_args=func_args,
            func_kwargs=func_kwargs,
            func_source=func_source,
        ),
        sort_keys=True,
        ensure_ascii=False,
    )

    key_hash = hashlib.sha256(key_string.encode()).hexdigest()
    return key_hash, key_string


def _prepare_cache_dir(cache_dir: Optional[str | Path] = None) -> None:
    """Directory where cached results will be stored"""

    if not cache_dir:
        raise ValueError("Cache directory must be specified")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)


def model_cache(
    cache_dir: Optional[str | Path] = None,
    include_source: bool = True,
    verbose: bool = False,
    offset: int = 128,
) -> Callable:
    _prepare_cache_dir(cache_dir)

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Create a unique cache key based on the input arguments
            digested, origin = _generate_cache_key(func, args, kwargs, include_source)
            cache_file_path = Path(cache_dir) / f"{digested}.pkl"

            # Check if the cached result exists on the filesystem
            if cache_file_path.exists():
                if verbose:
                    rich_print(
                        f"[cyan bold]Cache key hit.[/] [green]`{origin[:offset]}`[/]"
                    )
                with open(cache_file_path, "rb") as f:
                    result = pickle.load(f)
            else:
                if verbose:
                    rich_print(
                        f"[yellow bold]Cache key miss.[/] [green]`{origin[:offset]}`[/] "
                    )
                result = func(*args, **kwargs)

                # Save the result to the cache
                with open(cache_file_path, "wb") as f:
                    pickle.dump(result, f)

            return result

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            # Create a unique cache key based on the input arguments
            digested, origin = _generate_cache_key(func, args, kwargs, include_source)
            cache_file_path = Path(cache_dir) / f"{digested}.pkl"

            # Check if the cached result exists on the filesystem
            if cache_file_path.exists():
                if verbose:
                    rich_print(
                        f"[cyan bold]Cache key hit.[/] [green]`{origin[:offset]}`[/]"
                    )
                with open(cache_file_path, "rb") as f:
                    result = pickle.load(f)
            else:
                if verbose:
                    rich_print(
                        f"[yellow bold]Cache key miss.[/] [green]`{origin[:offset]}`[/] "
                    )
                result = await func(*args, **kwargs)

                # Save the result to the cache
                with open(cache_file_path, "wb") as f:
                    pickle.dump(result, f)

            return result

        return async_wrapper if iscoroutinefunction(func) else wrapper

    return decorator
