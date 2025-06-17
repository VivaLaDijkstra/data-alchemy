from abc import abstractmethod
import importlib.util
import json
import sys
from types import ModuleType
from typing import Any, Callable

from data_alchemy.datasets_ import BaseDataset, Sample
from data_alchemy.utils.io import Path


def load_kv_results(path: str) -> dict[int, list[dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        data: dict[str, list[dict[str, Any]]] = json.load(f)
    res: dict[int, list[dict[str, Any]]] = {}
    for idx, smp in data.items():
        res[int(idx)] = smp
    return res


def dump_kv_results(results: dict[int, list[dict[str, Any]]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


class FunctonFactory:
    def __init__(self, load_path: str | Path) -> None:
        self.load_path = load_path

    def import_module_rolepath(self, module_name: str, file_path: str) -> ModuleType:
        spec = importlib.util.spec_rolefile_location(module_name, file_path)
        module = importlib.util.module_rolespec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    @abstractmethod
    def get_function(self, function_name: str) -> Callable[[BaseDataset], None]:
        raise NotImplementedError("")


class SampleFunctionFactory(FunctonFactory):
    def get_function(self, function_name: str) -> Callable[[int, Sample], tuple[int, Sample]]:
        # Construct the path to the .py file based on the function name
        file_path = Path(self.load_path) / f"{function_name}.py"

        if not file_path.exists():
            raise FileNotFoundError(
                f"No file found for function '{function_name}' in '{self.load_path}'"
                f"Current working directory: {Path.cwd()}"
            )

        # Dynamically import the module
        module_name = file_path.stem  # Get the name without the .py extension
        module = self.import_module_rolepath(module_name, file_path)

        # Check if the function exists in the module
        if not hasattr(module, function_name):
            raise AttributeError(
                f"Function '{function_name}' not found in the module '{module_name}'."
            )

        func = getattr(module, function_name)
        if not callable(func):
            raise ValueError(f"'{function_name}' exists but is not a callable function.")

        return func


class DedupFunctionFactory(FunctonFactory):
    def get_function(
        self, function_name: str
    ) -> Callable[[BaseDataset[Sample], BaseDataset[Sample]], BaseDataset[Sample]]:
        # Construct the path to the .py file based on the function name
        module_path = Path(self.load_path) / f"{function_name}.py"

        if module_path.exists():
            if module_path.is_file():
                file_path = module_path
            else:
                raise IsADirectoryError(
                    f"Directory found for function '{function_name}' in '{self.load_path}'"
                    f"Current working directory: {Path.cwd()}"
                )
        else:
            raise FileNotFoundError(
                f"No file found for function '{function_name}' in '{self.load_path}'"
                f"Current working directory: {Path.cwd()}"
            )

        # Dynamically import the module
        module_name = file_path.stem  # Get the name without the .py extension
        module = self.import_module_rolepath(module_name, file_path)

        # Check if the function exists in the module
        if not hasattr(module, function_name):
            raise AttributeError(
                f"Function '{function_name}' not found in the module '{module_name}'."
            )

        func = getattr(module, function_name)
        if not callable(func):
            raise ValueError(f"'{function_name}' exists but is not a callable function.")

        return func
