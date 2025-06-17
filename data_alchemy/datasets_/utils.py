from functools import reduce

from pydantic import ValidationError

from data_alchemy.datasets_.generic import GenericSample
from data_alchemy.utils.io import load_first_sample
from data_alchemy.utils.logging import logger

from . import SCHEMAS
from .base import BaseDataset, BaseSample, RootSample, Sample


def identify_schema(path: str) -> type[BaseSample | RootSample]:
    try:
        sample = load_first_sample(path)
    except StopIteration as e:
        logger.warning(f"'{path}' has 0 sample")
        return SCHEMAS[0]

    for schema in SCHEMAS:
        try:
            logger.info(f"trying schema: {schema}")

            if isinstance(sample, dict):
                schema(**sample)
            elif isinstance(sample, list):
                if schema == GenericSample:
                    schema(messages=sample)
                else:
                    schema(root=sample)
            else:
                raise TypeError(f"Unsupported type for sample[0]: {type(sample)}\nsample: {sample}")

            logger.info(f"using schema: {schema}")
            return schema
        except ValidationError as e:
            logger.debug(e)

    raise TypeError(f"no dynamic schema matched for file {path}")


def load_dataset(
    path: str,
    schema: type[BaseSample | RootSample] | None = None,
) -> BaseDataset | tuple[BaseDataset, type[Sample]]:
    if schema is None:
        schema = identify_schema(path)

    # instantiate dataset
    dataset = BaseDataset(path, schema=schema)
    return dataset


def load_dataset_with_schema(
    path: str,
) -> tuple[BaseDataset, type[BaseSample | RootSample]]:
    schema = identify_schema(path)
    dataset = BaseDataset(path, schema=schema)
    return dataset, schema


def concat_datasets(
    datasets: list[BaseDataset[Sample]],
) -> BaseDataset[Sample]:
    return reduce(lambda x, y: x + y, datasets)
