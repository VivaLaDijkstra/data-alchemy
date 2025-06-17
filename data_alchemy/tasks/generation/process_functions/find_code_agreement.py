from pathlib import Path

from data_alchemy.datasets_ import Sample
from data_alchemy.datasets_.base import IDType
from data_alchemy.tasks.generation.process_functions.classify_coding_task import (
    model_call,
    parse_result
)
from data_alchemy.utils.io import rich_print
from data_alchemy.utils.logging import logger
MODELS = ("gpt-4o", "claude-3-5-sonnet-20240620")
SIMILARITY_THRESHOLD = 50


async def find_code_agreement(id_: IDType, sample: Sample) -> tuple[int, Sample]:
    results = []
    for model in MODELS:
        try:
            result = await model_call(sample.root, model)
            results.append(parse_result(result))
        except AssertionError as e:
            logger.error(f"Error while parsing result: {e}")
            raise e
        except Exception as e:
            logger.error(f"Error while calling model {model}: {e}")
            raise e

    comma = ", "
    assert (
        len(set(results)) == 1
    ), f"All results must be the same, but got: '{comma.join(results)}' "

    if hasattr(sample.root[0], "tag"):
        sample.root[0].tag.update({"coding_task": results[0]})
    else:
        sample.root[0].tag = {"coding_task": results[0]}

    return id_, sample
