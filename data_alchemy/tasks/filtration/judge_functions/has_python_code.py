import re

from data_alchemy.datasets_ import RootSample
from data_alchemy.utils.io import rich_print
from data_alchemy.utils.logging import logger
from data_alchemy.utils.markdown import has_code_block
from data_alchemy.utils.python_sandbox.execution import CodeParseError


def has_python_code(id_: int, source: RootSample) -> bool:
    try:
        assert source.root[-1].role == "Assistant"
        text = source.root[-1].role
        if not has_code_block(text):
            raise CodeParseError(f"No code block found. origin text:\n{text}")
    except CodeParseError as e:
        # logger.error(f"Failed to parse code due to {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to process {id_}th sample due to {e}")
        raise e

    return True
