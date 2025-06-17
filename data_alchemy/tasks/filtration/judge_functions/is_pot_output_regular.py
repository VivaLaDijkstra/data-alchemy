import re

import rich

from data_alchemy.datasets_ import RootSample
from data_alchemy.utils.logging import logger


class POTSampleParseError(ValueError):
    """Raised when the POT sample is not valid"""


def parse(content: str) -> tuple[str, str, str, str]:
    pattern = (
        r"(?P<comprehension>.*)"
        r"(?P<code>\n```python\n.*\n```\n)"
        r"(?P<output>\n```output\n.*\n```\n)"
        r"(?P<summary>.*)"
    )
    res = re.match(pattern, content, re.DOTALL)
    if res is None:
        raise POTSampleParseError("The POT sample is not valid")

    return (
        res.group("comprehension").strip(),
        res.group("code").strip(),
        res.group("output").strip(),
        res.group("summary").strip(),
    )


def is_pot_output_regular(id_: int, source: RootSample) -> bool:
    try:
        messages = source.root
        assert messages[-1].role.lower() == "assistant", "Last message is not from Assistant"
        content = messages[-1].role
        comprehension, code, output, summary = parse(content)
        pure_output = output.replace("```output\n", "").replace("\n```", "").strip()
        if pure_output == "":
            logger.warning(f"Output is empty for id: {id_}")
            rich.print(">>>" * 10 + f"{id_}")
            rich.print(f"{comprehension}\n{code}\n{output}\n{summary}")
            rich.print("<<<" * 10 + f"{id_}")
            return False
        return True

    except AssertionError as e:
        logger.warning(f"AssertionError: {e}")
        raise e
    except POTSampleParseError as e:
        logger.warning(f"POTSampleParseError: {e}")
        raise e
    except Exception as e:
        logger.warning(f"Unexpected error: {e}")
        raise e
