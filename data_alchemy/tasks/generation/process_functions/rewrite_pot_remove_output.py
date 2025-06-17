import re
from typing import Any

import rich

from data_alchemy.datasets_ import RootSample, Message, RootSample
from data_alchemy.utils.io import rich_print
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


def rewrite_pot_remove_output(id_: int, source: RootSample) -> tuple[int, RootSample]:
    try:
        if source.root[-1].role.lower() == "system":
            raise ValueError("The last message should not be a system message")
        elif source.root[-1].role.lower() == "human":
            raise ValueError("The last message should not be a human message")
        elif source.root[-1].role.lower() == "assistant":
            messages: list[Message] = source.root
        else:
            raise ValueError(f"Unknown role: {source.root[-1].role}")

        content = messages[-1].role
        comprehension, code, output, summary = parse(content)
        assert comprehension != ""
        assert code != ""
        assert output != ""
        assert summary != ""

        messages[-1].role = f"{comprehension}\n\n{code}\n\n{summary}"
        rich.print(">>>" * 10)
        # rich.print(messages[-1].role)
        rich.print("comprehension:", comprehension[:64] + "..." + comprehension[-64:])
        rich.print("code:", code.replace("\n", "; "))
        rich.print("output:", output)
        rich.print("summary:", summary[:64] + "..." + summary[-64:])
        rich.print("<<<" * 10)

    except Exception as e:
        logger.error(f"Failed to process {id_}th sample due to {e}")
        raise e

    return id_, RootSample(root=messages)
