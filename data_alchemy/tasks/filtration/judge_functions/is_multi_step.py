import re
from collections import Counter

from data_alchemy.datasets_.base import Message, RootSample
from data_alchemy.utils.io import rich_print
from data_alchemy.utils.logging import logger


def is_multi_step(id_: int, source: RootSample) -> bool:
    """check if the sample has multi coding-output steps"""

    messages: list[Message] = source.root
    assert len(messages) == 2, f"Unexpected number of messages in sample {id_}"
    assert messages[-1].role == "Assistant", f"Unexpected message sender in sample {id_}"
    raw_string: str = messages[-1].content

    pattern = re.compile(r"```python\n(.*?)```[\n]*```output\n(.*?)```", re.DOTALL)
    code_output_pairs = []
    try:
        for match in re.finditer(pattern, raw_string):
            code = match.group(1).strip()  # Extract the code block
            output = match.group(2).strip()  # Extract the output block
            code_output_pairs.append({"code": code, "output": output})
        return len(code_output_pairs) > 1
    except Exception as e:  # pylint: disable=broad-except
        logger.error(f"Error while parsing code-output pairs for sample {id_}: {e}")
        return False
