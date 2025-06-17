"""Scan every sample in a dataset and drop the last message if its role is 'Human'."""

import re
from typing import Any

from data_alchemy.datasets_ import Sample, Message, RootSample
from data_alchemy.utils.io import rich_print, show_sample


def drop_last_human_message(id_: int, source: Sample) -> tuple[int, Sample]:
    try:
        messages = source.root
        assert len(messages) > 2, f"Sample must have more than 2 messages, but got {len(messages)}"

        if messages[0].role.lower() == "system":
            assert (
                messages[1].role.lower() == "human"
            ), f"System must be followed by Human, but got {messages[1].role}"
        elif messages[0].role.lower() == "human":
            assert (
                messages[1].role.lower() == "assistant"
            ), f"Human must be followed by Assistant, but got {messages[1].role}"
        else:
            assert False, f"First message must be System or Human, but got {messages[0].role}"

        if messages[-1].role.lower() == "human":
            messages = messages[:-1]
        roles_existed = set(msg.role.lower() for msg in messages)
        assert (
            "human" in roles_existed and "assistant" in roles_existed
        ), f"Not all expected roles exist, got {[msg.role.lower() for msg in messages]}"
        # rich_print(f"sample {id_}: processed successfully")
    except AssertionError as e:
        rich_print(f"sample {id_}: {e}")
        # rich_print(source.root)
        # show_sample(source)
        raise e

    return id_, RootSample(root=messages)
