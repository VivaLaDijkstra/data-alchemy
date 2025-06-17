from collections import Counter

from data_alchemy.datasets_.base import Message, RootSample


def is_single_turn(id_: int, source: RootSample) -> bool:
    """check if the sample only has one turn"""
    messages: list[Message] = source.root

    counter = Counter([msg.role.lower() for msg in messages])

    if 2 <= len(counter) <= 3:
        if 0 <= counter["system"] <= 1 and counter["assistant"] == 1 and 1 <= counter["human"] <= 2:
            return True
    return False
