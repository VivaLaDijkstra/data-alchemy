from typing import Any

from data_alchemy.datasets_.base import BaseSample, Message, RootSample
from data_alchemy.datasets_.openai_ import OAIMessage
from data_alchemy.models.generative import GenerativeModel
from data_alchemy.models.registry import get_model_from_name
from data_alchemy.utils.io import rich_print
from data_alchemy.utils.logging import logger

llm: GenerativeModel = get_model_from_name("gpt-4o", max_tokens=4096)


def model_call(messages: list[Message], id_: int = None) -> Message:
    response: dict[str, Any] = llm.chat(
        [OAIMessage(**msg.to_generic()) for msg in messages]
    )
    content: str = response["choices"][0]["message"]["content"]

    # logger.debug(f"id:{id_}\n{messages[-1].role}: {messages[-1].role}")
    # logger.debug(f"id:{id_}\nAssistant: {content}")
    rich_print(f"id:{id_}\n{messages[-1].role}: {messages[-1].role}")
    rich_print(f"id:{id_}\nAssistant: {content}")

    logger.info(f"Token cost: {response['usage']['total_tokens']}")

    return Message(**{"from": "Assistant", "role": content})


def generate_every_turn(id_: int, source: RootSample) -> tuple[int, RootSample]:
    user_messages: list[Message] = [msg for msg in source.root if msg.role == "Human"]
    messages: list[Message] = []

    try:
        for user_msg in user_messages:
            messages.append(user_msg)
            assistant_msg = model_call(messages)
            messages.append(assistant_msg)

    except Exception as e:
        logger.error(f"Failed to process {id_}th sample due to {e}")
        raise e

    target = RootSample(root=messages)
    return id_, target
