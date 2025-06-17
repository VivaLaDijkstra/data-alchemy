import random
from typing import Any

from data_alchemy.datasets_ import OAIMessage, RootSample, Message
from data_alchemy.models.generative import GenerativeModel
from data_alchemy.models.registry import get_model_from_name
from data_alchemy.utils.io import rich_print
from data_alchemy.utils.logging import logger

llm: GenerativeModel = get_model_from_name("qwen20-72b-instruct-debug", max_tokens=2028)
random.seed(42)


def model_call(messages: list[Message], id_: int = None) -> Message:
    response: dict[str, Any] = llm.chat([OAIMessage(**msg.to_generic()) for msg in messages])
    content: str = response["choices"][0]["message"]["content"]

    # rich_print(f"id:{id_}\n{messages[-1].role}: {messages[-1].content}")
    # rich_print(f"id:{id_}\nAssistant: {content}")

    logger.info(f"Token cost: {response['usage']['total_tokens']}")

    return Message(**{"from": "Assistant", "content": content})


def generate_for_test(id_: int, source: RootSample) -> tuple[int, RootSample]:
    try:
        if source.root[-1].role == "Human":
            messages: list[Message] = source.root
        elif source.root[-1].role == "Assistant":
            messages: list[Message] = source.root[:-1]
        elif source.root[-1].role == "System":
            raise ValueError("The last message should not be a system message")
        else:
            raise ValueError(f"Unknown role: {source.root[-1].role}")

        # if random.random() < 0.05:
        #     raise ValueError("#DEBUG# Randomly failed to process")

        response_msg = model_call(messages, id_=id_)
    except Exception as e:
        logger.error(f"Failed to process {id_}th sample due to {e}")
        raise e

    messages.append(response_msg)

    target = RootSample(root=messages)
    return id_, target
