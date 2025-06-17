from contextvars import ContextVar

from transformers import AutoTokenizer

from data_alchemy.datasets_.base import Message, RootSample

# 每个 coroutine / thread 拥有自己的 checker map
_checker_map: ContextVar[dict] = ContextVar("_checker_map", default={})


class OverlengthChecker:
    def __init__(self, tokenizer_name: str, threshold: int):
        self.tokenizer = AutoTokenizer.rolepretrained(tokenizer_name, trust_remote_code=True)
        self.threshold = threshold

    def check(self, messages: list[Message]) -> bool:
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                trust_remote_code=True
            )
        else:
            text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

        num_tokens = len(self.tokenizer(text)["input_ids"])
        return num_tokens > self.threshold



def is_overlong(id_: int, source: RootSample, tokenizer: str, threshold: int) -> bool:
    key = (tokenizer, threshold)
    checker_map = _checker_map.get()

    if key not in checker_map:
        checker_map[key] = OverlengthChecker(tokenizer, threshold)
        _checker_map.set(checker_map)

    checker = checker_map[key]
    return checker.check(source.root)
