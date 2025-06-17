from pathlib import Path
from typing import Literal

from .base import BaseDataset, BaseSample
from .generic import GenericMessage, GenericSample


class GSM8kSample(BaseSample):
    question: str
    answer_only: str
    answer: str
    split: Literal["train", "validation", "test"]


class GSM8kEnSample(GSM8kSample):
    def to_str(self) -> str:
        return f"{self.question}\n{self.answer}"

    def to_generic(self) -> GenericSample:
        return GenericSample(
            messages=[
                GenericMessage(role="user", content=self.question),
                GenericMessage(role="assistant", content=self.answer),
            ]
        )


class GSM8kZhSample(GSM8kSample):
    question_zh: str
    answer_zh: str

    def to_str(self) -> str:
        return f"{self.question_zh}\n{self.answer_zh}"

    def to_generic(self) -> GenericSample:
        return GenericSample(
            messages=[
                GenericMessage(role="user", content=self.question_zh),
                GenericMessage(role="assistant", content=self.answer_zh),
            ]
        )


class GSM8kDataset(BaseDataset):
    def __init__(
        self,
        file_or_path: str | Path | list | dict,
        schema: type[GSM8kEnSample | GSM8kZhSample] = GSM8kEnSample,
    ) -> None:
        super().__init__(file_or_path, schema=schema)


class GSM8kEnDataset(GSM8kDataset):
    def __init__(self, file_or_path: str | Path | list | dict) -> None:
        super().__init__(file_or_path, schema=GSM8kEnSample)


class GSM8kZhDataset(GSM8kDataset):
    def __init__(self, file_or_path: str | Path | list | dict) -> None:
        super().__init__(file_or_path, schema=GSM8kZhSample)
