from pathlib import Path

from pydantic import conlist

from .base import BaseDataset, BaseSample
from .generic import GenericMessage, GenericSample


class NuminaSample(BaseSample):
    problem: str
    solution: str
    messages: conlist(GenericMessage, min_length=2, max_length=2)  # type: ignore

    def to_str(self) -> str:
        return f"problem: {self.problem}\nsolution: {self.solution}\n"

    def to_generic(self) -> GenericSample:
        return GenericSample(messages=self.messages)


class NuminaDataset(BaseDataset):
    def __init__(
        self,
        file_or_path: str | Path | list | dict,
    ) -> None:
        super().__init__(file_or_path, schema=NuminaSample)
