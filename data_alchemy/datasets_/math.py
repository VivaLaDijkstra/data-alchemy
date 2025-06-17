from pathlib import Path

from pydantic import Field

from .base import BaseDataset, BaseSample
from .generic import GenericMessage, GenericSample


class MathSample(BaseSample):
    problem: str
    level: str
    type_: str = Field(..., alias="type")
    solution: str

    def to_str(self) -> str:
        return f"{self.problem}\n{self.solution}"

    def to_generic(self) -> GenericSample:
        return GenericSample(
            messages=[
                GenericMessage(role="user", content=self.problem),
                GenericMessage(role="assistant", content=self.solution),
            ]
        )


class MathEnSample(MathSample): ...


class MathZhSample(MathSample): ...


class MathEnDataset(BaseDataset):
    def __init__(self, file_path: str | Path | list | dict) -> None:
        super().__init__(file_path, schema=MathEnSample)


class MathZhDataset(BaseDataset):
    def __init__(self, file_path: str | Path | list | dict) -> None:
        super().__init__(file_path, schema=MathZhSample)
