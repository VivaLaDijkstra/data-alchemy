from pathlib import Path

from pydantic import Field

from .base import BaseDataset, BaseSample
from .generic import GenericMessage, GenericSample


class MetaMathQASample(BaseSample):
    query: str
    response: str
    type_: str = Field(..., alias="type")


class MetaMathQAEnSample(MetaMathQASample):
    query: str
    response: str
    type_: str = Field(..., alias="type")

    def to_str(self) -> str:
        return f"{self.query}\n{self.response}"

    def to_generic(self) -> GenericSample:
        return GenericSample(
            messages=[
                GenericMessage(role="user", content=self.query),
                GenericMessage(role="assistant", content=self.response),
            ]
        )


class MetaMathQAZhSample(MetaMathQASample):
    query_zh: str
    response_zh: str

    def to_str(self) -> str:
        return f"{self.query_zh}\n{self.response_zh}"

    def to_generic(self) -> GenericSample:
        return GenericSample(
            messages=[
                GenericMessage(role="user", content=self.query_zh),
                GenericMessage(role="assistant", content=self.response_zh),
            ]
        )


class MetaMathQADataset(BaseDataset):
    def __init__(
        self,
        file_or_path: str | Path | list | dict,
        schema: type[MetaMathQASample] = MetaMathQASample,
    ) -> None:
        super().__init__(file_or_path, schema=schema)


class MetaMathQAEnDataset(MetaMathQADataset):
    def __init__(self, file_or_path: str | Path | list | dict) -> None:
        super().__init__(file_or_path, schema=MetaMathQAEnSample)


class MetaMathQAZhDataset(MetaMathQADataset):
    def __init__(self, file_or_path: str | Path | list | dict) -> None:
        super().__init__(file_or_path, schema=MetaMathQAZhSample)
