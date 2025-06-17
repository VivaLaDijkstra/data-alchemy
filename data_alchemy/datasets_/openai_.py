from pathlib import Path
from typing import Literal

from .base import BaseDataset, BaseMessage, RootSample
from .generic import GenericMessage, GenericSample


class OAIMessage(BaseMessage):
    role: Literal["system", "user", "assistant"]
    content: str

    def to_str(self) -> str:
        return f"{self.role}: {self.content}"

    def to_generic(self) -> GenericSample:
        return GenericMessage(role=self.role, content=self.content)


class OAISample(RootSample):
    root: list[OAIMessage]

    def to_str(self, with_system_prompt: bool = False) -> str:
        if with_system_prompt:
            return "\n".join(msg.to_str() for msg in self.root)
        else:
            return "\n".join(msg.to_str() for msg in self.root[1:])

    def to_generic(self) -> GenericSample:
        return GenericSample(
            messages=[msg.to_generic() for msg in self.root],
        )


class OAIDataset(BaseDataset):
    def __init__(
        self,
        file_or_path: str | Path | list | dict,
    ) -> None:
        super().__init__(file_or_path, schema=OAISample)
