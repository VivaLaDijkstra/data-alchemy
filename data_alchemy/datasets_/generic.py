from typing import Literal

from pydantic import ConfigDict

from .base import BaseMessage, BaseSample


class GenericMessage(BaseMessage):
    role: Literal["system", "user", "assistant", "human"]
    content: str

    model_config = ConfigDict(extra="allow")

    def to_generic(self) -> "GenericMessage":
        return self

    def to_str(self):
        return f"{self.role}: {self.content}"


class GenericSample(BaseSample):
    messages: list[GenericMessage]

    model_config = ConfigDict(extra="allow")

    def to_generic(self) -> "GenericSample":
        return self

    def to_str(self):
        return "\n".join(msg.to_str() for msg in self.messages)
