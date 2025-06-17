from typing import Any

from data_alchemy.datasets_.base import Message

from data_alchemy.models.base import BaseModel


class GenerativeModel(BaseModel):
    """Base class for generative models."""

    def __init__(self, model: str, *args: Any, **kwargs: Any):
        """Initialize the generative model."""
        self.model = model

        self.temperature = kwargs.get("temperature", None)
        self.top_p = kwargs.get("top_p", None)
        self.max_tokens = kwargs.get("max_tokens", None)
        self.timeout = kwargs.get("timeout", None)
        self.n = kwargs.get("n", None)


    def chat(
        self, messages: list[Message | dict[str, Any]], **kwargs
    ) -> dict[str, Any]:
        """Chat with the model."""
        raise NotImplementedError("Subclass must implement chat method")

    def complete(self, prompt: str, **kwargs) -> dict[str, Any]:
        """Complete a prompt."""
        raise NotImplementedError("Subclass must implement complete method")

    async def async_chat(
        self, messages: list[Message | dict[str, Any]], **kwargs
    ) -> dict[str, Any]:
        """Async chat with the model."""
        raise NotImplementedError("Subclass must implement achat method")

    async def async_complete(self, prompt: str, **kwargs) -> dict[str, Any]:
        """Async complete a prompt."""
        raise NotImplementedError("Subclass must implement async_complete method")


class ChatParams:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs: dict[str, Any] = kwargs

    def update(self, kwargs: dict[str, Any]) -> None:
        self.kwargs.update(kwargs)

    def keys(self) -> list[str]:
        return list(self.kwargs.keys())

    def __getitem__(self, key: str) -> Any:
        return self.kwargs[key]
