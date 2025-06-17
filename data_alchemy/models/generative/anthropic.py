from typing import Any

from data_alchemy.datasets_.base import Message

from ..proxy import SDKProxy
from .base import GenerativeModel


class ClaudeModel(GenerativeModel):
    def __init__(
        self,
        model: str,
        # temperature: Optional[float] = None,
        # max_tokens: Optional[int] = None,
        # stream: Optional[bool] = None,
        # top_p: Optional[float] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, *args, **kwargs)
        self.hyper_params = dict(model=model)
        self.hyper_params.update(kwargs)
        self.proxy = SDKProxy(timeout=self.timeout)

    def chat(self, messages: list[Message | dict[str, Any]], **kwargs) -> dict[str, Any]:
        self.hyper_params.update(kwargs)
        response = self.proxy.chat(
            messages=messages,
            **self.hyper_params,
        )
        return response

    def complete(self, prompt: str, **kwargs) -> dict[str, Any]:
        self.hyper_params.update(kwargs)
        response = self.proxy.complete(
            prompt=prompt,
            **self.hyper_params,
        )
        return response

    async def async_chat(
        self, messages: list[Message | dict[str, Any]], **kwargs
    ) -> dict[str, Any]:
        self.hyper_params.update(kwargs)
        response = await self.proxy.async_chat(
            messages=messages,
            **self.hyper_params,
        )
        return response

    async def async_complete(self, prompt: str, **kwargs) -> dict[str, Any]:
        self.hyper_params.update(kwargs)
        response = await self.proxy.async_complete(
            prompt=prompt,
            **self.hyper_params,
        )
        return response
