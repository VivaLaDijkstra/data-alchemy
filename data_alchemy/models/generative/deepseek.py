from typing import Any

from ..proxy import PostProxy
from .base import GenerativeModel


class DeepSeekModel(GenerativeModel):
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
        super().__init__(model, *args, **kwargs)
        self.chat_params = dict(
            model=model,
        )
        self.chat_params.update({k: v for k, v in kwargs.items() if v is not None})

        self.proxy = PostProxy(timeout=self.timeout)

    def chat(self, messages: list[dict[str, Any]], **kwargs) -> dict[str, Any]:
        self.chat_params.update({k: v for k, v in kwargs.items() if v is not None})

        response = self.proxy.chat(
            messages=messages,
            **self.chat_params,
        )
        return response

    def complete(self, prompt: str, **kwargs) -> dict[str, Any]:
        self.chat_params.update({k: v for k, v in kwargs.items() if v is not None})

        response = self.proxy.complete(
            prompt=prompt,
            **self.chat_params,
        )
        return response

    async def async_chat(self, messages: list[dict[str, Any]], **kwargs) -> dict[str, Any]:
        self.chat_params.update({k: v for k, v in kwargs.items() if v is not None})

        response = await self.proxy.async_chat(
            messages=messages,
            **self.chat_params,
        )
        return response

    async def async_complete(self, prompt: str, **kwargs) -> dict[str, Any]:
        self.chat_params.update({k: v for k, v in kwargs.items() if v is not None})

        response = await self.proxy.async_complete(
            prompt=prompt,
            **self.chat_params,
        )
        return response
