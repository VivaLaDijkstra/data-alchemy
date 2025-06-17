from typing import Any

from ..proxy import SDKProxy
from .base import GenerativeModel


class GPTModel(GenerativeModel):
    def __init__(
        self,
        model: str,
        # temperature: Optional[float] = None,
        # max_tokens: Optional[int] = None,
        # stream: Optional[bool] = None,
        # top_p: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.hyper_params: dict[str, Any] = dict(
            model=model,
        )
        assert (
            "model" not in kwargs
        ), f"model must be specified as the 1st argument, but got '{model}'"

        self.hyper_params.update({k: v for k, v in kwargs.items() if v is not None})

        self.proxy = SDKProxy(timeout=self.timeout)

    def chat(self, messages: list[dict[str, Any]], **kwargs) -> dict[str, Any]:
        self.hyper_params.update({k: v for k, v in kwargs.items() if v is not None})

        response = self.proxy.chat(
            messages=messages,
            **self.hyper_params,
        )
        return response

    def complete(self, prompt: str, **kwargs) -> dict[str, Any]:
        self.hyper_params.update({k: v for k, v in kwargs.items() if v is not None})

        response = self.proxy.complete(
            prompt=prompt,
            **self.hyper_params,
        )
        return response

    async def async_chat(self, messages: list[dict[str, Any]], **kwargs) -> dict[str, Any]:
        self.hyper_params.update({k: v for k, v in kwargs.items() if v is not None})

        response = await self.proxy.async_chat(
            messages=messages,
            **self.hyper_params,
        )
        return response

    async def async_complete(self, prompt: str, **kwargs) -> dict[str, Any]:
        self.hyper_params.update({k: v for k, v in kwargs.items() if v is not None})

        response = await self.proxy.async_complete(
            prompt=prompt,
            **self.hyper_params,
        )
        return response
