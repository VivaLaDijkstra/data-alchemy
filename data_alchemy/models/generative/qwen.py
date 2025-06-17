from typing import Any, Optional

from ..proxy import PostProxy
from .base import GenerativeModel


def apply_chat_template(
    conversation: list[dict[str, str]],
    # tools: Optional[list[dict]] = None,  # TODO: not supported yet
    # documents: Optional[list[dict[str, str]]] = None,  # TODO: not supported yet
    # chat_template: Optional[str] = None,  # TODO: not supported yet
    add_generation_prompt: bool = False,
    **kwargs,  # keep consistency with the huggingface transformers chat template
) -> str:
    """
    mimic the huggingface transformers chat template behavior

    reference:
        doc: https://huggingface.co/docs/transformers/main/en/chat_templating
        code: https://github.com/huggingface/transformers/blob/e878eaa9fc4da9cec1c74ae962e89092b6832db8/src/transformers/tokenization_utils_base.py#L1709
    """
    prompt = ""
    for msg in conversation:
        prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"

    if add_generation_prompt:
        prompt += "<|im_start|>assistant\n"

    return prompt


class QwenModel(GenerativeModel):
    def __init__(
        self,
        model: str,
        # temperature: Optional[float] = None,
        max_tokens: Optional[int] = 2048,
        # stream: Optional[bool] = None,
        # top_p: Optional[float] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, *args, **kwargs)
        self.chat_params = dict(model=model, max_tokens=max_tokens)
        self.chat_params.update({k: v for k, v in kwargs.items() if v is not None})

        self.proxy = PostProxy(timeout=self.timeout)

    def chat(self, messages: list[dict[str, Any]], **kwargs) -> dict[str, Any]:
        self.chat_params.update({k: v for k, v in kwargs.items() if v is not None})

        response = self.proxy.complete(
            prompt=apply_chat_template(messages, add_generation_prompt=True),
            **self.chat_params,
        )
        # remake the response to mimic the chat completion response format
        response["choices"][0]["message"] = {
            "role": "assistant",
            "content": response["choices"][0]["text"],
            "function_call": None,
            "tool_calls": None,
        }
        response["choices"][0].pop("text")

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

        response = await self.proxy.async_complete(
            prompt=apply_chat_template(messages, add_generation_prompt=True),
            **self.chat_params,
        )
        # remake the response to mimic the chat completion response format
        response["choices"][0]["message"] = {
            "role": "assistant",
            "content": response["choices"][0]["text"],
            "function_call": None,
            "tool_calls": None,
        }

        response["choices"][0].pop("text")

        return response

    async def async_complete(self, prompt: str, **kwargs) -> dict[str, Any]:
        self.chat_params.update({k: v for k, v in kwargs.items() if v is not None})

        response = await self.proxy.async_complete(
            prompt=prompt,
            **self.chat_params,
        )
        return response
