import os
from json import JSONDecodeError
from typing import Any, Callable, Iterable, Optional, cast

from httpx import AsyncClient, Client, HTTPStatusError, RequestError, TimeoutException
from openai import AsyncOpenAI, OpenAI, OpenAIError, RateLimitError
from openai.types.chat import ChatCompletionMessageParam
from data_alchemy.datasets_.base import Message
from data_alchemy.utils.logging import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter


class ProxyError(HTTPStatusError, OpenAIError, JSONDecodeError):
    """all errors raised by this module can be catched by this class"""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        logger.error(f"ProxyError: {self}")


def type_cast(msg: Message | dict[str, Any]) -> dict[str, Any]:
    if isinstance(msg, Message):
        return msg.to_raw()
    if isinstance(msg, dict):
        return msg
    raise TypeError(f"Invalid message type: {type(msg)}")


class ModelServiceProxy:
    def __init__(self) -> None:
        pass

    def chat(
        self, messages: list[Message | dict[str, str]], **kwargs
    ) -> dict[str, Any]:
        raise NotImplementedError("This method should be implemented by subclasses")

    def complete(self, prompt: str, **kwargs) -> dict[str, Any]:
        raise NotImplementedError("This method should be implemented by subclasses")

    async def async_chat(
        self, messages: list[Message | dict[str, str]], **kwargs
    ) -> dict[str, Any]:
        raise NotImplementedError("This method should be implemented by subclasses")

    async def async_complete(self, prompt: str, **kwargs) -> dict[str, Any]:
        raise NotImplementedError("This method should be implemented by subclasses")


class SDKProxy(ModelServiceProxy):
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        if api_key is None:
            self.api_key: str | None = os.getenv("OPENAI_API_KEY")
            assert self.api_key, "API key is not set"
        if base_url is None:
            self.base_url: str = os.getenv(
                "OPENAI_BASE_URL", "https://api.openai.com/v1"
            )
            assert self.base_url, "Base URL is not set"

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.async_client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

        self.timeout = timeout or (16 * 60.0)

        self.retry = retry(
            stop=stop_after_attempt(10),
            retry=retry_if_exception_type(RateLimitError),
            wait=wait_exponential_jitter(initial=0.5, max=30 * 60, jitter=2.0),
            reraise=True,
        )

    def _post(self, create: Callable, **kwargs) -> dict[str, Any]:
        @self.retry
        def __post(**kwargs) -> dict[str, Any]:
            try:
                logger.debug(f"Request to API by OPENAI SDK: {kwargs}")
                response = create(**kwargs)
                result = response.model_dump()
                logger.debug(f"Response from API by OPENAI SDK: {result}")
            except RateLimitError as e:
                logger.error(f"RateLimitError by SDK: {e}. API kwargs: {kwargs}")
                raise e
            except OpenAIError as e:
                logger.error(f"OpenAIError by SDK: {e}. API kwargs: {kwargs}")
                raise e

            return result

        return __post(**kwargs)

    def chat(
        self, messages: list[Message | dict[str, Any]], **kwargs
    ) -> dict[str, Any]:
        messages = [type_cast(msg) for msg in messages]
        response = self._post(
            self.client.chat.completions.create,
            messages=cast(Iterable[ChatCompletionMessageParam], messages),
            **kwargs,
        )
        return response

    def complete(self, prompt: str, **kwargs) -> dict[str, Any]:
        response = self._post(
            self.client.completions.create,
            prompt=prompt,
            **kwargs,
        )
        return response

    async def _async_post(self, create: Callable, **kwargs) -> dict[str, Any]:
        @self.retry
        async def __async_post(**kwargs) -> dict[str, Any]:
            try:
                logger.debug(f"Request to API by OPENAI SDK: {kwargs}")
                response = await create(**kwargs)
                result = response.model_dump()
                logger.debug(f"Response from API by OPENAI SDK: {result}")
            except RateLimitError as e:
                logger.error(f"RateLimitError by SDK: {e}. API kwargs: {kwargs}")
                raise e
            except OpenAIError as e:
                logger.error(f"OpenAIError by SDK: {e}. API kwargs: {kwargs}")
                raise e

            return result

        return await __async_post(**kwargs)

    async def async_chat(
        self, messages: list[Message | dict[str, Any]], **kwargs
    ) -> dict[str, Any]:
        messages = [type_cast(msg) for msg in messages]
        response = await self._async_post(
            self.async_client.chat.completions.create,
            messages=cast(Iterable[ChatCompletionMessageParam], messages),
            **kwargs,
        )
        return response

    async def async_complete(self, prompt: str, **kwargs) -> dict[str, Any]:
        response = await self._async_post(
            self.async_client.completions.create,
            prompt=prompt,
            **kwargs,
        )
        return response


class PostProxy(ModelServiceProxy):
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        retry_times: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> None:
        if api_key is None:
            api_key = "# Call from data-alchemy #"

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        if base_url is None:
            self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            assert isinstance(self.base_url, str), "base_url is not a string"
        else:
            self.base_url = base_url

        self.http2 = True
        self.timeout = timeout or (16 * 60.0)

        self.retry = retry(
            stop=stop_after_attempt(retry_times or 10),
            retry=retry_if_exception_type(
                (HTTPStatusError, RequestError, JSONDecodeError, TimeoutException)
            ),
            wait=wait_exponential_jitter(initial=0.5, max=30 * 60.0, jitter=2.0),
            reraise=True,
        )

    def _post(self, **kwargs) -> dict[str, Any]:
        @self.retry
        def __post(**kwargs) -> dict[str, Any]:
            with Client(
                headers=self.headers, timeout=self.timeout, http2=self.http2
            ) as client:
                try:
                    logger.debug(f"Request to API: {kwargs}")
                    # response = requests.post(**kwargs)  # pylint: disable=missing-timeout
                    response = client.post(**kwargs)
                    response.raise_for_status()
                    result = response.json()
                    logger.debug(f"Response from API: {result}")
                except HTTPStatusError as e:
                    logger.error(
                        f"Error in post request: {e}, request response: {response.text}."
                        f"API kwargs: {kwargs}"
                    )
                    raise e
                except RequestError as e:
                    logger.error(f"Error in post request: {e}. API kwargs: {kwargs}")
                    raise e
                except JSONDecodeError as e:
                    logger.error(f"Error in JSON decoding: {e}. API kwargs: {kwargs}")
                    raise e
                except Exception as e:
                    logger.error(f"Error in API call: {e}. API kwargs: {kwargs}")
                    raise e

            return result

        return __post(**kwargs)

    def chat(
        self, messages: list[Message | dict[str, Any]], **kwargs
    ) -> dict[str, Any]:
        payload = {"messages": [type_cast(msg) for msg in messages]}
        payload.update({k: v for k, v in kwargs.items() if v is not None})

        response = self._post(
            url=f"{self.base_url}/chat/completions",
            json=payload,
        )
        return response

    def complete(self, prompt: str, **kwargs):
        payload = {"prompt": prompt}
        payload.update({k: v for k, v in kwargs.items() if v is not None})

        response = self._post(
            url=f"{self.base_url}/completions",
            json=payload,
        )
        return response

    async def _async_post(self, **kwargs) -> dict[str, Any]:
        @self.retry
        async def __async_post(**kwargs) -> dict[str, Any]:
            async with AsyncClient(
                headers=self.headers, timeout=self.timeout, http2=self.http2
            ) as client:
                try:
                    logger.debug(f"Async request to API: {kwargs}")
                    response = await client.post(**kwargs)
                    response.raise_for_status()
                    result = response.json()
                    logger.debug(f"Async response from API: {result}")
                except HTTPStatusError as e:
                    logger.error(
                        f"HTTP error in async post request: {e}. API kwargs: {kwargs}"
                    )
                    raise e
                except RequestError as e:
                    logger.error(
                        f"Error in async post request: {e}. API kwargs: {kwargs}"
                    )
                    raise e
                except JSONDecodeError as e:
                    logger.error(f"Error in JSON decoding: {e}. API kwargs: {kwargs}")
                    raise e
                except Exception as e:
                    logger.error(f"Error in async API call: {e}. API kwargs: {kwargs}")
                    raise e

                return result

        return await __async_post(**kwargs)

    async def async_chat(
        self, messages: list[Message | dict[str, Any]], **kwargs
    ) -> dict[str, Any]:
        payload = {"messages": [type_cast(msg) for msg in messages]}
        payload.update({k: v for k, v in kwargs.items() if v is not None})

        response = await self._async_post(
            url=f"{self.base_url}/chat/completions",
            json=payload,
        )
        return response

    async def async_complete(self, prompt: str, **kwargs) -> dict[str, Any]:
        payload = {"prompt": prompt}
        payload.update({k: v for k, v in kwargs.items() if v is not None})

        response = await self._async_post(
            url=f"{self.base_url}/completions",
            json=payload,
        )
        return response
