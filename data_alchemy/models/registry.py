from itertools import chain

from data_alchemy.utils.logging import logger
from data_alchemy.utils.string import prefix_match

from . import BaseModel
from .embedding.bge import BGEM3EmbeddingModel
from .generative.anthropic import ClaudeModel
from .generative.deepseek import DeepSeekModel
from .generative.openai_ import GPTModel
from .generative.qwen import QwenModel

__OPENAI_MODELS__ = {
    # openai
    (
        "gpt3.5",
        "gpt4",
        "gpt-4o",
        "gpt-4o-mini",
        "o1-mini",
        "o1-preview",
        "gpt-4-0613",
        "gpt-4-1106-preview",
    ): GPTModel,
    # anthropic
    (
        "claude-3-opus",
        "claude-3-sonnet",
        "claude-3-5-sonnet-20240620",
        "claude-3-5-sonnet-20241022",
    ): ClaudeModel,
    # zhipuai
    (
        "glm-4",
        "chatglm_pro",
        "glm-web",
        "chatglm_turbo",
    ): GPTModel,
    # google
    ("gemini-pro",): None,
    (
        "SparkDesk",
        "SparkDesk-v3.5",
    ): None,
    # kimi
    (
        "kimi-web",
        "moonshot",
    ): GPTModel,
    # deepseek
    ("deepseek-chat", "deepseek-coder", "deepseek-reasoner"): GPTModel,
    # doubao
    ("Doubao-pro-32k",): GPTModel,
    # baidu
    ("ERNIE-Bot-4",): None,
    (
        "hunyuan-standard",
        "hunyuan-pro",
    ): None,
}


__EMBEDDING_MODELS__ = {
    ("bge-m3",): BGEM3EmbeddingModel,
}


def repr_model_args(model_name: str, **kwargs) -> str:
    return "\n".join(
        f"{k}={v}" for k, v in {"model_name": model_name, **kwargs}.items()
    )


def get_model_from_name(model_name: str, **kwargs) -> BaseModel:
    """

    Args:
        model_name (str): online or deployed model name

    Returns:
        BaseModel: model instance
    """
    for series, model_cls in __OPENAI_MODELS__.items():
        if model_name in series:
            logger.info(
                f"Instancializing {model_cls.__name__}:\n{repr_model_args(model_name, **kwargs)}"
            )
            return model_cls(model_name, **kwargs)

    for series, model_cls in chain(
        [], __EMBEDDING_MODELS__.items()
    ):
        if any(prefix_match(sery, model_name) for sery in series):
            logger.info(
                f"Instancializing {model_cls.__name__}:\n{repr_model_args(model_name, **kwargs)}"
            )
            return model_cls(model_name, **kwargs)

    raise ValueError(f"Model '{model_name}' not found")
