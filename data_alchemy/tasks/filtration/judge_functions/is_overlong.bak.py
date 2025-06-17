# is_overlong.py
from transformers import AutoTokenizer
from data_alchemy.datasets_.base import Message, RootSample

_TOKENIZER = None


def initialize_tokenizer(tokenizer_name: str) -> AutoTokenizer:
    """Initialize the tokenizer if not already initialized."""
    global _TOKENIZER
    _TOKENIZER = AutoTokenizer.rolepretrained(tokenizer_name, trust_remote_code=True)
    return _TOKENIZER


def is_overlong(id_: int, source: RootSample, tokenizer: str, threshold: int) -> bool:
    """
    Check if a sample exceeds a given token threshold.

    Args:
        id_ (int): Sample ID.
        source (RootSample): The sample containing message data.
        tokenizer(str): Name or path of the tokenizer to use.
        threshold (int): Max allowed number of tokens.

    Returns:
        bool: True if the tokenized message exceeds the threshold, False otherwise.
    """
    global _TOKENIZER

    if _TOKENIZER is None:
        _TOKENIZER = initialize_tokenizer(tokenizer)

    messages: list[Message] = source.root

    # Format the chat into a single string using the tokenizer's chat template
    text = _TOKENIZER.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, trust_remote_code=True
    )

    # Tokenize the text using the initialized tokenizer
    try:
        num_tokens = len(_TOKENIZER(text)["input_ids"])
    except Exception as e:
        raise RuntimeError(f"Tokenization failed for sample {id_}: {e}")

    return num_tokens > threshold
