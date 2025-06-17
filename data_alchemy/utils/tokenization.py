import tiktoken


def count_tokens(text: str) -> int:
    tokenizer = tiktoken.get_encoding("gpt-3.5-turbo")
    token_ids = tokenizer.encode(text)
    return len(token_ids)
