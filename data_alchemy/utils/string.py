import ast
import hashlib
from typing import Any


def replace_right_to_left(s: str, old: str, new: str, count=-1) -> str:
    # If count is -1, replace all occurrences
    if count == -1:
        count = s.count(old)

    # Replace from the rightmost occurrence
    parts = s.rsplit(old, count)
    return new.join(parts)


def ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return str(n) + suffix


def prefix_match(src: str, tgt: str) -> bool:
    src, tgt = src.strip().lower(), tgt.strip().lower()
    return tgt.startswith(src)


def gen_hash(obj: Any, method: str = "sha256") -> str:
    return hashlib.new(method, str(obj).encode()).hexdigest()


def parse_kwarg_str(s: str) -> dict[str, Any]:
    """e.g. s = 'tokenizer=Qwen/Qwen-14B, max_tokens=2048, temperature=0.8'
    support all python built-in types(int, float, str, tuple, list, set, dict, None).
    support recursive nesting for tuple, list, set and dict.
    """
    if not s:
        return {}

    result = {}
    current = ""
    depth = 0
    in_str = False
    str_char = ""

    args = []

    # Smart splitting on commas outside nested structures and strings
    for char in s:
        if char in ('"', "'"):
            if not in_str:
                in_str = True
                str_char = char
            elif str_char == char:
                in_str = False

        elif not in_str:
            if char in "([{":
                depth += 1
            elif char in ")]}":
                depth -= 1
            elif char == ',' and depth == 0:
                args.append(current.strip())
                current = ''
                continue

        current += char
    # end for

    if current.strip():
        args.append(current.strip())

    for item in args:
        if '=' not in item:
            raise ValueError(f"Invalid argument format: {item}")
        key, value = item.split('=', 1)
        key = key.strip()
        value = value.strip()

        try:
            parsed = ast.literal_eval(value)
        except Exception:
            parsed = value  # fallback to string

        result[key] = parsed

    return result
