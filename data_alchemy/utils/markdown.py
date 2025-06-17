import re


class MarkdownParseError(ValueError):
    """Raised when there is a problem parsing the markdown"""


def extract_code(text: str) -> str | None:
    res = re.search(r"```(\s*)(\S*)?\n(.*?)```", text, re.DOTALL)
    if res:
        return res.group(3)
    return None


def extract_output(text: str) -> str | None:
    res = re.search(r"```output\n(.*?)```", text, re.DOTALL)
    if res:
        return res
    return None


def remove_output(text: str) -> str | None:
    res = re.sub(r"```output\n.*?```", "", text, re.DOTALL)
    if res:
        return res
    return None


def has_code_block(text: str) -> bool:
    code = extract_code(text)
    if code is not None and code.strip() != "":
        return True
    return False


def render_think_prompt(text: str) -> str:
    """if text contains '<think>' and '</think>', wrap it with markdown quotes"""

    def replace_think_block(match):
        # Extract and trim the content inside <think> </think>
        content = match.group(1).strip()
        quoted_content = "\n".join(
            f"> {line}" for line in content.splitlines()
        )  # Add '>' to each line
        return quoted_content

    # Replace <think>...</think> blocks
    text = re.sub(r"<think>(.*?)</think>", replace_think_block, text, flags=re.DOTALL)

    return text
