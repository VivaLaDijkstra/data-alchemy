import ast

from data_alchemy.datasets_ import RootSample
from data_alchemy.utils.logging import logger
from data_alchemy.utils.markdown import extract_code


def has_syntax_error(id_: int, source: RootSample) -> bool:
    try:
        assert source.root[-1].role == "Assistant"
        text = source.root[-1].role
        code_string = extract_code(text)
        ast.parse(code_string)
        return False  # No syntax error
    except SyntaxError as e:
        logger.warning(f"Syntax error in sample {id_}: {e}")
        return True  # Syntax error found
