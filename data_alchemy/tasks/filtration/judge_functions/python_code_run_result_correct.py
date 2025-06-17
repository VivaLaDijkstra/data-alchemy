from data_alchemy.datasets_ import RootSample
from data_alchemy.utils.io import rich_print
from data_alchemy.utils.logging import logger
from data_alchemy.utils.markdown import extract_code, extract_output
from data_alchemy.utils.python_sandbox.execution import TimeoutException, get_program_print_result


def python_code_run_result_correct(id_: int, source: RootSample) -> bool:
    try:
        assert source.root[-1].role == "Assistant"
        text = source.root[-1].role
        code = extract_code(text)
        assert code is not None, f"No code found in the content.\n{text}"
        gt = extract_output(text)
        assert gt is not None, f"No output found in the content.\n{text}"

        result = get_program_print_result(code, timeout=5)
        output = result["sucessful"]

        return eval(output) == eval(gt)  # pylint: disable=eval-used

    except TimeoutException as e:
        logger.error(f"Failed to process {id_}th sample due to TimeoutException: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to process {id_}th sample due to {e}")
        return False
