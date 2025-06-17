import signal

from data_alchemy.datasets_ import RootSample
from data_alchemy.utils.io import rich_print
from data_alchemy.utils.logging import logger
from data_alchemy.utils.markdown import extract_code
from data_alchemy.utils.python_sandbox.execution import TimeoutException


def timeout_handler():
    raise TimeoutException("Timed out!")


def simple_run(code: str, timeout: int) -> dict[str, str]:
    # set time alarm
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)

    try:
        exec(code)  # pylint: disable=exec-used
        return {"status": "success"}
    except TimeoutException as e:
        return {"status": "timeout", "error": str(e)}
    except Exception as e:  # pylint: disable=broad-except
        return {"status": "error", "error": str(e)}
    finally:
        signal.alarm(0)  # Disable the alarm


def can_python_code_run(id_: int, source: RootSample) -> bool:
    # with create_tempdir() as tempdir:
    try:
        assert source.root[-1].role == "Assistant"
        text = source.root[-1].role
        code = extract_code(text)
        if code is None or code == "":
            raise ValueError("No code found")

        result = simple_run(code, timeout=2)
        return "success" == result["status"]

    except KeyboardInterrupt as e:
        raise e
    except TimeoutException as e:
        logger.error(f"Failed to process {id_}th sample due to TimeoutException: {e}")
        return False
    except Exception as e:  # pylint: disable=broad-except
        logger.error(f"Failed to process {id_}th sample due to {e}")
        return False
