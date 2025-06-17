import fcntl
import os
import signal
import time
from dataclasses import dataclass
from subprocess import DEVNULL, PIPE, STDOUT, Popen
from typing import IO, Any, List, Optional

from data_alchemy.utils.logging import logger


@dataclass
class ProcessResult:
    timeout: bool
    exit_code: int
    stdout: str
    stderr: str


class PythonSandbox:
    def __init__(self, max_byte_per_read: int = 1024, sleep_between_reads: float = 0.1) -> None:
        self.max_bytes_per_read = max_byte_per_read
        self.sleep_between_reads = sleep_between_reads

    @staticmethod
    def set_nonblocking(reader: IO[bytes]) -> None:
        """
        Set the file descriptor of the given reader to non-blocking mode so that main process can
        read from it without waiting for data to be available.
        """
        fd = reader.fileno()
        fl = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

    def run(
        self,
        args: List[str],
        timeout_seconds: int = 15,
        max_output_size: int = 2048,
        env: Optional[str] = None,
        cwd: Optional[str] = None,
    ) -> ProcessResult:
        """
        Runs the given program with arguments. After the timeout elapses, kills the process
        and all other processes in the process group. Captures at most max_output_size bytes
        of stdout and stderr each, and discards any output beyond that.
        """

        p = Popen(
            args,
            env=env,
            cwd=cwd,
            stdin=DEVNULL,
            stdout=PIPE,
            stderr=PIPE,
            start_new_session=True,
            bufsize=self.max_bytes_per_read,
        )

        self.set_nonblocking(p.stdout)
        self.set_nonblocking(p.stderr)

        process_group_id: int = os.getpgid(p.pid)
        max_iterations: int = int(timeout_seconds / self.sleep_between_reads)
        stdout_saved_bytes: List[bytes] = []
        stderr_saved_bytes: List[bytes] = []
        stdout_bytes_read: int = 0
        stderr_bytes_read: int = 0

        for _ in range(max_iterations):
            this_stdout_read: Optional[bytes] = p.stdout.read(self.max_bytes_per_read)
            this_stderr_read: Optional[bytes] = p.stderr.read(self.max_bytes_per_read)

            if this_stdout_read and stdout_bytes_read < max_output_size:
                stdout_saved_bytes.append(this_stdout_read)
                stdout_bytes_read += len(this_stdout_read)
            if this_stderr_read and stderr_bytes_read < max_output_size:
                stderr_saved_bytes.append(this_stderr_read)
                stderr_bytes_read += len(this_stderr_read)

            exit_code: Optional[int] = p.poll()
            if exit_code is not None:
                break
            time.sleep(self.sleep_between_reads)

        try:
            os.killpg(process_group_id, signal.SIGKILL)
        except ProcessLookupError as e:
            # logger.warning(f"Failed to kill process group {process_group_id}: {e}")
            pass

        return ProcessResult(
            timeout=exit_code is None,
            exit_code=exit_code if exit_code is not None else -1,
            stdout=b"".join(stdout_saved_bytes).decode("utf-8", errors="ignore"),
            stderr=b"".join(stderr_saved_bytes).decode("utf-8", errors="ignore"),
        )

    def resolve(self, result: ProcessResult) -> dict[str, Any]:
        if result.timeout:
            status = "Timeout"
        elif result.exit_code == 0:
            status = "OK"
        elif "SyntaxError" in result.stderr:
            status = "SyntaxError"
        elif "AssertionError" in result.stderr:
            status = "AssertionError"
        else:
            status = "Exception"

        return {
            "status": status,
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
