import os
import sys
from pathlib import Path
from typing import Any

import loguru

LOG_LEVELS = {
    "TRACE": 0,
    "DEBUG": 10,
    "INFO": 20,
    "SUCCESS": 25,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}


class Logger:

    def __init__(self, level: str = "INFO") -> None:
        self.logger = loguru.logger
        self.logger.remove()  # 移除默认 handler
        self.logger.add(sys.stdout, level=level)  # 终端日志

    def config(
        self,
        log_file: str | Path,
        rotation: str = "1 week",
        retention: str = "10 days",
        level: str = "INFO",
    ) -> None:
        if level not in LOG_LEVELS:
            raise ValueError(f"Invalid log level: {level}")

        log_file = Path(log_file)
        log_dir = log_file.parent
        if not log_dir.exists():
            log_dir.mkdir(parents=True, exist_ok=True)

        self.logger.add(
            log_file,
            rotation=rotation,
            retention=retention,
            level=level,
        )

    def __getattr__(self, name: str) -> Any:
        """let logger have the same interface as loguru"""
        if hasattr(self.logger, name):
            return getattr(self.logger, name)
        raise AttributeError(f"'Logger' object has no attribute '{name}'")


log_level = os.getenv("DATA_ALCHEMY_LOG_LEVEL", "INFO")

logger = Logger(level=log_level)
