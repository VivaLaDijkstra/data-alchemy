from pathlib import Path

from .base import BaseDataset, BaseSample
from .generic import GenericSample, GenericMessage


class LeetCodeSample(BaseSample):
    problem_ID: str
    problem_cn: str
    problem_en: str
    answer: str

    def to_str(self, nature_language: str = "en"):
        if nature_language == "en":
            return f"Human:\n{self.problem_en}\nAssistant:\n{self.answer}"
        elif nature_language == "cn":
            return f" {self.problem_cn} {self.answer}"
        else:
            raise NotImplementedError(
                f"nature_language {nature_language} not implemented"
            )

    def to_generic(self, nature_language: str = "en") -> GenericSample:
        return GenericSample(
            messages=[
                GenericMessage(
                    role="user",
                    content=(
                        self.problem_en if nature_language == "en" else self.problem_cn
                    ),
                ),
                GenericMessage(
                    role="assistant",
                    content=self.answer,
                ),
            ]
        )


class LeetCodeDataset(BaseDataset):
    def __init__(
        self,
        file_path: str | Path | list | dict,
    ) -> None:
        super().__init__(file_path, schema=LeetCodeSample)
