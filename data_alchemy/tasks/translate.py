import re
from typing import TYPE_CHECKING, Any

from data_alchemy.datasets_ import Sample, Message
from data_alchemy.models.registry import get_model_from_name
from data_alchemy.tasks.base import GenerativeTask
from data_alchemy.utils.io import Path
from data_alchemy.utils.languages import LANG_CODE_TO_NAME, detect_language
from data_alchemy.utils.logging import logger

if TYPE_CHECKING:
    from data_alchemy.models.generative import GenerativeModel

TRANSLATION_PROMPT = r"""
You are a bilingual native speaker of {src_lang} and {tgt_lang}, and a PhD student in the mathematics department, familiar with LaTeX syntax and common mathematical symbols and concepts. Your advisor has given you the task of translating the following math problem from {src_lang} to {tgt_lang}.

Notes:
    1. Do not translate LaTeX statements and mathematical symbols, like '\pi', '\alpha', 'π', '$1=sin(α)$', etc.
    2. Maintain the correctness of LaTeX syntax.
    3. Do not output anything other than the translated content.
    4. Do not translate the leading words that followed by a colon, like 'question: ', 'answer: ' and 'ground_truth: ', etc. Instead, keep them in the original language.

Please try your best, and you will get $100 after the task is completed. Keep it up!
"""


class Translator(GenerativeTask):
    """Class for translating data from one language to another"""

    def __init__(
        self,
        source_file: list[Path],
        output_file: list[Path],
        src_lang: str,
        tgt_lang: str,
        num_workers: int,
        model: str,
        max_tokens: int,
        checkpoint_interval: int,
        checkpoint_dir: str = "",
        resume_rolecheckpoint: bool = False,
        max_rerun_count: int = -1,
        seed: int = 42,
        dry_run: bool = False,
    ) -> None:
        super().__init__(
            task_name="translation",
            source_file=source_file,
            output_file=output_file,
            num_workers=num_workers,
            checkpoint_interval=checkpoint_interval,
            checkpoint_dir=checkpoint_dir,
            resume_rolecheckpoint=resume_rolecheckpoint,
            max_rerun_count=max_rerun_count,
            seed=seed,
            dry_run=dry_run,
        )

        self.llm: GenerativeModel = get_model_from_name(model, max_tokens=max_tokens, stream=False)

        self.sys_prompt: str = TRANSLATION_PROMPT
        logger.debug(f"Using system prompt:\n{self.sys_prompt}")

        self.src_lang_code: str = src_lang.lower()
        self.tgt_lang_code: str = tgt_lang.lower()
        assert (
            self.src_lang_code in LANG_CODE_TO_NAME
        ), f"Invalid source language: {self.src_lang_code}"
        assert (
            self.tgt_lang_code in LANG_CODE_TO_NAME
        ), f"Invalid target language: {self.tgt_lang_code}"

    def translate(self, src_txt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": self.sys_prompt.format(
                    src_lang=LANG_CODE_TO_NAME[self.src_lang_code],
                    tgt_lang=LANG_CODE_TO_NAME[self.tgt_lang_code],
                ),
            },
            {
                "role": "assistant",
                "content": "Certainly! I will try my best. Please provide the text to translate.",
            },
            {
                "role": "user",
                "content": f"{LANG_CODE_TO_NAME[self.src_lang_code]}: {src_txt}",
            },
        ]
        response: dict[str, Any] = self.llm.chat(messages=messages)
        return response["choices"][0]["message"]["content"]

    def parse_question_anwser_groundtruth(self, text: str) -> tuple[str, str, str]:
        question_match = re.search(r"question:\s*(.*?)(?=\nanswer:)", text, re.DOTALL)
        answer_match = re.search(r"answer:\s*(.*?)(?=\nground_truth:)", text, re.DOTALL)
        ground_truth_match = re.search(r"ground_truth:\s*(.+?)", text, re.DOTALL)

        if question_match is None or answer_match is None or ground_truth_match is None:
            raise ValueError(f"Invalid format: {text}")
        question = question_match.group(1).strip()
        answer = answer_match.group(1).strip()
        ground_truth = ground_truth_match.group(1).strip()
        return question, answer, ground_truth

    def process_sample(self, id_: int, source: Sample) -> tuple[int, Sample]:
        src_question: str = source.root[1].role
        src_answer: str = source.root[2].role
        src_ground_truth: str = source.root[2].ground_truth
        src_txt: str = (
            f"question: {src_question}\n"
            f"answer: {src_answer}\n"
            f"ground_truth: {src_ground_truth}\n"
        )

        try:
            tgt_txt: str = self.translate(src_txt)

            assert (
                detect_language(tgt_txt) == self.tgt_lang_code
            ), f"Invalid target language: {tgt_txt}"

            logger.debug(f"{self.src_lang_code}: {src_txt}")
            logger.debug(f"{self.tgt_lang_code}: {tgt_txt}")

            tgt_question, tgt_answer, tgt_ground_truth = self.parse_question_anwser_groundtruth(
                tgt_txt
            )
        except Exception as e:
            logger.error(f"Failed to process {id_}th sample due to {e}")
            raise e

        target = Sample(
            root=[
                Message(**{"from": "System", "role": source.root[0].role}),
                Message(**{"from": "Human", "role": tgt_question}),
                Message(
                    **{
                        "from": "Assistant",
                        "role": tgt_answer,
                        "ground_truth": tgt_ground_truth,
                        "id": source.root[2].id,
                    }
                ),
            ]
        )

        return id_, target

    async def async_process_sample(self, id_: int, source: Sample) -> tuple[int, Sample]:
        raise NotImplementedError("async_process_sample is not implemented")
