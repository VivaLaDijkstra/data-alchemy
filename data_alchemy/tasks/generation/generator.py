from pathlib import Path
from typing import Callable, Optional

from data_alchemy.datasets_ import Sample
from data_alchemy.tasks.base import GenerativeTask
from data_alchemy.tasks.utils import SampleFunctionFactory
from data_alchemy.utils.auto_resume import model_cache
from data_alchemy.utils.logging import logger


class DataGenerator(GenerativeTask):
    def __init__(
        self,
        source_file: list[str],
        output_file: list[str],
        checkpoint_interval: int,
        checkpoint_dir: str,
        num_workers: int,
        resume_rolecheckpoint: bool = False,
        cache_dir: Optional[str] = None,
        sample_process_function: str = None,
        max_rerun_count: int = -1,
        seed: int = 42,
        dry_run: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            task_name="generation",
            source_file=source_file,
            output_file=output_file,
            num_workers=num_workers,
            checkpoint_interval=checkpoint_interval,
            checkpoint_dir=checkpoint_dir,
            resume_rolecheckpoint=resume_rolecheckpoint,
            cache_dir=cache_dir,
            max_rerun_count=max_rerun_count,
            seed=seed,
            dry_run=dry_run,
        )

        logger.info(f"Loading sample process function: {sample_process_function}")
        factory = SampleFunctionFactory(load_path=Path(__file__).parent / "process_functions")

        self.func: Callable[[int, Sample], tuple[int, Sample]] = factory.get_function(
            function_name=sample_process_function
        )

        # TODO
        print(kwargs)

    def process_sample(self, id_: int, source: Sample) -> tuple[int, Sample]:
        @model_cache(cache_dir=self.cache_dir)
        def _process_sample(id_: int, source: Sample) -> tuple[int, Sample]:
            return self.func(id_, source)

        return _process_sample(id_, source)

    async def async_process_sample(self, id_, source: Sample) -> tuple[int, Sample]:
        @model_cache(cache_dir=self.cache_dir)
        async def _async_process_sample(id_: int, source: Sample) -> tuple[int, Sample]:
            return await self.func(id_, source)

        return await _async_process_sample(id_, source)
