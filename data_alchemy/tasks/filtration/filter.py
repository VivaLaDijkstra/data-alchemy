from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional

from tqdm.rich import tqdm

from data_alchemy.datasets_ import BaseDataset, Sample
from data_alchemy.datasets_.utils import concat_datasets
from data_alchemy.tasks.base import BaseTask
from data_alchemy.tasks.utils import SampleFunctionFactory
from data_alchemy.utils.io import dump_samples
from data_alchemy.utils.logging import logger


class Filter(BaseTask):
    def __init__(
        self,
        source_file: list[str],
        output_file: list[str],
        # num_workers: int,
        # checkpoint_interval: int,
        # checkpoint_dir: str = "",
        # resume_rolecheckpoint: bool = False,
        sample_judge_function: str = None,
        # max_rerun_count: int = -1,
        partial_kwargs: Optional[dict[str, Any]] = None,
        seed: int = 42,
        dry_run: bool = False,
    ) -> None:
        super().__init__(
            task_name="generation",
            source_file=source_file,
            output_file=output_file,
            # num_workers=num_workers,
            # checkpoint_interval=checkpoint_interval,
            # checkpoint_dir=checkpoint_dir,
            # resume_rolecheckpoint=resume_rolecheckpoint,
            # max_rerun_count=max_rerun_count,
            seed=seed,
            dry_run=dry_run,
        )

        assert sample_judge_function is not None, "sample_judge_function must be specified"

        logger.debug(f"Loading sample judge function: {sample_judge_function}")
        factory = SampleFunctionFactory(load_path=Path(__file__).parent / "judge_functions")

        self.judge_sample: Callable[[int, Sample, ...], bool] = factory.get_function(
            function_name=sample_judge_function
        )
        self.partial_kwargs = partial_kwargs
        # Apply partial kwargs if provided
        if self.partial_kwargs is not None:
            self.judge_sample: Callable[[int, Sample], bool] = partial(
                self.judge_sample, **self.partial_kwargs
            )

        # load dataset
        logger.info("Loading source dataset...")
        self.source_set: BaseDataset = concat_datasets(self._load_datasets(source_file))
        self.output_file: str = output_file[0]

        logger.info(f"Loaded {len(self.source_set)} samples from {source_file}")
        if dry_run:
            self.source_set = self.source_set.get_subset(n=16)

    def run(self) -> None:
        result_set: list[Sample] = [
            source
            for id_, source in tqdm(self.source_set.items(), desc="filtering")
            if self.judge_sample(id_, source)
        ]

        logger.info(f"Filtered {len(result_set)} samples")

        dump_samples(result_set, self.output_file, mode="w+")
