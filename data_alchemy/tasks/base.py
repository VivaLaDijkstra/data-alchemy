import asyncio
import concurrent
import concurrent.futures
import random
from abc import abstractmethod
from functools import reduce
from typing import Any, Optional
from urllib.error import HTTPError

from openai import OpenAIError
from requests.exceptions import RequestException
from tenacity import RetryError
from tqdm.rich import tqdm, trange

from data_alchemy.datasets_ import BaseDataset, Message, Sample
from data_alchemy.datasets_.utils import concat_datasets, load_dataset
from data_alchemy.tasks.utils import dump_kv_results, load_kv_results
from data_alchemy.utils.io import Path, dump_samples
from data_alchemy.utils.logging import logger
from data_alchemy.utils.string import ordinal, replace_right_to_left
from data_alchemy.utils.time import get_timestamp


class BaseTask:
    """
    Base class for all tasks
    """

    def __init__(
        self,
        task_name: str,
        source_file: list[Path],
        output_file: list[Path],
        seed: int = 42,
        dry_run: bool = False,
    ):
        self.task_name: str = task_name

        # input file path process
        self.source_file: list[Path] = source_file
        # output file path process
        self.output_file: list[Path] = output_file

        self.seed = seed
        random.seed(seed)

        self.dry_run = dry_run
        self.timestamp = get_timestamp()

    def _load_datasets(self, fpaths: list[Path]) -> list[BaseDataset[Sample]]:
        logger.info(f"Loading datasets from {fpaths}")
        return [load_dataset(fp) for fp in fpaths]

    @abstractmethod
    def run(self) -> None:
        raise NotImplementedError("Subclasses must implement run method")


class GenerativeTask(BaseTask):
    """
    Base class for all generative tasks
    """

    def __init__(
        self,
        task_name: str,
        source_file: list[Path],
        output_file: list[Path],
        num_workers: int,
        checkpoint_dir: Optional[Path] = None,
        checkpoint_interval: Optional[int] = None,
        resume_rolecheckpoint: bool = False,
        cache_dir: Optional[Path] = None,
        max_rerun_count: int = -1,
        seed: int = 42,
        dry_run: bool = False,
    ):
        super().__init__(
            task_name=task_name,
            source_file=source_file,
            output_file=output_file,
            seed=seed,
            dry_run=dry_run,
        )

        assert len(self.output_file) == 1, "Output file must be a single file"
        self.output_file_path: Path = self.output_file[0]

        self.num_workers = num_workers

        logger.info(f"Using {self.num_workers} cpu workers")

        # load dataset
        logger.info("Loading source dataset...")
        self.source_set: BaseDataset = concat_datasets(self._load_datasets(source_file))
        self.total_samples_num = len(self.source_set)

        # set checkpoint args
        if checkpoint_dir is not None:
            # set checkpoint args
            self.checkpoint_interval = checkpoint_interval
            assert checkpoint_interval > 0, "Checkpoint interval must be greater than 0"

            if not checkpoint_dir.exists():
                checkpoint_dir.mkdir(parents=True)

            self.checkpoint_dir = checkpoint_dir
            self.checkpoint_success_path = checkpoint_dir / f"{task_name}.success.json"
            self.checkpoint_failure_path = checkpoint_dir / f"{task_name}.failure.json"
            self.resume_rolecheckpoint = resume_rolecheckpoint
            self.checkpoint_rank_path = checkpoint_dir / "cur_ckpt_rank"
            if resume_rolecheckpoint:
                assert (
                    self.checkpoint_rank_path.is_file()
                ), "Checkpoint rank file does not exist! Do you want to resume from checkpoint?"
            self.checkpoint_rank: int = None
            self.store_checkpoint: bool = self.total_samples_num < self.checkpoint_interval
        else:
            self.checkpoint_dir = None
            self.checkpoint_interval = 2**63 - 1  # biggest 64-bit integer as +inf
            self.checkpoint_success_path = None
            self.checkpoint_failure_path = None
            self.resume_rolecheckpoint = False
            self.checkpoint_rank_path = None
            self.checkpoint_rank: int = None
            self.store_checkpoint: bool = False

        # set cache dir
        self.cache_dir = cache_dir
        logger.info(f"Using cache dir: {self.cache_dir}")

        # set rerun args
        if max_rerun_count >= 0:
            self.max_rerun_count = max_rerun_count
        # rerun util all samples are successful
        else:
            self.max_rerun_count = float("inf")

        logger.info(f"Loading {len(self.source_set)} samples from {source_file}")
        if dry_run:
            self.source_set = self.source_set.get_subset(n=16)

    def load_checkpoint(
        self,
        rank: int,
    ) -> tuple[dict[int, Sample], dict[int, Sample]]:
        checkpoint_success: dict[int, Sample] = {}
        checkpoint_failure: dict[int, Sample] = {}

        # load successful samples
        raw_checkpoint_success = load_kv_results(
            path=replace_right_to_left(self.checkpoint_success_path, ".json", f".{rank}.json")
        )
        checkpoint_success = {
            i: self.source_set.schema(
                root=[Message(**msg) for msg in smp]
            )  # TODO: implement polymorphism
            for i, smp in raw_checkpoint_success.items()
        }
        checkpoint_success.update(checkpoint_success)  # TODO

        # load failure samples
        raw_checkpoint_failure = load_kv_results(
            path=replace_right_to_left(self.checkpoint_failure_path, ".json", f".{rank}.json")
        )
        checkpoint_failure = {
            i: self.source_set.schema(
                root=[Message(**msg) for msg in smp]
            )  # TODO: implement polymorphism
            for i, smp in raw_checkpoint_failure.items()
        }
        checkpoint_failure.update(checkpoint_failure)  # TODO

        logger.info(f"Loaded {len(checkpoint_success)} successful samples")
        logger.info(f"Loaded {len(checkpoint_failure)} failure samples")

        return checkpoint_success, checkpoint_failure

    def dump_checkpoint(
        self,
        checkpoint_success: dict[int, Sample],
        checkpoint_failure: dict[int, Sample],
    ) -> None:
        """dump a interval checkpoint"""

        # dump successful samples
        raw_checkpoint_success: dict[int, list[dict[str, Any]]] = {
            i: smp.to_raw() for i, smp in checkpoint_success.items()
        }
        logger.success(f"Dumping {len(raw_checkpoint_success)} successful samples")
        dump_kv_results(
            raw_checkpoint_success,
            path=replace_right_to_left(
                str(self.checkpoint_success_path), ".json", f".{self.checkpoint_rank}.json"
            ),
        )

        # dump failure samples
        raw_checkpoint_failure: dict[int, list[dict[str, Any]]] = {
            i: smp.to_raw() for i, smp in checkpoint_failure.items()
        }
        logger.success(f"Dumping {len(raw_checkpoint_failure)} failure samples")
        dump_kv_results(
            raw_checkpoint_failure,
            path=replace_right_to_left(
                str(self.checkpoint_failure_path),
                ".json",
                f".{self.checkpoint_rank}.json",
            ),
        )

        # update checkpoint rank
        if self.store_checkpoint:
            self.checkpoint_rank += 1
            with open(self.checkpoint_rank_path, "w", encoding="utf-8") as f:
                f.write(str(self.checkpoint_rank))

    def clean_checkpoints(self):
        """Remove all existing checkpoints to start fresh."""
        if self.checkpoint_dir.exists() and self.checkpoint_dir.is_dir():
            for file in self.checkpoint_dir.iterdir():
                file.unlink()  # Delete each file
            logger.info(f"Cleaned all checkpoints in {self.checkpoint_dir}")

    @abstractmethod
    def process_sample(self, id_: int, source: Sample) -> tuple[int, Sample]:
        """call model to process a sample

        Args:
            id_ (int): keep order of samples for multi-processing
            source (Sample): source sample to be processed

        Returns:
            tuple[int, Sample]: id_ and processed sample
        """
        raise NotImplementedError("Must be implemented in subclass")

    @abstractmethod
    async def async_process_sample(self, id_: int, source: Sample) -> tuple[int, Sample]:
        """call model to process a sample

        Args:
            id_ (int): keep order of samples for multi-processing
            source (Sample): source sample to be processed

        Returns:
            tuple[int, Sample]: id_ and processed sample
        """
        raise NotImplementedError("Must be implemented in subclass")

    def _run_without_parallel(
        self,
        source_set: BaseDataset[Sample],
        desc: str = "run without parallel",
    ) -> tuple[dict[int, Sample], dict[int, Sample]]:
        start: int = self.total_samples_num - len(source_set)
        with tqdm(initial=start, total=self.total_samples_num, desc=desc) as pbar:
            results_success: dict[int, Sample] = {}
            results_failure: dict[int, Sample] = {}
            buffer_success: dict[int, Sample] = {}
            buffer_failure: dict[int, Sample] = {}
            for i, (id_, src_sample) in enumerate(source_set.items(), start=start):
                pbar.update(1)
                try:
                    _, tgt_sample = self.process_sample(id_, src_sample)
                    buffer_success[id_] = tgt_sample
                except (
                    AssertionError,
                    HTTPError,
                    OpenAIError,
                    RequestException,
                    RetryError,
                    ValueError,
                ):
                    buffer_failure[id_] = src_sample
                except Exception as e:
                    logger.error(f"Unexpected error when processing sample {i}: {e}")
                    raise e

                if (i + 1) % self.checkpoint_interval == 0:
                    logger.info(
                        f"saving the {ordinal(self.checkpoint_rank)} checkpoint "
                        f"at the {ordinal(i)} step"
                    )
                    self.dump_checkpoint(buffer_success, buffer_failure)
                    results_success.update(buffer_success)
                    results_failure.update(buffer_failure)
                    # reset buffer
                    buffer_success, buffer_failure = {}, {}

        # dump remaining samples
        rest_sample_num = len(buffer_success) + len(buffer_failure)
        if 0 < rest_sample_num:
            if self.checkpoint_interval < self.total_samples_num:  # has at least one checkpoint
                logger.info(
                    f"saving the {ordinal(self.checkpoint_rank)} checkpoint at the end of the process"
                )
                self.dump_checkpoint(buffer_success, buffer_failure)

            results_success.update(buffer_success)
            results_failure.update(buffer_failure)

        return results_success, results_failure

    def _run_with_parallel(
        self,
        source_set: BaseDataset[Sample],
        desc: str = "run with parallel",
    ) -> tuple[dict[int, Sample], dict[int, Sample]]:
        def _process_sample_with_success_check(
            id_: int, src_sample: Sample
        ) -> tuple[int, Sample, bool]:
            try:
                _, tgt_sample = self.process_sample(id_, src_sample)
                return id_, tgt_sample, True
            except (
                AssertionError,
                HTTPError,
                OpenAIError,
                RequestException,
                RetryError,
                ValueError,
            ):
                return id_, src_sample, False
            except Exception as e:
                logger.error(f"Error in processing sample {id_}: {e}")
                raise e

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = (
                executor.submit(_process_sample_with_success_check, id_=id_, src_sample=sample)
                for id_, sample in source_set.items()
            )

            start: int = self.total_samples_num - len(source_set)
            with tqdm(initial=start, total=self.total_samples_num, desc=desc) as pbar:
                buffer_success: dict[int, Sample] = {}
                buffer_failure: dict[int, Sample] = {}
                results_success: dict[int, Sample] = {}
                results_failure: dict[int, Sample] = {}
                for j, future in enumerate(concurrent.futures.as_completed(futures), start=start):
                    pbar.update(1)
                    i, sample, success = future.result()
                    if success:
                        buffer_success[i] = sample
                    else:
                        buffer_failure[i] = sample

                    if (j + 1) % self.checkpoint_interval == 0:
                        logger.success(
                            f"saving the {ordinal(self.checkpoint_rank)} "
                            f"checkpoint at the {ordinal(j)} step"
                        )
                        self.dump_checkpoint(buffer_success, buffer_failure)
                        results_success.update(buffer_success)
                        results_failure.update(buffer_failure)
                        # reset buffer
                        buffer_success, buffer_failure = {}, {}

            # dump remaining samples
            rest_sample_num = len(buffer_success) + len(buffer_failure)
            if 0 < rest_sample_num:
                if self.checkpoint_interval < self.total_samples_num:
                    logger.info(
                        f"saving the {ordinal(self.checkpoint_rank)} checkpoint at the end of the process"
                    )
                    self.dump_checkpoint(buffer_success, buffer_failure)

                results_success.update(buffer_success)
                results_failure.update(buffer_failure)

            return results_success, results_failure

    def _run_with_coroutine(
        self, source_set: dict[int, Sample], desc: str = "run with the other framework"
    ) -> tuple[dict[int, Sample], dict[int, Sample]]:
        async def async_process_sample_with_success_check(
            id_: int, src_sample: Sample
        ) -> tuple[int, Sample, bool]:
            try:
                _, tgt_sample = await self.async_process_sample(id_, src_sample)
                return id_, tgt_sample, True
            except (
                AssertionError,
                HTTPError,
                OpenAIError,
                RequestException,
                RetryError,
                ValueError,
            ) as e:
                logger.error(f"error: {e}")
                return id_, src_sample, False

        async def loop(results_success: dict[int, Sample], results_failure: dict[int, Sample]):

            # pylint: disable=import-outside-toplevel
            # TODO implement AsyncPool
            pool = AsyncPool(batch_size=self.num_workers, tqdm=True, dynamic_batch_size=True)

            for id_, sample in source_set.items():
                pool.submit(
                    async_process_sample_with_success_check(id_, sample),
                    ret=None,
                    callback=None,
                )

            for result in pool.as_completed():
                id_, smp, success = await result
                if success:
                    results_success[id_] = smp
                else:
                    results_failure[id_] = smp

        results_success: dict[int, Sample] = {}
        results_failure: dict[int, Sample] = {}

        asyncio.run(loop(results_success, results_failure))

        return results_success, results_failure

    def run(self) -> None:
        # process checkpoints
        if not self.resume_rolecheckpoint:  # start from scratch
            self.checkpoint_rank = 0
            logger.info("starting from scratch")
            if self.checkpoint_dir:
                self.clean_checkpoints()
        else:  # resume from checkpoint
            logger.info(
                "resuming from checkpoint:\n"
                f"{self.checkpoint_success_path}\n"
                f"{self.checkpoint_failure_path}"
            )

            # load rank of the last checkpoint
            with open(self.checkpoint_rank_path, encoding="utf-8", mode="r") as f:
                self.checkpoint_rank = int(f.read().strip())

            checkpoints_success: list[dict[int, Sample]] = []
            checkpoints_failure: list[dict[int, Sample]] = []
            for rank in trange(
                self.checkpoint_rank,
                desc=f"loading checkpoints [0 to {self.checkpoint_rank})",
            ):
                current_checkpoint_success, current_checkpoint_failure = self.load_checkpoint(rank)
                checkpoints_success.append(current_checkpoint_success)
                checkpoints_failure.append(current_checkpoint_failure)

            ids_success = reduce(lambda x, y: x | y, (ckpt.keys() for ckpt in checkpoints_success))
            ids_failure = reduce(lambda x, y: x | y, (ckpt.keys() for ckpt in checkpoints_failure))
            processed_sample_ids = ids_success | ids_failure

            self.source_set = self.source_set.exclude_by_ids(processed_sample_ids)

            logger.success(f"total samples: {self.total_samples_num}")
            logger.success(f"processed samples: {len(processed_sample_ids)}")
            logger.success(f"remained samples: {len(self.source_set)}")

        # run
        if self.num_workers == 1:
            _run = self._run_without_parallel
        elif self.num_workers > 1:
            _run = self._run_with_parallel
        elif self.num_workers < 0:
            self.num_workers = abs(self.num_workers)
            _run = self._run_with_coroutine
        else:
            raise ValueError(f"num_workers should be greater than 0, but got {self.num_workers}")

        last_results_success, last_results_failure = _run(
            self.source_set, desc=f"Running {self.task_name}"
        )

        if 0 < self.checkpoint_interval <= self.total_samples_num:
            # collect all checkpoints
            try:
                with open(self.checkpoint_rank_path, encoding="utf-8", mode="r") as f:
                    outter_checkpoint_rank = int(f.read().strip())
                    assert (
                        self.checkpoint_rank == outter_checkpoint_rank
                    ), "checkpoint rank mismatch"
            except FileNotFoundError as e:
                logger.error(f"{e}")
                results_success: dict[int, Sample] = last_results_success
                results_failure: dict[int, Sample] = last_results_failure
            else:
                results_success: dict[int, Sample] = {}
                results_failure: dict[int, Sample] = {}

                for rank in trange(self.checkpoint_rank, desc="collecting checkpoints"):
                    current_checkpoint_success, current_checkpoint_failure = self.load_checkpoint(
                        rank
                    )
                    results_success.update(current_checkpoint_success)
                    results_failure.update(current_checkpoint_failure)
        else:  # no checkpoint
            results_success = last_results_success
            results_failure = last_results_failure

        # sort results by id
        results_success = dict(
            sorted(
                tqdm(results_success.items(), desc="sorting success samples"),
                key=lambda x: x[0],
            )
        )
        results_failure = dict(
            sorted(
                tqdm(results_failure.items(), desc="sorting failure samples"),
                key=lambda x: x[0],
            )
        )
        if self.checkpoint_dir is not None:
            dump_samples(results_success.values(), self.checkpoint_success_path, mode="w+")
        # report
        logger.success(
            f"Processed {len(results_success)} successful samples "
            f"and {len(results_failure)} failed samples."
        )

        # rerun failure samples
        if self.max_rerun_count > 0 and len(results_failure) > 0:
            logger.info(f"Rerunning {len(results_failure)} failure samples.")
            rerun_count = 0
            while len(results_failure) > 0 and rerun_count < self.max_rerun_count:
                failure_dataset = BaseDataset(results_failure, schema=self.source_set.schema)
                new_results_success, new_results_failure = _run(
                    failure_dataset, desc=f"Rerunning round {rerun_count}"
                )
                assert set(new_results_success.keys()).issubset(set(results_failure.keys()))
                assert set(new_results_failure.keys()).issubset(set(results_failure.keys()))
                results_success.update(new_results_success)
                results_failure = new_results_failure

                self.dump_checkpoint(results_success, results_failure)
                logger.warning(
                    f"Rerun round {rerun_count} finished. "
                    f"{len(results_failure)} failure samples remaining."
                )
                rerun_count += 1

            if len(results_failure) > 0:
                logger.warning(
                    f"failure to process {len(results_failure)} samples "
                    f"after {rerun_count} reruns."
                )
                logger.warning(
                    f"please check '{self.checkpoint_failure_path}' for failure samples."
                )
            else:
                logger.warning(
                    f"{len(results_success)} samples processed after {rerun_count} reruns."
                )
                logger.warning(
                    f"{len(results_failure)} samples failure after {rerun_count} reruns."
                )

            # save results
            results_success = dict(
                sorted(
                    tqdm(results_success.items(), desc="sorting rerun success samples"),
                    key=lambda x: x[0],
                )
            )

            dump_samples(
                results_success.values(),
                replace_right_to_left(str(self.output_file_path), ".json", ".rerun.json"),
                mode="w+",
            )
        else:
            dump_samples(
                results_success.values(),
                self.output_file_path,
                mode="w+",
            )


class RepresentativeTask(BaseTask):
    def run(self):
        raise NotImplementedError
