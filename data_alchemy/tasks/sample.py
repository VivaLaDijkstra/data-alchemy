from data_alchemy.datasets_ import BaseDataset
from data_alchemy.datasets_.utils import concat_datasets
from data_alchemy.tasks.base import BaseTask
from data_alchemy.utils.io import Path, dump_samples
from data_alchemy.utils.logging import logger


class Sampler(BaseTask):
    def __init__(
        self,
        source_file: list[Path],
        output_file: list[Path],
        ratio: float = None,
        num: int = None,
        seed: int = 42,
        dry_run: bool = False,
    ):
        """Sample a dataset from a list of files to an output file

        Args:
            source_file (list[str]): List of source files to sample from
            output_dir (str): Output directory to save the sampled file
            ratio (float): Ratio of samples to take from each source file
            num (int): Number of samples to take from each source file
            dry_run (bool, optional): Dry run. Defaults to False.
        """
        super().__init__(
            task_name="sampling",
            source_file=source_file,
            output_file=output_file,
            seed=seed,
            dry_run=dry_run,
        )
        assert len(self.source_file) == 1, "Sampling task only supports one source file"
        assert len(self.output_file) == 1, "Sampling task only supports one output file"
        assert ratio is None or num is None, "Only one of ratio or num must be provided"

        self.output_file_path: str = self.output_file[0]
        self.dataset: BaseDataset = concat_datasets(self._load_datasets(source_file))
        if ratio is not None:
            self.subset_size: int = int(len(self.dataset) * ratio)
            self.ratio: float = ratio
        elif num is not None:
            assert num <= len(
                self.dataset
            ), "Number of samples must be less than or equal to the size of the dataset"
            self.subset_size: int = num
            self.ratio: float = num / len(self.dataset)
        else:
            raise ValueError("Either ratio or num must be provided")

    def run(self):
        """Sample the dataset"""

        logger.success(
            "Sampling datasets from size "
            f"{len(self.dataset)} to size {self.subset_size} with ratio {self.ratio:.2f}"
        )

        subset: BaseDataset = self.dataset.get_subset(self.subset_size)

        logger.success(f"Saving sampled dataset to {self.output_file_path}")
        dump_samples(
            subset,
            self.output_file_path,
        )
