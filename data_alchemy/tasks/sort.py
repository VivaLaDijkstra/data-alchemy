from data_alchemy.datasets_ import BaseDataset
from data_alchemy.datasets_.utils import concat_datasets
from data_alchemy.tasks.base import BaseTask
from data_alchemy.utils.io import Path, dump_samples
from data_alchemy.utils.logging import logger


class Sorter(BaseTask):
    def __init__(
        self,
        source_file: list[Path],
        output_file: list[Path],
        reverse: bool = False,
        seed: int = 42,
        dry_run: bool = False,
    ):
        """sort the datasets from a list of files to an output file

        Args:
            source_file (list[str]): List of source files to sample from
            output_dir (str): Output directory to save the sampled file
            seed(int, optional): Random seed. Defaults to 42.
            shuffle (bool, optional): Shuffle the dataset. Defaults to True.
        """
        super().__init__(
            task_name="sorting",
            source_file=source_file,
            output_file=output_file,
            seed=seed,
            dry_run=dry_run,
        )
        self.output_file_path = self.output_file[0]
        self.reverse = reverse
        self.dataset: BaseDataset = concat_datasets(self._load_datasets(source_file))

    def run(self):
        """sort the datasets"""

        logger.success(
            f"Sorting datasets with size {len(self.dataset)}",
        )

        sorted_dataset = self.dataset.sorted(key=lambda x: x[1].root[-1].id, reverse=self.reverse)

        logger.success(f"Saving sorted dataset to {self.output_file_path}")

        dump_samples(sorted_dataset, self.output_file_path)
