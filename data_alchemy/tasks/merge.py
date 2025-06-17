from data_alchemy.datasets_ import BaseDataset
from data_alchemy.datasets_.utils import concat_datasets
from data_alchemy.tasks.base import BaseTask
from data_alchemy.utils.io import Path, dump_samples
from data_alchemy.utils.logging import logger


class Merger(BaseTask):
    def __init__(
        self,
        source_file: list[Path],
        output_file: list[Path],
        seed: int = 42,
        shuffle: bool = False,
        dry_run: bool = False,
    ):
        """Merge the datasets from a list of files to an output file

        Args:
            source_file (list[str]): List of source files to sample from
            output_dir (str): Output directory to save the sampled file
            seed(int, optional): Random seed. Defaults to 42.
            shuffle (bool, optional): Shuffle the dataset. Defaults to True.
        """
        super().__init__(
            task_name="merging",
            source_file=source_file,
            output_file=output_file,
            seed=seed,
            dry_run=dry_run,
        )
        assert len(self.output_file) == 1, "Merging task only supports one output file"
        self.output_file_path: str = self.output_file[0]
        self.shuffle = shuffle
        self.datasets: list[BaseDataset] = self._load_datasets(source_file)

    def run(self):
        """Merge the datasets"""

        if self.dry_run:
            merged_dataset = BaseDataset([])
        else:
            merged_dataset: BaseDataset = concat_datasets(self.datasets)

        if self.shuffle:
            merged_dataset = merged_dataset.shuffle()

        logger.success(
            "Merging datasets with sizes "
            f"{', '.join(str(len(ds)) for ds in self.datasets)}"
            f"to size {len(merged_dataset)}"
        )

        logger.success(f"Saving merged dataset to {self.output_file_path}")
        dump_samples(merged_dataset, self.output_file_path)
