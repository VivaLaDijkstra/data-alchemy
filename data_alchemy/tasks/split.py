from tqdm.rich import tqdm, trange

from data_alchemy.datasets_ import BaseDataset
from data_alchemy.datasets_.base import IDType
from data_alchemy.datasets_.utils import concat_datasets
from data_alchemy.tasks.base import BaseTask
from data_alchemy.utils.io import Path, dump_samples
from data_alchemy.utils.logging import logger
from data_alchemy.utils.string import replace_right_to_left


class Splitter(BaseTask):
    def __init__(
        self,
        source_file: list[Path],
        output_file: list[Path],
        ratio: list[float],
        randomly: bool = False,
        seed: int = 42,
        dry_run: bool = False,
    ):
        """Split datasets from a list of files to n parts

        Args:
            source_file (list[str]): List of source files to split from
            output_dir (str): Output directory to save the sampled file
            ratio (list[float]): Ratio of samples to take from each source file
            dry_run (bool, optional): Dry run. Defaults to False.
        """
        super().__init__(
            task_name="splitting",
            source_file=source_file,
            output_file=output_file,
            seed=seed,
            dry_run=dry_run,
        )
        assert len(self.source_file) == 1, "Splitting task only supports one source file"
        assert len(self.output_file) > 1, "Splitting task requires more than one output file"
        assert len(self.output_file) == len(
            ratio
        ), "Must have same number of output files as ratios"

        self.ratio: list[float] = [r / sum(ratio) for r in ratio]
        assert (sum(self.ratio) - 1.0) < 1e-5, "Ratioes must sum to 1"

        self.randomly = randomly

        self.datasets: list[BaseDataset] = self._load_datasets(source_file)
        self.dataset: BaseDataset = concat_datasets(self.datasets)

    def run(self):
        """split the dataset"""
        subset_sizes = [int(len(self.dataset) * r) for r in self.ratio]
        subset_sizes[-1] = len(self.dataset) - sum(subset_sizes[:-1])

        logger.info(
            f"splitting dataset from size {len(self.dataset)} into "
            f"{'+'.join(str(sub_size) for sub_size in subset_sizes)} sizes "
            + ("randomly" if self.randomly else "sequentially")
        )

        if self.randomly:
            self.dataset = self.dataset.shuffle()

        cum_sizes = [0]
        for s in subset_sizes:
            cum_sizes.append(cum_sizes[-1] + s)

        ids = list(self.dataset.keys())
        subsets_ids: list[list[IDType]] = [
            ids[start:end] for start, end in zip(cum_sizes[:-1], cum_sizes[1:])
        ]
        subsets: list[BaseDataset] = []
        for sub_ids in tqdm(subsets_ids, desc="splitting"):
            subsets.append(self.dataset.select_by_ids(sub_ids))

        for i in trange(len(subsets), desc="dumping"):
            path = self.output_file[i]
            subset = subsets[i]
            fpath = replace_right_to_left(
                path, ".json", f"_{len(subset)}.ratio_{self.ratio[i]:.2f}.part_{i}.json"
            )
            logger.success(f"dumping {len(subset)} samples for part{i} to {fpath}")
            dump_samples(subset, fpath)
