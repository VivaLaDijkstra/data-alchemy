from pathlib import Path

from data_alchemy.datasets_.utils import concat_datasets
from data_alchemy.tasks.base import BaseTask
from data_alchemy.tasks.utils import DedupFunctionFactory
from data_alchemy.utils.logging import logger


class Deduplicator(BaseTask):
    """Class for deduplicating sentences using embeddings"""

    def __init__(
        self,
        source_file: list[Path],
        reference_file: list[Path],
        output_file: list[Path],
        dedup_function: str,
        seed: int = 42,
        dry_run: bool = False,
        # specific arguments for each deduplication function
        # embedding_model_name_or_path: Optional[str] = None,
        # max_length: int = 512,
        # batch_size: int = 32,
        # similarity_threshold: float = 0.9,
        # embedding_cache_dir: str = "./",
        # use_gpu: bool = False,
        **kwargs,
    ):
        """Initialize the deduplicator

        Arguments:
            source_file (list[str]): List of raw dataset files to deduplicate
            reference_file (list[str]): List of reference dataset files to deduplicate against
            output_file (list[str]): List of output files to save the deduplicated dataset
            embedding_model_name_or_path (str): Name or path of the embedding model
            max_length (int): Maximum length of the input sequence for the embedding model
            emb_batch_size (int): Batch size for the embedding model
            similarity_threshold (float): Threshold for similarity score
            store_embeddings_dir (str): Directory to store the embeddings
            use_stored_embeddings (bool): Whether to use stored embeddings
            use_gpu (bool): Whether to use GPU for embedding
            num_workers (int): Number of workers for parallel processing
            seed (int): Random seed
            dry_run (bool): Whether to run in dry run mode
        """
        super().__init__(
            task_name="dedup",
            source_file=source_file,
            output_file=output_file,
            seed=seed,
            dry_run=dry_run,
        )
        assert len(output_file) == 1, "Deduplication only supports one output file"

        # load dedup functiony
        logger.info(f"Loading sample process function: {dedup_function}")
        factory = DedupFunctionFactory(load_path=Path(__file__).parent / "dedup_functions")
        self.deduplicate = factory.get_function(dedup_function)

        # load source datasets
        logger.info(f"Loading source datasets from {self.source_file}")
        self.source_set = concat_datasets(self._load_datasets(source_file))

        if sorted(self.source_file) == sorted(self.output_file):  # self deduplication
            logger.info(
                "Source file and output file are the same, using source file as reference set"
            )
            self.reference_set = self.source_file
        else:
            # load reference datasets
            logger.info(f"Loading reference datasets from {reference_file}")
            self.reference_set = concat_datasets(self._load_datasets(reference_file))

        if dry_run:
            self.source_set = self.source_set[:4]
            self.reference_set = self.reference_set[:4]

        self.kwargs = {k: v for k, v in kwargs.items() if v is not None}

    def run(self) -> None:
        logger.info(
            f"Running deduplication task with "
            f"{len(self.source_set)} source samples, "
            f"{len(self.reference_set)} reference samples"
        )
        target_set = self.deduplicate(self.source_set, self.reference_set, **self.kwargs)

        logger.success("Target set size:", len(target_set))
        logger.success(f"Removed {len(self.source_set) - len(target_set)} samples")

        # save target_dataset and rest_dataset to file
        logger.info(f"Saving {len(target_set)} target samples to file: {self.output_file[0]}")
        target_set.save(self.output_file[0])
