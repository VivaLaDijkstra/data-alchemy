import asyncio
from functools import wraps
from pathlib import Path
from typing import Optional, Generator

import numpy as np
import torch
from tqdm.rich import tqdm

from data_alchemy.datasets_.base import BaseDataset, Sample
from data_alchemy.models.embedding.bge import BGEM3EmbeddingModel
from data_alchemy.tasks.deduplication.similarity import parallel_pairwise_similarity
from data_alchemy.utils.logging import logger
from data_alchemy.utils.string import gen_hash


def batch_generator(
    dataset: BaseDataset[Sample], batch_size: int
) -> Generator[list[Sample], None, None]:
    batch: list[Sample] = []
    for i, sample in enumerate(dataset):
        batch.append(sample)
        if len(batch) == batch_size or i == len(dataset) - 1:
            yield batch
            batch = []


# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-locals
class BGEM3EmbeddingDeduplicator:
    def __init__(self, *args, **kwargs) -> None:
        self.actually_initiated = False
        self.args = args
        self.kwargs = kwargs

    @staticmethod
    def ensure_initialized(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not self.actually_initiated:
                self.lazy_init(*args, **kwargs)
            return func(self, *args, **kwargs)

        return wrapper

    # pylint: disable=attribute-defined-outside-init
    # pylint: disable=too-many-arguments
    def lazy_init(
        self,
        source_file: list[Path],
        reference_file: list[Path],
        use_gpu: bool,
        embedding_cache_dir: Path,
        threshold: float,
        embedding_model_name_or_path: str = "BAAI/bge-m3",
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
    ):
        self.actually_initiated = True

        self.source_file = source_file
        self.reference_file = reference_file

        self.device: str = "cpu"
        if use_gpu:
            if torch.cuda.is_available():
                self.device = "cuda"
        logger.info(f"Using device: {self.device}")
        logger.info(f"Device count: {torch.cuda.device_count()}")

        self.embedding_model = BGEM3EmbeddingModel(
            model_name_or_path=embedding_model_name_or_path,
            batch_size=batch_size,
            max_length=max_length,
            # use_fp16=False,
            device=self.device,
        )
        self.batch_size = batch_size

        self.threshold: float = threshold

        if not embedding_cache_dir.is_dir():
            embedding_cache_dir.mkdir(parents=True, exist_ok=True)

        query_cache_name = gen_hash(
            embedding_model_name_or_path + ",".join(sorted(self.source_file))
        )
        self.query_embedding_cache_path: Path = embedding_cache_dir / f"{query_cache_name}.npy"

        self.self_dedup = sorted(self.source_file) == sorted(self.reference_file)
        if self.self_dedup:
            self.key_embedding_cache_path: Path = self.query_embedding_cache_path
        else:
            key_cache_name = gen_hash(
                embedding_model_name_or_path + ",".join(sorted(self.reference_file))
            )
            self.key_embedding_cache_path: Path = embedding_cache_dir / f"{key_cache_name}.npy"

        self.similarity_scores_cache_path = (
            embedding_cache_dir / f"{query_cache_name}_to_{key_cache_name}.npy"
        )
        self.duplicated_indices_cache_path = (
            embedding_cache_dir
            / f"{query_cache_name}_to_{key_cache_name}_threshold_{threshold}_duplicated_indices.npy"
        )

    def submit(
        self, embedding_model: BGEM3EmbeddingModel, sample: Sample, embeddings: list[np.ndarray]
    ):
        pass

    def calculate_matrix(
        self, dataset: BaseDataset[Sample]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
        # make messages to a string

        logger.info(f"Total samples: {len(dataset)}")

        embeddings: list[np.ndarray] = []
        for sample in tqdm(dataset, desc="Submitting samples"):
            self.submit(self.embedding_model, sample, embeddings)

        embeddings = asyncio.gather(*embeddings)

        logger.info(f"Embeddings Number: {len(embeddings)}")
        logger.info(f"Embedding Shape: {embeddings[0].shape}")

        matrix = np.vstack(embeddings).astype(np.float32)

        logger.info(f"Matrix Shape: {matrix.shape}")

        return matrix

    @ensure_initialized
    def __call__(
        self,
        source_set: BaseDataset[Sample],
        reference_set: BaseDataset[Sample],
        *args,
        **kwargs,
    ) -> BaseDataset[Sample]:
        if self.duplicated_indices_cache_path.exists():
            duplicated_pair_indices: np.ndarray[tuple[int, int], np.dtype[np.float32]] = np.load(
                self.duplicated_indices_cache_path, allow_pickle=True
            )
        else:
            if self.similarity_scores_cache_path.exists():
                similarity_scores: np.ndarray[tuple[int, int], np.dtype[np.float32]] = np.load(
                    self.similarity_scores_cache_path, allow_pickle=True
                )
            else:
                query: np.ndarray[tuple[int, int], np.dtype[np.float32]]
                key: np.ndarray[tuple[int, int], np.dtype[np.float32]]

                if self.query_embedding_cache_path.exists():
                    query = np.load(self.query_embedding_cache_path, allow_pickle=True)
                else:
                    query = self.calculate_matrix(source_set)
                    np.save(query, self.query_embedding_cache_path, allow_pickle=True)

                if self.key_embedding_cache_path.exists():
                    key = np.load(self.key_embedding_cache_path, allow_pickle=True)
                else:
                    key = self.calculate_matrix(reference_set)
                    np.save(key, self.key_embedding_cache_path, allow_pickle=True)

                similarity_scores: np.ndarray[tuple[int, int], np.dtype[np.float32]] = (
                    parallel_pairwise_similarity(
                        query,
                        key,
                        metric="cosine",
                        device=self.device,
                    )
                )

                if self.self_dedup:
                    # if query and key are the same, remove the diagonal
                    logger.success("Removing diagonal from similarity matrix")
                    diagonal_mask = np.eye(similarity_scores.shape[0], dtype=bool)
                    similarity_scores[diagonal_mask] = 0

                np.save(self.similarity_scores_cache_path, similarity_scores, allow_pickle=True)

            thresholds = np.arange(0, 1, step=0.1)
            buckets = dict(zip(thresholds, [0 for _ in thresholds]))
            with tqdm(
                total=len(thresholds) * len(similarity_scores), desc="Calculating buckets"
            ) as pbar:
                for threshold in thresholds:
                    for i, row in enumerate(similarity_scores):
                        if np.any(row > threshold):
                            buckets[threshold] += 1
                        pbar.update(1)

            for threshold, count in buckets.items():
                logger.info(
                    f"Threshold: {threshold:.2f}, "
                    f"Count, Percentage: {count}, "
                    f"{100.*count/len(similarity_scores):.2f}%"
                )

            logger.info(f"Matrix shape: {similarity_scores.shape}")
            logger.info(f"Score Mean: {similarity_scores.mean()}")  # cost O(len(query)*len(key))
            logger.info(f"Score Std: {similarity_scores.std()}")  # cost O(len(query)*len(key))
            logger.info(f"Score Max: {similarity_scores.max()}")  # cost O(len(query)*len(key))
            logger.info(f"Score Min: {similarity_scores.min()}")  # cost O(len(query)*len(key))

            logger.info("collecting duplicated sample indices")
            duplicated_pair_indices = np.argwhere(similarity_scores > self.threshold)

            np.save(self.duplicated_indices_cache_path, duplicated_pair_indices)

        # collect non-duplicated samples from source_set to target_set and duplicated samples from
        # source_set to rest_set
        duplicated_source_sample_indices: set[int] = set(duplicated_pair_indices[:, 0])

        target_set: list[Sample] = []
        for i, smp in enumerate(source_set):
            if i not in duplicated_source_sample_indices:
                target_set.append(smp)

        return target_set


# Callable[[BaseDataset[Sample], BaseDataset[Sample], ...], BaseDataset[Sample]]
embedding_similarity = BGEM3EmbeddingDeduplicator()
