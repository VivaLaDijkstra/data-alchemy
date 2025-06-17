import numpy as np
from tqdm.rich import tqdm

from data_alchemy.datasets_.base import BaseDataset, Sample
from data_alchemy.tasks.deduplication.similarity import parallel_pairwise_jarcard_similarity


def jarcard_dedup(
    source_set: BaseDataset[Sample], reference_set: BaseDataset[Sample]
) -> BaseDataset[Sample]:
    # calculate jarcard similarity between each pair of samples in source_set
    # remove duplicated samples from source_set to target_set
    # save target_dataset and rest_dataset to file
    similarity_scores: np.ndarray[tuple[int, int], np.dtype[np.float32]]
    with tqdm(
        total=len(source_set) * len(reference_set),
        desc="Calculating jarcard similarity",
    ) as pbar:
        similarity_scores = parallel_pairwise_jarcard_similarity(
            source_set,
            reference_set,
        )
        pbar.update(len(source_set))

    target_set: list[Sample] = []
    return target_set
