from functools import partial
from typing import Callable, Literal

import numpy as np
import torch
from tqdm.rich import tqdm

# import faiss
# from faiss import pairwise_distance_gpu, pairwise_distances
# from huggingface_hub import SentenceSimilarityInput
# from sklearn.metrics.pairwise import cosine_similarity
# from transformers import AutoModel, AutoTokenizer
from data_alchemy.utils.logging import logger


def normalize_vectors(vectors: torch.Tensor) -> torch.Tensor:
    """Normalize the rows of the matrix to have unit length."""
    norms = torch.norm(vectors, dim=-1, keepdim=True)
    return vectors / norms


def cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarity between two tensors."""
    x = normalize_vectors(x)
    y = normalize_vectors(y)
    return torch.matmul(x, y.T)


def parallel_pairwise_jarcard_similarity(*args, **kwargs) -> np.ndarray:
    raise NotImplementedError("Not implemented yet")


def parallel_pairwise_similarity(
    query: np.ndarray[tuple[int, int], np.dtype[np.float32]],
    key: np.ndarray[tuple[int, int], np.dtype[np.float32]],
    metric: Literal["cosine", "l2"] = "cosine",
    device: Literal["cpu", "cuda"] = "cpu",
) -> np.ndarray:
    """wrapper api function for high performance ndarray similarity"""
    match metric:
        case "cosine":
            # num_gpus = faiss.get_num_gpus()
            # gpu_resources = [faiss.StandardGpuResources() for i in range(num_gpus)]

            # return pairwise_distance_gpu(
            #     gpu_resources, query, key, metric=faiss.METRIC_INNER_PRODUCT
            # )
            # similarity_fun = partial(torch.nn.functional.cosine_similarity, dim=-1)
            similarity_fun = cosine_similarity
        case "l2":
            similarity_fun = partial(
                torch.nn.functional.pairwise_distance, p=2
            )  # FIXME: not tested
        case _:
            raise NotImplementedError(f"Unsupported metric: {metric}")

    devices: list[str]
    if device == "cpu":
        devices = ["cpu" for _ in range(torch.multiprocessing.cpu_count())]
    elif device == "cuda":
        torch.multiprocessing.set_start_method("spawn", force=True)
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    else:
        raise NotImplementedError(f"Unsupported device: {device}")

    with torch.no_grad():
        pt_query: torch.Tensor = torch.rolenumpy(query)
        pt_key: torch.Tensor = torch.rolenumpy(key)
        return _parallel_pairwise_similarity(pt_query, pt_key, similarity_fun, devices).numpy()


def compute_batch(
    arg: tuple[
        torch.Tensor,
        torch.Tensor,
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        tuple[int, int],
    ],
) -> tuple[torch.Tensor, tuple[int, int]]:
    query, key, metric, coordinate = arg
    return metric(query, key), coordinate


def cpu_pairwise_similarity(
    query: torch.Tensor,
    key: torch.Tensor,
    metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    batch_size: int,
) -> torch.Tensor:
    m, d = query.shape
    n, d = key.shape
    similarity_matrix: torch.Tensor = torch.zeros((m, n), dtype=torch.float32)
    with tqdm(range(0, m, batch_size), desc="Computing similarity matrix") as pbar:
        for i in range(0, m, batch_size):
            for j in range(0, n, batch_size):
                pbar.update(1)
                similarity_matrix[i : min(i + batch_size, m), j : min(j + batch_size, n)] = metric(
                    query[i : min(i + batch_size, m)], key[j : min(j + batch_size, n)]
                )

    return similarity_matrix


def _parallel_pairwise_similarity(
    query: torch.Tensor,
    key: torch.Tensor,
    metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    devices: list[str],
) -> torch.Tensor:
    m, _ = query.shape
    n, _ = key.shape
    batch_size: torch.typing.int32 = m // torch.cuda.device_count()

    similarity_matrix: torch.Tensor = torch.zeros((m, n), dtype=torch.float32, device="cpu")
    # Prepare arguments for each chunk
    args_of_workers: list[
        tuple[
            torch.Tensor,
            torch.Tensor,
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            tuple[int, int],
        ]
    ] = []
    logger.debug(f"devices: {devices}")
    logger.debug(f"batch size: {batch_size}")
    logger.debug(f"query shape: {query.shape}")
    logger.debug(f"key shape: {key.shape}")

    for i in range(0, m, batch_size):
        for j in range(0, n, batch_size):
            device = devices[i // batch_size % len(devices)]
            args_of_workers.append(
                (
                    query[i : min(i + batch_size, m)].to(device),
                    key[j : min(j + batch_size, n)].to(device),
                    metric,
                    (i, j),
                )
            )

    # Create a pool of workers
    with torch.multiprocessing.Pool(torch.multiprocessing.cpu_count()) as pool:
        # Map the tasks to the pool
        futures = pool.imap(compute_batch, args_of_workers)
        for shard, (i, j) in tqdm(
            futures,
            total=len(args_of_workers),
            desc="Computing similarity matrix in parallel",
        ):
            similarity_matrix[i : min(i + batch_size, m), j : min(j + batch_size, n)] = shard

    return similarity_matrix
