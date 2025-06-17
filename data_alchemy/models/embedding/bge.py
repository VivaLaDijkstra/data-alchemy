import asyncio
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from data_alchemy.utils.auto_resume import model_cache

from .base import EmbeddingModel


class BGEM3EmbeddingModel(EmbeddingModel):
    def __init__(
        self,
        model_name_or_path: str = "BAAI/bge-m3",
        batch_size: int = 12,
        max_length: int = 8192,
        use_fp16: str = True,
        device: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ):
        if device == "cuda" and torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                device = None  # FlagEmbedding takes None for multi-GPU
            else:
                device = "cuda"
        else:
            device = "cpu"

        # dynamic import to avoid heavy lib reliance
        # pylint: disable=import-outside-toplevel
        from FlagEmbedding import BGEM3FlagModel
        from FlagEmbedding.BGE_M3 import BGEM3ForInference

        self.model = BGEM3FlagModel(model_name_or_path, use_fp16=use_fp16, device=device)

        self.batch_size = batch_size
        self.max_length = max_length

    def embed(
        self, sentences: list[str], batch_size: int = None, max_length: int = None
    ) -> list[np.ndarray[tuple[np.int32], np.float64]]:
        dense_vecs: np.ndarray[tuple[np.int64, np.int64], float] = self.model.encode(
            sentences,
            batch_size=batch_size or self.batch_size,
            max_length=max_length or self.max_length,
        )["dense_vecs"]

        # unpack tensor to list
        embeddings = list(dense_vecs)
        return embeddings
