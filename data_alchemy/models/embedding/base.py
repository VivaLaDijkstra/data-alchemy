import numpy as np

from ..base import BaseModel


class EmbeddingModel(BaseModel):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("EmbeddingModel is an abstract class and cannot be instantiated")

    def embed(self, sentence: str) -> np.ndarray:
        raise NotImplementedError("embed method must be implemented in subclass")
