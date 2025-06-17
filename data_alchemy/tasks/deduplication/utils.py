import numpy as np
import numpy.typing as npt
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE


def reduce_dim(
    arr: np.ndarray[tuple[int, ...], np.dtype[np.float32]],
    dim: int = 1,
    method: str = "pca",
) -> np.ndarray[tuple[int, ...], np.dtype[np.float32]] | npt.NDArray:
    """Reduce dimension of the array using PCA, SVD or t-SNE."""

    assert method in ["pca", "svd", "t-sne"]

    n_components = min(*arr.shape, dim)

    match method:
        case "pca":
            # cost O(q * n * d)
            pca = PCA(n_components=n_components)
            return pca.fit_transform(arr)
        case "svd":
            # cost O(q * n * d)
            svd = TruncatedSVD(n_components=n_components)
            return svd.fit_transform(arr)
        case "tsne":
            # barnes hut method cost O(nlogn) time complexity
            tsne = TSNE(n_components=n_components, method="barnes_hut", perplexity=30.0)
            return tsne.fit_transform(arr)
        case _:
            raise ValueError(f"Unknown method: {method}")
