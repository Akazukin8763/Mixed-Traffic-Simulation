from typing import Tuple

import numpy as np
from numba import njit


@njit
def vector_angles(vecs: np.ndarray) -> np.ndarray:
     return np.arctan2(vecs[:, 1], vecs[:, 0])


@njit
def left_normal(vecs: np.ndarray) -> np.ndarray:
    vecs = np.fliplr(vecs) * np.array([-1.0, 1.0])
    return vecs


@njit
def right_normal(vecs: np.ndarray) -> np.ndarray:
    vecs = np.fliplr(vecs) * np.array([1.0, -1.0])
    return vecs


@njit
def normalize(vecs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    norm_factors = []
    for line in vecs:
        norm_factors.append(np.linalg.norm(line))
    norm_factors = np.array(norm_factors)
    normalized = vecs / np.expand_dims(norm_factors, -1)
    for i in range(norm_factors.shape[0]):
        if norm_factors[i] == 0:
            norm_factors[i] = 1e-10
            normalized[i] = np.zeros(vecs.shape[1])
    return normalized, norm_factors


@njit
def vec_diff(vecs: np.ndarray) -> np.ndarray:
    return np.expand_dims(vecs, 1) - np.expand_dims(vecs, 0)


def each_diff(vecs: np.ndarray, keepdims=False) -> np.ndarray:
    diff = vec_diff(vecs)
    diff = diff[~np.eye(diff.shape[0], dtype=bool), :]
    if keepdims:
        diff = diff.reshape(vecs.shape[0], -1, vecs.shape[1])
    return diff
