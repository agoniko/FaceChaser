from typing import Tuple
import torch.nn.functional as F
import numpy as np
from numpy.linalg import norm


class Person:
    def __init__(self, embedding: np.ndarray, bbox: np.ndarray):
        self.identity = Identity(embedding)
        self.bbox = bbox.copy()

    def get_coords(self) -> Tuple[float, float, float]:
        # TODO implement z
        if self.bbox is not None:
            x = (self.bbox[0] + self.bbox[2]) / 2
            y = (self.bbox[1] + self.bbox[3]) / 2
            z = 950 * 25 / (self.bbox[3] - self.bbox[1])
            return x, y, z
        else:
            return None, None, None

    def update(self, embedding: np.ndarray, bbox: np.ndarray):
        self.identity.update(embedding)
        self.bbox = bbox.copy()

    def copy(self):
        return Person(self.identity.embedding, self.bbox)

    def compare(self, embeddings: np.ndarray) -> float:
        return self.identity.similarity(embeddings)


class Identity:
    def __init__(self, embedding: np.ndarray):
        self.embedding = embedding.copy()
        self.n_updates = 0

    def similarity(self, embeddings: np.ndarray) -> np.ndarray:
        "returns a np.ndarray of similarity scores between self and the target embeddings"
        return np.dot(embeddings, self.embedding) / (
            norm(embeddings, axis=1) * norm(self.embedding)
        )

    def update(self, embedding: np.ndarray):
        self.embedding = self.embedding * self.n_updates + embedding
        self.n_updates += 1
        self.embedding /= self.n_updates
