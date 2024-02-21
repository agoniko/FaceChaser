from typing import Tuple
import numpy as np
from numpy.linalg import norm


class Person:
    def __init__(self, embedding: np.ndarray, bbox: np.ndarray):
        """
        A person is defined by their embedding and their bounding box
        """
        self.identity = Identity(embedding)
        self.bbox = bbox.copy()

    def get_coords(self) -> np.ndarray:
        """
        Returns the x, y, z coordinates of the person, where x and y are the center of the bounding box.
        z is an estimation of distance from the camera, in cm, calculated using the height of the bounding box.
        """
        if self.bbox is not None:
            x = (self.bbox[0] + self.bbox[2]) / 2
            y = (self.bbox[1] + self.bbox[3]) / 2
            z = self.bbox[3] - self.bbox[1]
            return np.array([x, y, z])
        else:
            return None

    def update(self, embedding: np.ndarray, bbox: np.ndarray):
        """
        updates embedding and bbox of the person
        """
        self.identity.update(embedding)
        self.bbox = bbox.copy()

    def copy(self):
        return Person(self.identity.embedding, self.bbox)

    def compare(self, embeddings: np.ndarray) -> float:
        """
        returns a similarity score between self and the target embeddings
        """
        return self.identity.similarity(embeddings)


class Identity:
    def __init__(self, embedding: np.ndarray):
        self.embedding = embedding.copy()
        self.n_updates = 0

    def similarity(self, embeddings: np.ndarray) -> np.ndarray:
        "returns a np.ndarray of cosine similarity scores between self and the target embeddings"
        return np.dot(embeddings, self.embedding) / (
            norm(embeddings, axis=1) * norm(self.embedding)
        )

    def update(self, embedding: np.ndarray):
        """
        Since a tracked person moves around and its face has different poses, 
        we update the embedding by averaging all the embeddings received.
        We empirically found that this method works better than keeping the first or last embedding received.
        It is like having a centroid of the person's face embeddings cluster.
        """
        self.embedding = self.embedding * self.n_updates + embedding
        self.n_updates += 1
        self.embedding /= self.n_updates
