from torch import Tensor
from typing import Tuple
import torch.nn.functional as F


class Person:
    def __init__(self, embedding: Tensor, bbox: Tensor):
        self.identity = Identity(embedding)
        self.bbox = bbox

    def get_coords(self) -> Tuple[float, float, float]:
        # TODO implement z
        x = (self.bbox[0] + self.bbox[2]) / 2
        y = (self.bbox[1] + self.bbox[3]) / 2
        z = 0
        return x, y, z


class Identity:
    def __init__(self, embedding: Tensor):
        self.embedding = embedding.clone()

    def similarity(self, embeddings: Tensor) -> Tensor:
        "returns a tensor of similarity scores between self and the target embeddings"
        return F.cosine_similarity(embeddings, self.embedding, dim=-1)

    def update(self, embedding: Tensor):
        self.embedding = embedding.clone()
