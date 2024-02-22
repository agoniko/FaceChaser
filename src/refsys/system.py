from typing import List, Self

from refsys.transformations import Transformation

class ReferenceSystem:
    def __init__(self, name: str):
        self.name = name
        self.parent = None

    @staticmethod
    def from_apply(
        other: Self,
        parent: Self | None,
        apply: Transformation | None,
    ) -> Self:
        pass

    def __repr__(self):
        return f"{self.name}__reference_system"
    
    def path_to(self, other_reference_system) -> List[Self] | None:
        raise NotImplementedError