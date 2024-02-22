from itertools import pairwise

import numpy as np

from .system import ReferenceSystem

class Vector:
    """Class representing a vector defined in a reference system"""

    def __init__(self, array: np.ndarray, reference_system: ReferenceSystem):
        if not isinstance(array, np.ndarray):
            raise TypeError(f"array must be of type np.ndarray")
        self.array = array
        
        if not isinstance(reference_system, ReferenceSystem):
            raise TypeError(f"reference_system must be of type ReferenceSystem, received instead {type(reference_system)}")
        self.reference_system = reference_system
        self.reference_system._vectors.append(self)
    
    def __repr__(self):
        return f"Vector {self.array.tolist()}, in {self.reference_system}"
    
    def to(self, rf: ReferenceSystem) -> None:
        """Convert to other_reference_system representation"""
        if self.reference_system is None:
            raise RuntimeError(f"{self} has been detached from its reference system")

        if not isinstance(rf, ReferenceSystem):
            raise TypeError(f"rf must be of type ReferenceSystem, received instead {type(rf)}") 
        
        if rf is self.reference_system:
            return

        # Find path in ReferenceSystem graph
        path = self.reference_system.path_to(rf)
        if path is None:
            raise ValueError(f"Cannot reach {rf} from {self.reference_system}") 

        # Apply transformations between reference systems
        for rs1, rs2 in pairwise(path):
            if rs1 is rs2:
                continue
            if rs1 is rs2.parent:
                rs2.from_parent(self)
            elif rs1 in rs2._children.values():
                rs1.to_parent(self)
    
    def detach(self) -> None:
        """Detach self from its reference system, make it possible to garbage collect it"""
        if self.reference_system is None:
            # Already detached
            return
        self.reference_system._vectors.remove(self)
        self.reference_system = None