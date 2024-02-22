from itertools import pairwise

import numpy as np

from refsys.system import ReferenceSystem

class Vector:
    """Class representing a vector defined in a reference system"""

    def __init__(self, array: np.ndarray, reference_system: ReferenceSystem):
        if not isinstance(array, np.ndarray):
            raise TypeError(f"array must be of type np.ndarray")
        self.array = array
        
        if not isinstance(reference_system, ReferenceSystem):
            raise TypeError(f"reference_system must be of type ReferenceSystem, received instead {type(reference_system)}")
        self.reference_system = reference_system
    
    def __repr__(self):
        return f"Vector {self.array}, in {self.reference_system}"
    
    def to(self, other_reference_system: ReferenceSystem) -> None:
        """Convert to other_reference_system representation"""

        if not isinstance(other_reference_system, ReferenceSystem):
            raise TypeError(f"other_reference_system must be of type ReferenceSystem, received instead {type(other_reference_system)}") 
        
        if other_reference_system is self.reference_system:
            return

        # Find path in ReferenceSystem graph
        path = self.reference_system.path_to(other_reference_system)
        if path is None:
            raise ValueError(f"Cannot reach {other_reference_system} from {self.reference_system}") 

        # Apply transformations between reference systems
        for rs1, rs2 in pairwise(path):
            if rs1 is rs2.parent:
                rs2.transformation_from_parent(self)
            elif rs1 in rs2.children:
                rs1.transformation_to_parent(self)