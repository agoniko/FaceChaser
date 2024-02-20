import itertools
import numpy as np
from typing import List, Callable, Optional


class ReferenceFrame:
    def __repr__(self) -> str:
        return(self.name)
    
    def __init__(self, name, **kwargs):
        self.basis = None
        self.parent = None
        self.children = []
        self.transformation_from_parent = None
        self.transformation_to_parent = None

        self.name = name

        if "parent" in kwargs.keys():
            self.parent = kwargs["parent"]
            self.parent.children.append(self)

        if "basis" in kwargs.keys():
            basis = kwargs['basis']

            basis_matrix = np.array(basis)
            coords = []
            for e in np.eye(3):
                coords.append(np.linalg.solve(basis_matrix, e))
            from_parent_basis_to_self_basis_matrix = np.array(coords)
        
            self.transformation_from_parent = get_transformation(
                self.parent,
                self,
                from_parent_basis_to_self_basis_matrix,
            )

            self.transformation_to_parent = get_transformation(
                self,
                self.parent,
                basis_matrix,
            )
            return

        inverse_rotation_matrix = np.eye(3)
        if "rotations" in kwargs.keys():
            for from_axis, to_axis, angle in kwargs['rotations']:
                axes = [from_axis, to_axis]
                _inverse_rotation_matrix = np.eye(3)
            
                _inverse_rotation_matrix[axes[0], axes[0]] = np.cos(angle)
                _inverse_rotation_matrix[axes[1], axes[0]] = -np.sin(angle)
                _inverse_rotation_matrix[axes[0], axes[1]] = np.sin(angle)
                _inverse_rotation_matrix[axes[1], axes[1]] = np.cos(angle)

                #np.array([
                #    [ np.cos(angle), np.sin(angle)],
                #    [-np.sin(angle), np.cos(angle)],
                #])

                inverse_rotation_matrix = _inverse_rotation_matrix @ inverse_rotation_matrix
        rotation_matrix = np.linalg.inv(inverse_rotation_matrix)

        translation = np.zeros(3)
        if "translations" in kwargs.keys():
            for t in kwargs["translations"]:
                translation += t
        
        if "rotations" in kwargs.keys() or "translations" in kwargs.keys():
            self.transformation_from_parent = get_transformation(
                    self.parent,
                    self,
                    inverse_rotation_matrix,
                    translation,
                )

            self.transformation_to_parent = get_transformation(
                    self,
                    self.parent,
                    inverse_rotation_matrix,
                    -translation,
                    True
                )
    
    def search_children_path(self, rf):
        path = [self]
        if rf is self:
            return path
        if len(self.children) == 0:
            return None
        
        for _rf in self.children:
            _path = _rf.search_children_path(rf)
            if _path is None:
                continue
            else:
                path.extend(_path)
                break
        else:
            return None
        return path
        
    
    def search_path(self, rf):
        """Returns a path from self to rf"""
        self_to_root_path = [self]
        root = self
        while root.parent is not None:
            root = root.parent
            self_to_root_path.append(root)
        self_to_root_path.pop()

        root_to_rf_path = root.search_children_path(rf)

        if root_to_rf_path is None:
            return None
        
        return self_to_root_path + root_to_rf_path
    
    def remove(self):
        self.parent.children.remove(self)

def get_transformation(
    from_reference_frame: ReferenceFrame,
    to_reference_frame: ReferenceFrame,
    linear_term: np.ndarray,
    affine_term: np.ndarray = np.zeros(3),
    first_affine=False,
):
    def t(v: ReferenceFrameAwareVector):
        if v.reference_frame is not from_reference_frame:
            raise ValueError(f"vector must belong to {from_reference_frame}")
        if not first_affine:
            v.vector = linear_term @ v.vector + affine_term
        else:
            v.vector = linear_term @ (v.vector + affine_term)
        v.reference_frame = to_reference_frame
    t.__name__ = "from_"+str(from_reference_frame)+"_to_"+str(to_reference_frame)
    return t
    

class ReferenceFrameAwareVector:
    def __repr__(self):
        return str(self.vector) + " in reference frame: " + str(self.reference_frame)
    def __init__(
            self,
            vector: np.ndarray,
            reference_frame: ReferenceFrame,
        ) -> None:
        self.vector = vector
        self.reference_frame = reference_frame

    def to(self, reference_frame: ReferenceFrame) -> None:
        """Express in-place the vector in the specified reference frame"""

        if self.reference_frame is reference_frame:
            return
        
        # Find path in the reference frame tree
        reference_frame_path = self.reference_frame.search_path(reference_frame)
        if reference_frame_path is None:
            raise ValueError(f"No path to {reference_frame}") 

        # Compute transformations between reference frames
        transformations = []
        for rf1, rf2 in itertools.pairwise(reference_frame_path):
            if rf1 is rf2.parent:
                transformations.append(rf2.transformation_from_parent)
            elif rf1 in rf2.children:
                transformations.append(rf1.transformation_to_parent)

        # Transform the vector in-place
        for t in transformations:
            t(self)