import itertools
import numpy as np
from typing import List, Callable, Optional, Tuple
import json
from src.timethis import timethis

class ReferenceFrame:
    reference_frame_tree = []

    def __repr__(self) -> str:
        return(self.name)
    
    def __init__(self, name, **kwargs):
        self.basis = None
        self.parent = None
        self.children = []
        self.transformation_from_parent = None
        self.transformation_to_parent = None

        self.name = name
        self.kwargs = kwargs

        ReferenceFrame.reference_frame_tree.append(self)

        if "parent" in kwargs.keys():
            self.parent = kwargs["parent"]
            self.parent.children.append(self)

        if "camera" in kwargs.keys():
            camera = kwargs['camera']
            reference_depth = camera['reference_depth']
            reference_size = camera['reference_size']
            reference_pixel_size = camera['reference_pixel_size']
            image_size = camera['image_size']
            self.transformation_from_parent = get_camera_transformation(
                self.parent,
                self,
                reference_depth,
                reference_size,
                reference_pixel_size,
                image_size,
            )

            self.transformation_to_parent = get_camera_transformation(
                self,
                self.parent,
                reference_depth,
                reference_size,
                reference_pixel_size,
                image_size,
                True,
            )
            return

        if "basis" in kwargs.keys():
            basis = kwargs['basis']

            basis_matrix = np.array(basis)
            coords = []
            for e in np.eye(3):
                coords.append(np.linalg.solve(basis_matrix, e))
            from_parent_basis_to_self_basis_matrix = np.array(coords)
        
            self.transformation_from_parent = get_affine_transformation(
                self.parent,
                self,
                from_parent_basis_to_self_basis_matrix,
            )

            self.transformation_to_parent = get_affine_transformation(
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
            self.transformation_from_parent = get_affine_transformation(
                    self.parent,
                    self,
                    inverse_rotation_matrix,
                    -translation,
                )

            self.transformation_to_parent = get_affine_transformation(
                    self,
                    self.parent,
                    inverse_rotation_matrix,
                    translation,
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
        print("Rimuovo me stesso dai genitori")
        self.parent.children.remove(self)
        print("Rimuovo me stesso dall'albero genealogico")
        ReferenceFrame.reference_frame_tree.remove(self)

def load_reference_frame_tree(file_name: str) -> ReferenceFrame:
    with open(file_name, 'r') as fp:
        data = json.load(fp)
    
    for name, kwargs in data:
        print(f"Creating ReferenceFrame \"{name}\"")
        for key, value in kwargs.items():
            match key:
                case "parent":
                    for rf in ReferenceFrame.reference_frame_tree:
                        if rf.name == value:
                            kwargs["parent"] = rf
                            break
                    else:
                        raise KeyError(f"No ReferenceFrame is called {value}")
                case "basis":
                    kwargs["basis"] = [np.array(b) for b in value]
                case "rotations":
                    kwargs["rotations"] = [(ax1, ax2, angle) for ax1, ax2, angle in value]
                case "translations":
                    kwargs["translations"] = [np.array(b) for b in value]
                case "camera":
                    kwargs["camera"] = value
                case _:
                    pass
        ReferenceFrame(name, **kwargs)

def save_reference_frame_tree(file_name: str) -> None:
    tree_list = []
    for rf in ReferenceFrame.reference_frame_tree:
        name = rf.name
        kwargs = dict()
        for key, value in rf.kwargs.items():
            print(key, value)
            match key:
                case "parent":
                    kwargs["parent"] = value.name
                case "basis":
                    kwargs["basis"] = [b.tolist() for b in value]
                case "rotations":
                    kwargs["rotations"] = [(ax1, ax2, angle) for ax1, ax2, angle in value]
                case "translations":
                    kwargs["translations"] = [b.tolist() for b in value]
                case "camera":
                    kwargs["camera"] = value
                case _:
                    pass
                    
        tree_list.append((name, kwargs))

    with open(file_name, 'w') as fp:
        json.dump(tree_list, fp)

def get_affine_transformation(
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
    t.__name__ = "affine_transformation_from_"+str(from_reference_frame)+"_to_"+str(to_reference_frame)
    return t

def get_camera_transformation(
    from_reference_frame: ReferenceFrame,
    to_reference_frame: ReferenceFrame,
    reference_depth: float,
    reference_size: float,
    reference_pixel_size: float,
    image_size: Tuple[int, int],
    inverse=False,
):
    focal = reference_depth * reference_pixel_size / reference_size
    if not inverse:
        def t(v: ReferenceFrameAwareVector):
            if v.reference_frame is not from_reference_frame:
                raise ValueError(f"vector must belong to {from_reference_frame}")
            v.vector[2] = focal * reference_size / max(v.vector[2], 1e-8)
            scale_factor = (v.vector[2] / reference_depth) * (reference_size / reference_pixel_size)
            max_0 = image_size[0] * scale_factor
            max_1 = image_size[1] * scale_factor
            v.vector[0] = v.vector[0] * scale_factor - max_0/2.
            v.vector[1] = v.vector[1] * scale_factor - max_1/2.
            v.reference_frame = to_reference_frame
    else:
        def t(v: ReferenceFrameAwareVector):
            if v.reference_frame is not from_reference_frame:
                raise ValueError(f"vector must belong to {from_reference_frame}")
            scale_factor = (v.vector[2] / reference_depth) * (reference_size / reference_pixel_size)
            max_0 = image_size[0] * scale_factor
            max_1 = image_size[1] * scale_factor
            v.vector[2] = v.vector[2] / focal * reference_size
            v.vector[0] = (v.vector[0] + max_0/2.) / scale_factor
            v.vector[1] = (v.vector[1] + max_1/2.) / scale_factor
            
            v.reference_frame = to_reference_frame
    t.__name__ = "camera_transformation_from_"+str(from_reference_frame)+"_to_"+str(to_reference_frame)
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

    @timethis
    def to(self, reference_frame: ReferenceFrame) -> None:
        """Convert in-place the vector to the specified reference frame"""

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