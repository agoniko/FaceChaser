from typing import List, TypeVar

from .transformations import Transformation

TReferenceSystem = TypeVar("TReferenceSystem", bound="ReferenceSystem")

class ReferenceSystem:
    """Defines a space in which a vector lives."""
    _ref_sys_graph = dict()
    _roots = dict()
    def __init__(self, name: str):
        """Creates a new root reference system"""
        if name in ReferenceSystem._ref_sys_graph.keys():
            raise ValueError(f"A reference system named {name} already exists")
        self.name = name
        self._parent = None
        self._from_parent = None
        self._children = dict()
        self._vectors = []
        ReferenceSystem._ref_sys_graph[name] = self
        ReferenceSystem._roots[name] = self

    @property
    def parent(self):
        return self._parent
    
    @parent.setter
    def parent(self):
        raise TypeError("A reference system's parent cannot be changed")

    def apply(
        self,
        name: str,
        transformation: Transformation,
    ) -> TReferenceSystem:
        """Creates a new reference system from self by applying the transformation.
        
        If v is a Vector in the returned reference system and t is the transformation,
        then t(v) is the Vector in the starting reference system.
        """
        rf = ReferenceSystem(name)
        rf._parent = self
        rf._from_parent = transformation
        assert name not in self._children.keys(), f"{name} should not be in {self.name} children"
        self._children[name] = rf
        return rf

    def to_parent(self, v):
        """Move v to the parent of self. v must live in self"""
        if v.reference_system is not self:
            raise ValueError(f"v must live in {self.name}")
        if self._parent is None:
            raise RuntimeError(f"{self.name} is root")
        v.array = self._from_parent.inverse(v.array)
        v.reference_system = self._parent
        self._vectors.remove(v)
        self._parent._vectors.append(v)

    def from_parent(self, v):
        """Move v to self. v must live in self's parent"""
        if v.reference_system is not self._parent:
            raise ValueError(f"v must live in {self._parent.name}")
        v.array = self._from_parent(v.array)
        v.reference_system = self
        self._vectors.append(v)
        self._parent._vectors.remove(v)

    @property
    def from_parent_transformation(self):
        return self._from_parent
    
    def _get_children_vectors(self):
        children_vectors = []
        path = []
        self._get_children_vectors_core(children_vectors, path)
        return children_vectors
    
    def _get_children_vectors_core(self, children_vectors, path):
        """Returns children vectors along with path from self to their reference system"""
        path = path + [self]
        for v in self._vectors:
            children_vectors.append((v, path))
        
        for rf in self._children:
            rf._get_children_vectors_core(children_vectors, path)
    
    @from_parent_transformation.setter
    def from_parent_transformation(self, value):
        """Set the transformation that apply to parent vectors for getting a representation in this reference frame"""
        if self._from_parent is None:
            raise RuntimeError(f"{self.name} is root")
        if value is self._from_parent:
            raise ValueError("Cannot reassign the same transformation, make sure to make a copy first")

        children_vectors = self._get_children_vectors()

        # Bring each vector array back to the parent of this reference system
        for v, path in children_vectors:
            for rf in reversed(path):
                v.array = rf._from_parent.inverse(v.array)
        
        # Set new transformation
        self._from_parent = value

        # Bring each vector array back to their reference system
        for v, path in children_vectors:
            for rf in path:
                v.array = rf._from_parent(v.array)

    def __repr__(self):
        return f"{self.name}__reference_system"
    
    def path_to(self, rf) -> List[TReferenceSystem] | None:
        """Returns a path to rf, not guaranteed to be minimal and repetitions can appear"""
        root_path = [self]
        root = self
        while root.parent is not None:
            root = root.parent
            root_path.append(root)

        children_path = root._children_path_to(rf)

        if children_path is None:
            return None
        
        return root_path + children_path
    
    def _children_path_to(self, rf: TReferenceSystem) -> List[TReferenceSystem] | None:
        """Recursively search a path from self to rf looking at children.
        
        Returns:
            the path as a list of ReferenceSystem object where the first element is self
            and the last element is rf. None if no path was found. There are not repetitions.
        """
        path = [self]
        if rf is self:
            return path
        if len(self._children) == 0:
            return None
        
        for _rf in self._children.values():
            _path = _rf._children_path_to(rf)
            if _path is None:
                continue
            else:
                path.extend(_path)
                break
        else:
            return None
        return path