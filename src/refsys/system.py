from typing import List, TypeVar, Self

from .transformations import Transformation

class ReferenceSystem:
    """Defines a space in which a vector lives."""
    def __init__(
            self,
            name: str,
            apply: Transformation | None = None,
            to: Self | None = None,
        ):
        """Creates a reference system.
        
        Creates an absolute reference system if only a name is provided.
        Otherwise creates a reference system applying the transformation provided to
        the reference system provided.

        Args:
            name:
                just a name
            apply:
                a transformation to apply to "to"
            to:
                a reference system to which apply "apply"
        """
        self.name = name

        if apply is None and to is None:
            self._parent = None
            self._from_parent = None
            self._to_parent = None
            self._children = []
            self._vectors = []
        else:
            if apply is not None:
                if to is None:
                    raise ValueError("To what reference system apply the transformation?")
            
            if to is not None:
                if apply is None:
                    raise ValueError("Apply what?")
            
            self._parent = to
            self._from_parent = apply.inverse
            self._to_parent = apply
            to._children.append(self)

    @property
    def parent(self):
        return self._parent
    
    @parent.setter
    def parent(self):
        raise TypeError("A reference system's parent cannot be changed")

    def transform(self, transformation: Transformation) -> None:
        raise NotImplementedError
    
    def apply(
        self,
        name: str,
        transformation: Transformation,
    ) -> Self:
        """Creates a new reference system from self by applying the transformation.
        
        If v is a Vector in the returned reference system and t is the transformation,
        then t(v) is the Vector in the starting reference system.
        """
        rf = ReferenceSystem(name)
        rf._parent = self
        rf._from_parent = transformation.inverse
        rf._to_parent = transformation
        self._children.append(rf)
        return rf

    @property
    def from_parent_transformation(self):
        return self._from_parent
    
    def _get_children_vectors(self):
        """Returns children vectors along with path from self to their reference system"""
        children_vectors = []
        path = []
        self._get_children_vectors_recursive(children_vectors, path)
        return children_vectors
    
    def _get_children_vectors_recursive(self, children_vectors, path):
        path = path + [self]
        for v in self._vectors:
            children_vectors.append((v, path))
        
        for rf in self._children:
            rf._get_children_vectors_recursive(children_vectors, path)
    
    @from_parent_transformation.setter
    def from_parent_transformation(self, value):
        """Modifies this reference system as it was created with self.parent.apply(self.name, value)"""
        if self._from_parent is None:
            raise RuntimeError(f"{self.name} is root")
        if value is self._from_parent:
            raise ValueError("Cannot reassign the same transformation, make sure to make a copy first")

        children_vectors = self._get_children_vectors()

        # Bring each vector array back to the parent of this reference system
        for v, path in children_vectors:
            for rf in reversed(path):
                v.array = rf._to_parent(v.array)
        
        # Set new transformation
        self._from_parent = value.inverse
        self._to_parent = value

        # Bring each vector array back to their reference system
        for v, path in children_vectors:
            for rf in path:
                v.array = rf._from_parent(v.array)
    
    def __repr__(self):
        return f"{self.name}__reference_system"
    
    def path_to(self, rf) -> List[Self] | None:
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
    
    def _children_path_to(self, rf: Self) -> List[Self] | None:
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
        
        for _rf in self._children:
            _path = _rf._children_path_to(rf)
            if _path is None:
                continue
            else:
                path.extend(_path)
                break
        else:
            return None
        return path