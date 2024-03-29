from abc import ABC, abstractmethod
from typing import Self

import numpy as np

class Transformation(ABC):
    """Performs a transformation on arrays.
    
    Attributes:
        has_inverse: a boolean indicating if the function has an inverse
        inverse: the inverse transformation if present, otherwise None
    """
    def __init__(self, params):
        self.has_inverse = True
        self._direct = lambda x: x
        self._inverse = lambda x: x
        self.params = params

    @abstractmethod
    def _compute_direct(self):
        """Compute direct transformation"""
        ...

    @abstractmethod
    def _compute_inverse(self):
        """Compute inverse transformation"""
        ...

    def direct(self, array: np.ndarray) -> np.ndarray:
        return self._direct(array)
    
    def inverse(self, array: np.ndarray) -> np.ndarray:
        return self._inverse(array)
    
    @property
    def params(self):
        return self._params
    
    @params.setter
    def params(self, params):
        self._params = params
        self._compute_direct()
        self._compute_inverse()
    
    @abstractmethod
    def copy(self):
        ...
        

    def __call__(self, array: np.ndarray) -> np.ndarray:
        return self.direct(array)

class Identity(Transformation):
    """Identity transformation, does what you expect"""
    def __init__(self):
        super().__init__(None)
    def _compute_direct(self):
        pass
    def _compute_inverse(self):
        pass
    def copy(self):
        return Identity()

class Rotation(Transformation):
    """Performs anticlockwise rotation"""
    ax_to_int = {'x': 0, 'y': 1, 'z': 2}
    def __init__(self, ax1: str, ax2: str, angle: float, unit: str):
        """Performs anticlockwise rotation
        
        Rotation is performed on the oriented plane ax1ax2. Axes must be different.

        Args:
            ax1: axis from which perform the rotation, can be 'x', 'y' or 'z' 
            ax2: axis towards which perform the rotation, can be 'x', 'y' or 'z'
            angle: angle of anticlockwise rotation (as usually done in xy cartesian plane)
            unit: unit of measure of angle, can be "deg" or "rad"
        """
        if ax1 not in Rotation.ax_to_int.keys():
            raise ValueError(f"ax1 expected to be either 'x', 'y' or 'z', received {ax1}")
        if ax2 not in Rotation.ax_to_int.keys():
            raise ValueError(f"ax2 expected to be either 'x', 'y' or 'z', received {ax2}")
        if ax1 == ax2:
            raise ValueError(f"ax1 and ax2 must be different")
        self._ax1 = ax1
        self._ax2 = ax2
        self._angle = angle
        if not unit in ['deg', 'rad']:
            raise ValueError("unit must be either 'deg' or 'rad'")
        self._unit = unit
        super().__init__({
            "ax1": self._ax1,
            "ax2": self._ax2,
            "angle": self._angle,
            "unit": self._unit
        })
    
    @property
    def ax1(self):
        return self._ax1
    
    @ax1.setter
    def ax1(self, value):
        raise TypeError("You cannot modify ax1")

    @property
    def ax2(self):
        return self._ax2
    
    @ax2.setter
    def ax2(self, value):
        raise TypeError("You cannot modify ax2")
    
    @property
    def angle(self):
        return self._angle
    
    @angle.setter
    def angle(self, value):
        self._angle = value
        self.params = {
            "ax1": self._ax1,
            "ax2": self._ax2,
            "angle": self._angle,
            "unit": self._unit,
        }
    
    @property
    def unit(self):
        return self._unit
    
    @unit.setter
    def unit(self, value):
        if not value in ['deg', 'rad']:
            raise ValueError("unit must be either 'deg' or 'rad'")
        if value == self._unit:
            return
        if self._unit == 'deg':
            rad_angle = self._angle * np.pi / 180.
            self._angle = rad_angle
            self._unit = 'rad'
        else:
            deg_angle = self._angle * 180. / np.pi
            self._angle = deg_angle
            self._unit = 'deg'
        self._params['unit'] = self._unit
        self._params['angle'] = self._angle
    
    def copy(self):
        return Rotation(self._ax1, self._ax2, self._angle, self._unit)

    def increment_angle(self, amount: float) -> Self:
        return Rotation(self._ax1, self.ax2, self._angle + amount, self._unit)
        
    def _compute_direct(self):
        if self._unit == "deg":
            rad_angle = self._angle * np.pi / 180.
        self.direct_rotation_matrix = np.eye(3)

        ax1 = Rotation.ax_to_int[self._ax1]
        ax2 = Rotation.ax_to_int[self._ax2]
        # ax1 is mapped to this
        self.direct_rotation_matrix[ax1, ax1] =  np.cos(rad_angle)
        self.direct_rotation_matrix[ax2, ax1] =  np.sin(rad_angle)
        # ax2 is mapped to this
        self.direct_rotation_matrix[ax1, ax2] = -np.sin(rad_angle)
        self.direct_rotation_matrix[ax2, ax2] =  np.cos(rad_angle)
        self._direct = lambda array: self.direct_rotation_matrix @ array
    
    def _compute_inverse(self):
        # The matrix is orthonormal
        self.inverse_rotation_matrix = self.direct_rotation_matrix.T.copy()
        self._inverse = lambda array: self.inverse_rotation_matrix @ array

class Translation(Transformation):
    def __init__(self, x: float, y: float, z: float):
        self._x = x
        self._y = y
        self._z = z
        super().__init__({
            "x": self._x,
            "y": self._y,
            "z": self._z,
        })

        @property
        def x(self):
            return self._x
        
        @x.setter
        def x(self, value):
            self._x = value
            self.params = {
            "x": self._x,
            "y": self._y,
            "z": self._z,
            }
        
        @property
        def y(self):
            return self._y
        
        @y.setter
        def y(self, value):
            self._y = value
            self.params = {
            "x": self._x,
            "y": self._y,
            "z": self._z,
            }
        
        @property
        def z(self):
            return self._z
        
        @z.setter
        def z(self, value):
            self._z = value
            self.params = {
            "x": self._x,
            "y": self._y,
            "z": self._z,
            }

    def copy(self):
        return Translation(self._x, self._y, self._z)

    def _compute_direct(self):
        self.translation_array = np.array([self._x, self._y, self._z])
        self._direct = lambda array: self.translation_array + array
    
    def _compute_inverse(self):
        # The matrix is orthonormal
        self.inverse_translation_array = - self.translation_array.copy()
        self._inverse = lambda array: self.inverse_translation_array + array

    def increment(self, ax: str, amount: float) -> Self:
        match ax:
            case 'x':
                return Translation(self._x + amount, self._y, self._z)
            case 'y':
                return Translation(self._x, self._y + amount, self._z)
            case 'z':
                return Translation(self._x, self._y, self._z + amount)
            case _:
                raise ValueError(f"ax must be either x, y or z, instead got {ax}")