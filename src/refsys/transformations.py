from abc import ABC, abstractmethod

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

class Rotation(Transformation):
    """Performs anticlockwise rotation"""

    def __init__(self, ax1: int, ax2: int, angle: float, unit: str):
        """Performs clockwise rotation
        
        Rotation is performed on the oriented plane ax1ax2.

        Args:
            ax1: axis from which perform the rotation (consider it as x axis)
            ax2: axis towards which perform the rotation (consider it as y axis)
            angle: angle of anticlockwise rotation (as usually done in xy cartesian plane)
            unit: unit of measure of angle, can be "deg" or "rad"
        """
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
        self.has_inverse = True
    
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

    def _compute_direct(self):
        if self._unit == "deg":
            rad_angle = self._angle * np.pi / 180.
        self.direct_rotation_matrix = np.eye(3)
        # ax1 is mapped to this
        self.direct_rotation_matrix[self._ax1, self._ax1] =  np.cos(rad_angle)
        self.direct_rotation_matrix[self._ax2, self._ax1] =  np.sin(rad_angle)
        # ax2 is mapped to this
        self.direct_rotation_matrix[self._ax1, self._ax2] = -np.sin(rad_angle)
        self.direct_rotation_matrix[self._ax2, self._ax2] =  np.cos(rad_angle)
        self._direct = lambda x: self.direct_rotation_matrix @ x
    
    def _compute_inverse(self):
        # The matrix is orthonormal
        self.inverse_rotation_matrix = self.direct_rotation_matrix.T.copy()
        self._inverse = lambda x: self.inverse_rotation_matrix @ x