import numpy as np

from src.refsys.system import ReferenceSystem
from src.refsys.transformations import Rotation
from src.refsys.transformations import Translation
from src.refsys.vector import Vector

computer_refsys = ReferenceSystem('')
v = Vector(np.array([0., 0., 50.]), computer_refsys)
print(f"Created: {v}")

rot = Rotation('y', 'z', 45, 'deg')
vis_refsys = computer_refsys.apply('vis', rot)
v.to(vis_refsys)
print(f"Converted: {v}")