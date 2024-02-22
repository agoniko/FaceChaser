import numpy as np

from src.refsys.system import ReferenceSystem
from src.refsys.transformations import Rotation
from src.refsys.transformations import Translation
from src.refsys.vector import Vector

rf0 = ReferenceSystem('rf0')

rot = Translation(1., 1., 0.)
rf1 = rf0.apply('rf1', rot)
v = Vector(np.array([0., 0., 0.]), rf1)
print(f"Created: {v}")

rot = Translation(1., 0., 0.)
rf2 = rf0.apply('rf2', rot)
v.to(rf2)
print(f"Converted: {v}")

v.to(rf0)
print(f"Converted: {v}")