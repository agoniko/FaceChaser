import numpy as np

from src.refsys.system import ReferenceSystem
from src.refsys.transformations import Rotation
from src.refsys.vector import Vector

rf1 = ReferenceSystem('')
v = Vector(np.array([1., 0., 0.]), rf1)
print(f"Created {v}")

print("We now create a new ref sys and move there v")
rot = Rotation('x', 'y', 90, 'deg')
print('rot version =', rot._version)
rf2 = rf1.apply('rotated', rot)

v.to(rf2)
print(f"After conversion: {v}\n")

print("Let's now change rot by first copying it")
new_rot = rot.copy()
new_rot.angle = new_rot.angle + 180
print('new rot version =', rot._version)

rf2.from_parent_transformation = new_rot

print("\nLet's see v:")
print(f"{v}\n")