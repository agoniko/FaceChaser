import numpy as np

from src.refsys.system import ReferenceSystem
from src.refsys.transformations import Rotation
from src.refsys.transformations import Translation
from src.refsys.vector import Vector

computer_refsys = ReferenceSystem('')
v = Vector(np.array([0., 0., 50.]), computer_refsys)
print(f"Created {v}")

transl = Translation(-30., 10., -30.)
vis_refsys = computer_refsys.apply('vis', transl)
v.to(vis_refsys)

print(f"Created {v}")

transl = Translation(-7., 22., 30.)
arduino_refsys = computer_refsys.apply('arduino', transl)

v.to(arduino_refsys)
print(f"Converted: {v}")

rot = Rotation('z', 'x', 10, 'deg')
arduino_rot_refsys = arduino_refsys.apply('rotated', rot)

v.to(arduino_rot_refsys)
print(f"Converted: {v}")

arduino_rot_refsys.from_parent_transformation = rot.increment_angle(10)

print(f"Updated: {v}")