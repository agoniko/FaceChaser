import numpy as np

from src.refsys.system import ReferenceSystem
from src.refsys.transformations import Identity
from src.refsys.transformations import Rotation
from src.refsys.transformations import Translation
from src.refsys.vector import Vector

# Generate all reference systems
computer_refsys = ReferenceSystem('computer')

translation_1 = Translation(-7., 22., 30.)
pos_1_refsys = computer_refsys.apply('pos1', translation_1)
pan_1 = Rotation('x', 'z', 0., 'deg')
pan_1_refsys = pos_1_refsys.apply('pan1', pan_1)
tilt_1 = Rotation('y', 'z', 0., 'deg')
tilt_1_refsys = pan_1_refsys.apply('tilt1', tilt_1)
arduino_1_refsys = tilt_1_refsys.apply("arduino1", Identity())

translation_2 = Translation(7., 22., 30.)
pos_2_refsys = computer_refsys.apply('pos2', translation_2)
pan_2 = Rotation('x', 'z', 0., 'deg')
pan_2_refsys = pos_2_refsys.apply('pan2', pan_2)
tilt_2 = Rotation('y', 'z', 0., 'deg')
tilt_2_refsys = pan_2_refsys.apply('tilt2', tilt_2)
arduino_2_refsys = tilt_2_refsys.apply("arduino2", Identity())