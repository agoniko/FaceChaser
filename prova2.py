import numpy as np
from src.reference_frame_aware_vector import *

computer_pixel_frame = ReferenceFrame("computer_pixel_frame")

computer_frame = ReferenceFrame(
    "computer_frame",
    camera={
        "reference_depth": 24,
        "reference_size": 25,
        "reference_pixel_size": 1080,
        "image_size": (1080, 1920),
    },
    parent=computer_pixel_frame,
)

arduino_1_frame = ReferenceFrame(
    "arduino_1_frame",
    translations=[
        np.array([-7., 22., 33.5]),
    ],
    parent=computer_frame,
)

arduino_2_frame = ReferenceFrame(
    "arduino_2_frame",
    translations=[
        np.array([7., 22., 33.5]),
    ],
    parent=computer_frame,
)

save_reference_frame_tree('config')