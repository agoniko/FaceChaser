import argparse
import numpy as np
from src.reference_frame_aware_vector import *

parser = argparse.ArgumentParser()
parser.add_argument("--anto", action='store_true')
args = parser.parse_args()

computer_pixel_frame = ReferenceFrame("computer_pixel_frame")

if not args.anto:
    computer_frame = ReferenceFrame(
        "computer_frame",
        camera={
            "reference_depth": 24,
            "reference_size": 25,
            "reference_pixel_size": 1080,
            "image_size": (1920, 1080),
        },
        parent=computer_pixel_frame,
    )
    arduino_1_frame = ReferenceFrame(
    "arduino_1_frame",
    rotations=[(2, 0, 0.), (1, 2, 0.)],
    translations=[
        np.array([-7., 22., 33.5]),
    ],
    parent=computer_frame,
    )

    arduino_2_frame = ReferenceFrame(
        "arduino_2_frame",
        rotations=[(2, 0, 0.), (1, 2, 0.)],
        translations=[
            np.array([7., 22., 33.5]),
        ],
        parent=computer_frame,
    )

else:
    computer_frame = ReferenceFrame(
        "computer_frame",
        camera={
            "reference_depth": 19,
            "reference_size": 25,
            "reference_pixel_size": 480,
            "image_size": (640, 480),
        },
        parent=computer_pixel_frame,
    )
    arduino_1_frame = ReferenceFrame(
    "arduino_1_frame",
    rotations=[(2, 0, 0.), (1, 2, 0.)],
    translations=[
        np.array([-7., 19., 32.]),
    ],
    parent=computer_frame,
    )

    arduino_2_frame = ReferenceFrame(
        "arduino_2_frame",
        rotations=[(2, 0, 0.), (1, 2, 0.)],
        translations=[
            np.array([7., 19., 32.]),
        ],
        parent=computer_frame,
    )

save_reference_frame_tree('config.json')