from typing import Tuple

import numpy as np
import cv2

from . import computer_refsys, arduino_1_refsys, arduino_2_refsys
from src.refsys.system import ReferenceSystem
from src.refsys.transformations import Translation
from src.refsys.vector import Vector
from src.timethis import timethis


translation = Translation(-30.0, -20.0, -5.0)
visualization_refsys = computer_refsys.apply("visualization", translation)


@timethis
def get_calib_views(
    top_size: Tuple[int, int], side_size: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    scaling = 10
    l = 5.0
    thickness = 3

    computer_center = Vector(np.array([0.0, 0.0, 0.0]), computer_refsys)
    computer_basis = [
        Vector(np.array([l, 0.0, 0.0]), computer_refsys),
        Vector(np.array([0.0, l, 0.0]), computer_refsys),
        Vector(np.array([0.0, 0.0, l]), computer_refsys),
    ]
    # Arduino 1 axes to draw
    arduino_1_center = Vector(np.array([0.0, 0.0, 0.0]), arduino_1_refsys)
    arduino_1_basis = [
        Vector(np.array([l, 0.0, 0.0]), arduino_1_refsys),
        Vector(np.array([0.0, l, 0.0]), arduino_1_refsys),
        Vector(np.array([0.0, 0.0, l]), arduino_1_refsys),
    ]

    # Arduino 2 axes to draw
    arduino_2_center = Vector(np.array([0.0, 0.0, 0.0]), arduino_2_refsys)
    arduino_2_basis = [
        Vector(np.array([l, 0.0, 0.0]), arduino_2_refsys),
        Vector(np.array([0.0, l, 0.0]), arduino_2_refsys),
        Vector(np.array([0.0, 0.0, l]), arduino_2_refsys),
    ]

    # Black images
    top_image = np.zeros((60 * scaling, 60 * scaling, 3), dtype=np.uint8)
    side_image = np.zeros((60 * scaling, 60 * scaling, 3), dtype=np.uint8)

    # Centers of the various reference systems
    computer_center.to(visualization_refsys)
    x0 = int(computer_center.array[0] * scaling)
    y0 = int(computer_center.array[1] * scaling)
    z0 = int(computer_center.array[2] * scaling)
    computer_center.detach()

    arduino_1_center.to(visualization_refsys)
    a1x0 = int(arduino_1_center.array[0] * scaling)
    a1y0 = int(arduino_1_center.array[1] * scaling)
    a1z0 = int(arduino_1_center.array[2] * scaling)
    arduino_1_center.detach()

    arduino_2_center.to(visualization_refsys)
    a2x0 = int(arduino_2_center.array[0] * scaling)
    a2y0 = int(arduino_2_center.array[1] * scaling)
    a2z0 = int(arduino_2_center.array[2] * scaling)
    arduino_2_center.detach()

    # Draw computer to arduino lines
    cv2.line(top_image, (x0, z0), (a1x0, a1z0), (0, 0, 255), thickness)
    cv2.line(top_image, (x0, z0), (a2x0, a2z0), (0, 0, 255), thickness)
    cv2.line(side_image, (z0, y0), (a1z0, a1y0), (0, 0, 255), thickness)
    cv2.line(side_image, (z0, y0), (a2z0, a2y0), (0, 0, 255), thickness)

    # Draw computer
    for b in computer_basis:
        b.to(visualization_refsys)
        x = int(b.array[0] * scaling)
        y = int(b.array[1] * scaling)
        z = int(b.array[2] * scaling)
        cv2.line(top_image, (x0, z0), (x, z), (255, 0, 0), thickness)
        cv2.line(side_image, (z0, y0), (z, y), (255, 0, 0), thickness)
        b.detach()

    # Draw arduino 1
    for b in arduino_1_basis:
        b.to(visualization_refsys)
        x = int(b.array[0] * scaling)
        y = int(b.array[1] * scaling)
        z = int(b.array[2] * scaling)
        cv2.line(top_image, (a1x0, a1z0), (x, z), (0, 255, 0), thickness)
        cv2.line(side_image, (a1z0, a1y0), (z, y), (0, 255, 0), thickness)
        b.detach()

    # Draw arduino 2
    for b in arduino_2_basis:
        b.to(visualization_refsys)
        x = int(b.array[0] * scaling)
        y = int(b.array[1] * scaling)
        z = int(b.array[2] * scaling)
        cv2.line(top_image, (a2x0, a2z0), (x, z), (0, 255, 0), thickness)
        cv2.line(side_image, (a2z0, a2y0), (z, y), (0, 255, 0), thickness)
        b.detach()

    top_image = cv2.resize(top_image, top_size)
    side_image = cv2.resize(side_image, side_size)
    return top_image, side_image
