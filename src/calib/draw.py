import numpy as np
import cv2

from . import computer_refsys, arduino_1_refsys, arduino_2_refsys
from src.refsys.system import ReferenceSystem
from src.refsys.transformations import Translation
from src.refsys.vector import Vector


translation = Translation(-30., 10., -5.)
visualization_refsys = computer_refsys.apply('visualization', translation)

l = 5.
thickness = 1
compute_center = Vector(np.array([0., 0., 0.]), computer_refsys)
computer_basis = [
    Vector(np.array([l, 0., 0.]), computer_refsys),
    Vector(np.array([0., l, 0.]), computer_refsys),
    Vector(np.array([0., 0., l]), computer_refsys),
]

def get_top_view_image(width: int, height: int) -> np.ndarray:
    arduino_1_center = Vector(np.array([0., 0., 0.]), arduino_1_refsys)
    arduino_1_basis = [
        Vector(np.array([l, 0., 0.]), arduino_1_refsys),
        Vector(np.array([0., l, 0.]), arduino_1_refsys),
        Vector(np.array([0., 0., l]), arduino_1_refsys),
    ]

    arduino_2_center = Vector(np.array([0., 0., 0.]), arduino_2_refsys)
    arduino_2_basis = [
        Vector(np.array([l, 0., 0.]), arduino_2_refsys),
        Vector(np.array([0., l, 0.]), arduino_2_refsys),
        Vector(np.array([0., 0., l]), arduino_2_refsys),
    ]

    image = np.zeros((60, 60, 3), dtype=np.uint8)

    compute_center.to(visualization_refsys)
    x0 = int(compute_center.array[0])
    y0 = int(compute_center.array[2])

    arduino_2_center.to(visualization_refsys)
    a2x0 = int(arduino_2_center.array[0])
    a2y0 = int(arduino_2_center.array[2])

    arduino_1_center.to(visualization_refsys)
    a1x0 = int(arduino_1_center.array[0])
    a1y0 = int(arduino_1_center.array[2])

    # Draw computer to arduino lines
    cv2.line(image, (x0, y0), (a1x0, a1y0), (0, 0, 255), thickness)
    cv2.line(image, (x0, y0), (a2x0, a2y0), (0, 0, 255), thickness)

    # Draw computer
    for b in computer_basis:
        b.to(visualization_refsys)
        x = int(b.array[0])
        y = int(b.array[2])
        cv2.line(image, (x0, y0), (x, y), (255, 0, 0), thickness)
        b.detach()
    
    # Draw arduino 1
    for b in arduino_1_basis:
        b.to(visualization_refsys)
        x = int(b.array[0])
        y = int(b.array[2])
        cv2.line(image, (a1x0, a1y0), (x, y), (0, 255, 0), thickness)
        b.detach()
    
    # Draw arduino 2
    for b in arduino_2_basis:
        b.to(visualization_refsys)
        x = int(b.array[0])
        y = int(b.array[2])
        cv2.line(image, (a2x0, a2y0), (x, y), (0, 255, 0), thickness)
        b.detach()
    
    image = cv2.resize(image, (width, height))
    return image