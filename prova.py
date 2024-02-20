from src.reference_frame_aware_vector import *

pixel_frame = ReferenceFrame('pixel')


computer_frame = ReferenceFrame(
    'computer',
    basis=[
        np.array([1., 0., 0.]),
        np.array([0., -1., 0.]),
        np.array([0., 0., 1.]),
    ],
    parent=pixel_frame,
)


# pixel
target = ReferenceFrameAwareVector(
    vector=np.array([0.5, 0.6, 0.7]),
    reference_frame=pixel_frame
)

# passo a computer
target.to(computer_frame)
print(target)

# ripasso a pixel
target.to(pixel_frame)
print(target)


arduino_frame = ReferenceFrame(
    "Arduino",
    basis=[
        np.array([0., 1., 0.]),
        np.array([1., 0., 0.]),
        np.array([0., 0., 1.]),
    ],
    parent=computer_frame,
)

# passo a arduino
target.to(arduino_frame)
print(target)

arduino_frame_rotated = ReferenceFrame(
    "Arduino Rot",
    rotations=[
        (1, 0, 0.4),
    ],
    translations=[
        np.array([1., 1., 1.])
    ],
    parent=computer_frame,
)

target.to(arduino_frame_rotated)
print(target)


"""""

target = ReferenceFrameAwareVector(np.array([]), reference_frame=computer_frame)
target.to(arduino_frame)

# Rotate existing reference frame
ReferenceFrame(
    parent=computer_frame,
    rotations=[
        (from_axis='x', to_axis='z', angle=30),
        (from_axis='x', to_axis='y', angle=45),
        ],
    translations=[],
)

# target has been rotated
print(target)"""