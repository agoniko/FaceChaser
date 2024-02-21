import json
from src.iomanager import IOManager
import cv2
from datetime import datetime
import numpy as np
from src.Engine import Engine
from src.utils import display_results
from src.Arduino import Arduino
from src.reference_frame_aware_vector import load_reference_frame_tree, ReferenceFrame, ReferenceFrameAwareVector
import argparse


def get_mouse_coords(event, x, y, flags, param):
    global selected_person, persons
    if event == cv2.EVENT_LBUTTONUP:
        print(f"Mouse coords: {x}, {y}")
        for p in persons:
            if p.bbox[0] < x < p.bbox[2] and p.bbox[1] < y < p.bbox[3]:
                selected_person = p
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rescale", type=float, default=0.4)
    parser.add_argument("--similarity", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument(
        "--serial_ports",
        dest="serial_ports",
        nargs="+",
        action="store",
    )
    # serial port = /dev/cu.usbmodem21201
    args = parser.parse_args()

    RESCALE_FACTOR = args.rescale
    SIMILARIY_THRESHOLD = args.similarity
    DEVICE = args.device

    if args.serial_ports is not None:
        MAX_TRACKED_PERSONS = len(args.serial_ports)
    else:
        args.serial_ports = []
        MAX_TRACKED_PERSONS = 2

    # Reference frames creation
    load_reference_frame_tree("config.json")
    for rf in ReferenceFrame.reference_frame_tree:
        match rf.name:
            case "computer_pixel_frame":
                computer_pixel_frame = rf
            case "computer_frame":
                computer_frame = rf
            case "arduino_1_frame":
                arduino_1_frame = rf
            case "arduino_2_frame":
                arduino_2_frame = rf
            case _:
                pass
            
    arduino_reference_frames = [arduino_1_frame, arduino_2_frame]
    engine = Engine(computer_pixel_frame, DEVICE, RESCALE_FACTOR, SIMILARIY_THRESHOLD, MAX_TRACKED_PERSONS)
    arduinos = {f"{i}": Arduino(port, rf) for i, (port, rf) in enumerate(zip(args.serial_ports, arduino_reference_frames), 1)}
    
    # created a *threaded* io manager
    def close():
        global io_manager
        io_manager.stop()

    def anticlockwise_rotate():
        global arduinos
        rf = arduinos["1"].reference_frame.parent
        translations = arduinos["1"].reference_frame.kwargs["translations"]
        rotations = arduinos["1"].reference_frame.kwargs["rotations"]
        new_rotations = []
        for ax1, ax2, angle in rotations:
            if ax1 == 2 and ax2 == 0:
                new_rotations.append((ax1, ax2, angle - 0.1))
            else:
                new_rotations.append((ax1, ax2, angle))
        rotated_rf = ReferenceFrame(
            "arduino_1_frame",
            translations=translations,
            rotations=new_rotations,
            parent=rf,
        )
        print(new_rotations)
        arduinos["1"].reference_frame.remove()
        arduinos["1"].reference_frame = rotated_rf
    
    def clockwise_rotate():
        global arduinos
        rf = arduinos["1"].reference_frame.parent
        translations = arduinos["1"].reference_frame.kwargs["translations"]
        rotations = arduinos["1"].reference_frame.kwargs["rotations"]
        new_rotations = []
        for ax1, ax2, angle in rotations:
            if ax1 == 2 and ax2 == 0:
                new_rotations.append((ax1, ax2, angle + 0.1))
            else:
                new_rotations.append((ax1, ax2, angle))
        rotated_rf = ReferenceFrame(
            "arduino_1_frame",
            translations=translations,
            rotations=new_rotations,
            parent=rf,
        )
        print(new_rotations)
        arduinos["1"].reference_frame.remove()
        arduinos["1"].reference_frame = rotated_rf


    key_callback_dict = {
        "r": engine.select_random,
        "u": engine.unset_targets,
        "q": close,
        "a": engine.select_left,
        "d": engine.select_right,
        "w": engine.select_up,
        "s": engine.select_down,
        "m": anticlockwise_rotate,
        "n": clockwise_rotate,
    }
    for i in range(1, MAX_TRACKED_PERSONS + 1):
        key_callback_dict[str(i)] = engine.set_target

    io_manager = IOManager(
        src=0, name="Multi Tracking", key_callback_dict=key_callback_dict
    ).start()

    while io_manager.is_running():
        # grab the frame from the threaded video stream and resize it
        frame = io_manager.get_frame()
        frame = engine.process_frame(frame)

        for key, arduino in arduinos.items():
            target = engine.get_coords(key)
            if target is not None:
                arduino.send_coordinates(target)
            else:
                target = ReferenceFrameAwareVector(
                    vector=np.array([0., 0., 1.]),
                    reference_frame=arduino.reference_frame,
                )
                arduino.send_coordinates(target)

        # TODO: Add a way to select a person by clicking on them
        io_manager.step(frame)
