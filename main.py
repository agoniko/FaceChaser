import argparse
from datetime import datetime
from functools import partial
import json


import numpy as np
import cv2

from src.Arduino import Arduino
from src.calib import computer_refsys
from src.calib import pos_1_refsys
from src.calib import pos_2_refsys
from src.calib import pan_1_refsys
from src.calib import pan_2_refsys
from src.calib import tilt_1_refsys
from src.calib import tilt_2_refsys
from src.calib import arduino_1_refsys
from src.calib import arduino_2_refsys
from src.Engine import Engine
from src.iomanager import IOManager
from src.refsys.vector import Vector
from src.utils import display_results

def main():
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
            
    arduino_refsys_list = [
        (pos_1_refsys, pan_1_refsys, tilt_1_refsys, arduino_1_refsys),
        (pos_2_refsys, pan_2_refsys, tilt_2_refsys, arduino_2_refsys),
    ]
    engine = Engine(computer_refsys, DEVICE, RESCALE_FACTOR, SIMILARIY_THRESHOLD, MAX_TRACKED_PERSONS)

    arduinos = {f"{i}": Arduino(port, rf) for i, (port, rf) in enumerate(zip(args.serial_ports, arduino_refsys_list), 1)}

    # created a *threaded* io manager
    def close():
        global io_manager
        io_manager.stop()

    selected_arduino = None
    def select_arduino(arduino: Arduino):
        nonlocal selected_arduino
        selected_arduino = arduino
    
    def pan_rotate(clockwise):
        nonlocal selected_arduino
        if selected_arduino is None:
            return
        pan_refsys = selected_arduino.ref_sys_list[1]
        if clockwise:
            rot = pan_refsys._to_parent.increment_angle(1.)
        else:
            rot = pan_refsys._to_parent.increment_angle(-1.)
        pan_refsys.from_parent_transformation = rot
    
    def tilt_rotate(clockwise):
        nonlocal selected_arduino
        if selected_arduino is None:
            return
        tilt_refsys = selected_arduino.ref_sys_list[2]
        if clockwise:
            rot = tilt_refsys._to_parent.increment_angle(-1.)
        else:
            rot = tilt_refsys._to_parent.increment_angle(1.)
        tilt_refsys.from_parent_transformation = rot

    key_callback_dict = {
        "q": close,

        # Target selection
        "r": engine.select_random,
        "u": engine.unset_targets,
        "a": engine.select_left,
        "d": engine.select_right,
        "w": engine.select_up,
        "s": engine.select_down,

        # Arduino calibration

        # pos


        # pan
        "m": partial(pan_rotate, clockwise=False),
        "n": partial(pan_rotate, clockwise=True),

        #tilt
        "k": partial(tilt_rotate, clockwise=False),
        "j": partial(tilt_rotate, clockwise=True),
    }

    maiusc_digits = ["!", "\"", "£"]
    for i in range(1, MAX_TRACKED_PERSONS + 1):
        key_callback_dict[str(i)] = engine.set_target
        if str(i) in arduinos.keys():
            key_callback_dict[maiusc_digits[i-1]] = partial(select_arduino, arduinos[str(i)])

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
                target = Vector(
                    array=np.array([0., 0., 1.]),
                    reference_system=arduino.ref_sys_list[-1],
                )
                arduino.send_coordinates(target)

        # TODO: Add a way to select a person by clicking on them
        io_manager.step(frame)

if __name__ == "__main__":
    main()