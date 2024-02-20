import json
from src.iomanager import IOManager
import cv2
from datetime import datetime
import numpy as np
from src.Engine import Engine
from src.utils import display_results
from src.Arduino import Arduino
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

    engine = Engine(DEVICE, RESCALE_FACTOR, SIMILARIY_THRESHOLD, MAX_TRACKED_PERSONS)
    arduinos = {f"{i}": Arduino(port) for i, port in enumerate(args.serial_ports, 1)}
    
    with open('config.json', 'r') as fp:
        arduinos_config = json.load(fp)
    
    for key, computer_pos in arduinos_config.items():
        if key in arduinos.keys():
            arduinos[key].set_computer_position(np.array(computer_pos))

    # created a *threaded* io manager
    def close():
        global io_manager
        io_manager.stop()

    key_callback_dict = {
        "r": engine.select_random,
        "u": engine.unset_targets,
        "q": close,
        "a": engine.select_left,
        "d": engine.select_right,
        "w": engine.select_up,
        "s": engine.select_down,
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

        for arduino in arduinos.items():
            x, y, z = engine.get_coords(arduino[0])
            if x is not None and y is not None and z is not None:
                z *= RESCALE_FACTOR
                arduino[1].send_coordinates(
                    x, y, z, frame.shape[:2][::-1], RESCALE_FACTOR
                )
            else:
                arduino[1].send_coordinates(
                    frame.shape[0] // 2,
                    frame.shape[1] // 2,
                    frame.shape[1] // 2,
                    frame.shape[:2],
                    1,
                )

        # TODO: Add a way to select a person by clicking on them
        io_manager.step(frame)
