# from imutils.video import WebcamVideoStream
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
    parser.add_argument("--num_servo", type=int, default=2)
    parser.add_argument("--serial_port", type=str)

    # serial port = /dev/cu.usbmodem21201
    args = parser.parse_args()

    RESCALE_FACTOR = args.rescale
    SIMILARIY_THRESHOLD = args.similarity
    DEVICE = args.device

    if args.serial_port is not None:
        arduino = Arduino(args.serial_port)
    else:
        arduino = None

    MAX_TRACKED_PERSONS = args.num_servo

    engine = Engine(DEVICE, RESCALE_FACTOR, SIMILARIY_THRESHOLD, MAX_TRACKED_PERSONS)

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

        for i in range(MAX_TRACKED_PERSONS):
            if arduino is not None:
                x, y, z = engine.get_coords(str(i+1))
                if x is not None and y is not None and z is not None:
                    z *= RESCALE_FACTOR
                    arduino.send_coordinates(
                        i, x, y, z, frame.shape[:2][::-1], RESCALE_FACTOR
                    )
                else:
                    arduino.send_coordinates(
                        i,
                        frame.shape[0] // 2,
                        frame.shape[1] // 2,
                        frame.shape[1] // 2,
                        frame.shape[:2],
                        1,
                    )

        # TODO: Add a way to select a person by clicking on them
        io_manager.step(frame)
