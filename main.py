from imutils.video import WebcamVideoStream
import cv2
from datetime import datetime
import numpy as np
from src.Engine import Engine
from src.utils import display_results
from src.Arduino import Arduino


def get_mouse_coords(event, x, y, flags, param):
    global selected_person, persons
    if event == cv2.EVENT_LBUTTONUP:
        print(f"Mouse coords: {x}, {y}")
        for p in persons:
            if p.bbox[0] < x < p.bbox[2] and p.bbox[1] < y < p.bbox[3]:
                selected_person = p
                break


# Cam res: 1920, 1080
MAX_TRACKED_PERSONS = 1
RESCALE_FACTOR = 0.2
distance_threshold = 0.5

if __name__ == "__main__":
    engine = Engine("mps", RESCALE_FACTOR, 0.6, MAX_TRACKED_PERSONS)
    arduinos = {
        "1": Arduino("/dev/cu.usbmodem1101"),
        #"2": Arduino("/dev/cu.usbmodem11301"),
    }

    num_frames = 0
    # created a *threaded* video stream, allow the camera sensor to warmup,
    # and start the FPS counter
    vs = WebcamVideoStream(src=0).start()

    # loop over some frames...this time using the threaded stream
    start = datetime.now()
    fps = 0
    prev_num_person = 0
    persons = []
    selected_person, tracked_person = None, None

    # cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    # cv2.setMouseCallback("Frame", get_mouse_coords)
    while True:
        # grab the frame from the threaded video stream and resize it
        frame = vs.read()
        frame = cv2.flip(frame, 1)

        frame = engine.process_frame(frame)

        for arduino in arduinos.items():
            x, y, z = engine.get_coords(arduino[0])
            if x is not None and y is not None:
                arduino[1].send_coordinates(
                    x, y, frame.shape[:2][::-1], RESCALE_FACTOR
                )
            else:
                arduino[1].send_coordinates(
                    frame.shape[0] // 2,
                    frame.shape[1] // 2,
                    frame.shape[:2],
                    1,
                )

        if num_frames > 0:
            fps_str = f"FPS: {fps}"
            # print(type(fps_str))
            cv2.putText(
                frame,
                fps_str,
                org=(10, 50),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=2,
                color=(0, 255, 0),
                thickness=1,
            )

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        if key == ord("1"):
            engine.set_target("1")
        if key == ord("2"):
            engine.set_target("2")

        if key == ord("r"):
            engine.select_random()

        if key == ord("u"):
            engine.unset_targets()

        # TODO: Add a way to select a person by clicking on them

        # update the FPS counter
        num_frames += 1
        if (datetime.now() - start).total_seconds() > 1:
            start = datetime.now()
            fps = num_frames
            num_frames = 0

    cv2.destroyAllWindows()
    vs.stop()
    # ser.close()
