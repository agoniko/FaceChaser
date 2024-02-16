from imutils.video import WebcamVideoStream
import cv2
from datetime import datetime
import numpy as np
from AI_engine import Engine
from helper_functions import display_results
from Arduino import Arduino


def get_mouse_coords(event, x, y, flags, param):
    global selected_person, persons
    if event == cv2.EVENT_LBUTTONUP:
        print(f"Mouse coords: {x}, {y}")
        for p in persons:
            if p.bbox[0] < x < p.bbox[2] and p.bbox[1] < y < p.bbox[3]:
                selected_person = p
                break


# Cam res: 1920, 1080
IMG_SIZE = (720, 405)
distance_threshold = 0.5

if __name__ == "__main__":
    engine = Engine()
    # arduino = Arduino(IMG_SIZE)

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

    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Frame", get_mouse_coords)
    while True:
        # grab the frame from the threaded video stream and resize it
        frame = vs.read()
        frame = cv2.flip(frame, 1)
        img_rgb = cv2.resize(frame, dsize=IMG_SIZE)
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)

        persons = engine.process_frame(img_rgb, persons)

        if len(persons) != prev_num_person or tracked_person is None:
            tracked_person = None
            best = -np.inf
            idx = -1
            # match based on emb sims
            for i, p in enumerate(persons):
                if p.similarity > best and p.similarity > distance_threshold:
                    best = p.similarity
                    idx = i

            if idx != -1:
                tracked_person = persons[idx]

        prev_num_person = len(persons)

        if selected_person not in persons:
            selected_person = None

        if len(persons) > 0:
            if selected_person is None:
                selected_person = persons[0]
            # arduino.send_coordinates(*selected_person.pos)
            display_results(frame, persons, IMG_SIZE, tracked_person, selected_person)
        else:
            #
            selected_person = None

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

        cv2.putText(
            frame,
            f"Persons: {len(persons)}",
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 255, 0),
            1,
        )

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        if selected_person is not None:
            if key == ord("s"):
                tracked_person = selected_person
                engine.set_target(tracked_person)

            selected_person.color = (255, 0, 0)
            if key == ord("a"):
                idx = persons.index(selected_person)
                selected_person = persons[idx - 1 if idx > 0 else idx]
            if key == ord("d"):
                idx = persons.index(selected_person)
                selected_person = persons[idx + 1 if idx < len(persons) - 1 else idx]

        if key == ord(" "):
            engine.unset_target()
            tracked_person = None
            persons = []
            continue

        # update the FPS counter
        num_frames += 1
        if (datetime.now() - start).total_seconds() > 1:
            start = datetime.now()
            fps = num_frames
            num_frames = 0

    cv2.destroyAllWindows()
    vs.stop()
    # ser.close()
