import time
from threading import Thread
from typing import Dict, Callable

import numpy as np
import cv2

import src.calib.draw as draw

class IOManager:
    def __init__(
        self,
        src=0,
        name="Frame",
        key_callback_dict: Dict[str, Callable] = dict(),
    ):
        self._stream = cv2.VideoCapture(src)
        if not self._stream.isOpened():
            print("Cannot open camera")
            exit()

        width = int(self._stream.get(3))
        height = int(self._stream.get(4))

        self.name = name
        self._stopped = False

        self.key_callback_dict = key_callback_dict

        # fps related variables
        self.t1 = time.time()
        self.t2 = time.time()
        self.fps = 0
        self.frame_counter = 0

        # Capture first frame
        self.starting_frame = cv2.resize(cv2.imread("images/logo.png"), (width, height))
        self.frame = self.starting_frame.copy()
        self.show_frame = self.starting_frame.copy()
        # delay to show the logo
        self.delay_in_seconds = 6
        self.starting_time = time.time()

    def ready(self):
        return (
            self.frame != self.starting_frame
        ).any() and time.time() - self.starting_time > self.delay_in_seconds

    def start(self):
        camera_thread = Thread(
            target=self.capture, name="IOManagerVideoStream", args=()
        )
        camera_thread.daemon = True
        camera_thread.start()

        return self

    def step(self, image):
        if self._stopped:
            return
        self.show_frame = self._overlay_fps(image)

        if self.ready():
            calib_top_view, calib_side_view = draw.get_calib_views(
                (self.show_frame.shape[1], 200), # top view
                (200, self.show_frame.shape[0]), # side view
            )
            logo = cv2.resize(self.starting_frame, (200, 200))
            side_panel = np.vstack((calib_side_view, logo))
            self.show_frame = np.vstack((self.show_frame, calib_top_view))
            self.show_frame = np.hstack((self.show_frame, side_panel))
            cv2.imshow(self.name, self.show_frame)
        else:
            cv2.imshow(self.name, self.starting_frame)
        key = cv2.waitKey(1) & 0xFF
        for k, fun in self.key_callback_dict.items():
            if key == k:
                if chr(k).isdigit():
                    fun(chr(k))
                else:
                    fun()

    def capture(self):
        while True:
            if self._stopped:
                return
            (grabbed, self.frame) = self._stream.read()
            if not grabbed:
                print("Can't receive frame (stream end?). Exiting ...")
                self.stop()

    def get_frame(self):
        # Measure fps
        self.frame_counter += 1
        self.t2 = time.time()
        if self.t2 - self.t1 >= 1.0:
            self.fps = self.frame_counter
            self.frame_counter = 0
            self.t1 = self.t2
        return cv2.flip(self.frame, 1)

    def _overlay_fps(self, image):
        fps_str = f"FPS: {self.fps}"
        cv2.putText(
            image,
            fps_str,
            org=(10, 50),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=2,
            color=(0, 255, 0),
            thickness=1,
        )
        return image

    def is_running(self):
        return not self._stopped

    def stop(self):
        self._stopped = True
        cv2.destroyAllWindows()
        self._stream.release()
