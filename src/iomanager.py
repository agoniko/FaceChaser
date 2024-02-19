import time
from threading import Thread
import cv2
from typing import Dict, Callable

class IOManager:
    def __init__(self, src=0, name="Frame", key_callback_dict: Dict[str, Callable]=dict(), show_fps=True):
        self._stream = cv2.VideoCapture(src)
        if not self._stream.isOpened():
            print("Cannot open camera")
            exit()
            
        self.name = name
        self._stopped = False
        
        self.key_callback_dict = key_callback_dict

        # fps related variables
        self.show_fps = show_fps
        self.t1 = time.time()
        self.t2 = time.time()
        self.fps = 0
        self.frame_counter = 0
        
        # Capture first frame
        self.capture()

    def start(self):
        camera_thread = Thread(target=self.update, name="IOManagerVideoStream", args=())
        camera_thread.daemon = True
        camera_thread.start()

        input_thread = Thread(target=self.read_input, name='KeyboardInputReader', args=())
        input_thread.daemon = True
        input_thread.start()

        return self
    
    def capture(self):
        (grabbed, self.frame) = self._stream.read()
        if not grabbed:
            print("Can't receive frame (stream end?). Exiting ...")
            self.stop()

    def read_input(self):
        while True:
            if self._stopped:
                return
            key = cv2.waitKey(1) & 0xFF
            for k, fun in self.key_callback_dict.items():
                if key == ord(k):
                    fun()

    def update(self):
        while True:
            if self._stopped:
                return
            self.capture()
                
    def read(self):
        # Measure fps
        self.frame_counter += 1
        self.t2 = time.time()
        if self.t2 - self.t1 >= 1.:
            self.fps = self.frame_counter
            self.frame_counter = 0
            self.t1 = self.t2
        return cv2.flip(self.frame, 1)
    
    def show(self, image):
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