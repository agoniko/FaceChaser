import numpy as np
import cv2
from Person import Person
import time

class TimingInfoSingleton:
    def __new__(cls):
        """ creates a singleton object, if it is not created, 
        or else returns the previous singleton object"""
        if not hasattr(cls, 'instance'):
            cls.instance = super().__new__(cls)
        return cls.instance
    
    def __init__(self):
        self.info = []

def timethis(func):
    def wrapper(*args, **kwargs):
        t1 = time.time()
        res = func(*args, **kwargs)
        t2 = time.time()
        print(f'{func.__name__} took {(t2 - t1)*1000:.3f} ms')
        return res
    return wrapper


def display_results(
    frame, persons, src_size, tracked_person=None, selected_person=None
):
    dst_size = (frame.shape[1], frame.shape[0])
    for p in persons:
        bbox = p.bbox.copy()
        bbox[0] = bbox[0] * dst_size[0] / src_size[0]
        bbox[1] = bbox[1] * dst_size[1] / src_size[1]
        bbox[2] = bbox[2] * dst_size[0] / src_size[0]
        bbox[3] = bbox[3] * dst_size[1] / src_size[1]

        # applying the bounding box to the frame for each person
        cv2.rectangle(
            frame,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            (0, 0, 255),
            2,
        )
        # if the target is set than the similarity is displayed
        if p.similarity != -1:
            xmin, ymin, _, _ = np.int32(bbox)
            cv2.putText(
                frame,
                f"Sim: {p.similarity:.2f}",
                (xmin, ymin - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0),
                1,
            )
    if selected_person is not None:
        bbox = selected_person.bbox.copy()
        bbox[0] = bbox[0] * dst_size[0] / src_size[0]
        bbox[1] = bbox[1] * dst_size[1] / src_size[1]
        bbox[2] = bbox[2] * dst_size[0] / src_size[0]
        bbox[3] = bbox[3] * dst_size[1] / src_size[1]
        # applying the bounding box to the frame for each person
        cv2.rectangle(
            frame,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            (255, 0, 0),
            2,
        )

    if tracked_person is not None:
        bbox = tracked_person.bbox.copy()
        bbox[0] = bbox[0] * dst_size[0] / src_size[0]
        bbox[1] = bbox[1] * dst_size[1] / src_size[1]
        bbox[2] = bbox[2] * dst_size[0] / src_size[0]
        bbox[3] = bbox[3] * dst_size[1] / src_size[1]
        # applying the bounding box to the frame for each person
        cv2.rectangle(
            frame,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            (0, 255, 0),
            2,
        )


