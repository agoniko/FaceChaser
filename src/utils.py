import numpy as np
import cv2
from Person import Person
import time
from typing import List


def timethis(func):
    def wrapper(*args, **kwargs):
        t1 = time.time()
        res = func(*args, **kwargs)
        t2 = time.time()
        print(f"{func.__name__} took {(t2 - t1)*1000:.3f} ms")
        return res

    return wrapper


def display_results(
    frame: np.ndarray,
    bboxes: np.ndarray,
    similarities: np.ndarray,
    scaling_factor: float,
    tracked_persons: List[Person],
    selected_person: Person = None,
):
    dst_size = (frame.shape[1], frame.shape[0])
    for i, bbox in enumerate(bboxes):
        bbox = bbox * scaling_factor

        # applying the bounding box to the frame for each person
        cv2.rectangle(
            frame,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            (0, 0, 255),
            2,
        )

        text = []
        j = 0
        for c, person in enumerate(tracked_persons):
            if isinstance(person, Person):
                text.append(f"{c+1}={similarities[j, i]:.2f}")
                j += 1 
            else:
                text.append(f"{c+1}=_")
        text = ", ".join(text)
        xmin, ymin, _, _ = np.int32(bbox)
        cv2.putText(
            frame,
            text,
            (xmin, ymin - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            1,
        )

    if selected_person is not None:
        bbox = selected_person.bbox * scaling_factor
        # applying the bounding box to the frame for each person
        cv2.rectangle(
            frame,
            (int(bbox[0])+2, int(bbox[1])+2),
            (int(bbox[2])-2, int(bbox[3])-2),
            (255, 0, 0),
            2,
        )

    for person in tracked_persons:
        if not isinstance(person, Person):
            continue
        # if person is not in frame
        if person.bbox is None:
            continue
        bbox = person.bbox * scaling_factor
        # applying the bounding box to the frame for each person
        cv2.rectangle(
            frame,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            (0, 255, 0),
            2,
        )
