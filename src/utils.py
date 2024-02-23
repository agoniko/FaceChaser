import numpy as np
import cv2
from src.person import person
import time
from typing import Dict

def display_results(
    frame: np.ndarray,
    bboxes: np.ndarray,
    similarities: np.ndarray,
    scaling_factor: float,
    tracked_persons: Dict[str, Person],
    selected_person: Person = None,
):
    
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
        for j, key in enumerate(tracked_persons.keys()):
                text.append(f"{key}={similarities[j, i]:.2f}")

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
            (int(bbox[0]) + 2, int(bbox[1]) + 2),
            (int(bbox[2]) - 2, int(bbox[3]) - 2),
            (255, 0, 0),
            2,
        )

    for person in tracked_persons.values():
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
