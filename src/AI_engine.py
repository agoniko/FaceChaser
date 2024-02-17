from typing import Tuple, Dict, List
import sys

sys.path.append("src")

from Person import Person
import numpy as np
import cv2
from facenet_pytorch import InceptionResnetV1
from tqdm import tqdm
import copy
import utils
from torchvision.transforms.v2 import functional as F

# inside retinaFace implementation, changed device management to be aligned with the rest of the code
# (you have to pass the string name to the constructor)
import torch
from torch import Tensor
import torch.nn.functional as F
from batch_face import RetinaFace
from torchvision.transforms import Resize, Lambda, Compose

emb_transform = Compose(
    [
        Lambda(lambda x: torch.from_numpy(x).float()),
        Lambda(lambda x: (x - 127.5) / 128.0),
        Lambda(lambda x: x.permute(0, 3, 1, 2)),
    ]
)


@utils.timethis
def retina_to_cv2_box(boxes):
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        w, h = xmax - xmin, ymax - ymin
        box[0], box[1], box[2], box[3] = xmin, ymin, w, h
    return np.array(boxes)


@utils.timethis
def retina_to_cv2_keypoints(keypoints):
    # we have to swap keypoints 0 and 1 and 3 and 4
    for keypoint in keypoints:
        keypoint[0], keypoint[1] = keypoint[1], keypoint[0]
        keypoint[3], keypoint[4] = keypoint[4], keypoint[3]
    return np.array(keypoints)


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class Engine(metaclass=Singleton):
    def __init__(
        self,
        device: str = "mps",
        rescale_factor: float = 1.0,
        similarity_threshold: float = 0.6,
        max_tracked_persons: int = 10,
    ):
        self.device = torch.device(device)
        self.detector = RetinaFace(device, network="mobilenet")
        self.sface = cv2.FaceRecognizerSF_create(
            "src/model/face_recognizer_fast.onnx", ""
        )
        self.num_faces = 0
        self.track_with_embeddings = False
        self.rescale_factor = rescale_factor
        self.similarity_threshold = similarity_threshold
        self.max_tracked_persons = max_tracked_persons
        self.tracked_persons = dict()
        self.selected_person = None
        self.random_selection = False

    @utils.timethis
    def _detect_faces(
        self, img_rgb: np.ndarray, threshold: float = 0.7
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        with torch.no_grad():
            pred = self.detector(img_rgb)
            # if the confidence of the prediction is less than 0.7, the prediction is discarded
            bboxes = np.array(
                [p[0] for p in pred if p[2] > threshold], dtype=np.float32
            )

            keypoints = np.array(
                [p[1] for p in pred if p[2] > threshold], dtype=np.float32
            )

            scores = np.array(
                [p[2] for p in pred if p[2] > threshold], dtype=np.float32
            )

            return bboxes, keypoints, scores

    def _create_faces(self, bboxes, keypoints, scores):
        assert len(bboxes) == len(keypoints) == len(scores)
        bboxes = retina_to_cv2_box(bboxes)
        keypoints = retina_to_cv2_keypoints(keypoints)
        bboxes = bboxes.reshape(bboxes.shape[0], -1)
        keypoints = keypoints.reshape(keypoints.shape[0], -1)
        scores = np.array(scores).reshape(-1)

        faces = np.hstack((bboxes, keypoints, scores.reshape(-1, 1)))
        return faces

    @utils.timethis
    def _get_embeddings_SFace(self, img_rgb, bboxes, keypoints, scores) -> np.ndarray:
        faces = self._create_faces(bboxes, keypoints, scores)

        return np.array(
            [
                self.sface.feature(self.sface.alignCrop(img_rgb, face))[0]
                for face in faces
            ]
        )

    def process_frame(self, image: np.ndarray) -> np.ndarray:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(
            img_rgb,
            (
                int(img_rgb.shape[1] * self.rescale_factor),
                int(img_rgb.shape[0] * self.rescale_factor),
            ),
        )

        pred_bboxes, pred_keypoints, pred_scores = self._detect_faces(img_rgb)

        # if the number of people is changed from the last frame
        # or if a tracked person has None as bbox (out of frame) we rely on embeddings
        if len(pred_bboxes) != self.num_faces or any(
            [person.bbox is None for person in self.tracked_persons.values()]
        ):
            self.num_faces = len(pred_bboxes)
            self.track_with_embeddings = True
        else:
            self.track_with_embeddings = False

        if len(pred_bboxes) == 0:
            return image

        # embs, sims = self._get_embeddings(img_rgb, pred_keypoints)
        embeddings = self._get_embeddings_SFace(
            img_rgb,
            copy.deepcopy(pred_bboxes),
            copy.deepcopy(pred_keypoints),
            copy.deepcopy(pred_scores),
        )
        similarities = np.array(
            [person.compare(embeddings) for person in self.tracked_persons.values()]
        )
        # matches the persons of the last frame with the persons of the current frame prediction
        self._match_tracked_persons(
            self.tracked_persons.values(), pred_bboxes, embeddings, similarities
        )

        if self.random_selection:
            idx = np.random.choice(range(len(pred_bboxes)))
            self.selected_person = Person(embeddings[idx], pred_bboxes[idx])
            self.random_selection = False

        if self.selected_person is not None:
            sims = np.array([self.selected_person.compare(embeddings)])
            self._match_tracked_persons(
                [self.selected_person], pred_bboxes, embeddings, sims
            )
            if self.selected_person.bbox is None:
                self.selected_person = None

        utils.display_results(
            image,
            pred_bboxes,
            similarities,
            1 / self.rescale_factor,
            self.tracked_persons,
            self.selected_person,
        )
        return image

    def _match_tracked_persons(
        self,
        tracked_persons: List[Person],
        pred_bboxes: np.ndarray,
        embeddings: np.ndarray,
        similarities: np.ndarray,
    ) -> None:
        if self.track_with_embeddings:
            for i, person in enumerate(tracked_persons):
                idx = np.argmax(similarities[i])
                best_value = similarities[i][idx]
                if best_value > self.similarity_threshold:
                    person.update(embeddings[idx], pred_bboxes[idx])
                    #embeddings = np.delete(embeddings, idx, axis=0)
                    #pred_bboxes = np.delete(pred_bboxes, idx, axis=0)
                    #similarities = np.delete(similarities, idx, axis=1)
                else:
                    person.bbox = None  # person is not in the frame
        else:
            for person in tracked_persons:
                dist = np.linalg.norm(pred_bboxes - person.bbox[None], axis=1)
                idx = np.argmin(dist)
                person.update(embeddings[idx], pred_bboxes[idx])
                #pred_bboxes = np.delete(pred_bboxes, idx, axis=0)
                #embeddings = np.delete(embeddings, idx, axis=0)

    def select_random(self) -> None:
        self.random_selection = True

    def set_target(self, slot_key: str) -> None:
        if (
            slot_key in self.tracked_persons.keys()
            or len(self.tracked_persons) < self.max_tracked_persons
        ):
            self.tracked_persons[slot_key] = self.selected_person.copy()
        else:
            raise ValueError("You have reached the maximum number of tracked persons")
