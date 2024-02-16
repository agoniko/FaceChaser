from Person import Person
import numpy as np
import cv2
from matlab_cp2tform import get_similarity_transform_for_cv2
from facenet_pytorch import InceptionResnetV1
from tqdm import tqdm
import os
import sys
import copy

sys.path.append("sphereface_pytorch")
# inside retinaFace implementation, changed device management to be aligned with the rest of the code
# (you have to pass the string name to the constructor)
import torch
import torch.nn.functional as F
from batch_face import RetinaFace
from net_sphere import sphere20a
from torchvision.transforms import Resize, Lambda, Compose

emb_transform = Compose(
    [
        Lambda(lambda x: torch.from_numpy(x).float()),
        Lambda(lambda x: (x - 127.5) / 128.0),
        Lambda(lambda x: x.permute(0, 3, 1, 2)),
    ]
)

MAX_PERSONS = 10


def retina_to_cv2_box(boxes):
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        w, h = xmax - xmin, ymax - ymin
        box[0], box[1], box[2], box[3] = xmin, ymin, w, h
    return np.array(boxes)


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
    def __init__(self, device: str = "mps"):
        self.device = torch.device(device)
        self.detector = RetinaFace(device, network="mobilenet")
        self.emb_model = sphere20a(feature=True)
        self.emb_model.load_state_dict(torch.load("model/sphere20a_20171020.pth"))
        self.emb_model = InceptionResnetV1(pretrained="vggface2").eval()
        self.emb_model = self.emb_model.to(device)

        self.sface = cv2.FaceRecognizerSF_create(
            "./model/face_recognizer_fast.onnx", ""
        )
        # self.emb_model.alignCrop()
        # self.emb_model.eval()
        self.target = None
        # self._warmup()

    def _warmup(self):
        sample_tensor = np.zeros((720, 400, 3))
        self.detector(sample_tensor)
        sample_tensor = np.zeros((1, 112, 96, 3))
        self.emb_model(emb_transform(sample_tensor).to(self.device))

    def _detect_faces(self, img_rgb: np.ndarray, threshold: float = 0.7) -> list:
        with torch.no_grad():
            pred = self.detector(img_rgb)
            # if the confidence of the prediction is less than 0.7, the prediction is discarded
            return (
                [p[0] for p in pred if p[2] > threshold],
                [p[1] for p in pred if p[2] > threshold],
                [p[2] for p in pred if p[2] > threshold],
            )

    def _get_embeddings(self, img_rgb, keypoints: list) -> np.ndarray:
        aligned = np.array([self._alignment(img_rgb, k)[0] for k in keypoints])
        num_persons = len(aligned)
        # This placeholders increase (A lot) the inference time.
        # With a variable batch size (number of persons in the frame), when adding e.g. 5 person in the frame the computation
        # increases and the video streams stops for a while. Whith placeholders the overall computation may be slower but the
        # video stream is not interrupted and the computation is more stable.
        placeholders = np.zeros((MAX_PERSONS, 112, 96, 3))
        placeholders[:num_persons] = aligned
        with torch.no_grad():
            embs = self.emb_model(emb_transform(placeholders).to(self.device))
            embs = embs[:num_persons]
            # embs = torch.tensor(embs)
            if self.target is not None:
                sims = F.cosine_similarity(embs, self.target, dim=-1)
                #sims = torch.cdist(embs, self.target.unsqueeze(0), p=2).reshape(-1)
            else:
                sims = -torch.ones(num_persons).to(self.device)

            return embs, sims

    def _SFace(self, img_rgb, bboxes, keypoints, scores):
        assert len(bboxes) == len(keypoints) == len(scores)
        bboxes = retina_to_cv2_box(bboxes)
        keypoints = retina_to_cv2_keypoints(keypoints)
        bboxes = bboxes.reshape(bboxes.shape[0], -1)
        keypoints = keypoints.reshape(keypoints.shape[0], -1)
        scores = np.array(scores).reshape(-1)

        faces = np.hstack((bboxes, keypoints, scores.reshape(-1, 1)))
        embs = []

        for face in faces:
            aligned_face = self.sface.alignCrop(img_rgb, face)
            embs.append(self.sface.feature(aligned_face))

        embs = torch.tensor(embs).to(self.device)
        if self.target is not None:
            sims = F.cosine_similarity(embs, self.target, dim=-1).reshape(-1)
        else:
            sims = -torch.ones(embs.shape[0])

        return embs, sims.cpu().numpy()

    def process_frame(self, img_rgb, persons: list[Person]):
        pred_bboxes, pred_keypoints, pred_scores = self._detect_faces(
            img_rgb
        )  # returns list
        if len(pred_bboxes) == 0:
            return []

        #embs, sims = self._get_embeddings(img_rgb, pred_keypoints)
        embs, sims = self._SFace(
            img_rgb,
            copy.deepcopy(pred_bboxes),
            copy.deepcopy(pred_keypoints),
            copy.deepcopy(pred_scores),
        )

        # matches the persons of the last frame with the persons of the current frame prediction
        self._match_and_update(persons, pred_bboxes, embs, sims)
        # sort the persons by their x coordinate
        persons.sort()

        return persons

    # computes the embedding of the target person and stores it in the target attribute
    def set_target(self, person: Person) -> None:
        self.target = person.embedding

    def unset_target(self) -> None:
        self.target = None

    def _match_and_update(
        self,
        persons: list[Person],
        boxes: list[tuple[int, int, int, int]],
        embs: np.ndarray,
        sims: np.ndarray,
    ) -> None:
        assert len(boxes) == len(
            embs
        ), "The number of bboxes and embeddings must be the same"
        taken = []
        if len(persons) == 0:
            for b, e, s in zip(boxes, embs, sims):
                persons.append(Person(b, e, s))
            return

        for b, e, s in zip(boxes, embs, sims):
            min_dist = np.inf
            idx = None
            for j, person in enumerate(persons):
                if len(taken) > 0 and j in taken:
                    continue
                dist = np.linalg.norm(person.bbox - b)
                if dist < min_dist:
                    min_dist = dist
                    idx = j

            if idx is not None:
                taken.append(idx)
                persons[idx].update(b, e, s)
            else:
                persons.append(Person(b, e, s))
                taken.append(len(persons) - 1)

        to_remove = [persons[idx] for idx in range(len(persons)) if idx not in taken]
        for p in to_remove:
            persons.remove(p)

    def _alignment(self, src_img, src_pts):
        ref_pts = [
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041],
        ]
        crop_size = (96, 112)
        src_pts = np.array(src_pts).reshape(5, 2)

        s = np.array(src_pts).astype(np.float32)
        r = np.array(ref_pts).astype(np.float32)

        tfm = get_similarity_transform_for_cv2(s, r)
        face_img = cv2.warpAffine(src_img, tfm, crop_size)
        return np.array([face_img])

    def _alignment_mine(self, src_img, src_pts):
        pass
