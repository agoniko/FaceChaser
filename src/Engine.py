import copy
from typing import Tuple, List


from batch_face import RetinaFace
from batch_face.face_detection.alignment import load_net
import cv2
import numpy as np
from skimage import transform
import torch
from torchvision.transforms import Lambda, Compose

from src.net_sphere import sphere20a
from src.Person import Person
from src.refsys.system import ReferenceSystem
from src.refsys.vector import Vector
import src.utils as utils
from src.timethis import timethis

def override_init_RetinaFace(
    self,
    device="cpu",
    model_path=None,
    network="mobilenet",
):
    """
    Override the __init__ method of RetinaFace to allow for the device to be passed as a string,
    This allows to use mps, cuda, cpu, etc. as device names
    """
    self.device = torch.device(device)
    self.model = load_net(model_path, self.device, network)


RetinaFace.__init__ = override_init_RetinaFace


emb_transform = Compose(
    [
        Lambda(lambda x: torch.from_numpy(x).float()),
        Lambda(lambda x: (x - 127.5) / 128.0),
        Lambda(lambda x: x.permute(0, 3, 1, 2)),
    ]
)

def create_faces(bboxes, keypoints, scores):
    """
    Converts the output of the retinaface detector to a format compatible with SFace from cv2
    """
    assert len(bboxes) == len(keypoints) == len(scores)
    faces = np.hstack((bboxes, keypoints, scores.reshape(-1, 1)))
    return faces


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

#For fixed batch size
MAX_PERSONS = 10


class Engine(metaclass=Singleton):
    """
    This class is the main class of the application. It is responsible for the following:
    - Detecting and tracking faces in the frame
    - computing embeddings for each detected face
    - Selecting bounding boxes to track
    - Setting and unsetting selected boxes as targets
    - Getting the coordinates of a tracked person
    """

    def __init__(
        self,
        reference_system: ReferenceSystem,
        device: str = "mps",
        rescale_factor: float = 1.0,
        similarity_threshold: float = 0.6,
        max_tracked_persons: int = 10,
    ):
        """
        args:
        - reference_frame
        - device: str, default="mps"
            The device to use for the detector and the embeddings generator
        - rescale_factor: float, default=1.0
            The factor by which to rescale the input image
        - similarity_threshold: float, default=0.6
            The threshold for the cosine similarity between embeddings
        - max_tracked_persons: int, default=10
            The maximum number of persons to track, this param is also used for a fixed batch size for the embeddings generator
        """
        self.reference_system = reference_system
        self.device = torch.device(device)
        if self.device == torch.device("cpu"):
            self.embedding_generator = cv2.FaceRecognizerSF_create(
                "src/model/face_recognizer_fast.onnx", ""
            )
            self.detector = cv2.FaceDetectorYN_create(
                "src/model/face_detection_yunet_2023mar_int8.onnx", "", (0, 0)
            )
        else:
            self.detector = RetinaFace(device, network="mobilenet")
            self.embedding_generator = sphere20a(feature=True)
            self.embedding_generator.load_state_dict(
                torch.load("./src/model/sphere20a_20171020.pth")
            )
            self.embedding_generator.to(self.device)
            self.embedding_generator.eval()

        self.num_faces = 0
        self.rescale_factor = rescale_factor
        self.similarity_threshold = similarity_threshold
        self.max_tracked_persons = max_tracked_persons
        self.tracked_persons = dict()
        self.selected_person = None

        # Flag for tracking with embeddings or not (heuristic)
        self.track_with_embeddings = False
        # Flags for selecting a person
        self.random_selection = False
        self.right = False
        self.left = False
        self.up = False
        self.down = False

    def _detect_faces_cpu(
        self, img_rgb: np.ndarray, threshold: float = 0.7
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        width, height = img_rgb.shape[1], img_rgb.shape[0]
        self.detector.setInputSize((width, height))
        self.detector.setScoreThreshold(threshold)

        _, faces = self.detector.detect(img_rgb)
        if faces is None:
            return np.array([]), np.array([]), np.array([])

        bboxes = np.array(
            [face[:4] for face in faces if face[-1] > threshold], dtype=np.float32
        )
        keypoints = np.array(
            [face[4 : len(face) - 1] for face in faces if face[-1] > threshold],
            dtype=np.float32,
        )
        scores = np.array(
            [face[-1] for face in faces if face[-1] > threshold], dtype=np.float32
        )

        # converting bboxes from [xmin, ymin, w, h] to [xmin, ymin, xmax, ymax]
        bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
        bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]

        return bboxes, keypoints, scores

    def _detect_faces_gpu(
        self, img_rgb: np.ndarray, threshold: float = 0.7
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detects faces in the input image and returns the bounding boxes, keypoints and scores
        It thresholds the predictions based on the confidence score of the detector
        """
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
    
    @timethis
    def _detect_faces(
        self, img_rgb: np.ndarray, threshold: float = 0.7
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Selects the appropriate method for detecting faces based on the device
        """
        if self.device == torch.device("cpu"):
            return self._detect_faces_cpu(img_rgb, threshold)
        else:
            return self._detect_faces_gpu(img_rgb, threshold)

    def _get_embeddings_gpu(
        self,
        img_rgb: np.ndarray,
        bboxes: np.ndarray,
        keypoints: np.ndarray,
        scores: np.ndarray,
    ) -> np.ndarray:
        """
        Aligns the faces and computes the embeddings for each face using the sphereface model on gpu
        """
        assert len(bboxes) == len(keypoints) == len(scores)

        faces = []
        for keypoint in keypoints:
            face = self._alignment(
                img_rgb,
                keypoint,
            )
            faces.append(face)
        faces = emb_transform(np.array(faces))
        # Dynamic batch causes instability, so we use a fixed batch size with placeholders
        placeholders = torch.zeros(MAX_PERSONS, 3, 112, 96)
        placeholders[: len(faces)] = faces
        placeholders = placeholders.to(self.device)
        with torch.no_grad():
            embeddings = self.embedding_generator(placeholders)[: len(faces)]
        return embeddings.cpu().numpy()

    def _get_embeddings_cpu(
        self,
        img_rgb: np.ndarray,
        bboxes: np.ndarray,
        keypoints: np.ndarray,
        scores: np.ndarray,
    ) -> np.ndarray:
        """
        Uses the SFace model from cv2 to align each face and compute the embeddings.
        This model is used only for cpu inference.
        """
        faces = create_faces(bboxes, keypoints, scores)
        return np.array(
            [
                self.embedding_generator.feature(
                    self.embedding_generator.alignCrop(img_rgb, face)
                )[0]
                for face in faces
            ]
        )

    @timethis
    def _get_embeddings(
        self,
        img_rgb: np.ndarray,
        bboxes: np.ndarray,
        keypoints: np.ndarray,
        scores: np.ndarray,
    ) -> np.ndarray:
        """
        Selects the appropriate method for computing the embeddings based on the device
        """
        if self.device == torch.device("cpu"):
            # deepcopy is used because order of coordinates is switched for the alignment function of SFace
            return self._get_embeddings_cpu(
                img_rgb,
                bboxes,
                keypoints,
                scores,
            )
        else:
            return self._get_embeddings_gpu(img_rgb, bboxes, keypoints, scores)

    def must_use_embeddings(
        self,
        pred_bboxes: np.ndarray,
        pred_keypoints: np.ndarray,
        pred_scores: np.ndarray,
    ) -> bool:
        """Heuristics for deciding when to use embeddings for tracking"""
        # if the number of people is changed from the last frame
        # or if a tracked person has None as bbox (out of frame) we rely on embeddings
        return len(pred_bboxes) != self.num_faces or any(
            [person.bbox is None for person in self.tracked_persons.values()]
        )

    def process_frame(self, image: np.ndarray) -> np.ndarray:
        """
        This method is the main method of the class. It processes the input image and returns the image overlayed with the
        bounding boxes color coded for the tracked persons and selected person.
        It also overlay similarity scores with the tracked persons.
        """

        # Rescale the image for model inference, results are rescaled on the original size
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(
            img_rgb,
            (
                int(img_rgb.shape[1] * self.rescale_factor),
                int(img_rgb.shape[0] * self.rescale_factor),
            ),
        )
        # getting prediction from face detector (RetinaFace)
        pred_bboxes, pred_keypoints, pred_scores = self._detect_faces(img_rgb)

        # choosing to track targets by embeddings or not based on the heuristic
        if self.must_use_embeddings(pred_bboxes, pred_keypoints, pred_scores):
            self.track_with_embeddings = True
        else:
            self.track_with_embeddings = False

        self.num_faces = len(pred_bboxes)
        
        # No person detected
        if len(pred_bboxes) == 0:
            # Handle tracked_persons
            for person in self.tracked_persons.values():
                person.bbox = None

            # Handle selection
            self.random_selection = False
            self.left, self.right = False, False
            self.up, self.down = False, False

            # Handle selected_person
            self.selected_person = None
        else:
            embeddings = self._get_embeddings(
                img_rgb, pred_bboxes, pred_keypoints, pred_scores
            )

            # Randomly select a person to track, a must if in the previous frame no person is selected
            if self.random_selection:
                idx = np.random.choice(range(len(pred_bboxes)))
                self.selected_person = Person(embeddings[idx], pred_bboxes[idx])
                self.random_selection = False

            # updates the tracked persons and the selected person bounding boxes and embeddings
            similarities = self.match_tracked_selected_persons(pred_bboxes, embeddings)

            # if the user wants to change the selected person, the selected person is updated
            self._change_selected_person(pred_bboxes, embeddings)

            # overlays the image with the bounding boxes and the similarity scores
            utils.display_results(
                image,
                pred_bboxes,
                similarities,
                1 / self.rescale_factor,
                self.tracked_persons,
                self.selected_person,
            )
        return image

    def _change_selected_person(self, pred_bboxes, embeddings):
        if self.selected_person is None:
            return
        # sorting bboxes by xmin that is in [:, 0] position
        # if self.right we select the bounding box immediately after self.selected person.bbox[0]
        # if self.left we select the bounding box immediately before self.selected person.bbox[0]
        if self.right:
            pred_bboxes = pred_bboxes[pred_bboxes[:, 0].argsort()]
            idx = np.where(pred_bboxes[:, 0] > self.selected_person.bbox[0])[0]
            self.right = False
            if len(idx) > 0:
                idx = idx[0]
                self.selected_person = Person(embeddings[idx], pred_bboxes[idx])
        elif self.left:
            pred_bboxes = pred_bboxes[pred_bboxes[:, 0].argsort()]
            idx = np.where(pred_bboxes[:, 0] < self.selected_person.bbox[0])[0]
            self.left = False
            if len(idx) > 0:
                idx = idx[-1]
                self.selected_person = Person(embeddings[idx], pred_bboxes[idx])
        # same as before but for up and down
        elif self.up:
            # sorting by ymin
            pred_bboxes = pred_bboxes[pred_bboxes[:, 1].argsort()]
            idx = np.where(pred_bboxes[:, 1] < self.selected_person.bbox[1])[0]
            self.up = False
            if len(idx) > 0:
                idx = idx[-1]
                self.selected_person = Person(embeddings[idx], pred_bboxes[idx])
        elif self.down:
            # sorting by ymin
            pred_bboxes = pred_bboxes[pred_bboxes[:, 1].argsort()]
            idx = np.where(pred_bboxes[:, 1] > self.selected_person.bbox[1])[0]
            self.down = False
            if len(idx) > 0:
                idx = idx[0]
                self.selected_person = Person(embeddings[idx], pred_bboxes[idx])

    def match_tracked_selected_persons(
        self,
        pred_bboxes: np.ndarray,
        embeddings: np.ndarray,
    ) -> None:
        """
        This method matches the tracked persons and the selected person with the persons detected in the current frame
        """
        similarities = np.array(
            [person.compare(embeddings) for person in self.tracked_persons.values()]
        )

        # Match tracked persons with persons detected in current frame, #updates embeddings and bouding boxes
        self._match(
            self.tracked_persons.values(), pred_bboxes, embeddings, similarities
        )

        if self.selected_person is not None:
            # Track selected person
            sims = np.array([self.selected_person.compare(embeddings)])
            self._match([self.selected_person], pred_bboxes, embeddings, sims)
            # Is selected person out of frame?
            if self.selected_person.bbox is None:
                self.selected_person = None

        return similarities

    def _match(
        self,
        tracked_persons: List[Person],
        pred_bboxes: np.ndarray,
        embeddings: np.ndarray,
        similarities: np.ndarray,
    ) -> None:
        """
        Given a list of Person objects, it updates the embeddings and the bounding boxes of the persons in the list,
        This match is mainly done computing the distance between the bounding boxes of the tracked persons
        and the detected persons in the current frame. If the heuristic says that this match is uncertain,
        the similarity between the embeddings is used to update the tracked persons.
        """
        if self.track_with_embeddings:
            for i, person in enumerate(tracked_persons):
                idx = np.argmax(similarities[i])
                best_value = similarities[i][idx]
                if best_value > self.similarity_threshold:
                    person.update(embeddings[idx], pred_bboxes[idx])
                else:
                    person.bbox = None  # person is not in the frame
        else:
            for i, person in enumerate(tracked_persons):
                dist = np.linalg.norm(pred_bboxes - person.bbox[None], axis=1)
                idx = np.argmin(dist)
                person.update(embeddings[idx], pred_bboxes[idx])


    def set_target(self, slot_key: str) -> None:
        """
        Sets the selected person as a target to track in the slot_key position entered by the user
        """
        if (
            slot_key in self.tracked_persons.keys()
            or len(self.tracked_persons) < self.max_tracked_persons
        ):
            if self.selected_person is not None:
                self.tracked_persons[slot_key] = self.selected_person.copy()
        else:
            raise ValueError("You have reached the maximum number of tracked persons")

    def unset_targets(self) -> None:
        self.tracked_persons = dict()

    def get_coords(self, slot_key: str) -> Tuple[float, float, float]:
        """
        returns the center of the bounding box of the person in the slot_key position
        """
        if slot_key in self.tracked_persons.keys():
            coords = self.tracked_persons[slot_key].get_coords()
            if coords is None:
                return None
            coords[0] /= self.rescale_factor
            coords[1] /= self.rescale_factor
            coords[2] /= self.rescale_factor
            
            face_height = 25.
            reference_face_pixel_height = 480.
            reference_depth = 19.

            focal = reference_depth * reference_face_pixel_height / face_height
            z = focal * face_height / coords[2]
            cm_per_pixel = (z / reference_depth) * (face_height / reference_face_pixel_height)
            x_max = 640 * cm_per_pixel # TODO image shape
            y_max = 480 * cm_per_pixel
            x = coords[0] * cm_per_pixel - x_max/2.
            y = coords[1] * cm_per_pixel - y_max/2.

            return Vector(array=np.array([x, y, z]), reference_system=self.reference_system)
        else:
            return None

    # Function to align faces based on facial landmark detection
    def _alignment(self, src_img, src_pts):
        """
        Given an image and the 5 facial landmarks, it aligns the face and returns the aligned face.
        The return size is 96x112, the same size used for training the sphereface model
        """
        # Define reference points for standardized landmark positions
        ref_pts = [
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041],
        ]

        # Estimate similarity transform matrix using scikit-image
        tform = transform.estimate_transform("similarity", src_pts, ref_pts)

        # Convert scikit-image Transform object into OpenCV format
        mtx = tform.params[0:2, :]

        # Apply affine transformation to input image
        dst_shape = (96, 112)  # Output image size
        face_img = cv2.warpAffine(src_img, mtx, dst_shape)

        return face_img

    # Setting flags for selecting a person
    def select_random(self) -> None:
        self.random_selection = True

    def select_right(self):
        self.right = True

    def select_left(self):
        self.left = True

    def select_up(self):
        self.up = True

    def select_down(self):
        self.down = True
