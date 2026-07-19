from dataclasses import dataclass
from threading import Lock

import cv2
import mediapipe as mp
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class HandObservation:
    landmarks: tuple[tuple[float, float, float], ...]
    bounding_box: tuple[int, int, int, int]
    crop: NDArray[np.uint8]
    handedness: str
    confidence: float


class HandDetector:
    def __init__(
        self,
        *,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
        crop_margin: float = 0.2,
    ) -> None:
        self._crop_margin = crop_margin
        self._lock = Lock()
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def close(self) -> None:
        with self._lock:
            self._hands.close()

    def detect(self, frame: NDArray[np.uint8]) -> HandObservation | None:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a BGR image")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        with self._lock:
            result = self._hands.process(rgb)

        if not result.multi_hand_landmarks:
            return None

        hand_landmarks = result.multi_hand_landmarks[0]
        handedness = "Unknown"
        confidence = 0.0
        if result.multi_handedness:
            classification = result.multi_handedness[0].classification[0]
            handedness = classification.label
            confidence = float(classification.score)

        height, width = frame.shape[:2]
        points = tuple((float(p.x), float(p.y), float(p.z)) for p in hand_landmarks.landmark)
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        x_min = max(0, int(min(x_values) * width))
        y_min = max(0, int(min(y_values) * height))
        x_max = min(width, int(max(x_values) * width) + 1)
        y_max = min(height, int(max(y_values) * height) + 1)

        box_width = max(1, x_max - x_min)
        box_height = max(1, y_max - y_min)
        margin_x = int(box_width * self._crop_margin)
        margin_y = int(box_height * self._crop_margin)
        x_min = max(0, x_min - margin_x)
        y_min = max(0, y_min - margin_y)
        x_max = min(width, x_max + margin_x)
        y_max = min(height, y_max + margin_y)
        crop = frame[y_min:y_max, x_min:x_max].copy()
        if crop.size == 0:
            return None

        return HandObservation(
            landmarks=points,
            bounding_box=(x_min, y_min, x_max, y_max),
            crop=crop,
            handedness=handedness,
            confidence=confidence,
        )

