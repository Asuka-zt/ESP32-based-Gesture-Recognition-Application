from dataclasses import asdict
from threading import Event, Lock, Thread
from time import monotonic

import cv2

from gesture_mouse.core.types import GesturePrediction
from gesture_mouse.services.classifier import GestureClassifier, PredictionStabilizer
from gesture_mouse.services.frame_buffer import LatestFrameBuffer
from gesture_mouse.services.hand_detector import HandDetector, HandObservation


class VisionService:
    def __init__(
        self,
        source_frames: LatestFrameBuffer,
        output_frames: LatestFrameBuffer,
        hand_detector: HandDetector,
        classifier: GestureClassifier,
    ) -> None:
        self._source_frames = source_frames
        self._output_frames = output_frames
        self._hand_detector = hand_detector
        self._classifier = classifier
        self._stabilizer = PredictionStabilizer()
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._lock = Lock()
        self._prediction = GesturePrediction(None, None, 0.0, {}, False, 0.0, 0)
        self._observation: HandObservation | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._classifier.load()
        self._stop_event.clear()
        self._thread = Thread(target=self._run, name="gesture-vision", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def prediction(self) -> GesturePrediction:
        with self._lock:
            return self._prediction

    def observation(self) -> HandObservation | None:
        with self._lock:
            return self._observation

    def status(self) -> dict[str, object]:
        prediction = asdict(self.prediction())
        return {
            "model_ready": self._classifier.ready,
            "model_error": self._classifier.error,
            "prediction": prediction,
        }

    def _set_prediction(self, prediction: GesturePrediction) -> None:
        with self._lock:
            self._prediction = prediction

    def _set_observation(self, observation: HandObservation | None) -> None:
        with self._lock:
            self._observation = observation

    def _run(self) -> None:
        source_sequence = 0
        while not self._stop_event.is_set():
            packet = self._source_frames.wait_for_new(source_sequence, timeout=0.5)
            if packet is None:
                continue
            source_sequence = packet.sequence
            started_at = monotonic()
            annotated = packet.image.copy()
            observation = self._hand_detector.detect(packet.image)
            self._set_observation(observation)

            if observation is None:
                self._stabilizer.reset()
                prediction = GesturePrediction(
                    raw_label=None,
                    stable_label=None,
                    confidence=0.0,
                    probabilities={},
                    hand_detected=False,
                    inference_ms=(monotonic() - started_at) * 1000,
                    frame_sequence=packet.sequence,
                )
                cv2.putText(
                    annotated,
                    "No hand",
                    (20, 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 80, 255),
                    2,
                )
            else:
                prediction = self._predict(observation, packet.sequence, started_at)
                self._draw_observation(annotated, observation, prediction)

            self._set_prediction(prediction)
            self._output_frames.publish(annotated)

    def _predict(
        self, observation: HandObservation, frame_sequence: int, started_at: float
    ) -> GesturePrediction:
        raw_label: str | None = None
        confidence = 0.0
        probabilities: dict[str, float] = {}
        if self._classifier.ready:
            raw_label, confidence, probabilities = self._classifier.predict(observation)
        stable_label = self._stabilizer.update(raw_label)
        return GesturePrediction(
            raw_label=raw_label,
            stable_label=stable_label,
            confidence=confidence,
            probabilities=probabilities,
            hand_detected=True,
            inference_ms=(monotonic() - started_at) * 1000,
            frame_sequence=frame_sequence,
        )

    @staticmethod
    def _draw_observation(
        image: object, observation: HandObservation, prediction: GesturePrediction
    ) -> None:
        x_min, y_min, x_max, y_max = observation.bounding_box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (70, 220, 120), 2)
        height, width = image.shape[:2]  # type: ignore[attr-defined]
        for x, y, _ in observation.landmarks:
            cv2.circle(image, (int(x * width), int(y * height)), 2, (255, 190, 50), -1)
        label = prediction.stable_label or prediction.raw_label or "unknown"
        text = f"{label} {prediction.confidence:.0%}"
        cv2.putText(
            image,
            text,
            (x_min, max(24, y_min - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (70, 220, 120),
            2,
        )
