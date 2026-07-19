from dataclasses import asdict

import httpx

from gesture_mouse.config import Settings
from gesture_mouse.services.capture import MjpegCaptureService
from gesture_mouse.services.classifier import GestureClassifier
from gesture_mouse.services.dataset import DatasetService
from gesture_mouse.services.frame_buffer import LatestFrameBuffer
from gesture_mouse.services.hand_detector import HandDetector
from gesture_mouse.services.mouse_backend import MacOSMouseBackend
from gesture_mouse.services.mouse_control import MouseControlService
from gesture_mouse.services.vision import VisionService


class ApplicationRuntime:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.frames = LatestFrameBuffer()
        self.annotated_frames = LatestFrameBuffer()
        self.capture = MjpegCaptureService(
            settings.stream_url,
            self.frames,
            reconnect_initial_seconds=settings.reconnect_initial_seconds,
            reconnect_max_seconds=settings.reconnect_max_seconds,
        )
        self.dataset = DatasetService(settings.dataset_root)
        self._hand_detector: HandDetector | None = None
        self._classifier = GestureClassifier(settings.model_path, settings.model_metadata_path)
        self._vision: VisionService | None = None
        self._vision_error: str | None = None
        self._mouse_control: MouseControlService | None = None
        self._mouse_backend_error: str | None = None

    @property
    def hand_detector(self) -> HandDetector:
        if self._hand_detector is None:
            self._hand_detector = HandDetector()
        return self._hand_detector

    @property
    def vision(self) -> VisionService:
        if self._vision_error is not None:
            raise RuntimeError(self._vision_error)
        if self._vision is None:
            try:
                self._vision = VisionService(
                    self.frames,
                    self.annotated_frames,
                    self.hand_detector,
                    self._classifier,
                )
            except RuntimeError as exc:
                self._vision_error = str(exc)
                raise
        return self._vision

    def vision_status(self) -> dict[str, object]:
        if self._vision is not None:
            return self._vision.status()
        if self._vision_error is not None:
            return {
                "model_ready": False,
                "model_error": self._vision_error,
                "prediction": None,
            }
        try:
            return self.vision.status()
        except RuntimeError:
            return {
                "model_ready": False,
                "model_error": self._vision_error,
                "prediction": None,
            }

    @property
    def mouse_control(self) -> MouseControlService | None:
        if not self.settings.enable_mouse_control:
            return None
        if self._mouse_control is None and self._mouse_backend_error is None:
            try:
                self._mouse_control = MouseControlService(
                    self.settings, self.vision, MacOSMouseBackend()
                )
            except (RuntimeError, ValueError) as exc:
                self._mouse_backend_error = str(exc)
        return self._mouse_control

    def start(self) -> None:
        self.capture.start()
        try:
            self.vision.start()
        except RuntimeError:
            pass
        if self.mouse_control is not None:
            self.mouse_control.start()

    def stop(self) -> None:
        if self._mouse_control is not None:
            self._mouse_control.stop()
            self._mouse_control = None
        if self._vision is not None:
            self._vision.stop()
            self._vision = None
        self.capture.stop()
        if self._hand_detector is not None:
            self._hand_detector.close()
            self._hand_detector = None

    async def status(self) -> dict[str, object]:
        device: dict[str, object] | None = None
        device_error: str | None = None
        try:
            async with httpx.AsyncClient(timeout=1.5) as client:
                response = await client.get(self.settings.device_status_url)
                response.raise_for_status()
                device = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            device_error = str(exc)

        metrics = asdict(self.capture.metrics())
        return {
            "stream": metrics,
            "device": device,
            "device_error": device_error,
            "vision": self.vision_status(),
            "control": (
                self.mouse_control.status()
                if self.mouse_control is not None
                else {
                    "available": False,
                    "reason": self._mouse_backend_error or "disabled_by_configuration",
                }
            ),
        }
