from dataclasses import dataclass

from gesture_mouse.config import Settings
from gesture_mouse.core.types import GesturePrediction
from gesture_mouse.services.hand_detector import HandObservation
from gesture_mouse.services.mouse_control import MouseControlService


class FakeBackend:
    def __init__(self) -> None:
        self.events: list[tuple[str, float, float]] = []

    def permission_granted(self) -> bool:
        return True

    def screen_size(self) -> tuple[float, float]:
        return 1000.0, 500.0

    def move(self, x: float, y: float, *, dragging: bool = False) -> None:
        self.events.append(("drag" if dragging else "move", x, y))

    def button_down(self, x: float, y: float) -> None:
        self.events.append(("down", x, y))

    def button_up(self, x: float, y: float) -> None:
        self.events.append(("up", x, y))


@dataclass
class FakeVision:
    current_prediction: GesturePrediction
    current_observation: HandObservation | None

    def prediction(self) -> GesturePrediction:
        return self.current_prediction

    def observation(self) -> HandObservation | None:
        return self.current_observation

    def status(self) -> dict[str, object]:
        return {"model_ready": True}


def observation_with_pinch(distance: float) -> HandObservation:
    landmarks = [(0.5, 0.5, 0.0)] * 21
    landmarks[4] = (0.5 - distance, 0.5, 0.0)
    landmarks[8] = (0.5, 0.5, 0.0)
    return HandObservation(tuple(landmarks), (0, 0, 10, 10), None, "Right", 1.0)  # type: ignore[arg-type]


def prediction(label: str, sequence: int) -> GesturePrediction:
    return GesturePrediction(label, label, 0.99, {label: 0.99}, True, 1.0, sequence)


def test_pinch_generates_balanced_down_and_up_events() -> None:
    backend = FakeBackend()
    vision = FakeVision(prediction("ok", 1), observation_with_pinch(0.04))
    service = MouseControlService(Settings(), vision, backend)  # type: ignore[arg-type]
    service.enable()

    service._tick()
    vision.current_prediction = prediction("ok", 2)
    vision.current_observation = observation_with_pinch(0.15)
    service._tick()

    event_names = [event[0] for event in backend.events]
    assert event_names.count("down") == 1
    assert event_names.count("up") == 1


def test_emergency_stop_releases_pressed_mouse() -> None:
    backend = FakeBackend()
    vision = FakeVision(prediction("ok", 1), observation_with_pinch(0.04))
    service = MouseControlService(Settings(), vision, backend)  # type: ignore[arg-type]
    service.enable()
    service._tick()

    service.emergency_stop("test")

    assert service.status()["state"] == "paused"
    assert service.status()["mouse_down"] is False
    assert [event[0] for event in backend.events].count("up") == 1

