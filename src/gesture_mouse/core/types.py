from dataclasses import dataclass
from enum import StrEnum
from time import monotonic

import numpy as np
from numpy.typing import NDArray


class ConnectionState(StrEnum):
    STOPPED = "stopped"
    CONNECTING = "connecting"
    ONLINE = "online"
    OFFLINE = "offline"


@dataclass(frozen=True, slots=True)
class FramePacket:
    image: NDArray[np.uint8]
    sequence: int
    captured_at: float

    @property
    def age_ms(self) -> float:
        return max(0.0, (monotonic() - self.captured_at) * 1000)


@dataclass(frozen=True, slots=True)
class StreamMetrics:
    state: ConnectionState
    frames_received: int
    decode_failures: int
    reconnects: int
    fps: float
    last_error: str | None
    latest_frame_age_ms: float | None


@dataclass(frozen=True, slots=True)
class GesturePrediction:
    raw_label: str | None
    stable_label: str | None
    confidence: float
    probabilities: dict[str, float]
    hand_detected: bool
    inference_ms: float
    frame_sequence: int

