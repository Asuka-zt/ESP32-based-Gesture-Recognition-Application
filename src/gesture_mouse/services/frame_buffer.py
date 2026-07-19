from threading import Condition, Lock
from time import monotonic

import numpy as np
from numpy.typing import NDArray

from gesture_mouse.core.types import FramePacket


class LatestFrameBuffer:
    def __init__(self) -> None:
        self._condition = Condition(Lock())
        self._latest: FramePacket | None = None
        self._sequence = 0

    def publish(self, image: NDArray[np.uint8]) -> FramePacket:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("frame must be a BGR image with shape HxWx3")

        with self._condition:
            self._sequence += 1
            packet = FramePacket(image=image, sequence=self._sequence, captured_at=monotonic())
            self._latest = packet
            self._condition.notify_all()
            return packet

    def latest(self, *, copy: bool = False) -> FramePacket | None:
        with self._condition:
            packet = self._latest
            if packet is None or not copy:
                return packet
            return FramePacket(
                image=packet.image.copy(),
                sequence=packet.sequence,
                captured_at=packet.captured_at,
            )

    def wait_for_new(self, after_sequence: int, timeout: float = 2.0) -> FramePacket | None:
        with self._condition:
            self._condition.wait_for(
                lambda: self._latest is not None and self._latest.sequence > after_sequence,
                timeout=timeout,
            )
            if self._latest is None or self._latest.sequence <= after_sequence:
                return None
            return self._latest

