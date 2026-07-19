import logging
from collections import deque
from threading import Event, Lock, Thread
from time import monotonic, sleep

import cv2

from gesture_mouse.core.types import ConnectionState, StreamMetrics
from gesture_mouse.services.frame_buffer import LatestFrameBuffer

logger = logging.getLogger(__name__)


class MjpegCaptureService:
    def __init__(
        self,
        stream_url: str,
        frame_buffer: LatestFrameBuffer,
        *,
        reconnect_initial_seconds: float = 0.5,
        reconnect_max_seconds: float = 8.0,
    ) -> None:
        self._stream_url = stream_url
        self._frame_buffer = frame_buffer
        self._reconnect_initial = reconnect_initial_seconds
        self._reconnect_max = reconnect_max_seconds
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._lock = Lock()
        self._state = ConnectionState.STOPPED
        self._frames_received = 0
        self._decode_failures = 0
        self._reconnects = 0
        self._last_error: str | None = None
        self._frame_times: deque[float] = deque(maxlen=60)

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = Thread(target=self._run, name="esp32-mjpeg-capture", daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
        with self._lock:
            self._state = ConnectionState.STOPPED

    def metrics(self) -> StreamMetrics:
        with self._lock:
            times = tuple(self._frame_times)
            fps = 0.0
            if len(times) >= 2 and times[-1] > times[0]:
                fps = (len(times) - 1) / (times[-1] - times[0])
            packet = self._frame_buffer.latest()
            return StreamMetrics(
                state=self._state,
                frames_received=self._frames_received,
                decode_failures=self._decode_failures,
                reconnects=self._reconnects,
                fps=round(fps, 2),
                last_error=self._last_error,
                latest_frame_age_ms=None if packet is None else round(packet.age_ms, 1),
            )

    def _set_state(self, state: ConnectionState, error: str | None = None) -> None:
        with self._lock:
            self._state = state
            self._last_error = error

    def _interruptible_sleep(self, seconds: float) -> None:
        self._stop_event.wait(seconds)

    def _run(self) -> None:
        backoff = self._reconnect_initial
        while not self._stop_event.is_set():
            self._set_state(ConnectionState.CONNECTING)
            capture = cv2.VideoCapture(self._stream_url, cv2.CAP_FFMPEG)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not capture.isOpened():
                capture.release()
                self._set_state(ConnectionState.OFFLINE, "unable to open ESP32 stream")
                with self._lock:
                    self._reconnects += 1
                self._interruptible_sleep(backoff)
                backoff = min(self._reconnect_max, backoff * 2)
                continue

            self._set_state(ConnectionState.ONLINE)
            backoff = self._reconnect_initial
            consecutive_failures = 0

            while not self._stop_event.is_set():
                ok, frame = capture.read()
                if not ok or frame is None:
                    consecutive_failures += 1
                    with self._lock:
                        self._decode_failures += 1
                    if consecutive_failures >= 5:
                        self._set_state(ConnectionState.OFFLINE, "stream read failed")
                        break
                    sleep(0.02)
                    continue

                consecutive_failures = 0
                self._frame_buffer.publish(frame)
                now = monotonic()
                with self._lock:
                    self._frames_received += 1
                    self._frame_times.append(now)
                    self._last_error = None

            capture.release()
            if not self._stop_event.is_set():
                with self._lock:
                    self._reconnects += 1
                self._interruptible_sleep(backoff)
                backoff = min(self._reconnect_max, backoff * 2)

