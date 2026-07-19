import math
from dataclasses import asdict
from threading import Event, Lock, Thread
from time import monotonic

from gesture_mouse.config import Settings
from gesture_mouse.core.control import ControlSnapshot, ControlState
from gesture_mouse.services.mouse_backend import MouseBackend
from gesture_mouse.services.vision import VisionService


class MouseControlService:
    def __init__(
        self,
        settings: Settings,
        vision: VisionService,
        backend: MouseBackend,
    ) -> None:
        if settings.pinch_up_threshold <= settings.pinch_down_threshold:
            raise ValueError("pinch_up_threshold must exceed pinch_down_threshold")
        self._settings = settings
        self._vision = vision
        self._backend = backend
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._lock = Lock()
        self._state = ControlState.PAUSED
        self._mouse_down = False
        self._cursor: tuple[float, float] | None = None
        self._last_action = "startup_paused"
        self._last_error: str | None = None
        self._last_frame_sequence = 0
        self._last_hand_at = monotonic()
        self._palm_started_at: float | None = None
        self._palm_latched = False

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = Thread(target=self._run, name="gesture-mouse-control", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self.emergency_stop("service_stopped")

    def snapshot(self) -> ControlSnapshot:
        with self._lock:
            cursor_x = None if self._cursor is None else self._cursor[0]
            cursor_y = None if self._cursor is None else self._cursor[1]
            return ControlSnapshot(
                state=self._state,
                permission_granted=self._safe_permission_check(),
                mouse_down=self._mouse_down,
                cursor_x=cursor_x,
                cursor_y=cursor_y,
                last_action=self._last_action,
                last_error=self._last_error,
            )

    def status(self) -> dict[str, object]:
        return asdict(self.snapshot())

    def enable(self, reason: str = "api_enable") -> None:
        if not self._safe_permission_check():
            raise PermissionError(
                "macOS Accessibility permission is required for the running terminal/Python process"
            )
        if not self._vision.status()["model_ready"]:
            raise RuntimeError("gesture model is not ready")
        with self._lock:
            self._state = ControlState.ACTIVE
            self._last_action = reason
            self._last_error = None

    def pause(self, reason: str = "api_pause") -> None:
        with self._lock:
            self._release_locked()
            self._state = ControlState.PAUSED
            self._last_action = reason

    def emergency_stop(self, reason: str = "emergency_stop") -> None:
        try:
            self.pause(reason)
        except Exception as exc:  # final safety path must not propagate
            with self._lock:
                self._state = ControlState.PAUSED
                self._mouse_down = False
                self._last_error = str(exc)

    def _safe_permission_check(self) -> bool:
        try:
            return self._backend.permission_granted()
        except Exception:
            return False

    def _run(self) -> None:
        while not self._stop_event.wait(0.015):
            try:
                self._tick()
            except Exception as exc:
                with self._lock:
                    self._last_error = str(exc)
                self.emergency_stop("runtime_error")

    def _tick(self) -> None:
        prediction = self._vision.prediction()
        if prediction.frame_sequence <= self._last_frame_sequence:
            self._handle_lost_hand()
            return
        self._last_frame_sequence = prediction.frame_sequence
        observation = self._vision.observation()

        if not prediction.hand_detected or observation is None:
            self._handle_lost_hand()
            return
        self._last_hand_at = monotonic()
        label = prediction.stable_label

        if label == "fist":
            self.emergency_stop("fist_emergency_stop")
            return

        self._handle_palm_toggle(label)
        with self._lock:
            if self._state == ControlState.PAUSED:
                return

        if label == "point":
            self._move_pointer(observation.landmarks[8][0], observation.landmarks[8][1])
            self._release_if_pressed("point_release")
        elif label == "ok":
            self._handle_pinch(observation.landmarks)
        else:
            self._release_if_pressed("gesture_release")

    def _handle_lost_hand(self) -> None:
        elapsed = monotonic() - self._last_hand_at
        if elapsed >= self._settings.lost_hand_release_seconds:
            self._release_if_pressed("lost_hand_release")
        if elapsed >= self._settings.lost_hand_pause_seconds:
            self.pause("lost_hand_pause")

    def _handle_palm_toggle(self, label: str | None) -> None:
        now = monotonic()
        if label != "palm":
            self._palm_started_at = None
            self._palm_latched = False
            return
        if self._palm_started_at is None:
            self._palm_started_at = now
            return
        if self._palm_latched or now - self._palm_started_at < self._settings.palm_toggle_seconds:
            return

        self._palm_latched = True
        with self._lock:
            state = self._state
        if state == ControlState.PAUSED:
            try:
                self.enable("palm_toggle_enable")
            except (PermissionError, RuntimeError) as exc:
                with self._lock:
                    self._last_error = str(exc)
        else:
            self.pause("palm_toggle_pause")

    def _move_pointer(self, normalized_x: float, normalized_y: float) -> None:
        margin = self._settings.pointer_margin
        usable = 1.0 - 2.0 * margin
        mapped_x = 1.0 - min(1.0, max(0.0, (normalized_x - margin) / usable))
        mapped_y = min(1.0, max(0.0, (normalized_y - margin) / usable))
        screen_width, screen_height = self._backend.screen_size()
        target = (mapped_x * (screen_width - 1), mapped_y * (screen_height - 1))

        with self._lock:
            if self._cursor is None:
                smoothed = target
            else:
                alpha = self._settings.pointer_smoothing_alpha
                smoothed = (
                    self._cursor[0] + alpha * (target[0] - self._cursor[0]),
                    self._cursor[1] + alpha * (target[1] - self._cursor[1]),
                )
                if math.dist(smoothed, self._cursor) < self._settings.pointer_deadzone_px:
                    return
            self._backend.move(smoothed[0], smoothed[1], dragging=self._mouse_down)
            self._cursor = smoothed
            self._last_action = "drag_move" if self._mouse_down else "pointer_move"

    def _handle_pinch(self, landmarks: tuple[tuple[float, float, float], ...]) -> None:
        thumb = landmarks[4]
        index = landmarks[8]
        distance = math.dist((thumb[0], thumb[1]), (index[0], index[1]))
        self._move_pointer(index[0], index[1])

        with self._lock:
            if self._cursor is None:
                return
            if not self._mouse_down and distance <= self._settings.pinch_down_threshold:
                self._backend.button_down(*self._cursor)
                self._mouse_down = True
                self._state = ControlState.PRESSED
                self._last_action = "mouse_down"
            elif self._mouse_down and distance >= self._settings.pinch_up_threshold:
                self._release_locked()
                self._state = ControlState.ACTIVE
                self._last_action = "mouse_up"

    def _release_if_pressed(self, reason: str) -> None:
        with self._lock:
            if self._mouse_down:
                self._release_locked()
                if self._state != ControlState.PAUSED:
                    self._state = ControlState.ACTIVE
                self._last_action = reason

    def _release_locked(self) -> None:
        if not self._mouse_down:
            return
        cursor = self._cursor or (0.0, 0.0)
        try:
            self._backend.button_up(*cursor)
        finally:
            self._mouse_down = False

