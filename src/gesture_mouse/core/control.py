from dataclasses import dataclass
from enum import StrEnum


class ControlState(StrEnum):
    PAUSED = "paused"
    ACTIVE = "active"
    PRESSED = "pressed"


@dataclass(frozen=True, slots=True)
class ControlSnapshot:
    state: ControlState
    permission_granted: bool
    mouse_down: bool
    cursor_x: float | None
    cursor_y: float | None
    last_action: str
    last_error: str | None

