from enum import StrEnum


class GestureLabel(StrEnum):
    POINT = "point"
    OK = "ok"
    PALM = "palm"
    FIST = "fist"
    V = "v"


GESTURE_LABELS = tuple(label.value for label in GestureLabel)

