from typing import Protocol


class MouseBackend(Protocol):
    def permission_granted(self) -> bool: ...

    def screen_size(self) -> tuple[float, float]: ...

    def move(self, x: float, y: float, *, dragging: bool = False) -> None: ...

    def button_down(self, x: float, y: float) -> None: ...

    def button_up(self, x: float, y: float) -> None: ...


class MacOSMouseBackend:
    def __init__(self) -> None:
        try:
            import Quartz
            from ApplicationServices import AXIsProcessTrusted
        except ImportError as exc:
            raise RuntimeError("macOS Quartz/ApplicationServices is unavailable") from exc
        self._quartz = Quartz
        self._is_trusted = AXIsProcessTrusted

    def permission_granted(self) -> bool:
        return bool(self._is_trusted())

    def screen_size(self) -> tuple[float, float]:
        bounds = self._quartz.CGDisplayBounds(self._quartz.CGMainDisplayID())
        return float(bounds.size.width), float(bounds.size.height)

    def move(self, x: float, y: float, *, dragging: bool = False) -> None:
        event_type = (
            self._quartz.kCGEventLeftMouseDragged
            if dragging
            else self._quartz.kCGEventMouseMoved
        )
        self._post(event_type, x, y)

    def button_down(self, x: float, y: float) -> None:
        self._post(self._quartz.kCGEventLeftMouseDown, x, y)

    def button_up(self, x: float, y: float) -> None:
        self._post(self._quartz.kCGEventLeftMouseUp, x, y)

    def _post(self, event_type: int, x: float, y: float) -> None:
        event = self._quartz.CGEventCreateMouseEvent(
            None,
            event_type,
            (x, y),
            self._quartz.kCGMouseButtonLeft,
        )
        if event is None:
            raise RuntimeError("failed to create macOS mouse event")
        self._quartz.CGEventPost(self._quartz.kCGHIDEventTap, event)

