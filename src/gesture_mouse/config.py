from pathlib import Path

from pydantic import Field, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="GESTURE_MOUSE_",
        extra="ignore",
    )

    esp32_base_url: HttpUrl = HttpUrl("http://esp32cam.local")
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    model_path: Path = Path("models/gesture_mobilenet_v3.pt")
    model_metadata_path: Path = Path("models/gesture_mobilenet_v3.json")
    dataset_root: Path = Path("data")
    enable_mouse_control: bool = False
    capture_width: int = Field(default=640, ge=160, le=1920)
    capture_height: int = Field(default=480, ge=120, le=1080)
    reconnect_initial_seconds: float = Field(default=0.5, ge=0.1, le=10)
    reconnect_max_seconds: float = Field(default=8.0, ge=1, le=60)
    pointer_smoothing_alpha: float = Field(default=0.25, gt=0, le=1)
    pointer_deadzone_px: float = Field(default=4.0, ge=0, le=100)
    pointer_margin: float = Field(default=0.15, ge=0, lt=0.45)
    palm_toggle_seconds: float = Field(default=1.0, ge=0.3, le=5)
    lost_hand_release_seconds: float = Field(default=0.35, ge=0.1, le=3)
    lost_hand_pause_seconds: float = Field(default=1.0, ge=0.3, le=10)
    pinch_down_threshold: float = Field(default=0.075, gt=0, lt=0.3)
    pinch_up_threshold: float = Field(default=0.105, gt=0, lt=0.4)

    @property
    def stream_url(self) -> str:
        return f"{str(self.esp32_base_url).rstrip('/')}/stream"

    @property
    def device_status_url(self) -> str:
        return f"{str(self.esp32_base_url).rstrip('/')}/status"


settings = Settings()
