import uvicorn

from gesture_mouse.config import settings


def run() -> None:
    uvicorn.run(
        "gesture_mouse.api.app:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    run()

