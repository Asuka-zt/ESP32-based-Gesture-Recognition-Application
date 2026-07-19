from fastapi.testclient import TestClient

from gesture_mouse.api.app import create_app


def test_health_endpoint() -> None:
    with TestClient(create_app(start_runtime=False)) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_dashboard_is_available() -> None:
    with TestClient(create_app(start_runtime=False)) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "OV3660" in response.text
