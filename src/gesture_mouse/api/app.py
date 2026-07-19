from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager

import cv2
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from gesture_mouse import __version__
from gesture_mouse.config import Settings, settings
from gesture_mouse.core.gestures import GestureLabel
from gesture_mouse.services.runtime import ApplicationRuntime


class SessionRequest(BaseModel):
    person: str = Field(min_length=1, max_length=64)
    session_id: str | None = Field(default=None, min_length=1, max_length=64)


class CaptureRequest(BaseModel):
    person: str = Field(min_length=1, max_length=64)
    session_id: str = Field(min_length=1, max_length=64)
    gesture: GestureLabel


def create_app(
    app_settings: Settings | None = None,
    *,
    runtime: ApplicationRuntime | None = None,
    start_runtime: bool = True,
) -> FastAPI:
    active_settings = app_settings or settings
    active_runtime = runtime or ApplicationRuntime(active_settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if start_runtime:
            active_runtime.start()
        try:
            yield
        finally:
            if start_runtime:
                active_runtime.stop()

    app = FastAPI(title="ESP32 Gesture Mouse", version=__version__, lifespan=lifespan)
    app.state.runtime = active_runtime

    @app.get("/health", tags=["system"])
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/api/status", tags=["system"])
    async def api_status(request: Request) -> dict[str, object]:
        return await request.app.state.runtime.status()

    @app.get("/api/live", tags=["stream"])
    async def api_live(request: Request) -> StreamingResponse:
        runtime = request.app.state.runtime
        frame_buffer = (
            runtime.annotated_frames
            if runtime.annotated_frames.latest() is not None
            else runtime.frames
        )
        if frame_buffer.latest() is None:
            raise HTTPException(status_code=503, detail="ESP32 stream has no frame")

        def generate() -> Iterator[bytes]:
            sequence = 0
            while True:
                packet = frame_buffer.wait_for_new(sequence, timeout=2.0)
                if packet is None:
                    continue
                sequence = packet.sequence
                ok, encoded = cv2.imencode(".jpg", packet.image, [cv2.IMWRITE_JPEG_QUALITY, 82])
                if not ok:
                    continue
                payload = encoded.tobytes()
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(payload)).encode("ascii")
                    + b"\r\n\r\n"
                    + payload
                    + b"\r\n"
                )

        return StreamingResponse(
            generate(), media_type="multipart/x-mixed-replace; boundary=frame"
        )

    @app.get("/api/prediction", tags=["recognition"])
    async def api_prediction(request: Request) -> dict[str, object]:
        return request.app.state.runtime.vision_status()

    @app.post("/api/control/enable", tags=["control"])
    async def enable_control(request: Request) -> dict[str, object]:
        controller = request.app.state.runtime.mouse_control
        if controller is None:
            raise HTTPException(status_code=503, detail="mouse control is unavailable")
        try:
            controller.enable()
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return controller.status()

    @app.post("/api/control/disable", tags=["control"])
    async def disable_control(request: Request) -> dict[str, object]:
        controller = request.app.state.runtime.mouse_control
        if controller is None:
            raise HTTPException(status_code=503, detail="mouse control is unavailable")
        controller.pause()
        return controller.status()

    @app.post("/api/control/emergency-stop", tags=["control"])
    async def emergency_stop(request: Request) -> dict[str, object]:
        controller = request.app.state.runtime.mouse_control
        if controller is None:
            raise HTTPException(status_code=503, detail="mouse control is unavailable")
        controller.emergency_stop()
        return controller.status()

    @app.post("/api/dataset/sessions", tags=["dataset"])
    async def create_dataset_session(
        payload: SessionRequest, request: Request
    ) -> dict[str, str]:
        try:
            session = request.app.state.runtime.dataset.create_session(
                payload.person, payload.session_id
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "person": session.person,
            "session_id": session.session_id,
            "created_at": session.created_at,
        }

    @app.post("/api/dataset/capture", tags=["dataset"])
    async def capture_dataset_image(
        payload: CaptureRequest, request: Request
    ) -> dict[str, object]:
        runtime = request.app.state.runtime
        packet = runtime.frames.latest(copy=True)
        if packet is None:
            raise HTTPException(status_code=503, detail="ESP32 stream has no frame")
        try:
            observation = runtime.hand_detector.detect(packet.image)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if observation is None:
            raise HTTPException(status_code=422, detail="no hand detected")
        try:
            path = runtime.dataset.save_crop(
                observation.crop,
                gesture=payload.gesture,
                person=payload.person,
                session_id=payload.session_id,
            )
        except (ValueError, OSError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "saved": True,
            "path": str(path.relative_to(runtime.settings.dataset_root)),
            "frame_sequence": packet.sequence,
            "handedness": observation.handedness,
        }

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def dashboard() -> str:
        return DASHBOARD_HTML

    return app


DASHBOARD_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>ESP32 手势鼠标</title>
  <style>
    :root { color-scheme: dark; font-family: ui-sans-serif, system-ui, sans-serif; }
    body { margin: 0; background: #0b1020; color: #e8ecf5; }
    main { width: min(1100px, 92vw); margin: 32px auto; }
    h1 { margin-bottom: 8px; }
    .muted { color: #9ca9c7; }
    .grid { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; margin-top: 24px; }
    .card { background: #151c31; border: 1px solid #29324d; border-radius: 14px; padding: 16px; }
    img { width: 100%; min-height: 280px; background: #050711; object-fit: contain;
          border-radius: 10px; }
    dl { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    dt { color: #9ca9c7; } dd { margin: 0; text-align: right; }
    button, input, select { font: inherit; border-radius: 8px; border: 1px solid #3a4668;
                            background: #0e1528; color: #e8ecf5; padding: 9px 12px; }
    button { cursor: pointer; background: #294e9a; }
    button.danger { background: #a72c3b; }
    .controls { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 12px; }
    .form { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
    .wide { grid-column: 1 / -1; }
    .online { color: #55d98a; } .offline { color: #ff6b78; }
    @media (max-width: 760px) {
      .grid, .form { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
<main>
  <h1>ESP32-CAM OV3660 手势鼠标</h1>
  <div class="muted">局域网实时视频与设备状态</div>
  <section class="grid">
    <div class="card"><img id="video" alt="等待 ESP32-CAM 视频"></div>
    <div class="card">
      <h2>运行状态</h2>
      <dl>
        <dt>视频流</dt><dd id="state">加载中</dd>
        <dt>传感器</dt><dd id="sensor">—</dd>
        <dt>FPS</dt><dd id="fps">0</dd>
        <dt>帧延迟</dt><dd id="age">—</dd>
        <dt>重连次数</dt><dd id="reconnects">0</dd>
        <dt>RSSI</dt><dd id="rssi">—</dd>
        <dt>手势</dt><dd id="gesture">—</dd>
        <dt>置信度</dt><dd id="confidence">—</dd>
        <dt>鼠标状态</dt><dd id="control-state">不可用</dd>
      </dl>
      <p class="muted" id="error"></p>
      <div class="controls">
        <button id="enable">启用控制</button>
        <button id="disable">暂停控制</button>
        <button id="stop" class="danger">急停</button>
      </div>
    </div>
    <div class="card wide">
      <h2>数据采集</h2>
      <div class="form">
        <input id="person" value="person_1" aria-label="操作者" placeholder="操作者">
        <input id="session" value="room_a_01" aria-label="场次" placeholder="采集场次">
        <select id="gesture-select" aria-label="手势类别">
          <option value="point">指向</option><option value="ok">OK/捏合</option>
          <option value="palm">张掌</option><option value="fist">握拳</option>
          <option value="v">V</option>
        </select>
      </div>
      <div class="controls">
        <button id="create-session">创建场次</button>
        <button id="capture">保存当前手部</button>
      </div>
      <p class="muted" id="dataset-message"></p>
    </div>
  </section>
</main>
<script>
const video = document.querySelector('#video');
async function refresh() {
  try {
    const response = await fetch('/api/status', {cache: 'no-store'});
    const data = await response.json();
    const online = data.stream.state === 'online';
    const state = document.querySelector('#state');
    state.textContent = data.stream.state;
    state.className = online ? 'online' : 'offline';
    document.querySelector('#sensor').textContent = data.device?.sensor ?? '—';
    document.querySelector('#fps').textContent = data.stream.fps.toFixed(1);
    document.querySelector('#age').textContent = data.stream.latest_frame_age_ms == null
      ? '—' : `${data.stream.latest_frame_age_ms} ms`;
    document.querySelector('#reconnects').textContent = data.stream.reconnects;
    document.querySelector('#rssi').textContent = data.device?.rssi == null
      ? '—' : `${data.device.rssi} dBm`;
    document.querySelector('#error').textContent =
      data.stream.last_error ?? data.device_error ?? '';
    const prediction = data.vision?.prediction;
    document.querySelector('#gesture').textContent =
      prediction?.stable_label ?? prediction?.raw_label ?? '—';
    document.querySelector('#confidence').textContent = prediction?.confidence == null
      ? '—' : `${Math.round(prediction.confidence * 100)}%`;
    document.querySelector('#control-state').textContent =
      data.control?.state ?? data.control?.reason ?? '不可用';
    if (online && !video.src) video.src = '/api/live';
    if (!online) video.removeAttribute('src');
  } catch (error) {
    document.querySelector('#error').textContent = String(error);
  }
}
async function post(path, body) {
  const response = await fetch(path, {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: body == null ? null : JSON.stringify(body)
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail ?? `HTTP ${response.status}`);
  return data;
}
document.querySelector('#enable').onclick = () => post('/api/control/enable').catch(showError);
document.querySelector('#disable').onclick = () => post('/api/control/disable').catch(showError);
document.querySelector('#stop').onclick = () =>
  post('/api/control/emergency-stop').catch(showError);
document.addEventListener('keydown', event => {
  if (event.key === 'Escape') post('/api/control/emergency-stop').catch(showError);
});
document.querySelector('#create-session').onclick = async () => {
  const payload = values();
  try {
    const data = await post('/api/dataset/sessions', {
      person: payload.person, session_id: payload.session_id
    });
    document.querySelector('#dataset-message').textContent =
      `场次已创建：${data.person}/${data.session_id}`;
  } catch (error) { showDatasetError(error); }
};
document.querySelector('#capture').onclick = async () => {
  try {
    const data = await post('/api/dataset/capture', values());
    document.querySelector('#dataset-message').textContent = `已保存：${data.path}`;
  } catch (error) { showDatasetError(error); }
};
function values() {
  return {
    person: document.querySelector('#person').value,
    session_id: document.querySelector('#session').value,
    gesture: document.querySelector('#gesture-select').value
  };
}
function showError(error) { document.querySelector('#error').textContent = String(error); }
function showDatasetError(error) {
  document.querySelector('#dataset-message').textContent = String(error);
}
setInterval(refresh, 1000); refresh();
</script>
</body>
</html>"""


app = create_app()
