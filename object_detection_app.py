"""
Object Detection Web App
FastAPI backend — MJPEG stream with YOLOv8 inference.
"""

import io
import threading
import time
from contextlib import asynccontextmanager
from typing import Generator

import cv2
import numpy as np
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Colours for bounding boxes (BGR for OpenCV)
# ---------------------------------------------------------------------------
BOX_COLOURS = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
    (49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
    (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),
    (147, 69, 52), (255, 115, 100), (236, 24, 0), (255, 56, 132),
    (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255),
]


# ---------------------------------------------------------------------------
# Global detection state (shared across requests)
# ---------------------------------------------------------------------------
class DetectionState:
    def __init__(self):
        self.model: YOLO | None = None
        self.cap: cv2.VideoCapture | None = None
        self.running = False
        self.keywords: list[str] = []
        self.conf_threshold: float = 0.40
        self.colour_map: dict[str, tuple[int, int, int]] = {}
        self._lock = threading.Lock()
        self._latest_annotated: bytes | None = None  # JPEG bytes
        self._thread: threading.Thread | None = None
        self.status: str = "idle"
        self.fps: float = 0.0

    def colour_for(self, label: str) -> tuple[int, int, int]:
        label_l = label.lower()
        for kw, col in self.colour_map.items():
            if kw in label_l:
                return col
        return BOX_COLOURS[hash(label) % len(BOX_COLOURS)]

    def label_matches(self, label: str) -> bool:
        if not self.keywords:
            return True
        return any(kw in label.lower() for kw in self.keywords)


state = DetectionState()


# ---------------------------------------------------------------------------
# App lifespan: load model on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    state.status = "loading_model"
    try:
        state.model = YOLO("yolov8n.pt")
        state.status = "idle"
    except Exception as e:
        state.status = f"model_error: {e}"
    yield
    _stop_stream()


app = FastAPI(lifespan=lifespan)


# ---------------------------------------------------------------------------
# Capture + inference loop (runs in a background thread)
# ---------------------------------------------------------------------------
def _capture_loop(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        state.status = f"error: cannot open source '{source}'"
        state.running = False
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    state.cap = cap
    state.status = "streaming"

    t0 = time.time()
    frames = 0

    while state.running:
        ret, frame = cap.read()
        if not ret:
            state.status = "error: stream ended"
            break

        # Run inference
        results = state.model.predict(
            frame, conf=state.conf_threshold, verbose=False
        )
        annotated = frame.copy()

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = state.model.names[cls_id]
                if not state.label_matches(label):
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                colour = state.colour_for(label)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)
                text = f"{label} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
                )
                cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), colour, -1)
                cv2.putText(
                    annotated, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
                )

        # Encode as JPEG
        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with state._lock:
            state._latest_annotated = buf.tobytes()

        frames += 1
        elapsed = time.time() - t0
        if elapsed >= 1.0:
            state.fps = round(frames / elapsed, 1)
            frames, t0 = 0, time.time()

    cap.release()
    state.cap = None
    if state.status == "streaming":
        state.status = "idle"


def _stop_stream():
    state.running = False
    if state._thread and state._thread.is_alive():
        state._thread.join(timeout=3)
    state._thread = None
    state._latest_annotated = None
    state.fps = 0.0


# ---------------------------------------------------------------------------
# MJPEG frame generator
# ---------------------------------------------------------------------------
def _mjpeg_frames() -> Generator[bytes, None, None]:
    placeholder = _make_placeholder("Waiting for stream…")
    while True:
        with state._lock:
            frame = state._latest_annotated
        if frame is None:
            frame = placeholder
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )
        time.sleep(0.03)  # ~30 fps cap on delivery


def _make_placeholder(msg: str) -> bytes:
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(img, msg, (60, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 180, 180), 2)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(HTML_PAGE)


@app.post("/start")
async def start_stream(
    source: str = Form("0"),
    keywords: str = Form(""),
    conf: float = Form(0.40),
):
    if state.model is None:
        return JSONResponse({"ok": False, "error": "Model not loaded yet."})

    _stop_stream()

    # Parse source
    src = int(source.strip()) if source.strip().isdigit() else source.strip()

    # Parse keywords
    kws = [k.strip().lower() for k in keywords.split(",") if k.strip()]
    state.keywords = kws
    state.conf_threshold = max(0.05, min(0.95, conf))
    # Assign colours
    state.colour_map = {}
    for i, kw in enumerate(kws):
        state.colour_map[kw] = BOX_COLOURS[i % len(BOX_COLOURS)]

    state.running = True
    state._thread = threading.Thread(target=_capture_loop, args=(src,), daemon=True)
    state._thread.start()
    return JSONResponse({"ok": True})


@app.post("/stop")
async def stop_stream():
    _stop_stream()
    state.status = "idle"
    return JSONResponse({"ok": True})


@app.get("/status")
async def get_status():
    return JSONResponse({
        "status": state.status,
        "fps": state.fps,
        "keywords": state.keywords,
        "conf": state.conf_threshold,
        "running": state.running,
        "model_classes": len(state.model.names) if state.model else 0,
    })


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        _mjpeg_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ---------------------------------------------------------------------------
# Inline HTML page
# ---------------------------------------------------------------------------
HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Object Detection</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', sans-serif; background: #1a1a2e; color: #eee; min-height: 100vh; }
  header { background: #16213e; padding: 16px 24px; display: flex; align-items: center; gap: 12px; border-bottom: 1px solid #0f3460; }
  header h1 { font-size: 1.4rem; color: #e94560; }
  header span { font-size: 0.85rem; color: #aaa; }
  .main { display: flex; gap: 16px; padding: 16px; height: calc(100vh - 65px); }
  .sidebar { width: 300px; flex-shrink: 0; display: flex; flex-direction: column; gap: 12px; }
  .card { background: #16213e; border-radius: 10px; padding: 16px; border: 1px solid #0f3460; }
  .card h3 { font-size: 0.9rem; color: #e94560; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 1px; }
  label { display: block; font-size: 0.8rem; color: #aaa; margin-bottom: 4px; margin-top: 10px; }
  label:first-of-type { margin-top: 0; }
  input[type=text], input[type=number] {
    width: 100%; padding: 8px 10px; border-radius: 6px;
    background: #0f3460; border: 1px solid #1a4a8a; color: #eee; font-size: 0.9rem;
  }
  input[type=range] { width: 100%; accent-color: #e94560; }
  .conf-row { display: flex; align-items: center; gap: 8px; }
  .conf-row input[type=range] { flex: 1; }
  .conf-val { font-size: 0.85rem; color: #e94560; min-width: 36px; text-align: right; }
  .btn { width: 100%; padding: 10px; border-radius: 6px; border: none; cursor: pointer; font-size: 0.95rem; font-weight: 600; transition: opacity 0.2s; }
  .btn:disabled { opacity: 0.4; cursor: default; }
  .btn-start { background: #e94560; color: #fff; }
  .btn-stop  { background: #444; color: #eee; margin-top: 6px; }
  .status-box { font-size: 0.78rem; padding: 8px 10px; background: #0f1923; border-radius: 6px; line-height: 1.6; }
  .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
  .dot-idle    { background: #888; }
  .dot-running { background: #2ecc71; }
  .dot-error   { background: #e74c3c; }
  .video-pane { flex: 1; display: flex; flex-direction: column; gap: 8px; }
  .video-wrap { flex: 1; background: #000; border-radius: 10px; overflow: hidden; display: flex; align-items: center; justify-content: center; border: 1px solid #0f3460; }
  .video-wrap img { max-width: 100%; max-height: 100%; object-fit: contain; }
  .tag-row { display: flex; flex-wrap: wrap; gap: 6px; min-height: 28px; }
  .tag { background: #0f3460; border: 1px solid #1a4a8a; border-radius: 20px; padding: 2px 10px; font-size: 0.78rem; color: #7ec8e3; }
  .tag-all { color: #aaa; font-style: italic; }
</style>
</head>
<body>
<header>
  <h1>&#127909; Object Detection</h1>
  <span>YOLOv8 · RTSP &amp; Webcam</span>
</header>
<div class="main">
  <div class="sidebar">
    <div class="card">
      <h3>Source</h3>
      <label>Camera index or RTSP URL</label>
      <input type="text" id="source" value="0" placeholder="0  or  rtsp://...">
    </div>
    <div class="card">
      <h3>Detection</h3>
      <label>Keywords (comma-separated)</label>
      <input type="text" id="keywords" value="person, car, dog" placeholder="person, car, dog — blank = all">
      <label>Confidence threshold</label>
      <div class="conf-row">
        <input type="range" id="conf" min="0.05" max="0.95" step="0.01" value="0.40" oninput="document.getElementById('confVal').textContent=parseFloat(this.value).toFixed(2)">
        <span class="conf-val" id="confVal">0.40</span>
      </div>
    </div>
    <div class="card">
      <h3>Controls</h3>
      <button class="btn btn-start" id="btnStart" onclick="startStream()">&#9654; Start</button>
      <button class="btn btn-stop"  id="btnStop"  onclick="stopStream()" disabled>&#9646;&#9646; Stop</button>
    </div>
    <div class="card">
      <h3>Status</h3>
      <div class="status-box" id="statusBox">Loading model…</div>
    </div>
  </div>
  <div class="video-pane">
    <div class="tag-row" id="tagRow"><span class="tag tag-all">No active keywords</span></div>
    <div class="video-wrap">
      <img id="videoFeed" src="/video_feed" alt="video feed">
    </div>
  </div>
</div>
<script>
let pollTimer = null;

async function startStream() {
  const source   = document.getElementById('source').value.trim();
  const keywords = document.getElementById('keywords').value.trim();
  const conf     = parseFloat(document.getElementById('conf').value);

  const fd = new FormData();
  fd.append('source', source);
  fd.append('keywords', keywords);
  fd.append('conf', conf);

  const res = await fetch('/start', { method: 'POST', body: fd });
  const data = await res.json();
  if (!data.ok) { alert(data.error); return; }

  document.getElementById('btnStart').disabled = true;
  document.getElementById('btnStop').disabled  = false;

  // Force img reload to restart MJPEG
  const img = document.getElementById('videoFeed');
  img.src = '/video_feed?t=' + Date.now();

  startPolling();
  updateTags(keywords);
}

async function stopStream() {
  await fetch('/stop', { method: 'POST' });
  document.getElementById('btnStart').disabled = false;
  document.getElementById('btnStop').disabled  = true;
  stopPolling();
  updateStatus({ status: 'idle', fps: 0, running: false });
}

function startPolling() {
  stopPolling();
  pollTimer = setInterval(pollStatus, 1500);
}
function stopPolling() {
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
}

async function pollStatus() {
  try {
    const res = await fetch('/status');
    const data = await res.json();
    updateStatus(data);
    if (!data.running && data.status !== 'idle') {
      // stream ended or error
      document.getElementById('btnStart').disabled = false;
      document.getElementById('btnStop').disabled  = true;
      stopPolling();
    }
  } catch(e) {}
}

function updateStatus(d) {
  const dotClass = d.running ? 'dot-running' : (d.status.startsWith('error') ? 'dot-error' : 'dot-idle');
  const fpsStr   = d.running ? ` &nbsp;|&nbsp; ${d.fps} fps` : '';
  const confStr  = d.running ? ` &nbsp;|&nbsp; conf ≥ ${d.conf}` : '';
  const classes  = d.running ? (d.model_classes ? ` &nbsp;|&nbsp; ${d.model_classes} classes` : '') : '';
  document.getElementById('statusBox').innerHTML =
    `<span class="dot ${dotClass}"></span><strong>${d.status}</strong>${fpsStr}${confStr}${classes}`;
}

function updateTags(keywords) {
  const row = document.getElementById('tagRow');
  const kws = keywords.split(',').map(k => k.trim()).filter(Boolean);
  if (!kws.length) {
    row.innerHTML = '<span class="tag tag-all">All classes (no filter)</span>';
  } else {
    row.innerHTML = kws.map(k => `<span class="tag">${k}</span>`).join('');
  }
}

// Initial status poll
(async () => {
  try {
    const res = await fetch('/status');
    const d   = await res.json();
    updateStatus(d);
  } catch(e) {}
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
