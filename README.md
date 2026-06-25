# Object Detection App

Real-time object detection from a webcam or RTSP stream.  
FastAPI backend · YOLOv8n inference · MJPEG stream · browser UI.

## Setup

```bash
pip install -r requirements.txt
python object_detection_app.py
# Open http://localhost:8000 in your browser
```

## Usage

| Control | Description |
|---------|-------------|
| **Source** | `0`, `1`, … for local cameras; `rtsp://user:pass@host/stream` for IP cameras |
| **Keywords** | Comma-separated label substrings, e.g. `person, car, dog`. Leave blank to detect everything. |
| **Conf** | Confidence threshold slider (0.05 – 0.95). |
| **Start / Stop** | Begin or halt the stream. |

### Keyword matching

Keywords are matched as **substrings** against YOLOv8's 80 COCO class names,
so `car` matches both `car` and `race car`.
Case-insensitive — `Person` and `PERSON` both work.

### Architecture

```
Browser  ──POST /start──►  FastAPI  ──spawn──►  Background thread
                                                    │
                                           cv2.VideoCapture
                                                    │
                                           YOLOv8 inference
                                                    │
                                        annotated JPEG → shared buffer
                                                    ▲
Browser  ──GET /video_feed──►  MJPEG generator ────┘
```

## Test results

| # | Test | Result |
|---|------|--------|
| 1 | `/status` returns JSON with model info | ✅ |
| 2 | HTML page loads with required elements | ✅ |
| 3 | Bad source → graceful error in status | ✅ |
| 4 | Keywords normalised (trimmed, lowercased) | ✅ |
| 5 | Blank keywords → detect-all mode | ✅ |
| 6 | Conf clamped to [0.05, 0.95] | ✅ |
| 7 | `/stop` resets running state | ✅ |
| 8 | `/video_feed` returns `multipart/x-mixed-replace` with JPEG frames | ✅ |
| 9 | Inference on blank image completes without error | ✅ |
| 10 | Inference on bundled bus.jpg detects bus + persons | ✅ |
