# Object Detection App

Real-time object detection from a webcam or RTSP stream using YOLOv8.

## Setup

```bash
pip install -r requirements.txt
python object_detection_app.py
```

## Usage

| Control | Description |
|---------|-------------|
| **Source** | `0`, `1`, … for local cameras; `rtsp://user:pass@host/stream` for IP cameras |
| **Keywords** | Comma-separated label substrings, e.g. `person, car, dog`. Leave blank to detect everything. |
| **Conf** | Confidence threshold slider (0.05 – 0.95). |
| **Start / Stop** | Begin or halt the stream. |

### Keyword matching

Keywords are matched as **substrings** against YOLOv8's COCO class names, so
`car` matches both `car` and `sportscar`, and `person` also matches `sportsperson`.
The 80 COCO classes include everyday objects: person, bicycle, car, motorcycle,
airplane, bus, train, truck, boat, traffic light, bird, cat, dog, horse, sheep,
cow, elephant, bear, backpack, umbrella, handbag, bottle, cup, fork, knife,
laptop, mouse, keyboard, phone, TV, clock, and more.

## Architecture

```
Main thread  →  Tkinter UI
Thread 1     →  cv2.VideoCapture grab loop  →  shared frame buffer
Thread 2     →  YOLOv8 inference  →  annotated frame  →  canvas update (via root.after)
```

The capture and detection threads are decoupled so a slow inference pass never
blocks the video grab, keeping buffer latency low for RTSP streams.
