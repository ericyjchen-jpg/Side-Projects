"""
Object Detection App
Accepts RTSP stream or built-in camera. User enters keyword labels;
YOLOv8 detects and highlights matching objects in real time.
"""

import threading
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO

# Colours for bounding boxes (cycle through these per label)
BOX_COLOURS = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
    (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
    (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
    (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
]


class ObjectDetectionApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Object Detection")
        self.root.resizable(True, True)

        self.model: YOLO | None = None
        self.cap: cv2.VideoCapture | None = None
        self.running = False
        self._frame_lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._capture_thread: threading.Thread | None = None
        self._detect_thread: threading.Thread | None = None

        self.keywords: list[str] = []  # lower-cased label substrings to match
        self.colour_map: dict[str, tuple[int, int, int]] = {}

        self._build_ui()
        self._load_model()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        # ---- top controls ----
        ctrl = ttk.Frame(self.root, padding=8)
        ctrl.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(ctrl, text="Source:").grid(row=0, column=0, sticky=tk.W)
        self.source_var = tk.StringVar(value="0")
        src_entry = ttk.Entry(ctrl, textvariable=self.source_var, width=36)
        src_entry.grid(row=0, column=1, padx=4)
        ttk.Label(ctrl, text="(camera index or rtsp://… URL)").grid(
            row=0, column=2, sticky=tk.W, padx=4
        )

        self.btn_start = ttk.Button(ctrl, text="Start", command=self._start)
        self.btn_start.grid(row=0, column=3, padx=4)
        self.btn_stop = ttk.Button(
            ctrl, text="Stop", command=self._stop, state=tk.DISABLED
        )
        self.btn_stop.grid(row=0, column=4, padx=4)

        # ---- keyword row ----
        kw_frame = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        kw_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(kw_frame, text="Keywords:").pack(side=tk.LEFT)
        self.kw_entry = ttk.Entry(kw_frame, width=40)
        self.kw_entry.insert(0, "person, car, dog")
        self.kw_entry.pack(side=tk.LEFT, padx=4)
        ttk.Button(kw_frame, text="Apply", command=self._apply_keywords).pack(
            side=tk.LEFT
        )
        ttk.Label(
            kw_frame, text="(comma-separated; leave blank = detect all)"
        ).pack(side=tk.LEFT, padx=6)

        # confidence slider
        ttk.Label(kw_frame, text="Conf:").pack(side=tk.LEFT, padx=(12, 0))
        self.conf_var = tk.DoubleVar(value=0.40)
        ttk.Scale(
            kw_frame, from_=0.05, to=0.95, variable=self.conf_var,
            orient=tk.HORIZONTAL, length=120,
        ).pack(side=tk.LEFT, padx=4)
        self.conf_label = ttk.Label(kw_frame, text="0.40")
        self.conf_label.pack(side=tk.LEFT)
        self.conf_var.trace_add("write", self._update_conf_label)

        # ---- active keywords display ----
        self.kw_display_var = tk.StringVar(value="Keywords: (all)")
        ttk.Label(self.root, textvariable=self.kw_display_var, foreground="blue").pack(
            side=tk.TOP, anchor=tk.W, padx=10
        )

        # ---- video canvas ----
        self.canvas = tk.Canvas(self.root, bg="black", width=800, height=480)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=8)

        # ---- status bar ----
        self.status_var = tk.StringVar(value="Ready — load a model first.")
        ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN).pack(
            side=tk.BOTTOM, fill=tk.X
        )

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _update_conf_label(self, *_):
        self.conf_label.config(text=f"{self.conf_var.get():.2f}")

    def _apply_keywords(self):
        raw = self.kw_entry.get()
        tokens = [t.strip().lower() for t in raw.split(",") if t.strip()]
        self.keywords = tokens
        if tokens:
            self.kw_display_var.set("Keywords: " + ", ".join(tokens))
            # assign stable colours
            for i, kw in enumerate(tokens):
                if kw not in self.colour_map:
                    self.colour_map[kw] = BOX_COLOURS[len(self.colour_map) % len(BOX_COLOURS)]
        else:
            self.kw_display_var.set("Keywords: (all — no filter)")

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self):
        self.status_var.set("Loading YOLOv8n model…")

        def _load():
            try:
                self.model = YOLO("yolov8n.pt")
                self.root.after(0, lambda: self.status_var.set(
                    "Model ready. Configure source and press Start."
                ))
            except Exception as exc:
                self.root.after(
                    0,
                    lambda: messagebox.showerror("Model Error", str(exc)),
                )

        threading.Thread(target=_load, daemon=True).start()

    # ------------------------------------------------------------------
    # Camera / stream control
    # ------------------------------------------------------------------

    def _start(self):
        if self.running:
            return
        if self.model is None:
            messagebox.showwarning("Not ready", "Model is still loading.")
            return

        source_raw = self.source_var.get().strip()
        # numeric index vs URL string
        source = int(source_raw) if source_raw.isdigit() else source_raw

        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", f"Cannot open source: {source_raw}")
            self.cap = None
            return

        # Optimise buffer for low latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._apply_keywords()
        self.running = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.status_var.set(f"Streaming from {source_raw} …")

        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._detect_thread = threading.Thread(target=self._detect_loop, daemon=True)
        self._capture_thread.start()
        self._detect_thread.start()

    def _stop(self):
        self.running = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.status_var.set("Stopped.")

    def _capture_loop(self):
        """Continuously grab frames into a shared buffer."""
        while self.running:
            if self.cap is None:
                break
            ret, frame = self.cap.read()
            if not ret:
                self.root.after(
                    0,
                    lambda: self.status_var.set("Stream ended or read error."),
                )
                break
            with self._frame_lock:
                self._latest_frame = frame
        if self.cap:
            self.cap.release()
            self.cap = None

    def _detect_loop(self):
        """Run inference on latest frame and push result to canvas."""
        while self.running:
            frame = None
            with self._frame_lock:
                if self._latest_frame is not None:
                    frame = self._latest_frame.copy()

            if frame is None:
                self.root.after(0, lambda: None)  # yield
                continue

            annotated = self._run_detection(frame)
            self._push_to_canvas(annotated)

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _label_matches(self, label: str) -> bool:
        """Return True if label matches any keyword (substring match)."""
        if not self.keywords:
            return True
        label_l = label.lower()
        return any(kw in label_l for kw in self.keywords)

    def _colour_for(self, label: str) -> tuple[int, int, int]:
        label_l = label.lower()
        for kw, col in self.colour_map.items():
            if kw in label_l:
                return col
        # default colour for unfiltered / "all" mode
        h = hash(label) % len(BOX_COLOURS)
        return BOX_COLOURS[h]

    def _run_detection(self, frame: np.ndarray) -> np.ndarray:
        conf_thresh = self.conf_var.get()
        results = self.model.predict(
            frame, conf=conf_thresh, verbose=False, stream=False
        )
        annotated = frame.copy()

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]
                if not self._label_matches(label):
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                colour = self._colour_for(label)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)
                text = f"{label} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
                )
                cv2.rectangle(
                    annotated,
                    (x1, y1 - th - 8),
                    (x1 + tw + 4, y1),
                    colour,
                    -1,
                )
                cv2.putText(
                    annotated,
                    text,
                    (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
        return annotated

    def _push_to_canvas(self, frame: np.ndarray):
        """Convert OpenCV BGR frame to Tk image and display on canvas."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize to fit canvas
        cw = self.canvas.winfo_width() or 800
        ch = self.canvas.winfo_height() or 480
        h, w = rgb.shape[:2]
        scale = min(cw / w, ch / h, 1.0)
        if scale < 1.0:
            nw, nh = int(w * scale), int(h * scale)
            rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)

        img = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.root.after(0, self._update_canvas, img)

    def _update_canvas(self, img: ImageTk.PhotoImage):
        self.canvas.delete("all")
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        self.canvas.create_image(cw // 2, ch // 2, anchor=tk.CENTER, image=img)
        # keep a reference so GC doesn't collect it
        self.canvas._img = img  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _on_close(self):
        self._stop()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
