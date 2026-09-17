# Project Workflow

This document walks through what actually happens between launching the
application and shutting it down. It is the textual companion to
[`diagrams/workflow.mmd`](diagrams/workflow.mmd) and
[`diagrams/sequence.mmd`](diagrams/sequence.mmd).

---

## 1. End-to-end flow

```
START
  │
  ├─ 1. Parse CLI arguments                     app/cli.py :: parse_args()
  │       • argparse converts and range-checks every value
  │       • AppConfig.validate() re-checks cross-field rules
  │       • invalid input → usage message, exit code 2
  │
  ├─ 2. Load the YOLO11n model                  app/detector.py :: Detector.load()
  │       • local .pt path is checked for existence
  │       • a bare official name (yolo11n.pt) is handed to Ultralytics,
  │         which downloads the weights on first use
  │       • failure → clear message, exit code 1
  │
  ├─ 3. Open the webcam                         app/camera.py :: Camera.open()
  │       • platform-appropriate backend is chosen
  │       • requested width/height are applied (the driver may ignore them)
  │       • failure → actionable message, exit code 1
  │
  ├─ 4. Warm up the model                       app/detector.py :: Detector.warmup()
  │       • one dummy inference so the first real frame is not slow
  │       • a warm-up failure is logged but never fatal
  │
  ├─ 5. Print the startup banner
  │       • model, device, confidence, IoU, inference size, capture source,
  │         resolution, output directory, keyboard controls
  │
  └─ 6. ENTER THE REAL-TIME LOOP ───────────────────────────────────────┐
                                                                        │
        ┌───────────────────────────────────────────────────────────────┘
        │
        ├─ 6.1  Capture a frame                Camera.read()
        │
        ├─ 6.2  Prepare the frame              (resizing/letterboxing is
        │                                       handled inside Ultralytics)
        │
        ├─ 6.3  YOLO inference                 Detector.detect()
        │         Input  → preprocessing → neural-network inference
        │         → bounding-box predictions → confidence filtering
        │         → non-maximum suppression (IoU) → list[Detection]
        │
        ├─ 6.4  Apply the confidence threshold (already applied by the model;
        │         the runtime +/- keys change it and take effect next frame)
        │
        ├─ 6.5  Update analytics
        │         compute_frame_analytics(detections) → FrameAnalytics
        │         FpsCounter.tick()                   → measured FPS
        │         SessionStats.update(...)            → session aggregates
        │
        ├─ 6.6  Draw                           Renderer.render()
        │         bounding boxes + labels + confidence values
        │         analytics panel (objects, classes, FPS, confidence,
        │         resolution, per-class counts, session counters)
        │         keyboard-help panel
        │         REC indicator if recording
        │         transient status message if one is active
        │
        ├─ 6.7  Persist the frame if recording  VideoRecorder.write()
        │
        ├─ 6.8  Display                        cv2.imshow() → OpenCV window
        │
        ├─ 6.9  Read the keyboard              cv2.waitKey(1)
        │         interpreted by app/controls.py :: interpret_key()
        │         applied to ControlState, returned as a ControlAction
        │
        └─ 6.10 Continue?  ───── YES ────────────────────────────────────┐
                │                                                        │
                │                                                        │
                NO (Q / ESC, Ctrl+C, --max-frames reached,               │
                │   stream lost)                                         │
                │                                                        │
                ▼                                                        │
  ┌──────────────────────────────────────────────────────────────────────┘
  │
  ├─ 7. Teardown (always runs, in a `finally` block)
  │       • stop the recording if one is active
  │       • release the camera
  │       • destroy the OpenCV window
  │
  ├─ 8. Report
  │       • print the session summary to the terminal
  │       • write outputs/session_summary.json
  │
  END  (exit code 0, or 130 after Ctrl+C, or 1 after a handled failure)
```

---

## 2. The real-time loop in detail

### 2.1 Capture

`Camera.read()` returns a `(ok, frame)` tuple rather than raising, so the loop can
distinguish *"this frame failed"* from *"the device is gone"*. Up to
`MAX_READ_FAILURES = 30` consecutive failures are tolerated, because USB webcams
and virtual cameras do occasionally drop a frame. Beyond that the loop stops with
a logged error and a clean teardown. For a file source, the end of the stream
ends the loop immediately.

### 2.2 Detection

`Detector.detect()` calls `model.predict()` once per frame with the current
confidence, IoU, image size, device and `max_det` settings, then converts the
result into a list of `Detection` dataclasses sorted by descending confidence.
The conversion is the only place in the project that touches the Ultralytics
result API; everything downstream works with plain Python objects.

### 2.3 Analytics

Three independent computations happen per frame:

| Computation | Where | Output |
| --- | --- | --- |
| Scene analytics | `compute_frame_analytics(detections)` | `FrameAnalytics` — object count, class counts, unique classes, mean and max confidence |
| Frame rate | `FpsCounter.tick()` | Rolling mean of the last 30 inter-frame intervals, expressed in FPS |
| Session aggregates | `SessionStats.update(analytics, fps)` | Cumulative counters, peak simultaneous detections, per-class totals, FPS samples |

The FPS value is **measured from real timestamps**. There is no hard-coded or
simulated value anywhere in the codebase.

### 2.4 Rendering

`Renderer.render()` draws the layers in a fixed order so that later layers are
never hidden by earlier ones:

1. bounding boxes and labels,
2. the analytics panel (top-left),
3. the keyboard-help panel (bottom-left),
4. the REC indicator (top-right),
5. the transient status message (bottom-centre).

Translucent panels are produced by blending a filled rectangle into a copy of the
region, which keeps the text readable over any background.

### 2.5 Keyboard dispatch

`interpret_key()` maps a key code to a `ControlAction`; `ControlState.apply()`
mutates the runtime state and returns the action; `Application.handle_action()`
performs the side effect. Splitting the three means the key mapping and the state
transitions can be unit tested without a window, which is exactly what
`tests/test_controls.py` does.

| Key | Action | Effect |
| --- | --- | --- |
| `Q` / `ESC` | `QUIT` | Leave the loop and tear down |
| `S` | `SNAPSHOT` | Write `outputs/snapshot_<timestamp>.jpg`, print the path |
| `R` | `TOGGLE_RECORDING` | Start or stop `outputs/recording_<timestamp>.mp4` |
| `C` | `RESET_STATS` | Zero the session counters (confidence and recording are unaffected) |
| `+` / `=` | `CONFIDENCE_UP` | Raise the threshold by 0.05, clamped to the allowed range |
| `-` / `_` | `CONFIDENCE_DOWN` | Lower the threshold by 0.05, clamped |
| `H` | `TOGGLE_OVERLAY` | Show or hide the analytics panel |

---

## 3. Outputs produced

| File | Produced when | Contents |
| --- | --- | --- |
| `outputs/snapshot_YYYY_MM_DD_HHMMSS.jpg` | `S` is pressed | The fully annotated frame, including boxes and overlay |
| `outputs/recording_YYYY_MM_DD_HHMMSS.mp4` | `R` starts and then stops a recording | The annotated video stream at the measured FPS |
| `outputs/session_summary.json` | At exit, unless `--no-summary` | Frames processed, total detections, peak simultaneous detections, average/peak FPS, average confidence, most frequent class, per-class totals, duration |

---

## 4. Modes of operation

| Mode | Command | Use |
| --- | --- | --- |
| Default live mode | `python -m app` | Normal use: camera 0, window shown, keyboard active |
| Custom camera and threshold | `python -m app --camera 1 --conf 0.40` | Second camera, stricter filtering |
| CPU only | `python -m app --device cpu` | Machines without CUDA or MPS |
| Smaller/faster inference | `python -m app --imgsz 416` | Low-power laptops; trades accuracy for speed |
| Camera discovery | `python -m app --list-cameras` | Find which device indices work |
| Headless smoke run | `python -m app --no-display --max-frames 60` | Servers, CI and automated verification; no window is created |

Headless mode exists because the full pipeline — capture, inference, analytics,
recording and summary writing — is meaningful without a display, and because it
allows the project to be verified automatically on a machine with no screen.
