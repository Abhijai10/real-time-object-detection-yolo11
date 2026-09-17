# System Architecture

This document describes the architecture of the **Real-Time Object Detection and
Scene Analytics Using YOLO11** application. Every statement here corresponds to
code that exists in the `app/` package; the diagram below is the rendered form of
[`diagrams/architecture.mmd`](diagrams/architecture.mmd).

---

## 1. Architectural style

The application is a **layered, modular command-line program** built around a
single control loop. It follows four principles:

1. **Separation of concerns.** Capture, detection, analytics, controls and
   rendering live in five separate modules, each with one reason to change.
2. **Dependency direction.** The controller (`app/cli.py`) depends on the
   modules; the modules never depend on the controller. `analytics.py` does not
   import `detector.py` at all — it accepts any object exposing `class_name` and
   `confidence` (a structural protocol), which is what makes the analytics layer
   independently testable.
3. **Lazy heavy imports.** `ultralytics`, `torch` and `cv2` are imported inside
   the functions that need them, not at module import time. Importing
   `app.analytics`, `app.controls` or `app.utils` therefore costs nothing and
   works on a machine without the model.
4. **Fail loudly at the boundary, gracefully inside.** User input is validated
   once, up front (`AppConfig.validate`); runtime failures are converted into
   messages the operator can act on.

---

## 2. Layer diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│  User at a terminal                                                  │
│      $ python -m app --camera 0 --conf 0.50                          │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────────────┐
│  PRESENTATION / ENTRY LAYER            app/__main__.py → app/cli.py  │
│  • build_parser()  argparse definition, --help, exit code 2 on error │
│  • parse_args()    namespace → AppConfig, then validate()            │
└───────────────────────────────┬──────────────────────────────────────┘
                                │ validated AppConfig
┌───────────────────────────────▼──────────────────────────────────────┐
│  APPLICATION LAYER                     app.cli.Application           │
│  Owns the control loop and the lifecycle of every other module.      │
│  Nothing else in the project knows the loop exists.                  │
└───┬───────────┬───────────┬────────────┬────────────┬───────────────┘
    │           │           │            │            │
    ▼           ▼           ▼            ▼            ▼
┌────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│ camera │ │ detector │ │analytics │ │ controls │ │ renderer │
│  .py   │ │   .py    │ │   .py    │ │   .py    │ │   .py    │
└────┬───┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘
     │          │            │            │            │
     │          ▼            │            │            │
     │   ┌──────────────┐    │            │            │
     │   │  YOLO11n     │    │            │            │
     │   │  Ultralytics │    │            │            │
     │   │ (3rd party)  │    │            │            │
     │   └──────┬───────┘    │            │            │
     │          │            │            │            │
     └──────────┴────────────┴────────────┴────────────┘
                          │
                          ▼
        ┌─────────────────────────────────────┐
        │  Annotated frame                    │
        ├──────────────────┬──────────────────┤
        │ OpenCV window    │ outputs/         │
        │ (live video)     │  snapshot_*.jpg  │
        │                  │  recording_*.mp4 │
        │                  │  session_summary │
        └──────────────────┴──────────────────┘
```

---

## 3. Module responsibilities

| Module | Public surface | Responsibility | Depends on |
| --- | --- | --- | --- |
| `app/utils.py` | `validate_*`, `save_snapshot`, `VideoRecorder`, `write_json`, `configure_logging` | Shared validation, logging setup, timestamped output paths, snapshot/recording/JSON persistence. | stdlib only (OpenCV imported lazily inside the two functions that need it) |
| `app/camera.py` | `Camera`, `CameraError`, `probe_camera`, `available_camera_indices` | Module 1. Webcam initialisation, resolution negotiation, frame capture, cleanup, and actionable error messages when a device cannot be opened. | `cv2` (defensive import) |
| `app/detector.py` | `Detector`, `Detection`, `filter_by_confidence`, `resolve_device` | Module 2. Loads the pretrained YOLO11 model, runs inference, converts raw Ultralytics results into `Detection` objects, applies the confidence threshold. | `ultralytics`, `torch` (lazy) |
| `app/analytics.py` | `FpsCounter`, `FrameAnalytics`, `SessionStats`, `compute_frame_analytics` | Module 3. Measured FPS, per-frame object/class counts, confidence statistics, and session-level aggregates. | `app.utils` only — no NumPy, no OpenCV |
| `app/controls.py` | `ControlAction`, `ControlState`, `interpret_key`, `adjust_confidence` | Module 4. Maps key codes to actions, owns the mutable runtime state (confidence, overlay, recording) and clamps the confidence threshold. | `app.utils` only |
| `app/renderer.py` | `Renderer`, `class_color` | Draws bounding boxes, labels, the analytics panel, the keyboard-help panel, the REC indicator and transient messages. | `cv2`, `numpy` |
| `app/cli.py` | `AppConfig`, `Application`, `build_parser`, `parse_args`, `main` | Argument parsing and validation, wiring, the real-time control loop, keyboard dispatch, teardown and reporting. | all of the above |

---

## 4. The control loop

`Application.run()` is the heart of the system. In pseudo-code:

```text
load model                      -> fail cleanly if weights cannot be loaded
open camera                     -> fail cleanly if no usable device
warm up the model               -> one dummy inference, errors are non-fatal
print the startup banner

while True:
    ok, frame = camera.read()
    if not ok:                  -> tolerate transient failures, stop after N in a row
        continue

    detections = detector.detect(frame)          # YOLO11 inference + confidence filter
    analytics  = compute_frame_analytics(detections)
    fps        = fps_counter.tick()              # measured, never hard-coded
    session.update(analytics, fps)

    renderer.render(frame, detections, analytics, fps, ...)   # boxes + overlay + help

    if recording:
        recorder.write(frame)

    key    = cv2.waitKey(1) if display else -1
    action = controls.apply(interpret_key(key))
    if not handle_action(action, frame):         # S, R, C, +/-, H, Q/ESC
        break

    if max_frames and session.frames_processed >= max_frames:
        break

finally:
    stop recording if active
    release the camera
    destroy the OpenCV window
    print the session summary
    write outputs/session_summary.json
```

The `finally` block guarantees that resources are released and the session
summary is written even if the loop exits because of an exception or Ctrl+C.

---

## 5. Data flow

```
  Camera.read()
        │  numpy.ndarray  (H, W, 3) uint8, BGR
        ▼
  Detector.detect(frame)
        │  list[Detection]   class_id, class_name, confidence, x1, y1, x2, y2
        ▼
  compute_frame_analytics(detections)
        │  FrameAnalytics    object_count, class_counts, unique_classes,
        │                    average_confidence, max_confidence
        ▼
  FpsCounter.tick()  ──────────────►  float fps
        │
        ▼
  SessionStats.update(analytics, fps)
        │  frames_processed, total_detections, peak_simultaneous_detections,
        │  class_totals, average_fps, ...
        ▼
  Renderer.render(...)
        │  annotated numpy.ndarray  (drawn in place)
        ├──────────────►  cv2.imshow()          live window
        ├──────────────►  VideoRecorder.write() recording file
        └──────────────►  save_snapshot()       JPEG on demand
```

Every arrow is a plain Python object. There is no global state and no singleton,
which is why the whole pipeline can be driven from a test with a fake detector.

---

## 6. Error-handling strategy

| Failure | Detected by | Behaviour |
| --- | --- | --- |
| Invalid `--conf`, `--camera`, `--width`, `--max-frames` | `argparse` type callables + `AppConfig.validate` | Usage message on stderr, **exit code 2**, no traceback |
| Missing local model file | `Detector.load()` | `ModelLoadError` with the path and the alternative, **exit code 1** |
| Ultralytics not installed | `Detector.load()` | `ModelLoadError` naming the exact `pip install` command |
| Webcam absent / busy / permission denied | `Camera.open()` | `CameraError` listing the four likely causes and suggesting `--camera 1` |
| Unsupported or missing video source | `Camera.open()` | `CameraError` naming the path and the supported extensions |
| Frame decoding stops mid-session | `Application.run()` | Up to 30 consecutive failures tolerated, then a logged error and clean teardown |
| Inference raises | `Detector.detect()` | `DetectionError`, logged, the frame is skipped and the loop continues |
| Snapshot cannot be written | `save_snapshot()` | `SnapshotError` shown as an on-screen message; the session continues |
| No usable video codec | `VideoRecorder.start()` | `RecordingError`, the recording flag is rolled back so the UI never lies |
| Unwritable output directory | `AppConfig.validate` / `ensure_directory` | Validation error before the camera is touched |
| Ctrl+C | `Application.run()` | `finally` teardown runs, **exit code 130** |

---

## 7. Testability

The architecture is designed so that everything except two things can be tested
without special hardware:

| Layer | Tested without a webcam? | Tested without the model? |
| --- | --- | --- |
| `utils` (validation, paths, JSON) | yes | yes |
| `utils` (snapshot, recorder) | yes — synthetic NumPy frames | yes |
| `analytics` | yes | yes |
| `controls` | yes | yes |
| `renderer` | yes — draws onto in-memory arrays | yes |
| `detector` (result conversion, device resolution) | yes | yes — fake Ultralytics result object |
| `detector` (real inference) | yes — synthetic frame | **no** — marked `slow` |
| `cli` (parsing, validation, action dispatch) | yes — `FakeDetector` double | yes |
| `camera` (real capture) | **no** — marked `display` | yes |

`pytest -q` therefore passes on any machine; the hardware- and
network-dependent tests are opt-in via `pytest -m slow` and `pytest -m display`.
