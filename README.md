# Real-Time Object Detection and Scene Analytics Using YOLO11

A command-line Computer Vision application that opens your webcam, detects objects
on every frame with the pretrained **YOLO11n** model, draws bounding boxes and
labels, and overlays live scene analytics (FPS, object counts, per-class counts,
confidence) in an OpenCV window.

No notebook, no IDE, no web framework, no browser. One command:

```bash
python -m app
```

---

## Table of contents

1. [Project overview](#1-project-overview)
2. [Problem statement](#2-problem-statement)
3. [Objectives](#3-objectives)
4. [Functional requirements](#4-functional-requirements)
5. [Non-functional requirements](#5-non-functional-requirements)
6. [Features](#6-features)
7. [Computer Vision concepts used](#7-computer-vision-concepts-used)
8. [The YOLO11 model](#8-the-yolo11-model)
9. [Why YOLO11n was selected](#9-why-yolo11n-was-selected)
10. [Dataset](#10-dataset)
11. [Technology stack](#11-technology-stack)
12. [System architecture](#12-system-architecture)
13. [Project workflow](#13-project-workflow)
14. [Folder structure](#14-folder-structure)
15. [Requirements](#15-requirements)
16. [Installation](#16-installation)
17. [Virtual environment setup](#17-virtual-environment-setup)
18. [Dependency installation](#18-dependency-installation)
19. [Model setup](#19-model-setup)
20. [How to run the project](#20-how-to-run-the-project)
21. [CLI options](#21-cli-options)
22. [Keyboard controls](#22-keyboard-controls)
23. [Output files](#23-output-files)
24. [Testing](#24-testing)
25. [Expected behaviour](#25-expected-behaviour)
26. [Project originality](#26-project-originality)
27. [Limitations](#27-limitations)
28. [Future enhancements](#28-future-enhancements)
29. [Troubleshooting](#29-troubleshooting)
30. [References](#30-references)
31. [Third-party attribution and licensing](#31-third-party-attribution-and-licensing)

---

## 1. Project overview

This project is a complete, self-contained Computer Vision practical. It performs
**real-time object detection on a live webcam feed** and enriches the raw
detections with **scene analytics** so that the output is genuinely useful rather
than a plain bounding-box demo.

The application is organised into four functional modules:

| Module | Responsibility |
| --- | --- |
| **1. Camera capture** | Open the webcam, negotiate resolution, capture frames continuously, handle device errors. |
| **2. Real-time object detection** | Load pretrained YOLO11n, run inference on every frame, extract bounding boxes, class names and confidence scores. |
| **3. Real-time scene analytics** | Measure FPS, count objects, group them by class, track session statistics. |
| **4. User controls and capture** | Keyboard controls for quit, snapshot, recording, statistics reset and confidence adjustment. |

Everything runs from the terminal. The only window is the OpenCV camera window
itself, which is part of the Computer Vision application.

**Important distinction:** the YOLO11 neural network is a **third-party pretrained
model** provided by Ultralytics. It was not designed, trained or invented by this
project. What this project provides is the **application layer** built around it:
the capture loop, the analytics, the rendering, the controls, the CLI, the error
handling, the persistence and the tests. See
[Project originality](#26-project-originality).

---

## 2. Problem statement

Interpreting a live video stream — detecting *what* is present, *where* it is and
*how many* instances are visible, continuously and within a practical time budget
— underpins surveillance, traffic monitoring, retail analytics, assistive
technology and industrial safety systems.

Designing and training a detector from scratch is not feasible within a
university practical: it requires large labelled datasets and substantial compute.
The realistic and academically valuable task is therefore to **integrate a modern
pretrained detector into a complete, well-engineered application** and to build
the surrounding engineering that turns a model into a usable tool: capture,
filtering, analytics, visualisation, interaction, persistence and robust failure
handling.

A further practical constraint shapes the design: the application must run on an
ordinary laptop CPU, without a GPU, and must be launchable with a single terminal
command.

---

## 3. Objectives

1. Build a modular Python application that performs object detection on a live
   webcam stream using a pretrained YOLO11 model.
2. Keep the architecture cleanly separated: capture, detection, analytics,
   controls, rendering and CLI each in their own module.
3. Provide a friendly, fully documented command-line interface that works with a
   single command and requires no GUI framework.
4. Add a meaningful analytics layer — measured FPS, object counts, per-class
   counts, confidence statistics — so the project goes beyond a minimal detector
   demo.
5. Support runtime interaction: quit, snapshot, recording, statistics reset and
   live confidence adjustment.
6. Handle normal user and hardware errors with clear, actionable messages instead
   of unexplained tracebacks.
7. Document the underlying Computer Vision concepts accurately, and clearly
   separate third-party model capabilities from what this project measures.
8. Deliver a test suite that runs on any machine, including machines with no
   webcam and no network access.
9. Produce a repository that is clean, documented and ready to publish publicly.

---

## 4. Functional requirements

| ID | Requirement |
| --- | --- |
| FR1 | The application must open a webcam at a configurable device index and capture frames continuously. |
| FR2 | It must detect whether a usable camera is available and report a clear error if not. |
| FR3 | It must load the pretrained YOLO11n model, downloading the weights automatically when they are not present locally. |
| FR4 | It must allow the user to supply a local model path instead of the default checkpoint. |
| FR5 | It must run object detection on every captured frame using the Ultralytics Python API. |
| FR6 | It must obtain bounding boxes, class IDs, class names and confidence scores for each detection. |
| FR7 | It must apply a configurable confidence threshold, adjustable at runtime. |
| FR8 | It must draw bounding boxes, class labels and confidence values on the frame. |
| FR9 | It must display the current FPS, the number of objects, the unique classes and the per-class counts. |
| FR10 | It must display the active confidence threshold and the camera resolution. |
| FR11 | It must show the processed video in an OpenCV window. |
| FR12 | It must support keyboard controls: `Q`/`ESC` quit, `S` snapshot, `R` recording toggle, `C` reset statistics, `+`/`-` confidence, `H` overlay toggle. |
| FR13 | It must save snapshots as timestamped images in the output directory. |
| FR14 | It must record the annotated stream to a timestamped video file when recording is enabled. |
| FR15 | It must maintain session statistics and write them to `outputs/session_summary.json` at exit. |
| FR16 | It must expose all options through an `argparse` CLI with `--help` and copy-pasteable examples. |
| FR17 | It must validate file paths, confidence values, camera indices and resolutions, and report problems clearly. |
| FR18 | It must support CPU execution and use a GPU automatically when one is available. |
| FR19 | It must allow the user to force a specific device (`--device cpu`, `--device 0`). |
| FR20 | It must run without any GUI application, notebook or web browser. |

---

## 5. Non-functional requirements

### NFR1 — Performance

The application must sustain interactive frame rates on ordinary hardware and must
never report a fabricated figure.

*FPS is measured from real monotonic timestamps* in `app/analytics.py`
(`FpsCounter` keeps a rolling window of the last 30 inter-frame intervals). The
default model is the smallest YOLO11 variant; the inference resolution is
configurable (`--imgsz`), and the device is selected automatically or forced by
the user.

### NFR2 — Usability

A new user must be able to start the project with one command and understand what
is happening.

`python -m app` works with zero arguments. A startup banner lists the model,
device, thresholds, capture source, resolution and output directory. All keyboard
controls are drawn on the video and repeated in the terminal banner, in `--help`,
in the README and in `examples/commands.md`.

### NFR3 — Reliability

A single bad frame, a dropped frame or a failed snapshot must not terminate the
session.

Up to 30 consecutive frame-read failures are tolerated before the loop stops. A
failed snapshot is reported on screen and the session continues. A failed
inference logs the error and skips the frame. A failed recording start rolls back
the recording flag so the interface never claims to be recording when it is not.
All resources are released in a `finally` block, which also guarantees that the
session summary is written.

### NFR4 — Maintainability

The codebase must be easy to read, extend and verify.

Seven focused modules, one responsibility each; type hints on every public
function; docstrings on every public class and function; a single logging
configuration; and a test suite of 240 tests across seven test files. The
layering is strict — `analytics.py` does not import `detector.py`, and no module
imports the controller — so any layer can be replaced or tested in isolation.

### NFR5 — Error handling

Normal user mistakes must never produce an unexplained traceback.

Invalid arguments produce an argparse usage message and exit code 2. A missing
camera produces a message listing the four likely causes. A missing model file
names the path and the alternative. An unusable output directory is reported
before the camera is touched. Ctrl+C is handled gracefully and still writes the
session summary.

### NFR6 — Resource efficiency

The application must be usable on a laptop without a GPU and must not leak
resources.

YOLO11n is deliberately the smallest model in the family. The camera, the video
writer and the OpenCV window are always released during teardown, including on
error paths. Heavy imports (`ultralytics`, `torch`) are performed lazily, so
starting the CLI, printing help or running the unit tests costs almost nothing.

---

## 6. Features

* **One-command startup** — `python -m app`.
* **Pretrained YOLO11n detection** — 80 COCO object classes, CPU-friendly, weights
  downloaded automatically on first use.
* **Configurable capture** — camera index, requested width/height, inference
  resolution, compute device.
* **Runtime confidence control** — raise or lower the threshold while running and
  see the effect immediately.
* **Live scene analytics overlay** — objects, unique classes, per-class counts,
  average confidence, measured FPS, active threshold, camera resolution, session
  counters.
* **Bounding boxes, labels and confidence values** drawn on every detection.
* **Snapshot capture** — one keypress writes a fully annotated, timestamped JPEG.
* **Session recording** — one keypress toggles recording of the annotated stream.
* **Session statistics** — frames processed, total detections, peak simultaneous
  detections, most frequent class, average and peak FPS, written to JSON at exit.
* **Camera discovery** — `--list-cameras` probes device indices and reports which
  ones work.
* **Headless mode** — `--no-display --max-frames N` runs the full pipeline
  without a window, for servers and automated verification.
* **Graceful failure everywhere** — validated arguments, actionable messages, no
  tracebacks for normal mistakes.

---

## 7. Computer Vision concepts used

### Computer Vision

The field concerned with extracting information about the world from images and
video. This project addresses the *detection* sub-problem: locating and naming
multiple objects within a single frame.

### Real-time video processing

A video is a sequence of still frames. Processing it "in real time" means the
per-frame pipeline (capture → infer → analyse → draw → display) completes faster
than the frame interval, so the displayed stream keeps up with reality. Whether
that holds depends on the machine, the model and the resolution — which is why
this project *measures* FPS rather than assuming it.

### Object detection

Producing, for one image, a set of *bounding boxes* each carrying a *class label*
and a *confidence score*. It is more informative than image classification
(one label for the whole image) and than localisation (one box only), because it
answers "what, and where, and how many".

### Bounding boxes

Rectangles that enclose a detected object, stored as corner coordinates. This
project uses the `(x1, y1, x2, y2)` convention: top-left and bottom-right in pixel
coordinates, with the origin at the top-left of the frame. See
`Detection.bbox_int` in `app/detector.py`.

### Object classes

The categories the model can recognise. YOLO11 detection models pretrained on
COCO recognise 80 classes, including `person`, `bicycle`, `car`, `bottle`,
`chair`, `laptop` and `cell phone`.

### Confidence scores

Each detection carries a value in `[0, 1]` expressing how strongly the model
believes the box contains an object of the predicted class. Higher is more
certain.

### Confidence thresholding

Discarding detections below a threshold. It is the single most useful runtime
knob: raising it removes false positives but can miss genuine objects; lowering it
does the opposite. This project exposes it as `--conf` and as the `+`/`-` keys,
and passes it directly to the model, which applies the filter internally.

### IoU (Intersection over Union)

The area of overlap between two boxes divided by the area of their union. It is
the standard measure of how well a predicted box matches a ground-truth box, and
it is also the criterion used to decide whether two predicted boxes refer to the
same object.

```
IoU = area(A ∩ B) / area(A ∪ B)
```

IoU = 1 means perfect overlap; IoU = 0 means no overlap at all.

### Non-maximum suppression (NMS)

A detector proposes many overlapping boxes for the same object. NMS sorts them by
confidence, keeps the best one, and discards every remaining box whose IoU with a
kept box exceeds a threshold. This is the **IoU threshold** exposed as `--iou`
(default `0.45`): lower values suppress more aggressively. NMS runs inside the
Ultralytics implementation; this project configures it and documents it.

### Neural-network inference

A single forward pass of the image through the trained network, producing raw
predictions. This project calls it via `model.predict()` and does not implement
any part of the network.

### FPS (frames per second)

The number of frames processed per second. This project measures it from real
timestamps and displays it live. It is a **runtime measurement of this
application**, not a property of the model.

### Precision, recall and mAP

Standard detection *evaluation* metrics:

| Metric | Question it answers |
| --- | --- |
| **Precision** | Of the objects I reported, what fraction were real? |
| **Recall** | Of the real objects present, what fraction did I find? |
| **mAP@50** | Mean average precision, counting a detection as correct when IoU ≥ 0.50. |
| **mAP@50-95** | The same, averaged over IoU thresholds from 0.50 to 0.95 in steps of 0.05. The stricter, more standard figure. |

**These metrics are not measured by this project.** Computing them requires a
labelled evaluation dataset with ground-truth boxes, which this repository does
not ship. They are explained here for completeness, and the values published by
Ultralytics for YOLO11n on the COCO validation set can be found on the official
model page (see [References](#30-references)). This application reports only what
it can genuinely measure: runtime FPS and per-frame detection statistics.

### Preprocessing

Before inference the frame is resized/letterboxed and normalised. Ultralytics
performs this internally; this project supplies the raw BGR frame and the
requested inference size (`--imgsz`).

---

## 8. The YOLO11 model

YOLO ("You Only Look Once") is a family of single-stage object detectors. A
single-stage detector predicts bounding boxes and class probabilities in **one
forward pass** of the network, in contrast to two-stage detectors that first
propose regions and then classify them. This is what makes YOLO-style models
suitable for real-time use.

YOLO11 is a recent generation of the YOLO family released by Ultralytics. It is
distributed with several detection variants of different sizes — commonly named
`n`, `s`, `m`, `l`, `x` — where the letter denotes the model scale.

The pretrained detection checkpoints are trained to recognise the 80 object
classes of the COCO dataset. When the application starts, it loads one of these
checkpoints and uses it as a **fixed, pretrained detector**: no training or
fine-tuning happens anywhere in this project.

The detection pipeline, conceptually:

```
Input frame
   → preprocessing (resize / letterbox, normalise)
   → neural-network inference (one forward pass)
   → bounding-box predictions with class scores
   → confidence filtering
   → non-maximum suppression (IoU threshold)
   → annotated output (boxes, labels, confidence)
   → analytics (counts, classes, FPS)
```

Steps 1–5 are performed by the third-party Ultralytics model. Steps 6 and 7 —
the annotation and the analytics — are implemented by this project.

---

## 9. Why YOLO11n was selected

**Why YOLO11 (the family).** Single-stage detectors offer the best accuracy-per-
millisecond trade-off for live video, the Ultralytics implementation is mature and
well documented, it exposes a clean Python API, and it handles preprocessing, NMS
and device placement internally so the application code can stay focused on the
application logic.

**Why the `n` (nano) variant specifically:**

| Consideration | Reason `n` fits this project |
| --- | --- |
| **Lightweight architecture** | The smallest variant, so it loads quickly and fits comfortably in memory. |
| **Inference speed** | The fastest variant, which matters because inference runs on *every* frame. |
| **CPU / GPU requirements** | Runs acceptably on an ordinary laptop CPU, so the project does not require a GPU to be demonstrated. A GPU is still used automatically when present. |
| **Pretrained weights available** | Official COCO-pretrained weights are published and downloaded automatically, so no training step is required. |
| **Suitability for webcam detection** | The webcam is a low-resolution, real-time source where per-frame latency dominates; a small, fast model is the right choice. |
| **Ease of integration** | One line to load (`YOLO("yolo11n.pt")`) and one call to infer, which keeps the student-authored code readable. |

**Trade-off.** Model scale trades accuracy for speed and size. A larger variant
would generally detect more objects, more reliably, at a lower frame rate and a
higher memory cost. `n` is the deliberate choice for a CPU-friendly, real-time
application; the `--model` option makes it trivial to try a larger checkpoint
(`yolo11s.pt`, `yolo11m.pt`, …) on a more capable machine.

**Acknowledgement.** Newer Ultralytics model families exist beyond YOLO11. This
project deliberately uses YOLO11n as a stable, lightweight, well-documented
choice, and does not claim it is the newest or the most accurate model available.

**No performance claim is made.** No FPS figure or accuracy figure for YOLO11n is
asserted by this project. Actual frame rate depends on the CPU/GPU, camera
resolution, model size, inference settings, operating system and background load.
The application measures and displays the real value for the machine it is running
on.

---

## 10. Dataset

This project **does not create, collect, annotate or train on any dataset**, and
it ships no image or video data.

The pretrained YOLO11n checkpoint used here was trained by Ultralytics on the
**COCO (Common Objects in Context)** dataset, which defines 80 object categories.
Those weights and that training procedure are third-party work; this project
simply consumes the published checkpoint.

The **runtime input** is the live webcam feed. In that sense the webcam *is* the
input source of this application — but it is an input stream, not a training or
evaluation dataset.

The unit tests use small **synthetic NumPy arrays** generated at runtime
(`tests/conftest.py`), so no third-party imagery is redistributed and no
copyrighted dataset is committed to the repository.

---

## 11. Technology stack

| Component | Role |
| --- | --- |
| **Python 3.10+** | Implementation language. |
| **Ultralytics** | Provides the pretrained YOLO11 model and the inference API. |
| **YOLO11n** | The specific pretrained detection checkpoint used by default. |
| **PyTorch** | The tensor/deep-learning backend pulled in by Ultralytics; performs inference. |
| **OpenCV (`opencv-python`)** | Webcam capture, drawing, display, video writing, image writing. |
| **NumPy** | Frame buffers and synthetic test images. |
| **pytest** | Unit test framework. |
| **pathlib / argparse / logging / dataclasses / enum** | Standard-library building blocks used throughout. |

There is **no** Flask, no Streamlit, no Django, no Jupyter, no database and no
JavaScript in this project.

---

## 12. System architecture

```
User
 │  $ python -m app --camera 0 --conf 0.50
 ▼
Command Line Interface                app/cli.py :: build_parser / parse_args
 │  validated AppConfig
 ▼
Application Controller                app.cli.Application  (the real-time loop)
 ├── Camera Module                    app/camera.py     :: Camera
 ├── Detection Module                 app/detector.py   :: Detector
 ├── Analytics Module                 app/analytics.py  :: FpsCounter, SessionStats
 ├── Controls Module                  app/controls.py   :: ControlState
 └── Rendering Module                 app/renderer.py   :: Renderer
 ▼
YOLO11n  (Ultralytics + PyTorch — third-party)
 ▼
Detection results  (boxes, class names, confidence scores)
 ▼
Analytics / annotated frame
 ├── OpenCV window (live video)
 └── outputs/  (snapshot_*.jpg, recording_*.mp4, session_summary.json)
```

Full design documentation, including UML-style use-case, sequence and component
diagrams, is in [`docs/`](docs/):

* [`docs/architecture.md`](docs/architecture.md) — layering, module
  responsibilities, data flow, error-handling strategy, testability.
* [`docs/workflow.md`](docs/workflow.md) — the end-to-end flow and the real-time
  loop in detail.
* [`docs/diagrams/`](docs/diagrams/) — Mermaid sources for the architecture,
  workflow, use-case, sequence and component diagrams.

---

## 13. Project workflow

```
START
 → Parse CLI arguments (validate; exit 2 on bad input)
 → Load YOLO11n model (download weights on first use)
 → Initialise webcam (exit 1 with a clear message if unavailable)
 → Warm up the model
 → Print the startup banner
 → ┌─ REAL-TIME LOOP ────────────────────────────────────────┐
   │  Capture frame                                          │
   │  → Preprocess / prepare the frame                       │
   │  → YOLO inference                                       │
   │  → Extract detections                                   │
   │  → Apply confidence threshold                           │
   │  → Update analytics (FPS, counts, session)              │
   │  → Draw bounding boxes and information                  │
   │  → Display the frame                                    │
   │  → Read keyboard input                                  │
   │  → Continue?  ── YES ──► back to "Capture frame"         │
   └────────────────── NO ────────────────────────────────────┘
 → Release camera, stop recording, destroy window
 → Save session statistics (outputs/session_summary.json)
END
```

---

## 14. Folder structure

```
real-time-object-detection-yolo11/
│
├── README.md                  ← this file
├── statement.md               ← problem statement, scope, users, features
├── VALIDATION.md              ← what was actually run and verified
├── requirements.txt           ← pip dependencies
├── pyproject.toml             ← package metadata + pytest configuration
├── .gitignore
├── LICENSE                    ← MIT (plus a third-party licensing notice)
│
├── app/                       ← application package
│   ├── __init__.py            ← package docstring and version
│   ├── __main__.py            ← enables `python -m app`
│   ├── cli.py                 ← argument parsing, validation, Application controller
│   ├── camera.py              ← Module 1: webcam capture and validation
│   ├── detector.py            ← Module 2: YOLO11 loading and inference
│   ├── analytics.py           ← Module 3: FPS, frame analytics, session statistics
│   ├── controls.py            ← Module 4: keyboard handling and runtime state
│   ├── renderer.py            ← drawing of boxes, labels and overlays
│   └── utils.py               ← logging, validation, output handling
│
├── tests/                     ← pytest suite (no hardware required by default)
│   ├── conftest.py            ← shared fixtures and the FakeDetector double
│   ├── test_utils.py
│   ├── test_analytics.py
│   ├── test_controls.py
│   ├── test_camera.py
│   ├── test_renderer.py
│   ├── test_detector.py
│   └── test_cli.py
│
├── outputs/                   ← generated at runtime (kept via .gitkeep)
│   └── .gitkeep
│
├── docs/
│   ├── architecture.md
│   ├── workflow.md
│   └── diagrams/
│       ├── architecture.mmd
│       ├── workflow.mmd
│       ├── use_case.mmd
│       ├── sequence.mmd
│       └── component.mmd
│
└── examples/
    └── commands.md            ← copy-pasteable command reference
```

---

## 15. Requirements

| Item | Requirement |
| --- | --- |
| **Python** | 3.10 or newer (tested with 3.13). |
| **Operating system** | Windows, macOS or Linux. |
| **Webcam** | Any camera OpenCV can open (built-in or USB). |
| **GPU** | Optional. The application runs on CPU and uses CUDA or Apple Silicon MPS automatically when available. |
| **Disk** | ~2 GB for PyTorch and its dependencies, plus ~5 MB for the model weights. |
| **Network** | Required only the first time, to download the YOLO11n weights. |
| **Permissions** | Camera access must be granted to the terminal application. |

---

## 16. Installation

```bash
git clone https://github.com/Abhijai10/real-time-object-detection-yolo11.git
cd real-time-object-detection-yolo11
```

Then follow sections [17](#17-virtual-environment-setup) and
[18](#18-dependency-installation).

---

## 17. Virtual environment setup

A virtual environment keeps the project's dependencies isolated from the rest of
your system. It is strongly recommended.

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Windows (cmd.exe)

```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

When the environment is active, your prompt is prefixed with `(.venv)`.

---

## 18. Dependency installation

```bash
pip install -r requirements.txt
```

This installs Ultralytics (which pulls in PyTorch), OpenCV, NumPy and pytest. The
download is large — mostly PyTorch — so allow a few minutes on the first run.

To verify the installation:

```bash
python -c "import cv2, ultralytics, numpy; print(cv2.__version__, ultralytics.__version__, numpy.__version__)"
```

Optional, for installing the project as a command:

```bash
pip install -e .
```

---

## 19. Model setup

**No manual step is required.** The first time the application runs, Ultralytics
downloads the pretrained `yolo11n.pt` checkpoint automatically (~5 MB) into its
local cache. Subsequent runs reuse the cached file and work offline.

```
Downloading https://github.com/ultralytics/assets/releases/download/.../yolo11n.pt to 'yolo11n.pt'...
```

If you already have the weights, or want to use a different checkpoint, pass the
path explicitly:

```bash
python -m app --model models/yolo11n.pt
python -m app --model yolo11s.pt          # a larger official checkpoint
```

**The weights are deliberately not committed to this repository.** They are large
binaries that Ultralytics distributes and can be re-downloaded at any time, so
keeping them out of Git is the correct practice. `models/` is not part of this
project's structure for that reason.

---

## 20. How to run the project

```bash
python -m app
```

That is the whole thing: camera 0, `yolo11n.pt`, confidence 0.25, window shown,
keyboard controls active.

Common variations:

```bash
# Choose a camera and a stricter confidence threshold
python -m app --camera 0 --conf 0.50

# Force CPU inference
python -m app --camera 0 --conf 0.40 --device cpu

# Request a higher capture resolution
python -m app --width 1280 --height 720

# Find which camera indices work on this machine
python -m app --list-cameras

# Run 60 frames without opening a window (servers, automated checks)
python -m app --no-display --max-frames 60
```

More examples: [`examples/commands.md`](examples/commands.md).

---

## 21. CLI options

| Option | Default | Description |
| --- | --- | --- |
| `--camera INDEX` | `0` | Webcam device index. A path to a video file is also accepted (headless development). |
| `--model PATH` | `yolo11n.pt` | Official checkpoint name or path to a local `.pt` file. |
| `--conf FLOAT` | `0.25` | Confidence threshold, between `0.01` and `1.0`. |
| `--iou FLOAT` | `0.45` | IoU threshold used by non-maximum suppression. |
| `--imgsz PIXELS` | `640` | Inference resolution. Lower is faster, higher is finer. |
| `--device DEVICE` | `auto` | `auto`, `cpu`, `mps`, or a CUDA index such as `0`. |
| `--width PIXELS` | driver default | Requested capture width. Must be given with `--height`. |
| `--height PIXELS` | driver default | Requested capture height. Must be given with `--width`. |
| `--output DIR` | `outputs` | Directory for snapshots, recordings and the session summary. |
| `--max-frames N` | `0` (unlimited) | Stop after N frames. Useful with `--no-display`. |
| `--no-display` | off | Run without creating an OpenCV window (headless). |
| `--no-help-overlay` | off | Do not draw the keyboard help panel on the video. |
| `--no-summary` | off | Do not write `outputs/session_summary.json` at exit. |
| `--list-cameras` | — | Probe camera indices, print which work, then exit. |
| `--verbose` | off | Enable debug logging. |
| `--quiet` | off | Only log warnings and errors. |
| `--version` | — | Print the version and exit. |
| `--help` | — | Print full help, including the keyboard controls and examples. |

Invalid input is rejected with a usage message and **exit code 2**; nothing is
started. A handled runtime failure (no camera, unloadable model) exits with
**exit code 1**. Ctrl+C exits with **exit code 130**.

---

## 22. Keyboard controls

Press these while the OpenCV window is focused.

| Key | Action |
| --- | --- |
| `Q` or `ESC` | Quit the application cleanly. |
| `S` | Save a snapshot of the current annotated frame to `outputs/snapshot_<timestamp>.jpg` and print the path in the terminal. |
| `R` | Start recording the annotated output. Press again to stop. The file is `outputs/recording_<timestamp>.mp4`. |
| `C` | Clear the accumulated session statistics. Confidence and recording are unaffected. |
| `+` or `=` | Increase the confidence threshold by 0.05 (clamped to the valid range). |
| `-` or `_` | Decrease the confidence threshold by 0.05 (clamped to the valid range). |
| `H` | Show or hide the on-screen help overlay. |

The same list is drawn inside the application window and printed in the startup
banner.

---

## 23. Output files

Everything is written to `outputs/` unless `--output` says otherwise.

| File | Created when | Contents |
| --- | --- | --- |
| `snapshot_YYYY_MM_DD_HHMMSS.jpg` | `S` is pressed | The annotated frame: boxes, labels, confidence values and overlay. |
| `recording_YYYY_MM_DD_HHMMSS.mp4` | A recording is started with `R` and then stopped | The annotated video stream, encoded at the measured FPS. |
| `session_summary.json` | At exit (unless `--no-summary`) | Session statistics — see below. |

Example `session_summary.json` (structure; values depend on the actual session):

```json
{
  "started_at": "2026-09-17T22:45:01",
  "duration_seconds": 62.4,
  "duration_human": "0:01:02",
  "frames_processed": 1487,
  "frames_with_detections": 1103,
  "total_detections": 2610,
  "peak_simultaneous_detections": 9,
  "average_detections_per_frame": 1.7552,
  "detection_rate": 0.7418,
  "average_confidence": 0.6123,
  "average_fps": 23.83,
  "peak_fps": 31.2,
  "most_frequent_class": "person",
  "most_frequent_class_count": 1204,
  "class_totals": {
    "chair": 318,
    "cup": 202,
    "laptop": 486,
    "person": 1204,
    "bottle": 400
  }
}
```

> The numbers above illustrate the **schema**, not a measured run. The actual
> values depend entirely on your camera, your scene and your hardware.

---

## 24. Testing

```bash
pytest -q
```

This runs the fast unit suite and **requires no webcam, no display and no network
access**: the tests use synthetic NumPy frames, a fake Ultralytics result object
and a `FakeDetector` double instead of the real model.

What is covered:

| Test file | Covers |
| --- | --- |
| `test_utils.py` | Confidence / camera-index / resolution validation, timestamped path generation, snapshot writing, JSON writing, video recording round-trip, logging configuration. |
| `test_analytics.py` | FPS measurement (including a known-rate case), frame analytics, session aggregates, peak simultaneous detections, most-frequent-class tie-breaking, JSON schema. |
| `test_controls.py` | Every key binding, confidence clamping at both bounds, recording/overlay toggles, action history, help-text consistency. |
| `test_camera.py` | Camera construction and validation, actionable error messages, missing-file and unsupported-extension handling, idempotent release, context-manager cleanup. |
| `test_renderer.py` | Deterministic class colours, box drawing, out-of-bounds clipping, every overlay layer, edge cases such as a 1×1 frame. |
| `test_detector.py` | `Detection` geometry, confidence filtering, device resolution, clean errors for missing weights, result conversion from a fake Ultralytics result, summary formatting. |
| `test_cli.py` | `--help`, `--version`, defaults, every invalid-argument path, `--list-cameras`, configuration validation, and application action dispatch. |

Hardware- and network-dependent tests are marked and **deselected by default**:

```bash
pytest -q -m slow       # loads the real YOLO11n weights (downloads ~5 MB)
pytest -m display       # requires a physical webcam
pytest -m "slow or display"   # everything
```

---

## 25. Expected behaviour

On a machine with a working webcam, `python -m app` should:

1. Print the startup banner listing the model, device, confidence, IoU, inference
   size, capture source, backend, resolution, output directory and controls.
2. Download `yolo11n.pt` on the very first run, then reuse the cached copy.
3. **On macOS, ask for camera permission the first time.** Approve it, then quit
   and reopen your terminal if the app still reports a denial. The permission is
   granted to the *terminal application* you launch from, so running the project
   from a different terminal (or from an IDE's integrated terminal) may require
   approving it again.
4. Open a window titled
   `Real-Time Object Detection and Scene Analytics - YOLO11`.
5. Show the live camera feed with bounding boxes and `class confidence` labels
   drawn around recognised objects.
6. Show an analytics panel in the top-left corner reporting objects, classes,
   FPS, the active confidence threshold, the camera resolution and the per-class
   counts.
7. Show the keyboard help panel in the bottom-left corner.
8. Update the FPS value continuously — it will fluctuate, because it is measured.
9. Respond to `S` by writing a JPEG and printing its path, and to `R` by showing
   a REC indicator and writing a video file.
10. On `Q` or `ESC`, close the window, print a session summary and write
    `outputs/session_summary.json`.

Actual FPS depends on the CPU/GPU, camera resolution, model size, inference
settings, operating system and background workload. **No specific FPS is
promised.**

---

## 26. Project originality

This project deliberately and explicitly separates third-party work from
student-authored work.

### Third-party (not written by this project)

| Component | Owner | Licence |
| --- | --- | --- |
| YOLO11 architecture and the pretrained `yolo11n.pt` weights | Ultralytics | AGPL-3.0 |
| The Ultralytics inference library and its preprocessing/NMS implementation | Ultralytics | AGPL-3.0 |
| PyTorch | Meta / PyTorch Foundation | BSD-3-Clause |
| OpenCV — capture, drawing, display, video and image writing | OpenCV team | Apache-2.0 |
| NumPy | NumPy developers | BSD-3-Clause |
| pytest | pytest developers | MIT |
| The COCO dataset used to pretrain the model | COCO consortium | see COCO terms |

**This project does not claim to have invented, designed or trained YOLO11.** It
uses the published pretrained model as a component.

### Student-developed (the original contribution)

* The **webcam application** as a whole, and its modular architecture.
* The **real-time detection pipeline**: capture → infer → analyse → render →
  dispatch, and the lifecycle management around it.
* The **scene analytics layer**: FPS measurement, per-frame object/class counting,
  confidence statistics, session aggregates, peak simultaneous detections, and
  the most-frequent-class calculation.
* The **rendering system**: bounding boxes with adaptive label placement,
  translucent panels, the analytics overlay, the help overlay, the REC indicator
  and transient messages, with a stable per-class colour assignment.
* The **runtime controls**: the key-to-action mapping, the confidence arithmetic
  with clamping, and the state machine behind recording/overlay/reset.
* The **CLI and validation layer**: every option, all range and cross-field
  validation, actionable error messages and defined exit codes.
* **Snapshot capture** and **session recording** with timestamped filenames and
  codec fallback.
* **Session persistence** (`session_summary.json`).
* The **error-handling strategy** across every failure mode.
* The **test suite** and the **documentation**, including all diagrams.

---

## 27. Limitations

1. **No custom training.** The model is used exactly as published; it recognises
   the 80 COCO classes and nothing else. Objects outside those classes are not
   detected.
2. **No evaluation metrics are measured.** Precision, recall and mAP require a
   labelled evaluation dataset, which this project does not ship. They are
   explained conceptually and not reported as results.
3. **No object tracking.** Each frame is detected independently. A person who
   leaves and re-enters the frame counts as a new detection; there are no
   persistent identities and no trajectories.
4. **Session statistics are raw counts.** They are cumulative over the session and
   are not de-duplicated across frames, so `total_detections` counts
   *detections*, not distinct objects.
5. **Single camera.** One capture source at a time.
6. **Frame rate is hardware-dependent.** The application measures FPS but cannot
   guarantee any particular value; on a slow CPU with a high-resolution camera it
   may be noticeably below real time.
7. **Display requires a graphical session.** `--no-display` exists for headless
   machines, but then the window and keyboard controls are unavailable.
8. **The window must be focused for keyboard input.** `cv2.waitKey` only receives
   keys while the OpenCV window is the active window.
9. **Small-object detection is limited.** This is a property of the nano model and
   of the inference resolution, not of the application logic; raising `--imgsz`
   helps at a cost in speed.
10. **The 80-class vocabulary is COCO-specific.** Domain-specific objects require a
    differently trained model.

---

## 28. Future enhancements

* **Object tracking** (ByteTrack or BoT-SORT, both available through Ultralytics)
  to give detections persistent IDs and enable unique-object counting and
  trajectories.
* **Region-of-interest zoning** to count objects inside user-defined polygons.
* **Line-crossing counters** for traffic-style use cases.
* **Alerts** when a class count crosses a threshold, or when a specific class is
  detected.
* **CSV logging** of per-frame detections for offline analysis.
* **A configuration file** so frequently used option sets can be saved and reused.
* **Optional model-level evaluation**, if a labelled dataset in YOLO format is
  supplied, reporting precision, recall and mAP without inventing values.
* **A quantised or ONNX-exported model** for faster CPU inference.
* **A small overlay theme/scale option** for very high-resolution displays.

---

## 29. Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `Error: The operating system denied camera access for camera index 0.` | The camera exists but macOS/Windows has not granted this program permission to use it. **macOS:** System Settings → Privacy & Security → Camera, enable the terminal app you are running from (Terminal, iTerm2, VS Code, …), then quit and reopen that terminal. **Windows:** Settings → Privacy & security → Camera. **Linux:** add your user to the `video` group and log back in. |
| `Error: Unable to open webcam at camera index 0.` | No camera connected, the camera is in use by another app, or the index is wrong. Run `python -m app --list-cameras`, then try `--camera 1`. |
| The window opens but the video is black | Another application is holding the camera, or the driver needs a moment. Close other camera applications and restart. |
| `The 'ultralytics' package is not installed` | The dependencies were not installed, or the virtual environment is not active. Run `pip install -r requirements.txt` with the environment activated. |
| The model download fails | No network access, or a proxy is required. Download `yolo11n.pt` manually from the Ultralytics releases page and pass it with `--model /path/to/yolo11n.pt`. |
| Very low FPS | Lower the inference size (`--imgsz 416`), lower the capture resolution (`--width 640 --height 480`), close other applications, or force CPU/GPU explicitly with `--device`. |
| `Confidence threshold must be between 0.01 and 1.0` | `--conf` must be a number in that range. `--conf 0.5` is valid; `--conf 50` is not. |
| `Camera width and height must be provided together` | Supply both `--width` and `--height`, or neither. |
| `Output path exists but is not a directory` | `--output` points at an existing file. Choose a directory path. |
| Recording produces no file, or `Recording failed` | OpenCV was built without the requested encoder. The application already falls back through `mp4v` → `avc1` → `MJPG`/`AVI`; try `--output` on a different filesystem. |
| Snapshot fails | The output directory is not writable, or the disk is full. |
| Keyboard controls do nothing | The OpenCV window must be focused. Click the window and try again. Note that `--no-display` disables keyboard input entirely. |
| `ImportError` mentioning `libGL` (Linux) | OpenCV needs system GL libraries. Install them (for example `sudo apt install libgl1`) or use `opencv-python-headless`. |

---

## 30. References

1. Ultralytics — *YOLO11 documentation and model page*.
   <https://docs.ultralytics.com/models/yolo11/>
2. Ultralytics — *Python usage / prediction mode*.
   <https://docs.ultralytics.com/modes/predict/>
3. Ultralytics — *Model training and the COCO-pretrained checkpoints*.
   <https://docs.ultralytics.com/models/>
4. Ultralytics — *Licensing*.
   <https://www.ultralytics.com/license>
5. Redmon, J., Divvala, S., Girshick, R., Farhadi, A. — *You Only Look Once:
   Unified, Real-Time Object Detection*, CVPR 2016. <https://arxiv.org/abs/1506.02640>
6. Lin, T.-Y. et al. — *Microsoft COCO: Common Objects in Context*, ECCV 2014.
   <https://arxiv.org/abs/1405.0312>
7. OpenCV — *Official documentation*, in particular the video I/O and drawing
   modules. <https://docs.opencv.org/>
8. Bradski, G. — *The OpenCV Library*, Dr. Dobb's Journal of Software Tools, 2000.
9. PyTorch — *Official documentation*. <https://pytorch.org/docs/>
10. NumPy — *Official documentation*. <https://numpy.org/doc/>
11. pytest — *Official documentation*. <https://docs.pytest.org/>
12. Everingham, M. et al. — *The PASCAL Visual Object Classes (VOC) Challenge*,
    IJCV 2010 (background on precision/recall and IoU-based evaluation).
13. Mermaid — *Diagram syntax*. <https://mermaid.js.org/>

---

## 31. Third-party attribution and licensing

### Ultralytics YOLO11 — AGPL-3.0

The YOLO11 model architecture, the pretrained `yolo11n.pt` weights, and the
inference library are the work of **Ultralytics** and are distributed under the
**GNU Affero General Public License v3.0 (AGPL-3.0)**, with a separate commercial
licence available from Ultralytics.

This project imports and links that library and uses those weights. Consequently,
**distributing or deploying this project as a whole is subject to the AGPL-3.0**,
which among other things requires that the complete corresponding source code be
made available to users interacting with it over a network.

Anyone intending to use this project commercially should review the Ultralytics
licensing terms first: <https://www.ultralytics.com/license>

The model weights are **not** redistributed in this repository; they are
downloaded by the Ultralytics library at runtime.

### OpenCV — Apache-2.0

OpenCV provides webcam capture, drawing primitives, the display window, video
encoding and image writing. <https://opencv.org/license/>

### PyTorch — BSD-3-Clause

PyTorch performs the tensor computation and the actual neural-network inference.
<https://github.com/pytorch/pytorch/blob/main/LICENSE>

### NumPy — BSD-3-Clause

NumPy provides the frame buffers and the synthetic test images.
<https://numpy.org/doc/stable/license.html>

### pytest — MIT

pytest runs the test suite. <https://docs.pytest.org/en/stable/license.html>

### This project's own code — MIT

The original application code (`app/`), the tests (`tests/`) and the documentation
are released under the MIT licence. See [`LICENSE`](LICENSE), which also contains
the full third-party licensing notice.

### No third-party source code was copied

No source code from Ultralytics, OpenCV, PyTorch or any other project has been
copied into this repository. Those libraries are used as installed dependencies
through their public APIs. No third-party imagery, dataset or personal data is
included.

---

## Validation performed

See [`VALIDATION.md`](VALIDATION.md) for the exact commands run against this
repository, their output, and an honest statement of what could and could not be
verified in the development environment.
