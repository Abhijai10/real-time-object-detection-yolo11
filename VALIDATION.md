# Validation Report

This document records exactly what was executed against this repository, what the
results were, and — importantly — what **could not** be verified in the
development environment.

No result in this file is estimated, simulated or invented. Where something could
not be tested, it says so.

---

## 1. Environment

| Item | Value |
| --- | --- |
| Operating system | macOS (Darwin), Apple Silicon (arm64) |
| Python | 3.13.12 |
| Ultralytics | 8.4.155 |
| PyTorch | 2.14.0 |
| OpenCV (`opencv-python`) | 5.0.0.93 |
| NumPy | 2.5.3 |
| pytest | 9.1.1 |
| Resolved compute device | Apple Silicon GPU (MPS), selected automatically by `--device auto` |
| Capture backend | AVFoundation |
| **Webcam** | **Not available in this environment** |
| **Graphical display** | **Not available in this environment** |

The webcam and display limitations are the reason sections 4 and 5 below
distinguish clearly between what was verified and what was not.

---

## 2. Final file tree

```
Object Detection Model/
├── .gitignore
├── LICENSE
├── README.md
├── VALIDATION.md                  ← this file
├── pyproject.toml
├── requirements.txt
├── statement.md
│
├── app/
│   ├── __init__.py                (29 lines)
│   ├── __main__.py                (14 lines)
│   ├── analytics.py               (375 lines)
│   ├── camera.py                  (395 lines)
│   ├── cli.py                     (810 lines)
│   ├── controls.py                (205 lines)
│   ├── detector.py                (440 lines)
│   ├── renderer.py                (452 lines)
│   └── utils.py                   (656 lines)
│
├── tests/
│   ├── conftest.py                (132 lines)
│   ├── test_analytics.py          (265 lines,  23 tests)
│   ├── test_camera.py             (184 lines,  20 tests)
│   ├── test_cli.py                (345 lines,  40 tests)
│   ├── test_controls.py           (195 lines,  32 tests)
│   ├── test_detector.py           (290 lines,  33 tests)
│   ├── test_renderer.py           (242 lines,  20 tests)
│   └── test_utils.py              (325 lines,  65 tests)
│
├── outputs/
│   └── .gitkeep
│
├── docs/
│   ├── architecture.md
│   ├── workflow.md
│   └── diagrams/
│       ├── architecture.mmd
│       ├── component.mmd
│       ├── sequence.mmd
│       ├── use_case.mmd
│       └── workflow.mmd
│
└── examples/
    └── commands.md
```

Total: **9 application modules (3,376 lines)** and **8 test files (1,978 lines, 233 tests)**.

`yolo11n.pt` is also present in the working tree after the first run. It is a
**generated artefact** downloaded automatically by Ultralytics and is excluded by
`.gitignore` (`*.pt`), so it is not part of the repository.

---

## 3. Verification performed

### 3.1 Syntax and imports

```bash
python -m compileall -q app tests
python -c "import app, app.utils, app.analytics, app.controls, app.camera, app.detector, app.renderer, app.cli"
```

**Result:** both commands exited 0 with no output. Every module compiles and
imports.

### 3.2 Lazy-import claim

```bash
python -c "
import sys
import app.utils, app.analytics, app.controls
print([m for m in ('cv2','torch','ultralytics','numpy') if m in sys.modules])
"
```

**Result:** `none`. The validation, analytics and controls layers really do load
without pulling in OpenCV, PyTorch, Ultralytics or even NumPy — which is what
makes the unit suite fast and hardware-independent.

### 3.3 Test suite

```bash
pytest -q
```

**Result:**

```
........................................................................ [ 31%]
........................................................................ [ 62%]
........................................................................ [ 93%]
...............                                                          [100%]
233 passed, 3 deselected in 1.57s
```

| Test file | Tests | Covers |
| --- | --- | --- |
| `test_utils.py` | 65 | Validation, path helpers, snapshot/JSON writing, recorder round-trip, logging |
| `test_cli.py` | 40 | Parser, defaults, every invalid-argument path, `--list-cameras`, config validation, action dispatch, adaptive program name |
| `test_detector.py` | 33 | `Detection` geometry, confidence filtering, device resolution, missing-weight errors, result conversion |
| `test_controls.py` | 32 | Every key binding, confidence clamping, state transitions, history |
| `test_analytics.py` | 23 | FPS measurement, frame analytics, session aggregates, tie-breaking |
| `test_camera.py` | 20 | Construction, validation, error messages, lifecycle, context manager |
| `test_renderer.py` | 20 | Colours, box drawing, clipping, every overlay layer, 1×1 edge case |

The 3 deselected tests are the hardware- and model-dependent ones.

```bash
pytest -q -m slow
```

**Result:** `2 passed, 234 deselected in 9.22s`

These load the **real** `yolo11n.pt` weights and run a real inference pass. The
model reported **80 object classes**, confirming the checkpoint loaded correctly.

### 3.4 CLI behaviour

```bash
python -m app --help
python -m app --version
```

**Result:** `--help` printed the full option list, the keyboard controls and the
examples; exit code 0. `--version` printed `python -m app 1.0.0`; exit code 0.

Invalid input handling — every case below printed a clear message and produced
**no traceback**:

| Command | Message | Exit code |
| --- | --- | --- |
| `python -m app --conf 5.0` | `Confidence threshold must be between 0.01 and 1.0, received: 5.0` | **2** |
| `python -m app --conf abc` | `Confidence threshold must be a number, received: 'abc'` | **2** |
| `python -m app --camera -3` | `Camera index must be zero or greater, received: -3` | **2** |
| `python -m app --width 1280` | `Camera width and height must be provided together` | **2** |
| `python -m app --max-frames -5` | `--max-frames must be zero or greater, received: -5` | **2** |
| `python -m app --bogus-flag` | `unrecognized arguments: --bogus-flag` | **2** |
| `python -m app --camera /tmp/does_not_exist.mp4` | path/extension rejection | **2** |
| `python -m app` (no webcam present) | see below | **1** |

### 3.5 Model download and loading

```bash
python -m app
```

**Result:** Ultralytics downloaded `yolo11n.pt` (5.4 MB) automatically on first
use, then:

```
INFO | app.detector | Loading YOLO11 model 'yolo11n.pt' on Apple Silicon GPU (MPS)
INFO | app.detector | Model ready - 80 object classes available
```

This confirms automatic download, local caching, correct device auto-selection
and successful checkpoint loading.

### 3.6 Camera absence handling

Continuing the same run, the application then attempted to open the webcam:

```
Error: Unable to open webcam at camera index 0.
Possible causes:
  - No webcam is connected to this machine.
  - The camera is already in use by another application.
  - The operating system has not granted camera permission to the terminal.
  - A different device index is required; try --camera 1.
```

Exit code **1**, no traceback. This is the correct, intended behaviour on a
machine with no camera.

```bash
python -m app --list-cameras
```

**Result:**

```
Probing camera indices 0-4 (backend: AVFoundation) ...
No usable camera was found.
Check that a webcam is connected and that camera permission is granted.
```

Exit code **1**. (The native OpenCV chatter that normally appears on stderr during
probing is suppressed by the application, so only the clean message is shown.)

### 3.7 Full pipeline, end to end

Because no webcam exists here, the pipeline was validated using a **synthetic
video file generated at runtime** (moving geometric shapes drawn with OpenCV — no
third-party imagery) as the capture source. This exercises every stage of the real
code path: capture → YOLO11 inference → detection extraction → analytics → render →
persist → summary.

```bash
python -m app --camera /tmp/odm_validation/clip.mp4 --no-display --output /tmp/odm_validation/out --conf 0.25
```

**Result:** exit code 0. Startup banner printed correctly, all 40 frames
processed, and the session summary was:

```
Frames processed        : 40
Total detections        : 41
Peak simultaneous       : 2
Average detections/frame: 1.02
Average confidence      : 0.588
Average FPS             : 56.1
Peak FPS                : 61.8
Duration                : 0:00:07
Most frequent class     : frisbee (40)
Top classes             : frisbee x40, sports ball x1
```

The real YOLO11n model genuinely detected the moving green disc in the synthetic
frames (classified as `frisbee` / `sports ball`). This is a **real inference
result**, not a fabricated one — but it is **not** an accuracy measurement,
because the input is synthetic and unlabelled. It only demonstrates that the
detection path works.

The FPS values (56.1 average, 61.8 peak) are real measurements from this machine
(Apple Silicon MPS, 640×480 input). They say nothing about performance on other
hardware.

`outputs/session_summary.json` was written with the documented schema:

```json
{
  "started_at": "2026-09-17T23:03:45",
  "duration_seconds": 6.6,
  "duration_human": "0:00:07",
  "frames_processed": 40,
  "frames_with_detections": 40,
  "total_detections": 41,
  "peak_simultaneous_detections": 2,
  "average_detections_per_frame": 1.025,
  "detection_rate": 1.0,
  "average_confidence": 0.5885,
  "average_fps": 56.13,
  "peak_fps": 61.78,
  "most_frequent_class": "frisbee",
  "most_frequent_class_count": 40,
  "class_totals": { "frisbee": 40, "sports ball": 1 }
}
```

### 3.8 Keyboard controls, end to end

The real `Application.run()` loop was driven with a scripted key sequence by
patching **only** the key-code translation step (`interpret_key`). Everything
else — snapshot writing, recording start/stop, confidence adjustment, statistics
reset, teardown and summary writing — was the genuine production code path.

Scripted sequence: `S` → `R` (start) → `+` `-` `-` → `C` (reset) → `R` (stop) → `Q`.

**Result:** exit code 0, and the following files were produced:

| File | Size | Verified |
| --- | --- | --- |
| `snapshot_2026_09_17_230416.jpg` | 45,030 bytes | Re-opened with OpenCV: `(480, 640, 3)` uint8. Visually confirmed to contain a bounding box labelled `frisbee 0.62`, the analytics panel, and the help panel. |
| `recording_2026_09_17_230416.mp4` | 129,552 bytes | Re-opened with OpenCV: 17 frames, 640×480, 38.81 FPS (the measured FPS at recording start); first frame decoded successfully. |
| `session_summary.json` | 473 bytes | Written after `C` was pressed; counters confirmed reset (8 frames counted after the reset, not 17). |

This confirms requirements FR11 (`S` saves a snapshot), FR14 (`R` toggles
recording), the `C` reset behaviour and the `+`/`-` confidence controls.

### 3.9 Documentation integrity

```bash
grep -rniE "TODO|FIXME|XXX|HACK|not implemented|NotImplementedError" app/ tests/ docs/ examples/ README.md statement.md
```

**Result:** no matches. There is no placeholder code and no TODO presented as
finished functionality.

Every path referenced by the README, the docs and `examples/commands.md` was
checked and exists.

### 3.10 Public repository and fresh clone

```bash
gh repo create real-time-object-detection-yolo11 --public --source=. --remote=origin --push
gh repo view Abhijai10/real-time-object-detection-yolo11 --json visibility,url,defaultBranchRef
```

**Result:** repository created and pushed.

```
https://github.com/Abhijai10/real-time-object-detection-yolo11
visibility: PUBLIC
default branch: main
```

Root contents confirmed through the GitHub API: `README.md`, `statement.md`,
`VALIDATION.md`, `requirements.txt`, `pyproject.toml`, `.gitignore`, `LICENSE`,
and the `app/`, `tests/`, `docs/`, `examples/`, `outputs/` directories.

A fresh clone was then made into a clean directory and verified independently:

```bash
git clone https://github.com/Abhijai10/real-time-object-detection-yolo11.git
cd real-time-object-detection-yolo11
pytest -q
python -m app --version
python -m app --help
python -m app --camera <clip>.mp4 --no-display --max-frames 10 --output <dir>
```

**Result:** the clone contains 12 root entries, `pytest -q` reported
**233 passed, 3 deselected**, `--version` printed `python -m app 1.0.0`,
`--help` exited 0, and the pipeline processed 10 frames producing 10 real
detections and a valid `session_summary.json`. The repository is therefore
independently reproducible from a clean checkout.

### 3.11 Editable install and the `detect-yolo11` console script

`examples/commands.md` documents installing the project as a command, so that
path was verified rather than assumed:

```bash
pip install -e .
detect-yolo11 --version
detect-yolo11 --help
detect-yolo11 --conf 9            # expect exit code 2
detect-yolo11 --camera <clip>.mp4 --no-display --max-frames 5 --output <dir>
```

**Result:** the editable wheel built and installed cleanly, the console script was
created, `--version` exited 0, `--help` printed the full option list, invalid
confidence exited **2**, and the pipeline ran to completion from a different
working directory (`/tmp`), writing a valid `session_summary.json`.

This test exposed a small defect: the program name in usage/version output was
hard-coded to `python -m app`, so the console script misreported itself as
`python -m app 1.0.0`. `build_parser()` now derives the name from `sys.argv[0]`,
falling back to the documented invocation. Verified:

```
$ python -m app --version     ->  python -m app 1.0.0
$ detect-yolo11 --version     ->  detect-yolo11 1.0.0
```

Two regression tests were added for this, which is why the suite count moved from
231 to 233. The `*.egg-info/` build artefact created by the editable install was
confirmed to be covered by `.gitignore` and was then removed.

### 3.12 Mermaid diagram syntax

The five diagrams in `docs/diagrams/` are referenced by the README and the design
docs, so a syntax error would show as a broken box on GitHub. Each file was
rendered through the public `mermaid.ink` service, which returns HTTP 400 for
invalid syntax:

```python
url = "https://mermaid.ink/img/" + base64.urlsafe_b64encode(diagram_text.encode()).decode()
```

**Result:** all five returned **HTTP 200** with real PNG payloads
(164 KB – 335 KB):

| Diagram | Result |
| --- | --- |
| `architecture.mmd` | HTTP 200, 164,332 bytes |
| `component.mmd` | HTTP 200, 298,407 bytes |
| `sequence.mmd` | HTTP 200, 218,869 bytes |
| `use_case.mmd` | HTTP 200, 188,251 bytes |
| `workflow.mmd` | HTTP 200, 335,186 bytes |

The validator was itself validated with four deliberately broken diagrams
(unclosed quote, malformed arrow, unknown diagram type, unclosed class body) —
all four were correctly rejected with **HTTP 400**. The HTTP 200 results above
are therefore meaningful rather than vacuous.

The rendered `component.mmd` was inspected visually and matches the actual code:
the classes, attributes, methods and relationships shown are those in
`app/analytics.py`, `app/camera.py`, `app/cli.py`, `app/controls.py`,
`app/detector.py`, `app/renderer.py` and `app/utils.py`.

Note: `npm install` is blocked in this environment, so the local Mermaid CLI could
not be used; the remote renderer was the available equivalent.

---

## 4. What could NOT be verified

Stated plainly, because it matters:

| Item | Status | Reason |
| --- | --- | --- |
| Opening a **physical webcam** | **Not verified** | No webcam is attached to this machine. `--list-cameras` confirmed that indices 0–4 all fail to open. |
| **Live OpenCV window** (`cv2.imshow`) | **Not verified** | No graphical display is available in this environment. Window creation, live display and the REC indicator on a real window could not be observed. |
| **Physical keyboard input** to the window | **Not verified** | `cv2.waitKey` requires a real focused window. Controls were verified by driving the same action-dispatch code path programmatically instead. |
| **Real-scene detection quality** | **Not verified** | No camera and no real-object imagery were available. The detections observed were on synthetic geometric input and are meaningless as an accuracy measure. |
| **Model accuracy metrics** (precision, recall, mAP) | **Not measured** | Requires a labelled evaluation dataset with ground-truth boxes, which this project deliberately does not ship. These are documented conceptually only. |
| **Performance on other hardware** | **Not verified** | The measured 56 FPS is specific to this machine (Apple Silicon MPS, 640×480). No claim is made for any other configuration. |
| **Windows / Linux behaviour** | **Not verified** | Only macOS was available. The code paths are platform-guarded (`CAP_AVFOUNDATION` / `CAP_DSHOW` / default), but they were not executed. |

The verification in section 3.7 and 3.8 covers everything the webcam and display
would otherwise have exercised, **except** the physical device and the physical
window themselves.

**The first thing to do on a machine with a webcam is:**

```bash
python -m app --list-cameras
python -m app
```

---

## 5. Assumptions

1. **Python 3.10+** is available and the user can create a virtual environment.
2. The user has **network access on first run** so Ultralytics can download
   `yolo11n.pt` (~5 MB). After that, the application works offline.
3. The webcam is accessible to the terminal process, i.e. the operating system has
   granted camera permission.
4. The OpenCV build in use includes video encoding support. The application
   already falls back through `mp4v` → `avc1` → `MJPG`/`AVI`, but a build with no
   encoder at all would prevent recording (snapshots would still work).
5. `outputs/` is writable, or `--output` points somewhere that is.
6. The synthetic test video used for validation was generated locally and is
   **not** part of the repository.

---

## 6. Requirement coverage

| Requirement | Status | Evidence |
| --- | --- | --- |
| FR1 — configurable camera index, continuous capture | Implemented | `app/camera.py`; verified against a file source |
| FR2 — detect camera availability, clear error | Implemented | §3.6 |
| FR3 — load YOLO11n, auto-download weights | Implemented | §3.5 |
| FR4 — allow a local model path | Implemented | `tests/test_detector.py` (missing-file error path) |
| FR5 — detect on every frame via Ultralytics | Implemented | §3.7 |
| FR6 — boxes, class ids, names, confidences | Implemented | `Detection`; §3.8 snapshot |
| FR7 — configurable confidence, runtime adjustable | Implemented | `--conf`, `+`/`-`; §3.8 |
| FR8 — draw boxes, labels, confidences | Implemented | §3.8, visually confirmed |
| FR9 — FPS, object count, unique classes, per-class counts | Implemented | §3.8 snapshot overlay |
| FR10 — confidence threshold and resolution displayed | Implemented | §3.8 snapshot overlay |
| FR11 — snapshot on `S` | Implemented | §3.8 |
| FR12 — full keyboard control set | Implemented | §3.8 (`Q`/`ESC`, `S`, `R`, `C`, `+`, `-`, `H`) |
| FR13 — timestamped snapshot filenames | Implemented | `snapshot_2026_09_17_230416.jpg` |
| FR14 — recording toggle on `R` | Implemented | §3.8 |
| FR15 — session summary JSON at exit | Implemented | §3.7 |
| FR16 — argparse CLI with `--help` | Implemented | §3.4 |
| FR17 — validation of paths, confidence, index, resolution | Implemented | §3.4 |
| FR18 — CPU support, automatic GPU use | Implemented | MPS auto-selected in §3.5; `--device cpu` tested in unit tests |
| FR19 — force a device | Implemented | `--device` |
| FR20 — no GUI app / notebook / browser required | Implemented | everything above ran from the terminal |

---

## 7. Exact commands used

```bash
# --- environment -------------------------------------------------------------
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# --- static checks -----------------------------------------------------------
python -m compileall -q app tests
python -c "import app, app.utils, app.analytics, app.controls, app.camera, app.detector, app.renderer, app.cli"
python -c "import sys; import app.utils, app.analytics, app.controls; print([m for m in ('cv2','torch','ultralytics','numpy') if m in sys.modules])"

# --- tests -------------------------------------------------------------------
pytest -q
pytest -q -m slow
pytest -m display          # deselected here: no webcam

# --- CLI ---------------------------------------------------------------------
python -m app --help
python -m app --version
python -m app --conf 5.0
python -m app --conf abc
python -m app --camera -3
python -m app --width 1280
python -m app --max-frames -5
python -m app --bogus-flag
python -m app --camera /tmp/does_not_exist.mp4
python -m app --list-cameras
python -m app                     # no webcam present -> clean error, exit 1

# --- full pipeline against a synthetic video ---------------------------------
python -m app --camera /tmp/odm_validation/clip.mp4 --no-display \
              --output /tmp/odm_validation/out --conf 0.25

# --- keyboard controls driven through the real loop --------------------------
# (scripted key sequence, patching only interpret_key)
python /tmp/drive_controls.py

# --- editable install and console script -------------------------------------
pip install -e .
detect-yolo11 --version
detect-yolo11 --help
detect-yolo11 --conf 9
detect-yolo11 --camera /tmp/odm_validation/clip.mp4 --no-display --max-frames 5 --output /tmp/odm_sc_out

# --- Mermaid diagram syntax --------------------------------------------------
# each docs/diagrams/*.mmd rendered through https://mermaid.ink/img/<base64>
# plus four deliberately broken diagrams as a control (all rejected with HTTP 400)
```

---

## 8. Known limitations of this validation

* The webcam, the display window and physical keyboard input were **not** tested,
  as explained in section 4. They are the only parts of the project that remain
  unverified.
* The detection results reported in §3.7 come from synthetic input and are
  evidence that the pipeline works, not that the model is accurate.
* The measured FPS is specific to this machine and this input resolution.
