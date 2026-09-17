# Real-Time Object Detection Using YOLO11

A Computer Vision project that opens the webcam, detects the objects in each
frame using a pretrained YOLO11n model, and draws the results on screen while
showing live statistics.

---

## 1. Description

The application captures a live video feed, runs object detection on every
frame, and displays bounding boxes with the class name and confidence of each
detected object. A small panel shows how many objects were found, which classes
they belong to and the current frame rate.

It runs entirely from the terminal:

```bash
python -m app
```

The detection model is the pretrained **YOLO11n** model provided by
[Ultralytics](https://docs.ultralytics.com/). This project does **not** train or
fine-tune a model — it is an application built around an existing one. The work
here is the capture loop, the integration with the model, the drawing of the
results, the statistics and the keyboard controls.

## 2. Features

- Live object detection on a webcam feed using pretrained YOLO11n
- Bounding boxes with class name and confidence score
- Real-time statistics: object count, class counts and FPS
- Adjustable confidence threshold while the program is running
- Save a snapshot of the annotated frame with one key press
- Session summary written to `outputs/session_summary.json`
- Automatic use of a GPU when one is available, otherwise CPU
- Clear error messages if the camera cannot be opened

## 3. Technologies used

| Technology | Purpose |
| --- | --- |
| Python 3.10+ | The language the project is written in |
| Ultralytics YOLO11n | The pretrained object detection model |
| PyTorch | Runs the neural network (installed with Ultralytics) |
| OpenCV | Webcam capture, drawing and the display window |
| NumPy | Frame buffers |
| pytest | Unit tests |

## 4. Project structure

```
real-time-object-detection-yolo11/
├── README.md
├── statement.md            problem statement and scope
├── requirements.txt
├── pyproject.toml
├── LICENSE
├── app/
│   ├── __init__.py
│   ├── __main__.py         entry point for `python -m app`
│   ├── main.py             argument parsing and the main loop
│   ├── camera.py           webcam capture
│   ├── detector.py         YOLO11 loading and inference
│   ├── analytics.py        FPS and object counts
│   ├── controls.py         keyboard controls
│   └── renderer.py         drawing the boxes and the statistics
├── tests/
│   ├── test_camera.py
│   ├── test_detector.py
│   ├── test_analytics.py
│   ├── test_controls.py
│   └── test_main.py
├── outputs/                snapshots and the session summary
└── docs/
    ├── architecture.md
    ├── workflow.md
    └── diagrams/           five Mermaid diagrams
```

## 5. Installation

Python 3.10 or newer is required.

```bash
git clone https://github.com/Abhijai10/real-time-object-detection-yolo11.git
cd real-time-object-detection-yolo11
```

Create and activate a virtual environment.

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

This installs Ultralytics, which also installs PyTorch. That download is large
(around 100 MB), so the first install takes a few minutes.

## 6. Running the application

```bash
python -m app
```

On the very first run, Ultralytics downloads the `yolo11n.pt` weights
(about 5 MB) and caches them. Later runs start immediately.

A window opens showing the live camera feed with the detection results drawn on
it. Press `Q` or `ESC` to quit.

### Command line options

| Option | Default | Meaning |
| --- | --- | --- |
| `--camera INDEX` | `0` | Which webcam to use |
| `--model PATH` | `yolo11n.pt` | Model weights to use |
| `--conf FLOAT` | `0.25` | Confidence threshold |
| `--width PIXELS` | driver default | Requested capture width |
| `--height PIXELS` | driver default | Requested capture height |

`--width` and `--height` must be given together.

```bash
python -m app --conf 0.40
python -m app --camera 1 --width 1280 --height 720
```

## 7. Keyboard controls

| Key | Action |
| --- | --- |
| `Q` or `ESC` | Quit |
| `S` | Save a snapshot of the current annotated frame |
| `C` | Clear the session statistics |
| `+` / `-` | Increase or decrease the confidence threshold |

The same list is drawn in the corner of the window.

## 8. How it works

```
Webcam
  -> read a frame with OpenCV
  -> run YOLO11n inference on the frame
  -> collect the boxes, class names and confidences
  -> update the FPS and object counts
  -> draw the boxes, labels and statistics
  -> show the frame in the OpenCV window
  -> check for a key press
  -> repeat
```

Each of those steps lives in its own module:

- `camera.py` opens the webcam and reads frames.
- `detector.py` loads YOLO11n and converts the model output into a list of
  `Detection` objects (class name, confidence, bounding box).
- `analytics.py` measures FPS and counts the objects per class.
- `renderer.py` draws everything onto the frame.
- `controls.py` turns a key code into an action.
- `main.py` runs the loop that ties these together.

The model receives the frame and returns bounding boxes, a class index and a
confidence for each object it recognises. The confidence is how sure the model
is, from 0 to 1. Only objects above the confidence threshold are kept, and the
model's own non-maximum suppression removes overlapping duplicate boxes for the
same object.

### Example session summary

`outputs/session_summary.json` records what happened during a run. This is a
real session recorded by the author on a laptop webcam, and is an example rather
than a benchmark:

```json
{
  "frames_processed": 1913,
  "total_detections": 3132,
  "average_detections_per_frame": 1.64,
  "average_confidence": 0.679,
  "average_fps": 15.2,
  "peak_fps": 19.8,
  "most_frequent_class": "person"
}
```

Frame rates vary a lot between machines. The numbers above are specific to one
laptop and one camera resolution and should not be read as a general
performance figure.

## 9. Limitations

- Detection quality depends entirely on the pretrained YOLO11n model. It
  recognises the 80 classes in the COCO dataset and nothing else.
- Some objects are missed or given the wrong label, especially small, distant or
  partly hidden ones.
- Performance depends on the hardware, the camera resolution and the model size.
- Low light, motion blur and unusual viewing angles all reduce accuracy.
- No custom training was carried out, and **no accuracy benchmark is included**.
  The project does not report precision, recall or mAP, because that would
  require a labelled evaluation dataset that is not part of this project.
- Only one camera feed is processed at a time, and objects are not tracked
  between frames.

## 10. Testing

```bash
pytest -q
```

The tests cover argument parsing, camera error handling, detection structure,
confidence filtering, class counting, FPS calculation, session statistics and
the key mappings. They do not download the model or need a webcam, so they run
in a couple of seconds on any machine.

The tests check that the code behaves correctly. They say nothing about how
accurate the model is — that is not something unit tests can measure.

## 11. Future improvements

- Track objects between frames so each one keeps a stable ID
- Add an option to run detection on a recorded video file
- Support a custom-trained model for a specific set of classes
- Count objects crossing a line in the frame
- Reduce the inference resolution automatically when FPS drops

## 12. References

- Ultralytics YOLO11 documentation — <https://docs.ultralytics.com/models/yolo11/>
- Ultralytics Python API — <https://docs.ultralytics.com/modes/predict/>
- YOLO11 paper — Khanam, R. and Hussain, M. (2024), *YOLOv11: An Overview of
  the Key Architectural Enhancements*
- OpenCV documentation — <https://docs.opencv.org/>
- Redmon, J. et al. (2016), *You Only Look Once: Unified, Real-Time Object
  Detection*
- COCO dataset — <https://cocodataset.org/>

## 13. Third-party attribution and licence

This project is released under the MIT licence (see `LICENSE`).

It depends on third-party software that is **not** part of this project:

| Component | Owner | Licence |
| --- | --- | --- |
| YOLO11 architecture and the pretrained `yolo11n.pt` weights | Ultralytics | AGPL-3.0 |
| Ultralytics inference library | Ultralytics | AGPL-3.0 |
| PyTorch | Meta / PyTorch Foundation | BSD-3-Clause |
| OpenCV | OpenCV team | Apache-2.0 |
| NumPy | NumPy developers | BSD-3-Clause |

The YOLO11 model and its weights are the work of Ultralytics. This project did
not create, train or modify them. Note that Ultralytics distributes its code and
models under the **AGPL-3.0** licence, which is stricter than the MIT licence
used here; anyone reusing this project should check those terms.
