# System Architecture

## Overview

The application is a loop that reads frames from a webcam, runs object detection
on each one, and draws the results. It is split into small modules so that each
part has one clear job.

```
User
  |
  v
app/main.py            argument parsing and the main loop
  |
  +-- app/camera.py        webcam capture
  +-- app/detector.py      YOLO11 loading and inference
  +-- app/analytics.py     FPS and object counts
  +-- app/controls.py      keyboard controls
  +-- app/renderer.py      drawing
  |
  v
YOLO11n (Ultralytics)  third-party pretrained model
  |
  v
annotated frame  ->  OpenCV window
                 ->  outputs/ (snapshot, session summary)
```

The diagram in `docs/diagrams/architecture.mmd` shows the same structure with
the data that flows between the modules.

## Modules

### `app/main.py`

Parses the command line arguments, creates the other objects, and runs the main
loop. The loop is the heart of the program:

1. Read a frame from the camera.
2. Run detection on it.
3. Update the statistics.
4. Draw the boxes, labels and statistics.
5. Show the frame in the window.
6. Check whether a key was pressed.

It also writes the session summary when the loop ends.

### `app/camera.py`

Wraps `cv2.VideoCapture`. It opens the device, reads frames and releases it
again. If the camera cannot be opened it raises `CameraError` with a message
explaining the likely causes, rather than letting OpenCV's own error surface.

### `app/detector.py`

Loads the pretrained YOLO11n model and runs inference. It converts the raw
output of the model into a list of `Detection` objects, each holding a class
name, a confidence and a bounding box. The rest of the program never touches the
Ultralytics API directly.

The model is a third-party component. This module does not implement any part of
the neural network.

### `app/analytics.py`

Measures the frame rate from real timestamps and counts the objects in each
frame. `SessionStats` accumulates totals for the whole run and produces the JSON
written to `outputs/session_summary.json`.

This module has no dependency on OpenCV or Ultralytics, which means the counting
logic can be tested on its own.

### `app/controls.py`

Maps the integer returned by `cv2.waitKey` to a named `Action`, and clamps the
confidence threshold to the allowed range. It contains no OpenCV code, so the
key handling can also be tested on its own.

### `app/renderer.py`

Contains every OpenCV drawing call: the bounding boxes, the labels, the
statistics panel and the controls panel. Class colours are derived from a
checksum of the class name so that a given class always gets the same colour.

## Error handling

The program is designed to fail with an explanation rather than a traceback.

| Situation | What happens |
| --- | --- |
| Invalid command line value | argparse prints a clear message and exits with code 2 |
| Model file missing | A message naming the path, exit code 1 |
| Ultralytics not installed | A message telling the user to install the requirements |
| Camera cannot be opened | A message listing the likely causes, exit code 1 |
| The video stream is lost | The loop stops after repeated failed reads |
| Snapshot cannot be written | A message is printed; the loop keeps running |

## Design notes

- **The model is loaded before the camera is opened.** A missing model is a more
  likely mistake than a missing camera, and reporting it first avoids opening a
  camera needlessly.
- **The confidence threshold is passed to the model**, so Ultralytics filters the
  detections during inference instead of the application filtering afterwards.
- **`main.py` owns the loop.** The other modules are called by it and do not call
  each other, apart from `renderer` reading `Detection` objects produced by
  `detector`.
