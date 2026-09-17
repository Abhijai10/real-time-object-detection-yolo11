# Project Statement

## Real-Time Object Detection and Scene Analytics Using YOLO11

---

## 1. Problem Statement

Computer vision systems are increasingly expected to interpret a live video
stream rather than a single still image. Detecting *what* is present, *where* it
is located and *how many* instances are visible, continuously and within a
practical time budget, is the foundation of applications such as surveillance,
traffic monitoring, assistive systems, retail analytics and industrial safety.

Building such a system from first principles is not feasible within the scope of
a university practical: designing, training and tuning a convolutional or
transformer-based detector requires large labelled datasets and significant
compute. A more realistic and academically valuable exercise is to **integrate a
modern pretrained detector into a complete, well-engineered application**, and
to build the surrounding engineering — capture, filtering, analytics, rendering,
controls and error handling — that turns a model into a usable tool.

The problem addressed by this project is therefore twofold:

1. **Perception:** reliably localise and classify objects in a live webcam
   stream using a pretrained object detection model.
2. **Engineering:** present the results in a form that a human operator can
   actually use — annotated video, live scene analytics (frame rate, object and
   class counts), keyboard controls, snapshot capture and session statistics —
   while remaining robust to the failure modes that occur in practice (no
   camera, unreadable device, invalid arguments, unavailable output directory).

A specific practical constraint motivates the design: the application must run
on an ordinary laptop CPU, without a GPU, and must be launchable from a terminal
with a single command.

---

## 2. Scope of the Project

### In scope

| Area | Included |
| --- | --- |
| **Input source** | A single live webcam device (configurable index), plus an optional video file used for headless development and automated testing. |
| **Detection** | Single-class-family object detection using the pretrained Ultralytics **YOLO11n** model. 80 COCO object classes. |
| **Preprocessing** | Frame acquisition, optional resolution negotiation, letterbox resizing performed internally by Ultralytics. |
| **Post-processing** | Confidence thresholding (adjustable at runtime), non-maximum suppression via the model's IoU parameter. |
| **Analytics** | Measured FPS, per-frame object count, unique classes, per-class counts, average confidence, camera resolution, and session-level aggregates. |
| **Visualisation** | Bounding boxes, class labels, confidence values, analytics overlay, keyboard-help overlay, recording indicator, transient status messages. |
| **Interaction** | Terminal CLI (`python -m app`) and in-window keyboard controls (quit, snapshot, record, reset, confidence up/down, overlay toggle). |
| **Persistence** | Timestamped JPEG snapshots, optional timestamped video recordings, and `outputs/session_summary.json` at exit. |
| **Quality assurance** | Unit test suite (pytest), type hints, docstrings, structured logging, explicit error handling. |

### Out of scope

The following are deliberately **excluded** to keep the project focused and to
match the constraints of a practical submission:

* Training, fine-tuning or re-training any neural network.
* Custom dataset collection, annotation or management.
* Multi-object tracking with persistent identities (no tracker is used; each
  frame is treated independently).
* Batch processing of image folders or offline video libraries.
* Web dashboards, REST APIs, databases, authentication or any GUI framework.
* Multi-camera synchronisation and distributed deployment.
* Model-level accuracy benchmarking. Evaluation metrics such as mAP, precision
  and recall are **discussed conceptually** in the documentation, but are not
  measured by this application, because measuring them requires a labelled
  evaluation dataset which this project does not ship. The application reports
  only what it can genuinely measure: runtime FPS and per-frame detection
  statistics.

---

## 3. Target Users

| User | Need |
| --- | --- |
| **University student / examiner** | A self-contained, documented Computer Vision practical that can be installed and demonstrated in minutes from a terminal. |
| **Computer Vision learner** | A readable reference for how a modern detector is integrated into a real application, including the concepts behind bounding boxes, confidence thresholds and IoU. |
| **Application developer** | A small, modular starting point (capture → detect → analyse → render) that can be extended with tracking, logging or domain-specific logic. |
| **Researcher / demonstrator** | A quick way to qualitatively inspect what a pretrained detector sees in a given physical environment, with adjustable confidence. |

---

## 4. High-Level Features

1. **Single-command startup.** `python -m app` opens the webcam and starts
   detecting, with no GUI framework, no notebook and no configuration file
   required.
2. **Pretrained YOLO11n detection.** Lightweight, CPU-friendly, 80 COCO classes.
   Weights are downloaded automatically on first use.
3. **Configurable capture.** Camera index, requested width/height, inference
   resolution and compute device are all command-line options.
4. **Runtime confidence control.** The threshold can be raised or lowered while
   the application is running, and the effect is visible immediately.
5. **Live scene analytics.** Objects, unique classes, per-class counts, measured
   FPS, active confidence threshold and camera resolution, all drawn on the
   video.
6. **Snapshot capture.** One keypress writes a timestamped, fully annotated JPEG
   to `outputs/` and prints its path in the terminal.
7. **Session recording.** A keypress toggles recording of the annotated stream to
   a timestamped video file.
8. **Session statistics.** Frames processed, total detections, peak simultaneous
   detections, most frequent class, average and peak FPS, written to
   `outputs/session_summary.json`.
9. **Graceful failure.** Missing camera, unreadable device, invalid arguments,
   bad confidence values, unwritable output directory and model-loading failures
   all produce clear, actionable messages instead of tracebacks.
10. **Tested and documented.** A pytest suite covering validation, analytics,
    controls, rendering, camera error paths and the CLI; plus architecture,
    workflow and UML-style diagrams in `docs/`.

---

## 5. Academic Positioning

This project does **not** claim to have invented YOLO11, its architecture, its
training procedure or its weights. Those are the work of Ultralytics and the
wider research community, and are used here as a third-party component.

What this project contributes is the **application layer**: the real-time capture
loop, the detection-to-analytics pipeline, the rendering and overlay system, the
runtime control scheme, the CLI and validation layer, the output and persistence
handling, the session statistics, the error-handling strategy, the test suite and
the documentation.

Section 20 of the README ("Project originality") states this boundary explicitly.
