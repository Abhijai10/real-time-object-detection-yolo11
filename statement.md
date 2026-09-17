# Project Statement

**Project title:** Real-Time Object Detection Using YOLO11

## Problem statement

Cameras are everywhere — in laptops, phones and security systems — but a camera
on its own only records what it sees. It cannot tell you *what* is in the
picture. Doing that by hand does not scale: watching a live video feed and
writing down every object that appears is slow and impossible to keep up with.

The problem this project addresses is how to make a live camera feed
understandable automatically. The goal is a program that watches the webcam,
identifies the objects in each frame as they appear, and presents the result in
a way a person can read at a glance.

Object detection is a well-studied Computer Vision problem, and pretrained
models now make it practical to build such an application without training a
network from scratch. This project applies one of those models — YOLO11 — to a
real-time webcam feed and builds the surrounding application.

## Scope of the project

The project covers:

- Capturing a live video feed from a webcam using OpenCV
- Running multi-class object detection on every frame with the pretrained
  YOLO11n model from Ultralytics
- Extracting the bounding boxes, class names and confidence scores from the
  model output
- Drawing the results onto the video and displaying it in a window
- Computing simple runtime statistics: FPS, objects per frame and per-class
  counts
- Letting the user change the confidence threshold, save a snapshot and clear
  the statistics while the program is running
- Writing a summary of the session to a JSON file at exit

The project deliberately does **not** cover:

- Training or fine-tuning a detection model
- Preparing or labelling a dataset
- Measuring model accuracy (precision, recall, mAP) — this needs a labelled
  evaluation dataset, which is outside the scope
- Object tracking between frames, face recognition, pose estimation or
  distance estimation
- Any web interface, database or cloud service

The application runs from the command line. It needs Python and a webcam, and
nothing else.

## Target users

- **Students** learning how Computer Vision models are applied in practice
- **Lecturers and examiners** assessing the implementation and documentation
- **Anyone** who wants a simple, readable example of running a pretrained
  detection model on a live camera feed

The program assumes no prior knowledge of the code. It can be started with a
single command.

## High-level features

1. **Camera capture** — opens the webcam, reads frames continuously and releases
   the device cleanly on exit. If the camera cannot be opened, the program
   explains the likely reason instead of failing with an unexplained error.

2. **Object detection** — loads the pretrained YOLO11n model and runs it on each
   frame, producing a list of detected objects with their class, confidence and
   bounding box. The model is downloaded automatically the first time it is
   needed.

3. **Visualisation and statistics** — draws a coloured box and a
   `class confidence` label around each object, and shows a panel with the
   object count, the classes present, the frame rate and the active confidence
   threshold.

4. **User controls** — the user can quit, save a snapshot of the current frame,
   clear the statistics, and raise or lower the confidence threshold, all
   without restarting the program.

5. **Session summary** — on exit the program prints a summary of the session and
   writes it to `outputs/session_summary.json`, recording how many frames were
   processed, how many objects were detected and which class was seen most
   often.

## A note on originality

The YOLO11 model and its pretrained weights are the work of Ultralytics and are
used here as a third-party component. This project is the application built
around that model: the capture loop, the integration with the model, the
processing of its output, the visualisation, the statistics, the keyboard
controls and the error handling. No part of the neural network itself was
written or trained as part of this project.
