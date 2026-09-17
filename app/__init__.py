"""Real-Time Object Detection and Scene Analytics Using YOLO11.

A command-line Computer Vision application that opens the webcam, detects
objects on every frame with the pretrained Ultralytics YOLO11n detector, draws
bounding boxes and shows live scene analytics (FPS, object counts, per-class
counts) in an OpenCV window.

Package layout
--------------
``camera``      Module 1 - webcam capture and validation.
``detector``    Module 2 - YOLO11 loading, inference and detection extraction.
``analytics``   Module 3 - FPS measurement and scene/session statistics.
``controls``    Module 4 - keyboard controls and runtime state.
``renderer``    Drawing of boxes, labels and the analytics overlay.
``cli``         Argument parsing, validation and the application controller.
``utils``       Shared helpers: logging, validation, output handling.

Third-party components: Ultralytics (YOLO11 model and inference API) and
OpenCV (capture, drawing and display).  The application layer -- capture loop,
analytics, controls, rendering, CLI and error handling -- is original work.
"""

from __future__ import annotations

__all__ = ["__version__", "PROJECT_TITLE"]

__version__ = "1.0.0"

PROJECT_TITLE = "Real-Time Object Detection and Scene Analytics Using YOLO11"
