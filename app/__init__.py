"""Real-Time Object Detection Using YOLO11.

A small Computer Vision project that runs object detection on a live webcam
feed using the pretrained YOLO11n model, and draws the results on screen.

The detection model itself is third-party software from Ultralytics. Everything
in this package is the application built around it.

Package layout
--------------
``camera``      Module 1 - webcam capture.
``detector``    Module 2 - YOLO11 loading and inference.
``analytics``   Module 3 - FPS and object counts.
``controls``    Module 4 - keyboard controls.
``renderer``    Drawing the boxes, labels and statistics.
``main``        Command line interface and the main loop.
"""

__version__ = "1.0.0"

PROJECT_TITLE = "Real-Time Object Detection Using YOLO11"
