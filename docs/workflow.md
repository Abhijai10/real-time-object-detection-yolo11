# Application Workflow

## Startup

1. **Parse the arguments.** `app/main.py` reads `--camera`, `--model`, `--conf`,
   `--width` and `--height`, and checks them. An invalid value stops the program
   with a message explaining what was wrong.

2. **Load the model.** `Detector.load()` asks Ultralytics for the YOLO11n
   weights. If they are not on disk yet they are downloaded once and cached. The
   device is chosen automatically: CUDA if a GPU is present, then Apple Silicon
   MPS, otherwise CPU.

3. **Open the camera.** `Camera.open()` starts the webcam. If it fails, the
   program prints the likely reasons and exits.

4. **Create the window.** OpenCV creates the display window, and the startup
   banner is printed to the terminal.

## The main loop

Each pass through the loop does the following:

```
read a frame from the webcam
        |
        v
run YOLO11n inference on the frame
        |
        v
get a list of detections (class, confidence, box)
        |
        v
update FPS and object counts
        |
        v
draw boxes, labels and the statistics panel
        |
        v
show the frame in the OpenCV window
        |
        v
read a key press
        |
        v
repeat, unless Q or ESC was pressed
```

If a frame cannot be read, the loop skips it. If this happens many times in a
row the stream is assumed to be lost and the loop stops.

## Handling a key press

`cv2.waitKey(1)` returns a key code, or `-1` when no key was pressed.
`app/controls.py` turns that code into an `Action`, and the loop responds:

| Action | What the loop does |
| --- | --- |
| `QUIT` | Leaves the loop |
| `SNAPSHOT` | Writes the current annotated frame to `outputs/snapshot_<timestamp>.jpg` |
| `RESET_STATS` | Clears the session totals |
| `CONFIDENCE_UP` | Raises the threshold by 0.05 and updates the detector |
| `CONFIDENCE_DOWN` | Lowers the threshold by 0.05 and updates the detector |
| `NONE` | Nothing |

A short message is drawn on the video for a couple of seconds after a control is
used, so the user can see that it worked.

## Shutdown

When the loop ends, whether the user pressed `Q` or the video stream ended:

1. The camera is released.
2. The OpenCV window is destroyed.
3. The session summary is printed to the terminal.
4. The same summary is written to `outputs/session_summary.json`.

Releasing the camera matters — if the program exits without doing so, the device
stays locked and the next run may fail to open it.

## What the statistics mean

| Value | Where it comes from |
| --- | --- |
| Objects | How many detections are in the current frame |
| Classes | The distinct class names in the current frame |
| FPS | Measured from the time between recent frames |
| Confidence | The threshold currently in use |
| Frames / Detections | Running totals since the start or the last `C` press |

The FPS figure is measured, not assumed. It will fluctuate while the program
runs, and it depends on the machine, the camera resolution and the model.
