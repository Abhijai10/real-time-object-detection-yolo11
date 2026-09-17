# Examples and Commands

Copy-pasteable commands for every supported workflow. Run all of them from the
**project root** (the directory containing `README.md` and the `app/` folder).

---

## 1. Setup

```bash
# Create and activate a virtual environment
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (cmd.exe)
.venv\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt
```

---

## 2. Basic usage

```bash
# Start with defaults: camera 0, yolo11n.pt, confidence 0.25
python -m app

# Show all options
python -m app --help

# Print the version
python -m app --version
```

---

## 3. Camera selection

```bash
# Built-in camera
python -m app --camera 0

# A second USB camera
python -m app --camera 1

# Find out which indices actually work on this machine
python -m app --list-cameras
```

---

## 4. Confidence threshold

```bash
# Default
python -m app --conf 0.25

# Fewer, more confident detections
python -m app --conf 0.50

# More detections, including uncertain ones
python -m app --conf 0.10
```

The threshold can also be changed **while the application is running** with the
`+` and `-` keys.

---

## 5. Compute device

```bash
# Let the application choose the best available device (default)
python -m app --device auto

# Force CPU inference
python -m app --device cpu

# Use a CUDA GPU (index 0)
python -m app --device 0

# Apple Silicon GPU
python -m app --device mps
```

---

## 6. Camera resolution and inference size

```bash
# Request 1280x720 capture (width and height must be given together)
python -m app --width 1280 --height 720

# Lower the inference resolution for higher FPS on a slow CPU
python -m app --imgsz 416

# Combine both
python -m app --width 1280 --height 720 --imgsz 512
```

---

## 7. Custom model

```bash
# Use a specific local checkpoint
python -m app --model models/yolo11n.pt

# Use a different official Ultralytics checkpoint (downloaded on first use)
python -m app --model yolo11s.pt
```

---

## 8. Output directory

```bash
# Send snapshots, recordings and the session summary to a custom folder
python -m app --output runs/session_01
```

---

## 9. Headless and bounded runs

```bash
# Run 60 frames without opening a window (servers, CI, automated checks)
python -m app --no-display --max-frames 60

# Same, but write no session summary
python -m app --no-display --max-frames 60 --no-summary

# Hide the on-screen keyboard help panel
python -m app --no-help-overlay

# Verbose logging
python -m app --verbose

# Errors and warnings only
python -m app --quiet
```

---

## 10. Combined examples

```bash
# Typical lecture demonstration
python -m app --camera 0 --conf 0.45 --width 1280 --height 720

# Low-power laptop
python -m app --device cpu --imgsz 416 --conf 0.35

# Record a demo to a specific folder
python -m app --camera 0 --conf 0.40 --output demo_output
```

---

## 11. Testing

```bash
# Fast unit tests: no webcam, no model download, no network
pytest -q

# Verbose listing of every test
pytest -v

# Include the tests that load the real YOLO11n weights (~5 MB download)
pytest -q -m slow

# Include the test that requires a physical webcam
pytest -m display

# Everything, including hardware-dependent tests
pytest -m "slow or display"
```

---

## 12. Headless development with a video file

On a machine with no webcam (for example a remote server), a video file can be
used as the capture source. This is a development affordance, not the intended
use of the project.

```bash
python -m app --camera path/to/clip.mp4 --no-display --max-frames 100
```

---

## 13. Using the application as an installed command

```bash
# Install the package into the current environment
pip install -e .

# Then the console script is available
detect-yolo11 --help
detect-yolo11 --camera 0 --conf 0.40
```
