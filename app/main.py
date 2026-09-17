"""Command line interface and main control loop.

This is where the application is put together::

    parse arguments
    load the YOLO11n model
    open the webcam
    while the camera is running:
        read a frame, detect objects, update statistics,
        draw the results, show the frame, handle a key press
    release the camera

Run it with ``python -m app``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import cv2

from app.analytics import FpsCounter, SessionStats, count_classes
from app.camera import Camera, CameraError, backend_name
from app.controls import (
    MAX_CONFIDENCE,
    MIN_CONFIDENCE,
    Action,
    adjust_confidence,
    control_help_lines,
    interpret_key,
)
from app.detector import DEFAULT_MODEL, Detector, ModelError, describe_device
from app.renderer import Renderer

__all__ = ["main", "build_parser", "save_snapshot", "WINDOW_NAME"]

#: Title of the OpenCV window.
WINDOW_NAME = "Real-Time Object Detection Using YOLO11"

#: Where snapshots and the session summary are written.
OUTPUT_DIR = Path("outputs")

#: How long a short status message stays on screen, in seconds.
MESSAGE_SECONDS = 2.5

#: Stop if the camera fails to deliver this many frames in a row.
MAX_READ_FAILURES = 30


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #


def _confidence(value: str) -> float:
    """Validate a ``--conf`` value."""
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not a number") from None
    if not MIN_CONFIDENCE <= number <= MAX_CONFIDENCE:
        raise argparse.ArgumentTypeError(
            f"confidence must be between {MIN_CONFIDENCE} and {MAX_CONFIDENCE}, got {number}"
        )
    return number


def _dimension(value: str) -> int:
    """Validate a ``--width`` or ``--height`` value."""
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not a whole number") from None
    if number <= 0:
        raise argparse.ArgumentTypeError(f"must be greater than zero, got {number}")
    return number


def build_parser() -> argparse.ArgumentParser:
    """Build the command line argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m app",
        description=(
            "Real-Time Object Detection Using YOLO11.\n"
            "Opens the webcam, detects objects on every frame with the pretrained "
            "YOLO11n model and draws the results on screen."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Controls inside the window:\n  "
            + "\n  ".join(control_help_lines())
            + "\n\nExamples:\n"
            "  python -m app\n"
            "  python -m app --conf 0.40\n"
            "  python -m app --camera 1 --width 1280 --height 720\n"
        ),
    )
    parser.add_argument(
        "--camera", type=int, default=0, metavar="INDEX",
        help="Webcam device index (default: 0).",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, metavar="PATH",
        help=f"YOLO11 weights to use (default: {DEFAULT_MODEL}, downloaded on first run).",
    )
    parser.add_argument(
        "--conf", type=_confidence, default=0.25, metavar="FLOAT",
        help="Confidence threshold (default: 0.25). Lower detects more objects.",
    )
    parser.add_argument(
        "--width", type=_dimension, default=None, metavar="PIXELS",
        help="Requested capture width. Must be given together with --height.",
    )
    parser.add_argument(
        "--height", type=_dimension, default=None, metavar="PIXELS",
        help="Requested capture height. Must be given together with --width.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse and validate the command line arguments."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if (args.width is None) != (args.height is None):
        parser.error("--width and --height must be given together")
    if args.camera < 0:
        parser.error("--camera must be zero or greater")

    return args


# --------------------------------------------------------------------------- #
# Output files
# --------------------------------------------------------------------------- #


def save_snapshot(frame, output_dir: Path = OUTPUT_DIR) -> Path:
    """Write the current annotated frame to a timestamped JPEG."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"snapshot_{time.strftime('%Y_%m_%d_%H%M%S')}.jpg"

    if not cv2.imwrite(str(path), frame):
        raise OSError(f"could not write the snapshot to {path}")
    return path


def write_session_summary(stats: SessionStats, output_dir: Path = OUTPUT_DIR) -> Path:
    """Write the session statistics to ``session_summary.json``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "session_summary.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(stats.as_dict(), handle, indent=2)
        handle.write("\n")
    return path


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #


def run(args: argparse.Namespace) -> int:
    """Run the application and return the process exit code."""
    detector = Detector(model=args.model, confidence=args.conf)
    camera = Camera(index=args.camera, width=args.width, height=args.height)

    # Load the model first, so a missing model is reported before the camera opens.
    try:
        detector.load()
    except ModelError as error:
        print(f"\n{error}\n", file=sys.stderr)
        return 1
    print(f"Loaded {args.model} with {len(detector.class_names)} object classes.", flush=True)

    try:
        camera.open()
    except CameraError as error:
        print(f"\n{error}\n", file=sys.stderr)
        return 1

    print("=" * 66)
    print(f"  Model      : {args.model}")
    print(f"  Device     : {describe_device(detector.device)}")
    print(f"  Confidence : {args.conf:.2f}")
    print(f"  Capture    : {camera.describe()}  ({backend_name()})")
    print(f"  Output     : {OUTPUT_DIR.resolve()}")
    print("-" * 66)
    for line in control_help_lines():
        print(f"  {line}")
    print("=" * 66)
    print()

    renderer = Renderer()
    fps_counter = FpsCounter()
    session = SessionStats()
    help_lines = control_help_lines()
    confidence = args.conf
    message = ""
    message_until = 0.0
    read_failures = 0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                read_failures += 1
                if camera.is_file_source or read_failures >= MAX_READ_FAILURES:
                    break
                continue
            read_failures = 0

            # 1. Detect the objects in this frame.
            detections = detector.detect(frame)

            # 2. Update the statistics.
            fps = fps_counter.tick(time.perf_counter())
            session.update(detections, fps)
            class_counts = count_classes(detections)

            # 3. Draw the results onto the frame.
            renderer.draw_detections(frame, detections)
            renderer.draw_stats(
                frame,
                object_count=len(detections),
                class_counts=class_counts,
                fps=fps,
                confidence=confidence,
                session_frames=session.frames,
                session_total=session.total_detections,
            )
            renderer.draw_help(frame, help_lines)
            if message and time.perf_counter() < message_until:
                # Sit just above the help panel so the two do not overlap.
                cv2.putText(
                    frame, message, (12, frame.shape[0] - 112),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
                )

            # 4. Show the frame.
            cv2.imshow(WINDOW_NAME, frame)

            # 5. Handle a key press, if there is one.
            action = interpret_key(cv2.waitKey(1) & 0xFF)
            if action is Action.QUIT:
                break
            elif action is Action.SNAPSHOT:
                try:
                    message = f"Snapshot saved: {save_snapshot(frame)}"
                    print(message)
                except OSError as error:
                    message = f"Snapshot failed: {error}"
                    print(message, file=sys.stderr)
                message_until = time.perf_counter() + MESSAGE_SECONDS
            elif action is Action.RESET_STATS:
                session.reset()
                message = "Statistics cleared"
                message_until = time.perf_counter() + MESSAGE_SECONDS
            elif action in (Action.CONFIDENCE_UP, Action.CONFIDENCE_DOWN):
                step = 0.05 if action is Action.CONFIDENCE_UP else -0.05
                confidence = adjust_confidence(confidence, step)
                detector.set_confidence(confidence)
                message = f"Confidence: {confidence:.2f}"
                message_until = time.perf_counter() + MESSAGE_SECONDS

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        camera.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    print("=" * 66)
    print("  Session summary")
    print("=" * 66)
    for line in session.summary():
        print(f"  {line}")
    print("=" * 66)

    try:
        print(f"\nSession statistics written to: {write_session_summary(session)}\n")
    except OSError as error:
        print(f"\nWarning: could not write the session summary: {error}\n", file=sys.stderr)

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Application entry point."""
    args = parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        print()
        return 130


if __name__ == "__main__":
    sys.exit(main())
