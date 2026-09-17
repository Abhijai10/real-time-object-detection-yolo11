"""Command-line interface and application controller.

This module owns the *orchestration* of the project: it parses and validates the
command line, wires the five functional modules together, runs the capture loop
and prints the terminal summary.

Responsibilities split:

* :func:`build_parser` / :func:`parse_args` -- argument parsing and validation.
* :class:`Application` -- the real-time control loop (Application Controller in
  the architecture diagram).
* :func:`main` -- process entry point used by ``python -m app``.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from app.analytics import FpsCounter, SessionStats, compute_frame_analytics
from app.camera import Camera, CameraError, available_camera_indices, capture_backend_name
from app.controls import CONTROL_HELP, ControlAction, ControlState, interpret_key
from app.detector import DEFAULT_MODEL, DetectionError, Detector, ModelLoadError, describe_device, resolve_device
from app.renderer import Renderer
from app.utils import (
    MAX_CONFIDENCE,
    MIN_CONFIDENCE,
    RecordingError,
    SnapshotError,
    VideoRecorder,
    configure_logging,
    ensure_directory,
    format_duration,
    get_logger,
    is_supported_video_extension,
    save_snapshot,
    session_summary_path,
    validate_camera_index,
    validate_confidence,
    validate_dimension,
    validate_resolution,
    write_json,
)

__all__ = ["AppConfig", "Application", "build_parser", "main", "parse_args"]

logger = get_logger(__name__)

#: Exit codes returned by :func:`main`.
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_INTERRUPTED = 130

#: Title of the OpenCV window.
WINDOW_NAME = "Real-Time Object Detection and Scene Analytics - YOLO11"

#: How long a transient on-screen message stays visible, in seconds.
MESSAGE_DURATION = 2.5

#: Number of consecutive failed frame reads tolerated before the loop stops.
MAX_READ_FAILURES = 30

PROJECT_VERSION = "1.0.0"


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass
class AppConfig:
    """Validated runtime configuration.

    Attributes:
        camera: Device index, or a path to a video file (headless development).
        model: YOLO11 checkpoint name or path to a local ``.pt`` file.
        confidence: Confidence threshold in ``[0.01, 1.0]``.
        iou: IoU threshold used by non-maximum suppression.
        image_size: Inference resolution in pixels.
        device: ``"auto"``, ``"cpu"``, ``"mps"`` or a CUDA index.
        width: Requested capture width, or ``None`` for the driver default.
        height: Requested capture height, or ``None`` for the driver default.
        output_dir: Directory for snapshots, recordings and the session summary.
        max_frames: Stop after this many frames; ``0`` means run until the user
            quits.
        display: When ``False`` no OpenCV window is created (headless mode).
        show_help: Whether the on-screen keyboard help panel is drawn.
        save_summary: Whether ``session_summary.json`` is written at exit.
        verbose: Enable debug logging.
        quiet: Only log warnings and errors.
    """

    camera: int | str = 0
    model: str = DEFAULT_MODEL
    confidence: float = 0.25
    iou: float = 0.45
    image_size: int = 640
    device: str = "auto"
    width: int | None = None
    height: int | None = None
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    max_frames: int = 0
    display: bool = True
    show_help: bool = True
    save_summary: bool = True
    verbose: bool = False
    quiet: bool = False

    @property
    def is_file_source(self) -> bool:
        """``True`` when the capture source is a video file, not a camera."""
        return isinstance(self.camera, str) and not str(self.camera).isdigit()

    def validate(self) -> "AppConfig":
        """Validate every field, returning ``self`` for convenient chaining.

        Returns:
            The validated configuration.

        Raises:
            ValueError: If any field is invalid.
        """
        self.confidence = validate_confidence(self.confidence)
        self.width, self.height = validate_resolution(self.width, self.height)

        if self.iou <= 0 or self.iou > 1:
            raise ValueError(f"IoU threshold must be between 0 and 1, received: {self.iou}")
        if self.image_size <= 0:
            raise ValueError(f"Inference size must be greater than zero, received: {self.image_size}")
        if self.max_frames < 0:
            raise ValueError(f"--max-frames must be zero or greater, received: {self.max_frames}")

        if self.is_file_source:
            path = Path(str(self.camera)).expanduser()
            if not path.exists():
                raise ValueError(f"Video source not found: {path}")
            if not is_supported_video_extension(path):
                raise ValueError(
                    f"Unsupported video extension '{path.suffix}' for source: {path}"
                )
            self.camera = str(path)
        else:
            self.camera = validate_camera_index(self.camera)

        try:
            ensure_directory(self.output_dir)
        except (OSError, NotADirectoryError) as exc:
            raise ValueError(f"Output directory is not usable: {exc}") from exc

        return self


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #


def _confidence_type(value: str) -> float:
    """Argparse type for ``--conf``.

    Args:
        value: Raw command line value.

    Returns:
        The validated confidence threshold.

    Raises:
        argparse.ArgumentTypeError: If the value is not a valid threshold.
    """
    try:
        return validate_confidence(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _camera_type(value: str) -> int | str:
    """Argparse type for ``--camera``.

    Accepts either a non-negative device index or the path of a video file.

    Args:
        value: Raw command line value.

    Returns:
        An ``int`` device index or a ``str`` path.

    Raises:
        argparse.ArgumentTypeError: If the value is neither a valid index nor an
            existing, supported video file.
    """
    text = str(value).strip()

    if text.lstrip("-").isdigit():
        try:
            return validate_camera_index(text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc)) from exc

    path = Path(text).expanduser()
    if not path.exists():
        raise argparse.ArgumentTypeError(
            f"'{text}' is neither a camera index nor an existing file"
        )
    if not is_supported_video_extension(path):
        raise argparse.ArgumentTypeError(f"Unsupported video file extension: '{path.suffix}'")
    return str(path)


def _dimension_type(value: str) -> int:
    """Argparse type for ``--width`` and ``--height``.

    Args:
        value: Raw command line value.

    Returns:
        The validated dimension.

    Raises:
        argparse.ArgumentTypeError: If the value is not a positive integer.
    """
    try:
        return validate_dimension(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _default_prog() -> str:
    """Return the program name to show in usage and help messages.

    When the application is launched as ``python -m app`` the interpreter sets
    ``sys.argv[0]`` to ``app/__main__.py``, so the documented invocation is used.
    When it is launched through the installed ``detect-yolo11`` console script,
    that script's own name is shown instead.

    Returns:
        Either ``"python -m app"`` or the invoked script name.
    """
    invoked = Path(sys.argv[0]).name if sys.argv and sys.argv[0] else ""
    if invoked and invoked not in {"__main__.py", "python", "python3", "py", "-c"}:
        return invoked
    return "python -m app"


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    """Build the command line argument parser.

    Args:
        prog: Program name shown in usage messages.  Defaults to
            :func:`_default_prog`.

    Returns:
        The configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog=prog or _default_prog(),
        description=(
            "Real-Time Object Detection and Scene Analytics Using YOLO11.\n"
            "Opens a webcam, detects objects on every frame with the pretrained "
            "YOLO11n model, draws bounding boxes and shows live scene analytics."
        ),
        epilog=(
            "Keyboard controls inside the window:\n  "
            + "\n  ".join(f"{key} - {description}" for key, description in CONTROL_HELP)
            + "\n\nExamples:\n"
            "  python -m app\n"
            "  python -m app --camera 0 --conf 0.50\n"
            "  python -m app --camera 0 --conf 0.40 --device cpu\n"
            "  python -m app --camera 0 --width 1280 --height 720 --output outputs\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--camera",
        type=_camera_type,
        default=0,
        metavar="INDEX",
        help=(
            "Webcam device index (default: 0). A path to a video file is also "
            "accepted, which is useful for headless testing on machines without a camera."
        ),
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        metavar="PATH",
        help=(
            "YOLO11 checkpoint name or path to a local .pt file "
            f"(default: {DEFAULT_MODEL}, downloaded automatically on first use)."
        ),
    )
    parser.add_argument(
        "--conf",
        type=_confidence_type,
        default=0.25,
        metavar="FLOAT",
        help=(
            f"Confidence threshold between {MIN_CONFIDENCE} and {MAX_CONFIDENCE} "
            "(default: 0.25). Lower values detect more objects but add false positives."
        ),
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        metavar="FLOAT",
        help="IoU threshold used by non-maximum suppression (default: 0.45).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        metavar="PIXELS",
        help="Inference resolution in pixels (default: 640). Lower is faster, higher is finer.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        metavar="DEVICE",
        help="Inference device: 'auto' (default), 'cpu', 'mps', or a CUDA index such as '0'.",
    )
    parser.add_argument(
        "--width",
        type=_dimension_type,
        default=None,
        metavar="PIXELS",
        help="Requested capture width. Must be given together with --height.",
    )
    parser.add_argument(
        "--height",
        type=_dimension_type,
        default=None,
        metavar="PIXELS",
        help="Requested capture height. Must be given together with --width.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs"),
        metavar="DIR",
        help="Directory for snapshots, recordings and session_summary.json (default: outputs).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Stop after N frames (default: 0, run until you press Q/ESC). "
            "Useful together with --no-display for automated smoke tests."
        ),
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help=(
            "Run without creating an OpenCV window. Intended for servers, CI and "
            "headless smoke tests; keyboard controls are unavailable in this mode."
        ),
    )
    parser.add_argument(
        "--no-help-overlay",
        action="store_true",
        help="Do not draw the keyboard help panel on the video.",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Do not write outputs/session_summary.json at exit.",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="Probe the first camera indices, print which ones work, then exit.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only log warnings and errors.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {PROJECT_VERSION}",
    )

    return parser


def parse_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, AppConfig]:
    """Parse ``argv`` into a namespace and a validated :class:`AppConfig`.

    Args:
        argv: Argument list.  Defaults to :data:`sys.argv[1:]`.

    Returns:
        A ``(namespace, config)`` tuple.  ``config`` is ``None``-safe only when
        ``--list-cameras`` was requested, in which case it is still returned but
        the caller may ignore it.

    Raises:
        SystemExit: On invalid arguments (argparse behaviour) or when
            configuration validation fails.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    config = AppConfig(
        camera=args.camera,
        model=args.model,
        confidence=args.conf,
        iou=args.iou,
        image_size=args.imgsz,
        device=args.device,
        width=args.width,
        height=args.height,
        output_dir=Path(args.output),
        max_frames=args.max_frames,
        display=not args.no_display,
        show_help=not args.no_help_overlay,
        save_summary=not args.no_summary,
        verbose=args.verbose,
        quiet=args.quiet,
    )

    try:
        config.validate()
    except ValueError as exc:
        parser.error(str(exc))

    return args, config


# --------------------------------------------------------------------------- #
# Application controller
# --------------------------------------------------------------------------- #


class Application:
    """Real-time detection control loop.

    Args:
        config: Validated application configuration.
        camera: Optional pre-built camera (used by tests to inject a fake).
        detector: Optional pre-built detector (used by tests to inject a fake).
        renderer: Optional renderer override.
    """

    def __init__(
        self,
        config: AppConfig,
        camera: Camera | None = None,
        detector: Detector | None = None,
        renderer: Renderer | None = None,
    ) -> None:
        self.config = config
        self.camera = camera or Camera(
            index=config.camera if not config.is_file_source else 0,
            width=config.width,
            height=config.height,
            source=config.camera if config.is_file_source else None,
        )
        self.detector = detector or Detector(
            model=config.model,
            confidence=config.confidence,
            iou=config.iou,
            device=config.device,
            image_size=config.image_size,
        )
        self.renderer = renderer or Renderer()

        self.fps_counter = FpsCounter()
        self.session = SessionStats()
        self.controls = ControlState(confidence=config.confidence, step=0.05)
        self.recorder = VideoRecorder(output_dir=config.output_dir, fps=30.0)

        self.help_lines = [f"{key} - {description}" for key, description in CONTROL_HELP]
        self._message: str = ""
        self._message_expiry: float = 0.0

        self._cv2 = None  # populated on first use in display mode

    # -- terminal reporting ------------------------------------------------ #

    def print_banner(self) -> None:
        """Print the startup banner describing the active configuration."""
        resolved_device = resolve_device(self.config.device)
        width, height = self.camera.actual_width, self.camera.actual_height
        resolution = f"{width}x{height}" if width and height else "driver default"

        lines = [
            "",
            "=" * 72,
            "  Real-Time Object Detection and Scene Analytics Using YOLO11",
            "=" * 72,
            f"  Model            : {self.config.model}",
            f"  Device           : {describe_device(resolved_device)}",
            f"  Confidence       : {self.config.confidence:.2f}",
            f"  IoU (NMS)        : {self.config.iou:.2f}",
            f"  Inference size   : {self.config.image_size} px",
            f"  Capture source   : {self.camera.describe()}",
            f"  Capture backend  : {capture_backend_name()}",
            f"  Resolution       : {resolution}",
            f"  Output directory : {Path(self.config.output_dir).resolve()}",
            f"  Display window   : {'enabled' if self.config.display else 'disabled (headless)'}",
            "-" * 72,
            "  Controls: " + " | ".join(f"{key} = {description}" for key, description in CONTROL_HELP),
            "=" * 72,
            "",
        ]
        print("\n".join(lines))

    def print_summary(self) -> None:
        """Print the end-of-session statistics to the terminal."""
        lines = [
            "",
            "=" * 72,
            "  Session summary",
            "=" * 72,
        ]
        lines += [f"  {line}" for line in self.session.summary_lines()]
        lines.append("=" * 72)
        print("\n".join(lines))

    def _set_message(self, message: str) -> None:
        """Show a transient message on the video for a short time.

        Args:
            message: Text to display.
        """
        self._message = message
        self._message_expiry = time.monotonic() + MESSAGE_DURATION
        logger.info(message)

    def _active_message(self) -> str:
        """Return the currently visible transient message, if any."""
        if self._message and time.monotonic() < self._message_expiry:
            return self._message
        return ""

    # -- action handling --------------------------------------------------- #

    def handle_action(self, action: ControlAction, frame) -> bool:
        """Service a keyboard action.

        Args:
            action: Action returned by :class:`ControlState`.
            frame: Current annotated frame, needed by the snapshot action.

        Returns:
            ``True`` to keep running, ``False`` when the user asked to quit.
        """
        if action is ControlAction.QUIT:
            logger.info("Quit requested by the user")
            return False

        if action is ControlAction.SNAPSHOT:
            try:
                path = save_snapshot(frame, self.config.output_dir)
                self._set_message(f"Snapshot saved: {path}")
            except SnapshotError as exc:
                self._set_message(f"Snapshot failed: {exc}")

        elif action is ControlAction.TOGGLE_RECORDING:
            if self.controls.recording:
                self._start_recording(frame)
            else:
                self._stop_recording()

        elif action is ControlAction.RESET_STATS:
            self.session.reset()
            self._set_message("Session statistics cleared")

        elif action is ControlAction.CONFIDENCE_UP:
            self.detector.set_confidence(self.controls.confidence)
            self._set_message(f"Confidence threshold: {self.controls.confidence:.2f}")

        elif action is ControlAction.CONFIDENCE_DOWN:
            self.detector.set_confidence(self.controls.confidence)
            self._set_message(f"Confidence threshold: {self.controls.confidence:.2f}")

        return True

    def _start_recording(self, frame) -> None:
        """Start recording the annotated output.

        Args:
            frame: Frame whose size defines the recording resolution.
        """
        height, width = frame.shape[:2]
        fps = self.fps_counter.fps or self.camera.reported_fps or 20.0
        try:
            path = self.recorder.start((width, height), fps=fps)
            self._set_message(f"Recording started: {path}")
        except RecordingError as exc:
            self.controls.recording = False
            self._set_message(f"Recording failed: {exc}")

    def _stop_recording(self) -> None:
        """Stop the active recording and report the resulting file."""
        path = self.recorder.stop()
        if path is not None:
            self._set_message(f"Recording saved: {path} ({self.recorder.frames_written} frames)")
        else:
            self._set_message("Recording stopped")

    # -- main loop --------------------------------------------------------- #

    def run(self) -> int:
        """Run the capture/detect/render loop until the user quits.

        Returns:
            A process exit code: ``EXIT_OK`` on a clean exit, ``EXIT_ERROR`` on a
            handled failure.
        """
        try:
            self.detector.load()
        except ModelLoadError as exc:
            print(f"\nError: {exc}\n", file=sys.stderr)
            return EXIT_ERROR

        try:
            self.camera.open()
        except CameraError as exc:
            print(f"\n{exc}\n", file=sys.stderr)
            return EXIT_ERROR

        self.detector.warmup()
        self.print_banner()

        if self.config.display:
            import cv2  # noqa: PLC0415 - only needed when a window is shown

            self._cv2 = cv2
            try:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
            except Exception:  # noqa: BLE001 - window creation can fail on headless builds
                logger.warning("Could not create the display window; continuing headlessly")
                self.config.display = False

        exit_code = EXIT_OK
        read_failures = 0

        try:
            while True:
                ok, frame = self.camera.read()
                if not ok or frame is None:
                    read_failures += 1
                    if read_failures >= MAX_READ_FAILURES or self.camera.is_file_source:
                        if self.camera.is_file_source:
                            logger.info("End of video source reached")
                        else:
                            logger.error(
                                "Lost the video stream after %d consecutive failed reads",
                                read_failures,
                            )
                            exit_code = EXIT_ERROR
                        break
                    continue

                read_failures = 0

                try:
                    detections = self.detector.detect(frame)
                except DetectionError as exc:
                    logger.error("%s", exc)
                    continue

                analytics = compute_frame_analytics(detections)
                fps = self.fps_counter.tick()
                self.session.update(analytics, fps)

                self.renderer.render(
                    frame=frame,
                    detections=detections,
                    analytics=analytics,
                    fps=fps,
                    confidence=self.controls.confidence,
                    session=self.session,
                    help_lines=self.help_lines if self.config.show_help else [],
                    show_overlay=True,
                    recording=self.controls.recording,
                    message=self._active_message(),
                )

                if self.recorder.is_recording:
                    self.recorder.write(frame)

                if self.config.display:
                    self._cv2.imshow(WINDOW_NAME, frame)
                    key = self._cv2.waitKey(1) & 0xFF
                else:
                    key = -1

                action = self.controls.apply(interpret_key(key))
                if not self.handle_action(action, frame):
                    break

                if self.config.max_frames and self.session.frames_processed >= self.config.max_frames:
                    logger.info("Reached the --max-frames limit (%d)", self.config.max_frames)
                    break

        except KeyboardInterrupt:
            print()
            logger.info("Interrupted by the user (Ctrl+C)")
            exit_code = EXIT_INTERRUPTED
        finally:
            self._shutdown()

        return exit_code

    def _shutdown(self) -> None:
        """Release every resource and persist the session statistics."""
        if self.recorder.is_recording:
            path = self.recorder.stop()
            if path is not None:
                logger.info("Recording saved: %s", path)

        self.camera.release()

        if self.config.display and self._cv2 is not None:
            try:
                self._cv2.destroyAllWindows()
                # Give the window manager a moment to process the destroy request.
                self._cv2.waitKey(1)
            except Exception:  # noqa: BLE001 - teardown must never raise
                logger.debug("Ignoring error while destroying windows", exc_info=True)

        self.print_summary()

        if self.config.save_summary:
            try:
                destination = write_json(
                    session_summary_path(self.config.output_dir), self.session.as_dict()
                )
                print(f"\nSession statistics written to: {destination}\n")
            except OSError as exc:
                print(f"\nWarning: could not write the session summary: {exc}\n", file=sys.stderr)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def _list_cameras(limit: int = 5) -> int:
    """Probe camera indices and print the usable ones.

    Args:
        limit: Number of indices to probe, starting at ``0``.

    Returns:
        ``EXIT_OK`` when at least one camera was found, otherwise ``EXIT_ERROR``.
    """
    print(f"\nProbing camera indices 0-{limit - 1} (backend: {capture_backend_name()}) ...")
    try:
        found = available_camera_indices(limit)
    except CameraError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    if found:
        print(f"Usable camera indices: {', '.join(str(index) for index in found)}")
        print(f"Start the application with: python -m app --camera {found[0]}\n")
        return EXIT_OK

    print("No usable camera was found.")
    print("Check that a webcam is connected and that camera permission is granted.\n")
    return EXIT_ERROR


def main(argv: Sequence[str] | None = None) -> int:
    """Application entry point.

    Args:
        argv: Argument list, defaulting to :data:`sys.argv[1:]`.

    Returns:
        A process exit code.
    """
    args, config = parse_args(argv)
    configure_logging(verbose=config.verbose, quiet=config.quiet)

    if args.list_cameras:
        return _list_cameras()

    logger.debug("Configuration: %s", config)

    try:
        return Application(config).run()
    except KeyboardInterrupt:
        print()
        return EXIT_INTERRUPTED
    except (CameraError, ModelLoadError) as exc:
        print(f"\nError: {exc}\n", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover - exercised via python -m app
    sys.exit(main())
