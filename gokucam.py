"""GokuCam: turn Super Saiyan and use Instant Transmission on your webcam."""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from contextlib import ExitStack
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

try:
    from .goku_effect import GokuPoseDetector, GokuTrigger, TELEPORT_SECONDS, render_teleport
    from .saiyan_effect import SaiyanState, golden_hair, render_saiyan
except ImportError:
    from goku_effect import GokuPoseDetector, GokuTrigger, TELEPORT_SECONDS, render_teleport
    from saiyan_effect import SaiyanState, golden_hair, render_saiyan


APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
WINDOW_NAME = "GokuCam - Preview"

SEGMENTER_MODEL = MODEL_DIR / "selfie_segmenter_landscape.tflite"
HAIR_MODEL = MODEL_DIR / "hair_segmenter.tflite"
HAND_MODEL = APP_DIR / "hand_landmarker.task"
FACE_MODEL = APP_DIR / "face_landmarker.task"
BACKGROUND_FILE = APP_DIR / "background.png"

MODEL_URLS = {
    SEGMENTER_MODEL: (
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
        "selfie_segmenter_landscape/float16/latest/"
        "selfie_segmenter_landscape.tflite"
    ),
    HAIR_MODEL: (
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
        "hair_segmenter/float32/latest/hair_segmenter.tflite"
    ),
}

PERSON_NOTICE_SECONDS = 4.0
BACKGROUND_SAMPLE_SECONDS = 0.55
GESTURE_INTERVAL_SECONDS = 0.08


@dataclass
class AppState:
    phase: str = "calibrating"
    calibration_started: float = 0.0
    vanish_started: float | None = None
    last_gesture_check: float = 0.0
    fullscreen: bool = False


def positive_fps(value: str) -> float:
    fps = float(value)
    if not np.isfinite(fps) or not 1 <= fps <= 120:
        raise argparse.ArgumentTypeError("FPS must be between 1 and 120.")
    return fps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Turn Super Saiyan and use Instant Transmission on your webcam."
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    parser.add_argument("--width", type=int, default=960, help="Video width")
    parser.add_argument("--height", type=int, default=540, help="Video height")
    parser.add_argument(
        "--fullscreen", action="store_true", help="Start in fullscreen"
    )
    parser.add_argument(
        "--no-mirror", action="store_true", help="Do not mirror the webcam"
    )
    parser.add_argument("--fps", type=positive_fps, default=30.0, help="Maximum virtual camera FPS (default: 30)")
    parser.add_argument(
        "--virtual-camera", action=argparse.BooleanOptionalAction, default=None,
        help="Require the virtual camera; by default it is used when available. --no-virtual-camera turns it off",
    )
    parser.add_argument(
        "--virtual-backend", choices=("obs", "unitycapture", "v4l2loopback"),
        help="Virtual camera driver to use",
    )
    parser.add_argument("--virtual-device", help="Virtual camera device name to use")
    parser.add_argument(
        "--background-file", type=Path, default=BACKGROUND_FILE,
        help="Where to save the background so the next start reuses it (SPACE captures a new one)",
    )
    parser.add_argument(
        "--blur-background", action="store_true",
        help="Blur the background; you vanish into the blurred background",
    )
    parser.add_argument(
        "--gesture-debug", action="store_true",
        help="Print gesture recognition details to the terminal (nothing is drawn on the video)",
    )
    parser.add_argument("--self-test", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def open_virtual_camera(args: argparse.Namespace, width: int, height: int):
    if args.virtual_camera is False:
        return None
    try:
        import pyvirtualcam

        camera = pyvirtualcam.Camera(
            width=width, height=height, fps=args.fps,
            fmt=pyvirtualcam.PixelFormat.BGR,
            backend=args.virtual_backend, device=args.virtual_device,
        )
    except (ImportError, RuntimeError, OSError, ValueError) as error:
        message = (
            "Virtual camera not available. Install OBS Studio (with OBS Virtual "
            "Camera) or Unity Capture. If you use OBS Virtual Camera, stop "
            "'Start Virtual Camera' in OBS. Run ./start.sh to install pyvirtualcam. "
            f"Details: {error}"
        )
        if args.virtual_camera is True or args.virtual_backend or args.virtual_device:
            raise RuntimeError(message) from error
        print(f"[Webcam] {message}\n[Webcam] Continuing with the local preview only.")
        return None
    print(f"[Webcam] Virtual camera on: {camera.device} ({width}x{height}, up to {args.fps:g} FPS)")
    return camera


def send_virtual_frame(camera, frame: np.ndarray) -> None:
    if frame.shape[:2] != (camera.height, camera.width):
        frame = cv2.resize(frame, (camera.width, camera.height), interpolation=cv2.INTER_LINEAR)
    camera.send(np.ascontiguousarray(frame))


def download_models() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for destination, url in MODEL_URLS.items():
        if destination.exists() and destination.stat().st_size > 100_000:
            continue

        temporary = destination.with_suffix(destination.suffix + ".part")
        print(f"[GokuCam] Downloading {destination.name}...")
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "GokuCam/1.0"}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                total = int(response.headers.get("Content-Length", 0))
                received = 0
                with temporary.open("wb") as output:
                    while chunk := response.read(1024 * 256):
                        output.write(chunk)
                        received += len(chunk)
                        if total:
                            percent = int(received * 100 / total)
                            print(f"\r  {percent:3d}%", end="", flush=True)
            print()
            os.replace(temporary, destination)
        except (OSError, urllib.error.URLError) as error:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                "Could not download the MediaPipe models. "
                "Check your internet connection and try again."
            ) from error


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    if platform.system() == "Windows":
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(index)
    else:
        capture = cv2.VideoCapture(index)

    if not capture.isOpened():
        raise RuntimeError(
            f"Webcam {index} is not available. Close other apps using it or try --camera 1."
        )

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def create_goku_detector() -> GokuPoseDetector:
    for model in (HAND_MODEL, FACE_MODEL, HAIR_MODEL):
        if not model.is_file():
            raise RuntimeError(f"Model not found: {model}")
    hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(HAND_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.35,
        min_hand_presence_confidence=0.35,
        min_tracking_confidence=0.45,
    ))
    try:
        face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(FACE_MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.4,
            min_face_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        ))
        hair = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
            base_options=python.BaseOptions(model_asset_path=str(HAIR_MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            output_confidence_masks=True,
            output_category_mask=False,
        ))
    except Exception:
        hands.close()
        if "face" in locals():
            face.close()
        raise
    return GokuPoseDetector(hands, face, hair)


def create_tasks() -> tuple[vision.ImageSegmenter, GokuPoseDetector]:
    segmenter_options = vision.ImageSegmenterOptions(
        base_options=python.BaseOptions(model_asset_path=str(SEGMENTER_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        output_confidence_masks=True,
        output_category_mask=False,
    )
    detector = create_goku_detector()
    try:
        return vision.ImageSegmenter.create_from_options(segmenter_options), detector
    except Exception:
        detector.close()
        raise


def person_mask(
    segmenter: vision.ImageSegmenter,
    image: mp.Image,
    timestamp_ms: int,
    width: int,
    height: int,
) -> np.ndarray:
    result = segmenter.segment_for_video(image, timestamp_ms)
    if not result.confidence_masks:
        return np.zeros((height, width), dtype=np.float32)
    raw = np.squeeze(result.confidence_masks[0].numpy_view()).astype(np.float32)
    resized = cv2.resize(raw, (width, height), interpolation=cv2.INTER_LINEAR)
    resized = cv2.GaussianBlur(resized, (0, 0), sigmaX=2.2, sigmaY=2.2)
    return np.clip((resized - 0.12) / 0.68, 0.0, 1.0)


def person_fraction(
    segmenter: vision.ImageSegmenter, frame: np.ndarray, timestamp_ms: int
) -> float:
    """Share of the frame covered by a person; one left in the background
    would stay visible after vanishing."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    height, width = frame.shape[:2]
    mask = person_mask(segmenter, image, timestamp_ms, width, height)
    return float((mask > 0.5).mean())


def blur_background(background: np.ndarray) -> np.ndarray:
    sigma = max(4.0, background.shape[1] / 70.0)
    return cv2.GaussianBlur(background, (0, 0), sigmaX=sigma, sigmaY=sigma)


def alpha_composite(person: np.ndarray, background: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    alpha_3 = alpha[:, :, None]
    return (
        person.astype(np.float32) * alpha_3
        + background.astype(np.float32) * (1.0 - alpha_3)
    ).astype(np.uint8)


def update_goku(detector: GokuPoseDetector, trigger: GokuTrigger, image: mp.Image,
                timestamp_ms: int, now: float, debug: bool) -> bool:
    """Feed one detection to the hold countdown; True when the teleport fires."""
    pose = detector.detect(image, timestamp_ms)
    event = trigger.update(pose, now)
    if event == "start":
        print("[Goku] Pose recognized: hold it for 1 second...")
    elif event == "broken":
        print("[Goku] Pose lost: the countdown restarts.")
    elif event == "trigger":
        print("[Goku] Instant Transmission!")
    if debug:
        print(f"[Goku] hands={detector.last_hand_count}, forehead={detector.forehead is not None}, "
              f"pose={pose}, remaining={trigger.remaining(now)}")
    return event == "trigger"


def update_saiyan(detector: GokuPoseDetector, trigger: GokuTrigger, saiyan: SaiyanState,
                  now: float) -> None:
    """Both fists held for a second toggle Super Saiyan on and off."""
    event = trigger.update(detector.fists, now)
    if event == "start":
        print("[Saiyan] Fists recognized: hold them for 1 second...")
    elif event == "broken":
        print("[Saiyan] Fists lowered: the countdown restarts.")
    elif event == "trigger":
        saiyan.toggle(now)
        print("[Saiyan] Super Saiyan!" if saiyan.active else "[Saiyan] Back to normal.")


def set_fullscreen(enabled: bool) -> None:
    mode = cv2.WINDOW_FULLSCREEN if enabled else cv2.WINDOW_NORMAL
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, mode)


def self_test() -> int:
    """Check bundled resources and model initialization without opening a webcam."""
    required = (SEGMENTER_MODEL, HAIR_MODEL, HAND_MODEL, FACE_MODEL)
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing resources: {', '.join(missing)}")
    segmenter, detector = create_tasks()
    try:
        detector.close()
    finally:
        segmenter.close()
    print("[Webcam] Self-test passed: all models load.")
    return 0


def run() -> int:
    args = parse_args()
    if getattr(args, "self_test", False):
        return self_test()
    with ExitStack() as resources:
        return run_camera(args, resources)


def run_camera(args: argparse.Namespace, resources: ExitStack) -> int:
    download_models()
    capture = open_camera(args.camera, args.width, args.height)
    resources.callback(capture.release)
    capture.set(cv2.CAP_PROP_FPS, args.fps)
    segmenter, detector = create_tasks()
    resources.callback(segmenter.close)
    resources.callback(detector.close)
    goku_trigger = GokuTrigger()
    # Two hands track less steadily than one, so forgive longer dropouts.
    saiyan_trigger = GokuTrigger(grace_seconds=0.6)
    saiyan = SaiyanState()
    hair_mask: np.ndarray | None = None
    hair_mask_at = float("-inf")
    resources.callback(cv2.destroyAllWindows)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, args.width, args.height)

    state = AppState(
        calibration_started=time.perf_counter(), fullscreen=args.fullscreen
    )
    if state.fullscreen:
        set_fullscreen(True)

    background: np.ndarray | None = None
    # Reuse the last clean background so the person does not have to leave
    # the frame on every start; SPACE captures a new one.
    saved_background = None
    if args.background_file is not None and args.background_file.is_file():
        saved_background = cv2.imread(str(args.background_file))
    background_samples: list[tuple[float, np.ndarray]] = []
    empty_since: float | None = None
    last_mask: np.ndarray | None = None
    last_mask_at = float("-inf")
    vanished_frame: np.ndarray | None = None
    virtual_camera = None
    virtual_camera_initialized = False
    person_notice_at = float("-inf")
    started = time.perf_counter()
    print("[Webcam] Step out of the frame: the background is captured as soon as it is empty.")
    print("[Webcam] Fists up for 1 s: Super Saiyan. Two fingers on the forehead for 1 s: "
          "Instant Transmission.")
    print("[Webcam] Window keys: SPACE new background, F fullscreen, Q quit.")

    try:
        while True:
            ok, camera_frame = capture.read()
            if not ok:
                raise RuntimeError("The webcam stopped returning images.")

            if not args.no_mirror:
                camera_frame = cv2.flip(camera_frame, 1)

            height, width = camera_frame.shape[:2]
            if not virtual_camera_initialized:
                virtual_camera = open_virtual_camera(args, width, height)
                virtual_camera_initialized = True
                if virtual_camera is not None:
                    resources.callback(virtual_camera.close)
            now = time.perf_counter()

            elapsed_ms = int((now - started) * 1000)
            mp_image = None

            if state.phase == "calibrating":
                if saved_background is not None:
                    if saved_background.shape == camera_frame.shape:
                        background = (blur_background(saved_background) if args.blur_background
                                      else saved_background)
                        state.phase = "ready"
                        print("[Webcam] Using the saved background. Press SPACE to capture a new one.")
                    saved_background = None
                # Capture as soon as the frame has been empty for a moment, so
                # the person does not have to time leaving with a countdown.
                elif (fraction := person_fraction(segmenter, camera_frame, elapsed_ms)) > 0.02:
                    background_samples.clear()
                    empty_since = None
                    if now - person_notice_at >= PERSON_NOTICE_SECONDS:
                        print(f"[Webcam] You are still in the frame ({fraction:.0%} "
                              "of the image): step all the way out.")
                        person_notice_at = now
                else:
                    if empty_since is None:
                        empty_since = now
                    background_samples.append((now, camera_frame.copy()))
                    background_samples = [
                        sample for sample in background_samples
                        if sample[0] >= now - BACKGROUND_SAMPLE_SECONDS
                    ]
                    if now - empty_since >= BACKGROUND_SAMPLE_SECONDS:
                        stack = np.stack([frame for _, frame in background_samples[-18:]], axis=0)
                        candidate = np.median(stack, axis=0).astype(np.uint8)
                        background_samples.clear()
                        empty_since = None
                        if args.background_file is not None:
                            cv2.imwrite(str(args.background_file), candidate)
                        # Every effect composites over this plate, so blurring it
                        # once also blurs what is left behind after vanishing.
                        background = blur_background(candidate) if args.blur_background else candidate
                        state.phase = "ready"
                        print("[Webcam] Background ready. You can step back in.")
                display = camera_frame

            elif background is None:
                state.phase = "calibrating"
                state.calibration_started = now
                display = camera_frame

            elif state.phase == "vanished":
                if vanished_frame is None:
                    vanished_frame = background.copy()
                display = vanished_frame
                # You come back with the same gesture, so keep watching while invisible.
                if now - state.last_gesture_check >= GESTURE_INTERVAL_SECONDS:
                    state.last_gesture_check = now
                    rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    if update_goku(detector, goku_trigger, mp_image, elapsed_ms, now,
                                   args.gesture_debug):
                        state.phase = "appearing"
                        state.vanish_started = now
                        vanished_frame = None
                        last_mask = None

            elif state.phase in ("vanishing", "appearing"):
                appearing = state.phase == "appearing"
                progress = min(1.0, (now - state.vanish_started) / TELEPORT_SECONDS)
                # Once the person is gone, freeze the silhouette so the lines
                # linger where they stood rather than following the real body.
                frozen = not appearing and progress >= 0.5 and last_mask is not None
                if not frozen and (last_mask is None or now - last_mask_at >= 0.05):
                    rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    last_mask = person_mask(segmenter, mp_image, elapsed_ms, width, height)
                    last_mask_at = now
                teleport_frame = camera_frame
                level = saiyan.level(now)
                if level > 0.0 and hair_mask is not None:
                    # The golden hair teleports with the person.
                    teleport_frame = golden_hair(camera_frame, hair_mask, last_mask, level)
                display = render_teleport(
                    teleport_frame, background, last_mask,
                    1.0 - progress if appearing else progress, now,
                )
                if progress >= 1.0:
                    state.phase = "ready" if appearing else "vanished"
                    state.vanish_started = None
                    last_mask = None
                    last_mask_at = float("-inf")

            else:
                mask = last_mask
                if now - state.last_gesture_check >= GESTURE_INTERVAL_SECONDS:
                    state.last_gesture_check = now
                    rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    if update_goku(detector, goku_trigger, mp_image, elapsed_ms, now,
                                   args.gesture_debug):
                        state.phase = "vanishing"
                        state.vanish_started = now
                        last_mask = None
                    else:
                        update_saiyan(detector, saiyan_trigger, saiyan, now)

                display = camera_frame
                level = saiyan.level(now)
                if args.blur_background or level > 0.0:
                    if mp_image is None:
                        rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    mask = person_mask(segmenter, mp_image, elapsed_ms, width, height)
                if args.blur_background:
                    display = alpha_composite(camera_frame, background, mask)
                if level > 0.0:
                    # Hair segmentation is the slowest model; refresh it
                    # every other frame, which still follows the head.
                    if hair_mask is None or now - hair_mask_at >= 0.06:
                        hair_mask = detector.hair_mask(mp_image, elapsed_ms)
                        hair_mask_at = now
                    display = render_saiyan(display, mask, hair_mask, level,
                                            saiyan.burst(now), now)

            if virtual_camera is not None:
                # Video apps mirror the self-view themselves and send the camera
                # image as is, so undo our preview mirroring for the output.
                send_virtual_frame(virtual_camera, display if args.no_mirror else cv2.flip(display, 1))
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                state.phase = "calibrating"
                state.calibration_started = now
                state.vanish_started = None
                last_mask = None
                last_mask_at = float("-inf")
                vanished_frame = None
                background_samples.clear()
                empty_since = None
                person_notice_at = float("-inf")
                goku_trigger.reset()
                saiyan_trigger.reset()
                print("[Webcam] Step out of the frame: the new background is captured as soon as it is empty.")
            elif key == ord("f"):
                state.fullscreen = not state.fullscreen
                set_fullscreen(state.fullscreen)
            if virtual_camera is not None:
                virtual_camera.sleep_until_next_frame()
    finally:
        resources.close()

    return 0


def main() -> int:
    try:
        return run()
    except KeyboardInterrupt:
        return 0
    except Exception as error:
        print(f"\n[GokuCam] ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
