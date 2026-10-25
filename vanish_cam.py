"""VanishCam: vanish from your webcam with a gesture, or teleport like Goku."""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
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
WINDOW_NAME = "VanishCam - Preview"

SEGMENTER_MODEL = MODEL_DIR / "selfie_segmenter_landscape.tflite"
GESTURE_MODEL = MODEL_DIR / "gesture_recognizer.task"
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
    GESTURE_MODEL: (
        "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
        "gesture_recognizer/float16/1/gesture_recognizer.task"
    ),
    HAIR_MODEL: (
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
        "hair_segmenter/float32/latest/hair_segmenter.tflite"
    ),
}

PERSON_NOTICE_SECONDS = 4.0
BACKGROUND_SAMPLE_SECONDS = 0.55
VANISH_SECONDS = 3.6
GESTURE_INTERVAL_SECONDS = 0.14


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
        description="Vanish from your webcam with a V sign, or teleport like Goku."
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
        "--mode", choices=("normal", "goku"),
        help="Skip the start menu and run this mode directly",
    )
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


def choose_mode() -> str | None:
    """Simple keyboard/mouse menu shown before the clean camera feed."""
    title = "Choose an effect"
    selected = [None]
    window = np.zeros((420, 700, 3), dtype=np.uint8)
    window[:] = (26, 29, 34)
    cv2.putText(window, "CHOOSE YOUR VANISH", (68, 88),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (236, 239, 242), 2, cv2.LINE_AA)
    buttons = (
        ("normal", "1  NORMAL VANISH", "V sign  /  dissolve", 140),
        ("goku", "2  GOKU", "Two fingers on the forehead  /  teleport", 260),
    )
    for mode, heading, detail, top in buttons:
        cv2.rectangle(window, (65, top), (635, top + 94), (48, 54, 62), -1, cv2.LINE_AA)
        cv2.rectangle(window, (65, top), (635, top + 94), (111, 142, 153), 2, cv2.LINE_AA)
        cv2.putText(window, heading, (90, top + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.77, (240, 245, 247), 2, cv2.LINE_AA)
        cv2.putText(window, detail, (90, top + 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.53, (179, 195, 201), 1, cv2.LINE_AA)

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONUP and 65 <= x <= 635:
            for mode, _, _, top in buttons:
                if top <= y <= top + 94:
                    selected[0] = mode

    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(title, on_mouse)
    try:
        while selected[0] is None:
            cv2.imshow(title, window)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("1"):
                selected[0] = "normal"
            elif key == ord("2"):
                selected[0] = "goku"
            elif key in (ord("q"), 27):
                return None
            if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                return None
        return selected[0]
    finally:
        cv2.destroyWindow(title)


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


def download_models(mode: str = "normal") -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for destination, url in MODEL_URLS.items():
        if mode != "normal" and destination == GESTURE_MODEL:
            continue
        if mode != "goku" and destination == HAIR_MODEL:
            continue
        if destination.exists() and destination.stat().st_size > 100_000:
            continue

        temporary = destination.with_suffix(destination.suffix + ".part")
        print(f"[VanishCam] Downloading {destination.name}...")
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "VanishCam/1.0"}
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


def create_tasks(mode: str = "normal") -> tuple[vision.ImageSegmenter, MultiViewGestureRecognizer | GokuPoseDetector]:
    segmenter_options = vision.ImageSegmenterOptions(
        base_options=python.BaseOptions(model_asset_path=str(SEGMENTER_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        output_confidence_masks=True,
        output_category_mask=False,
    )
    if mode == "goku":
        detector = create_goku_detector()
        try:
            return vision.ImageSegmenter.create_from_options(segmenter_options), detector
        except Exception:
            detector.close()
            raise
    classifier_options = mp.tasks.components.processors.ClassifierOptions(
        score_threshold=0.55,
        category_allowlist=["Victory"],
    )
    options = vision.GestureRecognizerOptions(
        base_options=python.BaseOptions(model_asset_path=str(GESTURE_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.35,
        min_hand_presence_confidence=0.35,
        min_tracking_confidence=0.5,
        canned_gesture_classifier_options=classifier_options,
    )

    segmenter = vision.ImageSegmenter.create_from_options(segmenter_options)
    try:
        primary = vision.GestureRecognizer.create_from_options(options)
        try:
            alternate = vision.GestureRecognizer.create_from_options(
                replace(options, running_mode=vision.RunningMode.IMAGE)
            )
        except Exception:
            primary.close()
            raise
    except Exception:
        segmenter.close()
        raise
    return segmenter, MultiViewGestureRecognizer(primary, alternate)


def smoothstep(edge0: float, edge1: float, values: np.ndarray) -> np.ndarray:
    values = np.clip((values - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return values * values * (3.0 - 2.0 * values)


def ease_in_out_cubic(value: float) -> float:
    if value < 0.5:
        return 4.0 * value * value * value
    return 1.0 - ((-2.0 * value + 2.0) ** 3) / 2.0


def make_dissolve_fields(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7391)
    y, x = np.mgrid[0:height, 0:width]
    y_normal = y.astype(np.float32) / max(height - 1, 1)
    x_normal = x.astype(np.float32) / max(width - 1, 1)
    low_noise = rng.random((max(2, height // 10), max(2, width // 10)), dtype=np.float32)
    low_noise = cv2.resize(low_noise, (width, height), interpolation=cv2.INTER_CUBIC)
    fine_noise = rng.random((height, width), dtype=np.float32)
    field = (
        (1.0 - y_normal) * 0.72
        + low_noise * 0.17
        + fine_noise * 0.08
        + np.sin(x_normal * 18.0) * 0.025
    )
    return np.clip(field, 0.0, 1.0), fine_noise


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


def apply_dissolve(
    frame: np.ndarray,
    background: np.ndarray,
    mask: np.ndarray,
    progress: float,
    field: np.ndarray,
    sparkle_noise: np.ndarray,
) -> np.ndarray:
    if progress <= 0.002:
        return alpha_composite(frame, background, mask)
    if progress >= 0.999:
        return background.copy()

    points = cv2.findNonZero((mask > 0.01).astype(np.uint8))
    if points is None:
        return background.copy()
    x, y, width, height = cv2.boundingRect(points)
    left, top = max(0, x - 18), max(0, y - 18)
    right = min(mask.shape[1], x + width + 18)
    bottom = min(mask.shape[0], y + height + 18)
    area = np.s_[top:bottom, left:right]
    mask = mask[area]
    field = field[area]
    sparkle_noise = sparkle_noise[area]
    eased = ease_in_out_cubic(progress)
    survival = smoothstep(eased - 0.055, eased + 0.055, field)
    alpha = mask * survival
    output = alpha_composite(frame[area], background[area], alpha)
    edge = (
        (np.abs(field - eased) < 0.018)
        & (mask > 0.48)
        & (sparkle_noise > 0.54)
    ).astype(np.float32)
    glow = cv2.GaussianBlur(edge, (0, 0), sigmaX=5.0, sigmaY=5.0)
    glow = np.clip(glow * 5.0 + edge, 0.0, 1.0)[:, :, None]
    cyan = np.full_like(output, (255, 228, 92), dtype=np.uint8)
    result = np.clip(
        output.astype(np.float32) * (1.0 - glow * 0.8)
        + cyan.astype(np.float32) * glow * 0.8,
        0,
        255,
    ).astype(np.uint8)
    canvas = background.copy()
    canvas[area] = result
    return canvas


def victory_landmark_score(points: np.ndarray) -> float:
    """Recognize straight index and middle fingers from 3D distances, not orientation."""
    if points.shape != (21, 3) or not np.isfinite(points).all():
        return 0.0

    straightness = []
    extended = []
    folded = []
    lengths = []
    # MCP, PIP, DIP and tip: index, middle, ring, pinky.
    for index, base in enumerate((5, 9, 13, 17)):
        finger = points[base : base + 4]
        length = float(np.linalg.norm(np.diff(finger, axis=0), axis=1).sum())
        if length < 1e-6:
            if index < 2:
                return 0.0
            straightness.append(0.0)
            lengths.append(0.0)
            extended.append(False)
            folded.append(True)
            continue
        ratio = float(np.linalg.norm(finger[3] - finger[0])) / length
        tip_distance = float(np.linalg.norm(finger[3] - points[0]))
        joint_distance = float(np.linalg.norm(finger[1] - points[0]))
        straightness.append(ratio)
        lengths.append(length)
        extended.append(ratio >= 0.82 and tip_distance > joint_distance * 1.10)
        folded.append(ratio < 0.78 or tip_distance <= joint_distance * 1.05)

    # Seen from the side, folded fingers can look straight but very short:
    # do not require clearly visible joints for the ring and pinky fingers.
    reference_length = min(lengths[:2])
    for index, base in ((2, 13), (3, 17)):
        reach = float(np.linalg.norm(points[base + 3] - points[base]))
        folded[index] = folded[index] or reach < reference_length * 0.65
    if not (extended[0] and extended[1] and folded[2] and folded[3]):
        return 0.0
    # The two tips must be apart; the thumb may stay hidden.
    separation = float(np.linalg.norm(points[8] - points[12]))
    if separation < min(lengths[:2]) * 0.20:
        return 0.0
    # Geometric quality of the straight fingers, not a classifier probability.
    return float(np.clip(min(straightness[:2]), 0.0, 1.0))


def classify_gesture_result(
    result: vision.GestureRecognizerResult,
    image: mp.Image,
) -> tuple[str, float]:
    # Landmarks are returned even when the classifier does not recognize
    # "Victory" (for example with the back of the hand facing the webcam).
    for landmarks in result.hand_world_landmarks:
        points = np.array([(point.x, point.y, point.z) for point in landmarks])
        score = victory_landmark_score(points)
        if score:
            return "Victory", score
    # ALWAYS check the visible geometry too: the estimated depth can be
    # inaccurate when the palm and folded fingers are hidden.
    for landmarks in result.hand_landmarks:
        points = np.array([(point.x * image.width, point.y * image.height, 0.0)
                           for point in landmarks])
        score = victory_landmark_score(points)
        if score:
            return "Victory", score
    for categories in result.gestures:
        for category in categories:
            if category.category_name == "Victory" and category.score >= 0.62:
                return "Victory", float(category.score)
    if result.hand_landmarks or result.hand_world_landmarks:
        return "Hand_Detected", 0.0
    return "No_Hand", 0.0


class MultiViewGestureRecognizer:
    """Retry detection on rotated frames without disturbing video tracking."""

    def __init__(self, primary, alternate):
        self.primary = primary
        self.alternate = alternate
        self.preferred_rotation: int | None = None
        self.preferred_crop: str | None = None
        self.next_retry_ms = 0
        self.retry_cursor = 0

    def recognize_gesture(self, image: mp.Image, timestamp_ms: int) -> tuple[str, float]:
        result = self.primary.recognize_for_video(image, timestamp_ms)
        best = classify_gesture_result(result, image)
        if best[0] == "Victory":
            self.preferred_rotation = None
            self.preferred_crop = None
            return best
        if timestamp_ms < self.next_retry_ms:
            return best

        # IMAGE mode forces a fresh detection while the VIDEO model stays on the
        # original frame, avoiding inconsistent timestamps across rotations.
        rgb = image.numpy_view()
        height, width = rgb.shape[:2]
        # A sideways palm is often too small in the full frame. The two lower
        # crops overlap and cover both hands, even when the image is mirrored.
        crops = {
            "full": rgb,
            "lower_left": rgb[int(height * 0.28):, :int(width * 0.78)],
            "lower_right": rgb[int(height * 0.28):, int(width * 0.22):],
        }
        views = [("full", rotation) for rotation in (0, 1, 3, 2)]
        views += [(crop, rotation) for crop in ("lower_left", "lower_right")
                  for rotation in (0, 1, 3)]
        preferred = (self.preferred_crop, self.preferred_rotation)
        selected = []
        if preferred in views:
            selected.append(preferred)
        # Spread the expensive IMAGE attempts over successive video frames.
        while len(selected) < 3:
            view = views[self.retry_cursor % len(views)]
            self.retry_cursor += 1
            if view not in selected:
                selected.append(view)
        for crop, rotation in selected:
            rotated = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(np.rot90(crops[crop], rotation)),
            )
            result = self.alternate.recognize(rotated)
            candidate = classify_gesture_result(result, rotated)
            if candidate[0] == "Victory":
                self.preferred_rotation = rotation
                self.preferred_crop = crop
                self.next_retry_ms = timestamp_ms + int(GESTURE_INTERVAL_SECONDS * 1000)
                return candidate
            if candidate[0] == "Hand_Detected":
                best = candidate
        self.preferred_rotation = None
        self.preferred_crop = None
        self.next_retry_ms = timestamp_ms + int(GESTURE_INTERVAL_SECONDS * 1000)
        return best

    def close(self) -> None:
        try:
            self.primary.close()
        finally:
            self.alternate.close()


def recognize_gesture(
    recognizer: MultiViewGestureRecognizer,
    image: mp.Image,
    timestamp_ms: int,
) -> tuple[str, float]:
    if isinstance(recognizer, MultiViewGestureRecognizer):
        return recognizer.recognize_gesture(image, timestamp_ms)
    return classify_gesture_result(recognizer.recognize_for_video(image, timestamp_ms), image)


def start_vanish_on_gesture(state: AppState, name: str, score: float, now: float) -> bool:
    """Start the dissolve on the first reliable V sign."""
    if state.phase == "ready" and name == "Victory" and score >= 0.62:
        state.phase = "vanishing"
        state.vanish_started = now
        return True
    return False


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
    required = (SEGMENTER_MODEL, GESTURE_MODEL, HAIR_MODEL, HAND_MODEL, FACE_MODEL)
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing resources: {', '.join(missing)}")
    for mode in ("normal", "goku"):
        segmenter, recognizer = create_tasks(mode)
        try:
            recognizer.close()
        finally:
            segmenter.close()
    print("[Webcam] Self-test passed: all models load.")
    return 0


def run() -> int:
    args = parse_args()
    if getattr(args, "self_test", False):
        return self_test()
    mode = args.mode or choose_mode()
    if mode is None:
        return 0
    with ExitStack() as resources:
        return run_camera(args, resources, mode)


def run_camera(args: argparse.Namespace, resources: ExitStack, mode: str) -> int:
    download_models(mode)
    capture = open_camera(args.camera, args.width, args.height)
    resources.callback(capture.release)
    capture.set(cv2.CAP_PROP_FPS, args.fps)
    segmenter, recognizer = create_tasks(mode)
    resources.callback(segmenter.close)
    resources.callback(recognizer.close)
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
    dissolve_field: np.ndarray | None = None
    sparkle_noise: np.ndarray | None = None
    vanished_frame: np.ndarray | None = None
    virtual_camera = None
    virtual_camera_initialized = False
    person_notice_at = float("-inf")
    last_gesture_report = 0.0
    started = time.perf_counter()
    print("[Webcam] Step out of the frame: the background is captured as soon as it is empty.")
    if mode == "goku":
        print("[Webcam] Window keys: SPACE new background, F fullscreen, Q quit.")
    else:
        print("[Webcam] Window keys: SPACE new background, R reappear, V vanish, F fullscreen, Q quit.")

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
            if mode == "normal" and (dissolve_field is None or dissolve_field.shape != (height, width)):
                dissolve_field, sparkle_noise = make_dissolve_fields(height, width)
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
                # Goku comes back with the same gesture, so keep watching while invisible.
                if mode == "goku" and now - state.last_gesture_check >= 0.08:
                    state.last_gesture_check = now
                    rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    if update_goku(recognizer, goku_trigger, mp_image, elapsed_ms, now,
                                   args.gesture_debug):
                        state.phase = "goku_appearing"
                        state.vanish_started = now
                        vanished_frame = None
                        last_mask = None

            elif state.phase in ("goku_vanishing", "goku_appearing"):
                appearing = state.phase == "goku_appearing"
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
                if (
                    state.phase == "ready"
                    and now - state.last_gesture_check >=
                    (0.08 if mode != "normal" else GESTURE_INTERVAL_SECONDS)
                ):
                    state.last_gesture_check = now
                    rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    if mode == "goku":
                        if update_goku(recognizer, goku_trigger, mp_image, elapsed_ms, now,
                                       args.gesture_debug):
                            state.phase = "goku_vanishing"
                            state.vanish_started = now
                            last_mask = None
                        else:
                            update_saiyan(recognizer, saiyan_trigger, saiyan, now)
                    else:
                        name, score = recognize_gesture(recognizer, mp_image, elapsed_ms)
                        if args.gesture_debug and now - last_gesture_report >= 1.0:
                            print(
                                f"[Gesture] {name}, score={score:.2f}, "
                                f"crop={recognizer.preferred_crop or 'full'}, "
                                f"rotation={recognizer.preferred_rotation or 0}"
                            )
                            last_gesture_report = now
                        start_vanish_on_gesture(state, name, score, now)

                # The dissolve tracks the silhouette at 20 Hz.
                if state.phase == "vanishing" and (mask is None or now - last_mask_at >= 0.05):
                    if mp_image is None:
                        rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    mask = person_mask(segmenter, mp_image, elapsed_ms, width, height)
                    last_mask = mask
                    last_mask_at = now

                if state.phase in ("ready", "goku_vanishing"):
                    display = camera_frame
                    level = saiyan.level(now) if mode == "goku" else 0.0
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
                            hair_mask = recognizer.hair_mask(mp_image, elapsed_ms)
                            hair_mask_at = now
                        display = render_saiyan(display, mask, hair_mask, level,
                                                saiyan.burst(now), now)
                else:
                    progress = 0.0
                    if state.phase == "vanishing" and state.vanish_started is not None:
                        progress = min(
                            1.0, (time.perf_counter() - state.vanish_started) / VANISH_SECONDS
                        )
                        if progress >= 1.0:
                            state.phase = "vanished"
                    display = apply_dissolve(
                        camera_frame, background, mask, progress, dissolve_field, sparkle_noise
                    )
                    if state.phase == "vanished":
                        vanished_frame = display

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
            # In Goku mode only the gesture makes you vanish and reappear.
            elif key == ord("r") and mode != "goku":
                if background is not None:
                    state.phase = "ready"
                    state.vanish_started = None
                    vanished_frame = None
                    last_mask = None
                    last_mask_at = float("-inf")
            elif key == ord("v") and background is not None and mode != "goku":
                state.phase = "vanishing"
                state.vanish_started = time.perf_counter()
                vanished_frame = None
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
        print(f"\n[VanishCam] ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
