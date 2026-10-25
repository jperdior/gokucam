"""Instant Transmission: two fingers on the forehead and comic speed lines."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


HOLD_SECONDS = 1.0
TELEPORT_SECONDS = 1.15
# Short detection dropouts while holding the pose do not restart the countdown.
POSE_GRACE_SECONDS = 0.25
FACE_MEMORY_MS = 1000

# Face mesh landmarks: mid forehead and the left/right face contour.
FOREHEAD = 151
FACE_LEFT, FACE_RIGHT = 234, 454


def _points(landmarks, width: int, height: int) -> np.ndarray:
    return np.array([(p.x * width, p.y * height) for p in landmarks], dtype=np.float32)


def forehead_from_face(face, width: int, height: int) -> tuple[np.ndarray, float] | None:
    """Return the forehead point and face width in pixels."""
    if face is None or len(face) <= max(FOREHEAD, FACE_LEFT, FACE_RIGHT):
        return None
    points = _points(face, width, height)
    face_width = float(np.linalg.norm(points[FACE_LEFT] - points[FACE_RIGHT]))
    if not np.isfinite(points).all() or face_width < 8.0:
        return None
    return points[FOREHEAD], face_width


def two_finger_pose(hand: np.ndarray) -> bool:
    """Index and middle straight and side by side, ring and pinky folded."""
    if hand.shape != (21, 2) or not np.isfinite(hand).all():
        return False
    lengths = []
    for base in (5, 9):
        finger = hand[base:base + 4]
        length = float(np.linalg.norm(np.diff(finger, axis=0), axis=1).sum())
        if length < 1e-6 or np.linalg.norm(finger[3] - finger[0]) / length < 0.78:
            return False
        lengths.append(length)
    reference = min(lengths)
    together = np.linalg.norm(hand[8] - hand[12]) < reference * 0.5
    folded = all(np.linalg.norm(hand[base + 3] - hand[base]) < reference * 0.75
                 for base in (13, 17))
    return bool(together and folded)


def goku_pose_from_landmarks(hands, forehead: tuple[np.ndarray, float] | None,
                             width: int, height: int) -> bool:
    """Recognize two joined, straight fingers touching the forehead."""
    if forehead is None:
        return False
    point, face_width = forehead
    for landmarks in hands:
        if len(landmarks) != 21:
            continue
        hand = _points(landmarks, width, height)
        if not two_finger_pose(hand):
            continue
        # The finger pads rest on the forehead, so any of the upper joints counts.
        distance = np.linalg.norm(hand[[6, 7, 8, 10, 11, 12]] - point, axis=1).min()
        if distance < face_width * 0.4:
            return True
    return False


class GokuPoseDetector:
    """Track hands and face in video; remember the face while fingers cover it."""

    def __init__(self, hands, face):
        self.hands = hands
        self.face = face
        self.forehead: tuple[np.ndarray, float] | None = None
        self.forehead_ms = -FACE_MEMORY_MS
        self.last_hand_count = 0

    def detect(self, image, timestamp_ms: int) -> bool:
        faces = self.face.detect_for_video(image, timestamp_ms).face_landmarks
        found = forehead_from_face(faces[0] if faces else None, image.width, image.height)
        if found is not None:
            self.forehead, self.forehead_ms = found, timestamp_ms
        elif timestamp_ms - self.forehead_ms > FACE_MEMORY_MS:
            self.forehead = None
        hands = self.hands.detect_for_video(image, timestamp_ms).hand_landmarks
        self.last_hand_count = len(hands)
        return goku_pose_from_landmarks(hands, self.forehead, image.width, image.height)

    def close(self) -> None:
        try:
            self.hands.close()
        finally:
            self.face.close()


@dataclass
class GokuTrigger:
    """Fire once the pose is held for HOLD_SECONDS; re-arm after the hand drops."""

    hold_started: float | None = None
    last_pose_at: float | None = None
    armed: bool = True

    def update(self, pose: bool, now: float) -> str | None:
        if pose:
            self.last_pose_at = now
            if not self.armed:
                return None
            if self.hold_started is None:
                self.hold_started = now
                return "start"
            if now - self.hold_started >= HOLD_SECONDS:
                self.hold_started = None
                self.armed = False
                return "trigger"
            return None
        if self.last_pose_at is not None and now - self.last_pose_at <= POSE_GRACE_SECONDS:
            return None
        broken = self.hold_started is not None
        self.hold_started = None
        self.armed = True
        return "broken" if broken else None

    def remaining(self, now: float) -> float | None:
        if self.hold_started is None:
            return None
        return max(0.0, HOLD_SECONDS - (now - self.hold_started))

    def reset(self) -> None:
        self.hold_started = None
        self.last_pose_at = None
        self.armed = True


def _smoothstep(edge0: float, edge1: float, value: float) -> float:
    value = min(1.0, max(0.0, (value - edge0) / (edge1 - edge0)))
    return value * value * (3.0 - 2.0 * value)


_columns: dict[int, tuple[np.ndarray, ...]] = {}


def _line_columns(width: int) -> tuple[np.ndarray, ...]:
    """Per-column dash pattern, fixed so the lines do not flicker between frames."""
    if width not in _columns:
        rng = np.random.default_rng(4114)
        active = (rng.random(width) < 0.16).astype(np.float32)
        # Thicken some lines to 2-3 px for a hand-drawn comic look.
        thick = np.convolve(active, np.ones(3, np.float32), mode="same") * (rng.random(width) < 0.5)
        active = np.clip(active + thick, 0.0, 1.0)
        dash = rng.uniform(40.0, 140.0, width).astype(np.float32)
        phase = rng.random(width, dtype=np.float32)
        speed = rng.uniform(2.5, 5.0, width).astype(np.float32)
        duty = rng.uniform(0.35, 0.75, width).astype(np.float32)
        _columns[width] = active, dash, phase, speed, duty
    return _columns[width]


def speed_lines(height: int, left: int, right: int, top: int, bottom: int,
                width: int, t: float) -> np.ndarray:
    active, dash, phase, speed, duty = (values[left:right] for values in _line_columns(width))
    y = np.arange(top, bottom, dtype=np.float32)[:, None]
    position = (y / dash + phase - t * speed) % 1.0
    return (position < duty).astype(np.float32) * active


def render_teleport(frame: np.ndarray, background: np.ndarray, mask: np.ndarray,
                    progress: float, t: float) -> np.ndarray:
    """Draw the teleport at progress 0..1; play it backwards to reappear."""
    progress = min(1.0, max(0.0, progress))
    gone = _smoothstep(0.42, 0.6, progress)
    lines = _smoothstep(0.0, 0.25, progress) * (1.0 - _smoothstep(0.72, 1.0, progress))
    flash = _smoothstep(0.25, 0.45, progress) * (1.0 - gone)
    canvas = background.copy()
    points = cv2.findNonZero((mask > 0.05).astype(np.uint8))
    if points is None:
        return canvas
    x, y, w, h = cv2.boundingRect(points)
    height, width = mask.shape
    left, top = max(0, x - 30), max(0, y - 30)
    right, bottom = min(width, x + w + 30), min(height, y + h + 30)
    area = np.s_[top:bottom, left:right]

    person = frame[area].astype(np.float32)
    person += (255.0 - person) * flash * 0.65
    alpha = (mask[area] * (1.0 - gone))[:, :, None]
    output = person * alpha + background[area].astype(np.float32) * (1.0 - alpha)
    if lines > 0.001:
        region = cv2.GaussianBlur(mask[area], (0, 0), sigmaX=7.0)
        core = speed_lines(height, left, right, top, bottom, width, t) * region * lines
        glow = cv2.GaussianBlur(core, (0, 0), sigmaX=3.0)
        glow = np.clip(glow * 1.6, 0.0, 1.0)[:, :, None]
        core = core[:, :, None]
        output = output * (1.0 - glow * 0.7) + np.array((255, 225, 150), np.float32) * glow * 0.7
        output = output * (1.0 - core) + 255.0 * core
    canvas[area] = np.clip(output, 0, 255).astype(np.uint8)
    return canvas
