"""Instant Transmission: two fingers on the forehead and comic speed lines."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

try:
    from .saiyan_effect import Head, both_fists, head_from_face
except ImportError:
    from saiyan_effect import Head, both_fists, head_from_face


HOLD_SECONDS = 1.0
TELEPORT_SECONDS = 0.8
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

    def __init__(self, hands, face, hair=None):
        self.hands = hands
        self.face = face
        self.hair = hair
        self.forehead: tuple[np.ndarray, float] | None = None
        self.forehead_ms = -FACE_MEMORY_MS
        self.head: Head | None = None
        self.fists = False
        self.last_hand_count = 0

    def track_face(self, image, timestamp_ms: int) -> None:
        faces = self.face.detect_for_video(image, timestamp_ms).face_landmarks
        face = faces[0] if faces else None
        found = forehead_from_face(face, image.width, image.height)
        if found is not None:
            self.forehead, self.forehead_ms = found, timestamp_ms
            head = head_from_face(face, image.width, image.height)
            if head is not None:
                self.head = head.blend(self.head)
        elif timestamp_ms - self.forehead_ms > FACE_MEMORY_MS:
            self.forehead = None
            self.head = None

    def detect(self, image, timestamp_ms: int) -> bool:
        self.track_face(image, timestamp_ms)
        hands = self.hands.detect_for_video(image, timestamp_ms).hand_landmarks
        self.last_hand_count = len(hands)
        self.fists = both_fists(hands, self.head, image.width, image.height)
        return goku_pose_from_landmarks(hands, self.forehead, image.width, image.height)

    def hair_mask(self, image, timestamp_ms: int) -> np.ndarray | None:
        """Hair confidence per pixel, for the Super Saiyan recolouring."""
        if self.hair is None:
            return None
        masks = self.hair.segment_for_video(image, timestamp_ms).confidence_masks
        if len(masks) < 2:
            return None
        # The model returns background and hair; hair is the second mask.
        hair = np.squeeze(masks[1].numpy_view()).astype(np.float32)
        return cv2.resize(hair, (image.width, image.height), interpolation=cv2.INTER_LINEAR)

    def close(self) -> None:
        try:
            self.hands.close()
        finally:
            try:
                self.face.close()
            finally:
                if self.hair is not None:
                    self.hair.close()


@dataclass
class GokuTrigger:
    """Fire once the pose is held for HOLD_SECONDS; re-arm after the hand drops."""

    hold_started: float | None = None
    last_pose_at: float | None = None
    armed: bool = True
    hold_seconds: float = HOLD_SECONDS
    grace_seconds: float = POSE_GRACE_SECONDS

    def update(self, pose: bool, now: float) -> str | None:
        if pose:
            self.last_pose_at = now
            if not self.armed:
                return None
            if self.hold_started is None:
                self.hold_started = now
                return "start"
            if now - self.hold_started >= self.hold_seconds:
                self.hold_started = None
                self.armed = False
                return "trigger"
            return None
        if self.last_pose_at is not None and now - self.last_pose_at <= self.grace_seconds:
            return None
        broken = self.hold_started is not None
        self.hold_started = None
        self.armed = True
        return "broken" if broken else None

    def remaining(self, now: float) -> float | None:
        if self.hold_started is None:
            return None
        return max(0.0, self.hold_seconds - (now - self.hold_started))

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
        active = (rng.random(width) < 0.08).astype(np.float32)
        # Thicken some lines to 2-3 px for a hand-drawn comic look.
        thick = np.convolve(active, np.ones(3, np.float32), mode="same") * (rng.random(width) < 0.5)
        active = np.clip(active + thick, 0.0, 1.0)
        dash = rng.uniform(40.0, 140.0, width).astype(np.float32)
        phase = rng.random(width, dtype=np.float32)
        speed = rng.uniform(4.0, 8.0, width).astype(np.float32)
        duty = rng.uniform(0.35, 0.75, width).astype(np.float32)
        _columns[width] = active, dash, phase, speed, duty
    return _columns[width]


def speed_lines(height: int, left: int, right: int, top: int, bottom: int,
                width: int, t: float) -> np.ndarray:
    active, dash, phase, speed, duty = (values[left:right] for values in _line_columns(width))
    y = np.arange(top, bottom, dtype=np.float32)[:, None]
    position = (y / dash + phase - t * speed) % 1.0
    return (position < duty).astype(np.float32) * active


AURA_COLOR = np.array((255, 230, 170), np.float32)  # Pale blue-white, BGR.


def _squeeze(layer: np.ndarray, center_x: float, scale: float) -> np.ndarray:
    """Compress a layer horizontally toward a vertical centre line."""
    height, width = layer.shape[:2]
    source_x = center_x + (np.arange(width, dtype=np.float32) - center_x) / scale
    map_x = np.ascontiguousarray(np.broadcast_to(source_x, (height, width)))
    map_y = np.ascontiguousarray(np.broadcast_to(
        np.arange(height, dtype=np.float32)[:, None], (height, width)))
    return cv2.remap(layer, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def render_teleport(frame: np.ndarray, background: np.ndarray, mask: np.ndarray,
                    progress: float, t: float) -> np.ndarray:
    """Draw Instant Transmission at progress 0..1; play it backwards to reappear.

    Speed lines and an aura build up, the body flickers, then squeezes into a
    bright vertical line that blinks out while the lines fade where it stood.
    """
    progress = min(1.0, max(0.0, progress))
    lines = _smoothstep(0.15, 0.45, progress) * (1.0 - _smoothstep(0.8, 1.0, progress))
    aura = _smoothstep(0.0, 0.3, progress) * (1.0 - _smoothstep(0.6, 0.75, progress))
    flash = _smoothstep(0.25, 0.55, progress)
    squeeze = _smoothstep(0.45, 0.75, progress)
    gone = _smoothstep(0.74, 0.82, progress)
    beam = _smoothstep(0.6, 0.74, progress) * (1.0 - _smoothstep(0.76, 0.84, progress))
    blink = 0.25 if 0.3 <= progress < 0.5 and int(progress * 60) % 2 else 1.0

    canvas = background.copy()
    points = cv2.findNonZero((mask > 0.05).astype(np.uint8))
    if points is None:
        return canvas
    x, y, w, h = cv2.boundingRect(points)
    height, width = mask.shape
    left, top = max(0, x - 40), max(0, y - 40)
    right, bottom = min(width, x + w + 40), min(height, y + h + 40)
    area = np.s_[top:bottom, left:right]
    silhouette = mask[area]
    columns = np.arange(right - left, dtype=np.float32)
    center_x = float((silhouette.sum(axis=0) * columns).sum() / max(float(silhouette.sum()), 1e-6))

    person = frame[area].astype(np.float32)
    person += (255.0 - person) * flash * 0.75
    alpha = silhouette * blink * (1.0 - gone)
    if squeeze > 0.001:
        scale = 1.0 - 0.97 * squeeze
        person = _squeeze(person, center_x, scale)
        alpha = _squeeze(alpha, center_x, scale)
    output = person * alpha[:, :, None] + background[area].astype(np.float32) * (1.0 - alpha[:, :, None])

    if aura > 0.001:
        ring = np.clip(cv2.GaussianBlur(alpha, (0, 0), sigmaX=9.0) * 2.0 - alpha, 0.0, 1.0)
        ring = (ring * aura * 0.8)[:, :, None]
        output = output * (1.0 - ring) + AURA_COLOR * ring
    if beam > 0.001:
        # The thin bright line the body collapses into just before it vanishes.
        rows = cv2.GaussianBlur(silhouette.max(axis=1), (0, 0), sigmaX=6.0).ravel()
        across = np.exp(-((columns - center_x) / 4.0) ** 2)
        halo = np.exp(-((columns - center_x) / 16.0) ** 2)
        core = (beam * rows[:, None] * across[None, :])[:, :, None]
        glow = (beam * 0.8 * rows[:, None] * halo[None, :])[:, :, None]
        output = output * (1.0 - glow) + AURA_COLOR * glow
        output = output * (1.0 - core) + 255.0 * core
    if lines > 0.001:
        region = cv2.GaussianBlur(silhouette, (0, 0), sigmaX=7.0)
        core = speed_lines(height, left, right, top, bottom, width, t) * region * lines
        glow = cv2.GaussianBlur(core, (0, 0), sigmaX=3.0)
        glow = np.clip(glow * 1.6, 0.0, 1.0)[:, :, None]
        core = core[:, :, None]
        output = output * (1.0 - glow * 0.7) + AURA_COLOR * glow * 0.7
        output = output * (1.0 - core) + 255.0 * core
    canvas[area] = np.clip(output, 0, 255).astype(np.uint8)
    return canvas
