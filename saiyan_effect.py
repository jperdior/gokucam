"""Super Saiyan: both fists raised to power up, golden hair and aura."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


TRANSFORM_SECONDS = 1.2
POWER_DOWN_SECONDS = 0.6
HEAD_SMOOTHING = 0.5

# Face mesh landmarks: top of the forehead, chin and the face contour.
HAIRLINE, CHIN = 10, 152
FACE_LEFT, FACE_RIGHT = 234, 454

AURA_INNER = np.array((190, 250, 255), np.float32)
AURA_OUTER = np.array((0, 170, 255), np.float32)

@dataclass
class Head:
    hairline: np.ndarray  # Top of the forehead, pixels.
    up: np.ndarray        # Unit vector from chin to forehead.
    width: float
    height: float

    @property
    def right(self) -> np.ndarray:
        return np.array((-self.up[1], self.up[0]), np.float32)

    def blend(self, previous: Head | None, weight: float = HEAD_SMOOTHING) -> Head:
        if previous is None:
            return self
        up = previous.up * (1.0 - weight) + self.up * weight
        return Head(
            previous.hairline * (1.0 - weight) + self.hairline * weight,
            (up / max(float(np.linalg.norm(up)), 1e-6)).astype(np.float32),
            previous.width * (1.0 - weight) + self.width * weight,
            previous.height * (1.0 - weight) + self.height * weight,
        )


def head_from_face(face, width: int, height: int) -> Head | None:
    if face is None or len(face) <= max(HAIRLINE, CHIN, FACE_LEFT, FACE_RIGHT):
        return None
    points = np.array([(face[i].x * width, face[i].y * height)
                       for i in (HAIRLINE, CHIN, FACE_LEFT, FACE_RIGHT)], np.float32)
    if not np.isfinite(points).all():
        return None
    up = points[0] - points[1]
    face_height = float(np.linalg.norm(up))
    face_width = float(np.linalg.norm(points[2] - points[3]))
    if face_height < 8.0 or face_width < 8.0:
        return None
    return Head(points[0], up / face_height, face_width, face_height)


def fist(hand: np.ndarray) -> bool:
    """All four fingers curled: every fingertip is closer to the wrist than its knuckle."""
    if hand.shape != (21, 2) or not np.isfinite(hand).all():
        return False
    wrist = hand[0]
    palm = float(np.linalg.norm(hand[9] - wrist))
    if palm < 1e-6:
        return False
    return all(np.linalg.norm(hand[base + 3] - wrist) < np.linalg.norm(hand[base] - wrist) * 1.15
               for base in (5, 9, 13, 17))


def both_fists(hands, head: Head | None, width: int, height: int) -> bool:
    """Two clenched fists raised in front of the body, as when powering up."""
    fists = []
    for landmarks in hands:
        if len(landmarks) != 21:
            continue
        hand = np.array([(p.x * width, p.y * height) for p in landmarks], np.float32)
        if fist(hand):
            fists.append(hand.mean(axis=0))
    if len(fists) < 2:
        return False
    if head is None:
        return True
    # Below the face but not resting far down, measured along the head axis.
    chin = head.hairline - head.up * head.height
    rise = [float(np.dot(center - chin, head.up)) / head.height for center in fists]
    return all(-2.2 < value < 0.6 for value in rise)


class SaiyanState:
    """Toggle between normal and Super Saiyan with a transform animation."""

    def __init__(self):
        self.active = False
        self.changed_at = -math.inf
        self.start_level = 0.0

    def toggle(self, now: float) -> None:
        self.start_level = self.level(now)
        self.active = not self.active
        self.changed_at = now

    def level(self, now: float) -> float:
        age = now - self.changed_at
        if self.active:
            ramp = min(1.0, max(0.0, age / TRANSFORM_SECONDS))
            return self.start_level + (1.0 - self.start_level) * ramp
        ramp = min(1.0, max(0.0, age / POWER_DOWN_SECONDS))
        return self.start_level * (1.0 - ramp)

    def burst(self, now: float) -> float:
        """Bright flash at the start of the transformation."""
        if not self.active:
            return 0.0
        age = (now - self.changed_at) / TRANSFORM_SECONDS
        if age < 0.0 or age > 0.6:
            return 0.0
        return math.sin(math.pi * age / 0.6)


_flames: dict[tuple[int, int], np.ndarray] = {}


def _flame_texture(height: int, width: int) -> np.ndarray:
    """Vertically stretched noise that scrolls upwards like flames."""
    key = (height, width)
    if key not in _flames:
        rng = np.random.default_rng(9001)
        noise = rng.random((height, width), dtype=np.float32)
        noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=1.5, sigmaY=6.0)
        noise = (noise - noise.min()) / max(float(noise.max() - noise.min()), 1e-6)
        _flames[key] = noise
    return _flames[key]


def render_aura(display: np.ndarray, mask: np.ndarray, level: float, burst: float,
                t: float) -> np.ndarray:
    """Golden flame aura behind the person, computed at a third of the size."""
    if level <= 0.001 and burst <= 0.001:
        return display
    height, width = mask.shape
    small_size = (max(8, width // 3), max(8, height // 3))
    small = cv2.resize(mask, small_size, interpolation=cv2.INTER_AREA)
    sh, sw = small.shape
    spread = 4.0 + 6.0 * level + 5.0 * burst
    body = cv2.GaussianBlur(small, (0, 0), sigmaX=spread)
    # Let the flames lick upwards: stretch the halo above the silhouette.
    rise = int(sh * 0.06 * (0.5 + level))
    lifted = np.zeros_like(body)
    lifted[:sh - rise] = body[rise:]
    halo = np.clip(np.maximum(body, lifted) * 2.2, 0.0, 1.0)
    flames = _flame_texture(sh, sw)
    shift = int(t * sh * 1.6) % sh
    flicker = np.roll(flames, -shift, axis=0)
    strength = halo * (0.45 + 0.75 * flicker) * min(1.0, level + burst)
    strength = np.clip(strength, 0.0, 1.0)
    # Only blend the area the aura reaches; full-frame float maths is slow.
    points = cv2.findNonZero((strength > 0.01).astype(np.uint8))
    if points is None:
        return display
    x, y, w, h = cv2.boundingRect(points)
    scale_x, scale_y = width / sw, height / sh
    left, top = int(x * scale_x), int(y * scale_y)
    right, bottom = min(width, int((x + w) * scale_x) + 1), min(height, int((y + h) * scale_y) + 1)
    big = cv2.resize(strength, (width, height), interpolation=cv2.INTER_LINEAR)[top:bottom, left:right]
    body = np.clip(mask[top:bottom, left:right], 0.0, 1.0)
    # Behind the person: no aura over the body itself.
    big *= 1.0 - np.clip(body * 1.3, 0.0, 1.0)
    big = big[:, :, None] * 0.85
    heat = np.clip(big * 1.6, 0.0, 1.0)
    color = AURA_OUTER * (1.0 - heat) + AURA_INNER * heat
    area = np.s_[top:bottom, left:right]
    region = display[area].astype(np.float32) * (1.0 - big) + color * big
    if burst > 0.001:
        flash = (body * burst * 0.6)[:, :, None]
        region = region * (1.0 - flash) + AURA_INNER * flash
    output = display.copy()
    output[area] = np.clip(region, 0, 255).astype(np.uint8)
    return output


DARK_GOLD = np.array((20, 135, 210), np.float32)   # BGR
LIGHT_GOLD = np.array((165, 245, 255), np.float32)


def golden_hair(frame: np.ndarray, hair: np.ndarray, person: np.ndarray, level: float) -> np.ndarray:
    """Recolour the real hair gold, keeping its strands and shading."""
    # Only confident hair on the body: a dark chair behind the head is not hair.
    confident = np.clip((hair - 0.45) / 0.35, 0.0, 1.0)
    body = np.clip((person - 0.4) / 0.4, 0.0, 1.0)
    weight = confident * body * min(1.0, level)
    points = cv2.findNonZero((weight > 0.02).astype(np.uint8))
    if points is None:
        return frame
    x, y, w, h = cv2.boundingRect(points)
    area = np.s_[y:y + h, x:x + w]
    region = frame[area].astype(np.float32)
    # Stretch the hair's own brightness range over the gold ramp so light and
    # dark strands stay visible instead of a flat yellow.
    luminance = cv2.cvtColor(frame[area], cv2.COLOR_BGR2GRAY).astype(np.float32)
    inside = luminance[weight[area] > 0.5]
    if inside.size < 20:
        return frame
    low, high = np.percentile(inside, (5, 95))
    shade = np.clip((luminance - low) / max(float(high - low), 1.0), 0.0, 1.0)
    shade = (0.15 + 0.85 * shade ** 0.8)[:, :, None]
    gold = DARK_GOLD * (1.0 - shade) + LIGHT_GOLD * shade
    blend = cv2.GaussianBlur(weight[area], (0, 0), sigmaX=1.5)[:, :, None] * 0.9
    output = frame.copy()
    output[area] = np.clip(region * (1.0 - blend) + gold * blend, 0, 255).astype(np.uint8)
    return output


def render_saiyan(display: np.ndarray, mask: np.ndarray, hair: np.ndarray | None,
                  level: float, burst: float, t: float) -> np.ndarray:
    if hair is not None:
        display = golden_hair(display, hair, mask, level)
    return render_aura(display, mask, level, burst, t)
