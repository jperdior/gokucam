"""Genki Dama: step back, raise both hands high and gather a Spirit Bomb."""

from __future__ import annotations

import math

import cv2
import numpy as np

try:
    from .saiyan_effect import Head
except ImportError:
    from saiyan_effect import Head


# The face must be at most this fraction of the frame width: you are far back.
FAR_FACE_WIDTH = 0.16
# Hands must be at least this many face heights above the hairline.
HANDS_ABOVE_HEAD = 0.2
CHARGE_SECONDS = 3.0
THROW_SECONDS = 0.9
# Lowering the hands for this long throws the bomb.
RELEASE_SECONDS = 0.5
SMOOTHING = 0.35

CORE_COLOR = np.array((255, 250, 235), np.float32)   # BGR, near white.
BALL_COLOR = np.array((255, 205, 120), np.float32)   # Light blue.
GLOW_COLOR = np.array((255, 170, 70), np.float32)    # Deeper blue.
SPARKS = 48

# Energy specks flying into the ball from all around, fixed per spark.
_sparks = np.random.default_rng(777).random((SPARKS, 4)).astype(np.float32)


def far_enough(head: Head | None, width: int) -> bool:
    return head is not None and head.width < width * FAR_FACE_WIDTH


def hands_raised(hands, head: Head | None, width: int, height: int,
                 minimum: int = 2) -> np.ndarray | None:
    """Point above the head, level with the highest raised hand, or None if too
    few hands are up. It stays on the head's axis so a lost hand does not pull it aside."""
    if head is None:
        return None
    tops = []
    for landmarks in hands:
        if len(landmarks) != 21:
            continue
        hand = np.array([(p.x * width, p.y * height) for p in landmarks], np.float32)
        if not np.isfinite(hand).all():
            continue
        rise = float(np.dot(hand.mean(axis=0) - head.hairline, head.up)) / head.height
        if rise > HANDS_ABOVE_HEAD:
            tops.append(float(np.max((hand - head.hairline) @ head.up)))  # Highest point.
    if len(tops) < minimum:
        return None
    return (head.hairline + head.up * max(tops)).astype(np.float32)


class GenkiDama:
    """Charges while the hands stay up, then is thrown when they come down."""

    def __init__(self):
        self.started: float | None = None
        self.thrown_at: float | None = None
        self.last_seen = -math.inf
        self.center: np.ndarray | None = None
        self.unit = 100.0  # Face height in pixels, sets the ball size.
        self.hint_at = -math.inf

    @property
    def active(self) -> bool:
        return self.started is not None

    def start(self, now: float) -> None:
        self.started, self.thrown_at, self.last_seen = now, None, now

    def follow(self, point: np.ndarray | None, head: Head | None, now: float) -> str | None:
        """Track the raised hands; return "throw" once they come down."""
        if not self.active or self.thrown_at is not None:
            return None
        if point is not None:
            self.last_seen = now
            if head is not None:
                self.unit = head.height
                target = point + head.up * self.radius(now)
            else:
                target = point
            self.center = target if self.center is None else (
                self.center * (1.0 - SMOOTHING) + target * SMOOTHING)
            return None
        if now - self.last_seen >= RELEASE_SECONDS:
            self.thrown_at = now
            return "throw"
        return None

    def charge(self, now: float) -> float:
        if self.started is None:
            return 0.0
        end = self.thrown_at if self.thrown_at is not None else now
        return min(1.0, max(0.0, (end - self.started) / CHARGE_SECONDS))

    def radius(self, now: float) -> float:
        return self.unit * (0.3 + 1.4 * self.charge(now))

    def throw_progress(self, now: float) -> float | None:
        if self.thrown_at is None:
            return None
        return min(1.0, (now - self.thrown_at) / THROW_SECONDS)

    def done(self, now: float) -> bool:
        progress = self.throw_progress(now)
        return progress is not None and progress >= 1.0

    def reset(self) -> None:
        hint_at = self.hint_at
        self.__init__()
        self.hint_at = hint_at


def _smoothstep(edge0: float, edge1: float, value):
    value = np.clip((value - edge0) / (edge1 - edge0), 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _screen(region: np.ndarray, amount: np.ndarray, color: np.ndarray) -> np.ndarray:
    """Add light: region and colour in 0..255, amount 0..1 per pixel."""
    return 255.0 - (255.0 - region) * (1.0 - amount[:, :, None] * color / 255.0)


def render_genki(display: np.ndarray, genki: GenkiDama, t: float) -> np.ndarray:
    if not genki.active or genki.center is None:
        return display
    height, width = display.shape[:2]
    center = genki.center.astype(np.float32)
    radius = genki.radius(t)
    throw = genki.throw_progress(t)
    flash = 0.0
    if throw is not None:
        # Fly at the camera: towards the middle of the frame and much bigger.
        ease = float(_smoothstep(0.0, 1.0, throw))
        center = center * (1.0 - ease) + np.array((width / 2, height / 2), np.float32) * ease
        radius *= 1.0 + 5.0 * ease ** 2
        flash = float(_smoothstep(0.55, 0.8, throw) * (1.0 - _smoothstep(0.85, 1.0, throw)))

    reach = radius * 1.9
    left, right = int(max(0, center[0] - reach)), int(min(width, center[0] + reach))
    top, bottom = int(max(0, center[1] - reach)), int(min(height, center[1] + reach))
    output = display.copy()
    if right - left > 4 and bottom - top > 4:
        # Shade the ball at half size; it is soft, so nobody sees the difference.
        size = (right - left, bottom - top)
        ys, xs = np.mgrid[top:bottom:2, left:right:2].astype(np.float32)
        dx, dy = xs - center[0], ys - center[1]
        d = np.sqrt(dx * dx + dy * dy) / max(radius, 1.0)
        angle = np.arctan2(dy, dx)
        pulse = 0.9 + 0.1 * math.sin(t * 12.0)
        # Swirling energy on the surface.
        swirl = 0.5 + 0.5 * np.sin(angle * 5.0 + d * 9.0 - t * 5.0) * np.sin(angle * 3.0 - d * 6.0 + t * 3.0)
        body = (1.0 - _smoothstep(0.9, 1.02, d)).astype(np.float32) * 0.92
        glow = (np.exp(-((d - 1.0) / 0.45) ** 2) * (d > 0.9) * 0.8 * pulse).astype(np.float32)
        heat = (1.0 - _smoothstep(0.0, 0.85, d)).astype(np.float32)[:, :, None]
        color = (BALL_COLOR * (1.0 - heat) + CORE_COLOR * heat) * (0.8 + 0.2 * swirl[:, :, None])
        body, glow = (cv2.resize(layer, size, interpolation=cv2.INTER_LINEAR) for layer in (body, glow))
        color = cv2.resize(color.astype(np.float32), size, interpolation=cv2.INTER_LINEAR)
        region = output[top:bottom, left:right].astype(np.float32)
        region = _screen(region, glow, GLOW_COLOR)
        opacity = body[:, :, None]
        region = region * (1.0 - opacity) + color * opacity
        output[top:bottom, left:right] = np.clip(region, 0, 255).astype(np.uint8)

    if throw is None:
        # Energy gathering from all around into the ball, drawn at half size.
        layer = np.zeros((height // 2, width // 2), np.float32)
        far = max(width, height) * 0.6
        for angle, phase, speed, size in _sparks:
            life = (t * (0.4 + 0.5 * speed) + phase) % 1.0
            distance = radius + (far - radius) * (1.0 - life)
            direction = np.array((math.cos(angle * 2.0 * math.pi), math.sin(angle * 2.0 * math.pi)),
                                 np.float32)
            spark = (center + distance * direction) / 2
            tail = (center + (distance + 40.0 + 60.0 * size) * direction) / 2
            cv2.line(layer, tuple(np.round(spark).astype(int)), tuple(np.round(tail).astype(int)),
                     float(math.sin(math.pi * life)), 1 + int(size * 1.5), cv2.LINE_AA)
        light = np.clip(cv2.GaussianBlur(layer, (0, 0), sigmaX=1.3) * 1.4 + layer * 0.6, 0.0, 1.0)
        light = cv2.resize(light, (width, height), interpolation=cv2.INTER_LINEAR)
        output = cv2.add(output, (light[:, :, None] * BALL_COLOR).astype(np.uint8))

    if flash > 0.001:
        output = np.clip(output.astype(np.float32) * (1.0 - flash) + CORE_COLOR * flash,
                         0, 255).astype(np.uint8)
    return output
