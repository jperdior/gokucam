import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import vanish_cam as app
from goku_effect import (
    HOLD_SECONDS, GokuPoseDetector, GokuTrigger, forehead_from_face,
    goku_pose_from_landmarks, render_teleport,
)
from test_video_output import arguments


def point(x, y):
    return SimpleNamespace(x=x, y=y)


def finger(x, base_y, tip_y):
    return [point(x, base_y + (tip_y - base_y) * step / 3) for step in range(4)]


def hand(x=0.5, top=0.30, spread=0.02, ring_folded=True):
    points = [point(x, 0.60)] + [point(x - 0.03, 0.55)] * 4
    points += finger(x - spread / 2, 0.45, top) + finger(x + spread / 2, 0.45, top)
    ring_tip = 0.48 if ring_folded else top
    points += finger(x + 0.03, 0.46, ring_tip) + finger(x + 0.05, 0.47, ring_tip + 0.01)
    return points


def face(forehead_y=0.28):
    points = [point(0.5, 0.4) for _ in range(478)]
    points[151] = point(0.5, forehead_y)
    points[234], points[454] = point(0.4, 0.35), point(0.6, 0.35)
    return points


class GokuPoseTests(unittest.TestCase):
    def forehead(self, y=0.28):
        return forehead_from_face(face(y), 640, 480)

    def test_two_joined_fingers_on_forehead(self):
        self.assertTrue(goku_pose_from_landmarks([hand()], self.forehead(), 640, 480))

    def test_pose_needs_the_forehead_nearby(self):
        self.assertFalse(goku_pose_from_landmarks([hand()], self.forehead(0.02), 640, 480))
        self.assertFalse(goku_pose_from_landmarks([hand()], None, 640, 480))

    def test_open_hand_or_v_sign_is_not_the_pose(self):
        forehead = self.forehead()
        self.assertFalse(goku_pose_from_landmarks([hand(ring_folded=False)], forehead, 640, 480))
        self.assertFalse(goku_pose_from_landmarks([hand(spread=0.12)], forehead, 640, 480))

    def test_detector_remembers_face_hidden_by_fingers(self):
        faces = [[face()], []]
        detector = GokuPoseDetector(
            SimpleNamespace(detect_for_video=lambda image, ts: SimpleNamespace(hand_landmarks=[hand()])),
            SimpleNamespace(detect_for_video=lambda image, ts: SimpleNamespace(face_landmarks=faces.pop(0))),
        )
        image = SimpleNamespace(width=640, height=480)
        self.assertTrue(detector.detect(image, 1000))
        self.assertTrue(detector.detect(image, 1500))


class GokuTriggerTests(unittest.TestCase):
    def test_pose_must_be_held_then_released_before_firing_again(self):
        trigger = GokuTrigger()
        self.assertEqual(trigger.update(True, 0.0), "start")
        self.assertIsNone(trigger.update(True, HOLD_SECONDS * 0.5))
        self.assertEqual(trigger.update(True, HOLD_SECONDS), "trigger")
        # Still holding after the teleport: no second trigger.
        self.assertIsNone(trigger.update(True, HOLD_SECONDS + 2.0))
        self.assertIsNone(trigger.update(False, HOLD_SECONDS + 3.0))
        self.assertEqual(trigger.update(True, 10.0), "start")
        self.assertEqual(trigger.update(True, 10.0 + HOLD_SECONDS), "trigger")

    def test_breaking_the_pose_restarts_the_countdown(self):
        trigger = GokuTrigger()
        trigger.update(True, 0.0)
        self.assertIsNone(trigger.update(False, 0.1))  # Brief dropout is tolerated.
        self.assertIsNone(trigger.update(True, 0.2))
        self.assertEqual(trigger.update(False, 0.9), "broken")
        self.assertEqual(trigger.update(True, 1.0), "start")
        self.assertIsNone(trigger.update(True, 1.5))


class TeleportRenderTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((60, 80, 3), 90, np.uint8)
        self.background = np.full((60, 80, 3), 20, np.uint8)
        self.mask = np.zeros((60, 80), np.float32)
        self.mask[10:50, 30:50] = 1.0

    def test_start_shows_person_and_end_shows_background(self):
        start = render_teleport(self.frame, self.background, self.mask, 0.0, 0.0)
        np.testing.assert_array_equal(start[20, 40], self.frame[20, 40])
        end = render_teleport(self.frame, self.background, self.mask, 1.0, 0.0)
        np.testing.assert_array_equal(end, self.background)

    def test_lines_are_drawn_mid_effect(self):
        middle = render_teleport(self.frame, self.background, self.mask, 0.7, 0.3)
        self.assertGreater(middle[10:50, 30:50].max(), 200)
        self.assertEqual(middle.shape, self.frame.shape)


class GokuVideoLoopTests(unittest.TestCase):
    def test_gesture_vanishes_and_brings_the_person_back(self):
        frames = [np.full((24, 40, 3), color, np.uint8) for color in (10, 40, 80, 120, 160, 200, 230)]
        capture = Mock()
        capture.read.side_effect = [(True, frame) for frame in frames]
        detector = Mock()
        displayed = []
        with ExitStack() as mocks:
            mocks.enter_context(patch.object(app, "parse_args", return_value=arguments(
                mode="goku", virtual_camera=False)))
            mocks.enter_context(patch.object(app, "download_models"))
            mocks.enter_context(patch.object(app, "open_camera", return_value=capture))
            mocks.enter_context(patch.object(app, "create_tasks", return_value=(Mock(), detector)))
            mocks.enter_context(patch.object(app, "person_mask", return_value=np.ones((24, 40), np.float32)))
            mocks.enter_context(patch.object(app, "frame_has_person", return_value=False))
            # Fires on the first check while ready and the first one while vanished.
            mocks.enter_context(patch.object(app, "update_goku", return_value=True))
            times = (0, 0, 0.1, 4.1, 4.3, 4.4, 6.0, 6.1, 8.0)
            mocks.enter_context(patch.object(app.time, "perf_counter", side_effect=times))
            for name in ("namedWindow", "resizeWindow", "destroyAllWindows"):
                mocks.enter_context(patch.object(app.cv2, name))
            mocks.enter_context(patch.object(app.cv2, "imshow",
                                             side_effect=lambda name, frame: displayed.append(frame.copy())))
            # V and R are ignored in Goku mode.
            keys = (0, 0, 0, ord("r"), ord("v"), 0, ord("q"))
            mocks.enter_context(patch.object(app.cv2, "waitKey", side_effect=keys))
            mocks.enter_context(patch("builtins.print"))
            self.assertEqual(app.run(), 0)
        detector.close.assert_called_once()
        np.testing.assert_array_equal(displayed[2], frames[2])  # Ready: gesture fires.
        np.testing.assert_array_equal(displayed[4], frames[1])  # Vanished: background only.
        np.testing.assert_array_equal(displayed[6], frames[6])  # Back after reappearing.


if __name__ == "__main__":
    unittest.main()
