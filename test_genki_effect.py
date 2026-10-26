import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import gokucam as app
from genki_effect import (
    CHARGE_SECONDS, RELEASE_SECONDS, THROW_SECONDS, GenkiDama, far_enough, hands_raised,
    render_genki,
)
from goku_effect import GokuTrigger, HOLD_SECONDS
from saiyan_effect import Head
from test_saiyan_effect import hand


def far_head():
    # A small face low in a 640x480 frame: standing far back.
    return Head(np.array((320.0, 300.0), np.float32), np.array((0.0, -1.0), np.float32), 60.0, 70.0)


class PoseTests(unittest.TestCase):
    def test_both_hands_above_the_head(self):
        head = far_head()
        self.assertIsNotNone(hands_raised([hand(0.4, 0.4), hand(0.6, 0.4)], head, 640, 480))
        self.assertIsNone(hands_raised([hand(0.4, 0.4)], head, 640, 480))
        self.assertIsNone(hands_raised([hand(0.4, 0.9), hand(0.6, 0.9)], head, 640, 480))

    def test_must_stand_far_back(self):
        self.assertTrue(far_enough(far_head(), 640))
        close = Head(np.array((320.0, 150.0), np.float32), np.array((0.0, -1.0), np.float32), 200.0, 240.0)
        self.assertFalse(far_enough(close, 640))
        self.assertFalse(far_enough(None, 640))


class GenkiTests(unittest.TestCase):
    def test_hold_charges_and_lowering_throws(self):
        genki, trigger = GenkiDama(), GokuTrigger()
        up = np.array((320.0, 120.0), np.float32)
        detector = SimpleNamespace(raised=up, raised_one=up, head=far_head())
        with patch("builtins.print"):
            app.update_genki(detector, trigger, genki, 0.0, 640)
            self.assertFalse(genki.active)
            app.update_genki(detector, trigger, genki, HOLD_SECONDS, 640)
            self.assertTrue(genki.active)
            app.update_genki(detector, trigger, genki, HOLD_SECONDS + CHARGE_SECONDS, 640)
            self.assertEqual(genki.charge(HOLD_SECONDS + CHARGE_SECONDS), 1.0)
            detector.raised = detector.raised_one = None
            end = HOLD_SECONDS + CHARGE_SECONDS + RELEASE_SECONDS
            app.update_genki(detector, trigger, genki, end, 640)
            self.assertIsNotNone(genki.throw_progress(end))
            self.assertTrue(genki.done(end + THROW_SECONDS))

    def test_too_close_does_not_start(self):
        genki, trigger = GenkiDama(), GokuTrigger()
        close = Head(np.array((320.0, 150.0), np.float32), np.array((0.0, -1.0), np.float32), 200.0, 240.0)
        up = np.array((320.0, 20.0), np.float32)
        detector = SimpleNamespace(raised=up, raised_one=up, head=close)
        with patch("builtins.print"):
            app.update_genki(detector, trigger, genki, 0.0, 640)
            app.update_genki(detector, trigger, genki, HOLD_SECONDS, 640)
        self.assertFalse(genki.active)

    def test_ball_is_drawn_above_the_hands(self):
        frame = np.full((480, 640, 3), 60, np.uint8)
        np.testing.assert_array_equal(render_genki(frame, GenkiDama(), 0.0), frame)
        genki = GenkiDama()
        genki.start(0.0)
        genki.follow(np.array((320.0, 200.0), np.float32), far_head(), 0.0)
        output = render_genki(frame, genki, 2.0)
        self.assertGreater(int(output[int(genki.center[1]), 320, 0]), 150)  # Blue-white ball.


if __name__ == "__main__":
    unittest.main()
