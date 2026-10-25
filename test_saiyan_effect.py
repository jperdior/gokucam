import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import vanish_cam as app
from goku_effect import GokuTrigger, HOLD_SECONDS
from saiyan_effect import (
    POWER_DOWN_SECONDS, TRANSFORM_SECONDS, Head, SaiyanState, both_fists, fist,
    golden_hair, render_aura,
)


def point(x, y):
    return SimpleNamespace(x=x, y=y)


def hand(x, y, closed=True):
    """A hand with its wrist at (x, y) and fingers pointing up."""
    points = [point(x, y)] + [point(x - 0.04, y - 0.03)] * 4  # Wrist, then the thumb.
    for offset in (-0.03, -0.01, 0.01, 0.03):
        knuckle = (x + offset, y - 0.08)
        if closed:
            joints = [knuckle, (x + offset, y - 0.10), (x + offset, y - 0.07), (x + offset, y - 0.05)]
        else:
            joints = [knuckle] + [(x + offset, y - 0.08 - 0.03 * step) for step in (1, 2, 3)]
        points += [point(*joint) for joint in joints]
    return points


def upright_head(width=640, height=480):
    return Head(np.array((320.0, 150.0), np.float32), np.array((0.0, -1.0), np.float32),
                120.0, 150.0)


class FistTests(unittest.TestCase):
    def test_closed_hand_is_a_fist_but_open_hand_is_not(self):
        closed = np.array([(p.x, p.y) for p in hand(0.5, 0.8)], np.float32)
        opened = np.array([(p.x, p.y) for p in hand(0.5, 0.8, closed=False)], np.float32)
        self.assertTrue(fist(closed))
        self.assertFalse(fist(opened))

    def test_needs_two_raised_fists(self):
        head = upright_head()
        raised = [hand(0.3, 0.85), hand(0.7, 0.85)]
        self.assertTrue(both_fists(raised, head, 640, 480))
        self.assertFalse(both_fists(raised[:1], head, 640, 480))
        self.assertFalse(both_fists([hand(0.3, 0.85), hand(0.7, 0.85, closed=False)], head, 640, 480))
        # Fists far above the head are not the power-up pose.
        self.assertFalse(both_fists([hand(0.3, 0.1), hand(0.7, 0.1)], head, 640, 480))


class SaiyanStateTests(unittest.TestCase):
    def test_transform_and_power_down_ramp_smoothly(self):
        saiyan = SaiyanState()
        self.assertEqual(saiyan.level(0.0), 0.0)
        saiyan.toggle(1.0)
        self.assertAlmostEqual(saiyan.level(1.0 + TRANSFORM_SECONDS / 2), 0.5)
        self.assertEqual(saiyan.level(1.0 + TRANSFORM_SECONDS), 1.0)
        self.assertGreater(saiyan.burst(1.0 + TRANSFORM_SECONDS * 0.3), 0.5)
        saiyan.toggle(5.0)
        self.assertEqual(saiyan.level(5.0), 1.0)
        self.assertAlmostEqual(saiyan.level(5.0 + POWER_DOWN_SECONDS), 0.0)
        self.assertEqual(saiyan.burst(5.1), 0.0)

    def test_fists_toggle_after_a_one_second_hold(self):
        saiyan, trigger = SaiyanState(), GokuTrigger()
        detector = SimpleNamespace(fists=True)
        with patch("builtins.print"):
            app.update_saiyan(detector, trigger, saiyan, 0.0)
            self.assertFalse(saiyan.active)
            app.update_saiyan(detector, trigger, saiyan, HOLD_SECONDS)
            self.assertTrue(saiyan.active)
            detector.fists = False
            app.update_saiyan(detector, trigger, saiyan, HOLD_SECONDS + 1.0)
            detector.fists = True
            app.update_saiyan(detector, trigger, saiyan, 5.0)
            app.update_saiyan(detector, trigger, saiyan, 5.0 + HOLD_SECONDS)
            self.assertFalse(saiyan.active)


class SaiyanRenderTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((480, 640, 3), 60, np.uint8)
        self.mask = np.zeros((480, 640), np.float32)
        self.mask[140:480, 240:400] = 1.0

    def test_only_hair_on_the_body_turns_gold_and_keeps_its_strands(self):
        frame = self.frame.copy()
        frame[100:140:2, 260:380] = 20   # Dark strands...
        frame[101:140:2, 260:380] = 70   # ...between lighter ones.
        hair = np.zeros((480, 640), np.float32)
        hair[100:140, 260:380] = 1.0
        hair[100:140, 450:500] = 1.0     # "Hair" on a chair, off the body.
        person = np.zeros((480, 640), np.float32)
        person[90:480, 240:400] = 1.0
        output = golden_hair(frame, hair, person, 1.0)
        strands = output[110:130, 300:340].astype(int)
        self.assertTrue(np.all(strands[:, :, 2] > strands[:, :, 0]))  # Gold: more red than blue.
        self.assertGreater(abs(int(output[110, 300].sum()) - int(output[111, 300].sum())), 40)
        np.testing.assert_array_equal(output[110:130, 460:490], frame[110:130, 460:490])
        np.testing.assert_array_equal(output[300:, :], frame[300:, :])
        np.testing.assert_array_equal(golden_hair(frame, hair, person, 0.0), frame)

    def test_aura_glows_around_but_not_over_the_body(self):
        output = render_aura(self.frame, self.mask, 1.0, 0.0, 0.0)
        np.testing.assert_array_equal(output[300, 320], self.frame[300, 320])
        self.assertGreater(int(output[300, 225].sum()), int(self.frame[300, 225].sum()))
        np.testing.assert_array_equal(render_aura(self.frame, self.mask, 0.0, 0.0, 0.0), self.frame)


if __name__ == "__main__":
    unittest.main()
