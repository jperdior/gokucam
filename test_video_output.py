import argparse
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import gokucam as app


def arguments(**overrides):
    values = dict(
        camera=0, width=960, height=540, fps=30.0, fullscreen=False,
        no_mirror=True, virtual_camera=None, virtual_backend=None, virtual_device=None,
        gesture_debug=False, blur_background=False, background_file=None,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


class VirtualCameraTests(unittest.TestCase):
    def test_bgr_format_and_actual_frame_dimensions(self):
        camera = Mock(device="OBS Virtual Camera")
        module = SimpleNamespace(Camera=Mock(return_value=camera), PixelFormat=SimpleNamespace(BGR="BGR"))
        with patch.dict("sys.modules", pyvirtualcam=module):
            self.assertIs(app.open_virtual_camera(arguments(), 640, 480), camera)
        module.Camera.assert_called_once_with(
            width=640, height=480, fps=30.0, fmt="BGR", backend=None, device=None,
        )

    def test_unavailable_driver_auto_falls_back_but_explicit_mode_fails(self):
        module = SimpleNamespace(Camera=Mock(side_effect=RuntimeError("driver missing")),
                                 PixelFormat=SimpleNamespace(BGR="BGR"))
        with patch.dict("sys.modules", pyvirtualcam=module), patch("builtins.print"):
            self.assertIsNone(app.open_virtual_camera(arguments(), 640, 480))
            for options in (dict(virtual_camera=True), dict(virtual_backend="obs")):
                with self.assertRaisesRegex(RuntimeError, "driver missing"):
                    app.open_virtual_camera(arguments(**options), 640, 480)

    def test_disabled_camera_does_not_open_device(self):
        module = SimpleNamespace(Camera=Mock())
        with patch.dict("sys.modules", pyvirtualcam=module):
            self.assertIsNone(app.open_virtual_camera(arguments(virtual_camera=False), 640, 480))
        module.Camera.assert_not_called()

    def test_send_preserves_colors_and_handles_noncontiguous_frames(self):
        camera = Mock(width=40, height=24)
        frame = np.zeros((24, 40, 3), np.uint8)
        frame[:] = (10, 70, 200)  # BGR: do not swap red and blue.
        app.send_virtual_frame(camera, frame[:, ::-1])
        sent = camera.send.call_args.args[0]
        self.assertTrue(sent.flags.c_contiguous)
        np.testing.assert_array_equal(sent, frame)

    def test_send_resizes_if_input_resolution_changes(self):
        camera = Mock(width=40, height=24)
        app.send_virtual_frame(camera, np.full((48, 80, 3), 42, np.uint8))
        sent = camera.send.call_args.args[0]
        self.assertEqual(sent.shape, (24, 40, 3))
        self.assertTrue((sent == 42).all())


class VideoLoopTests(unittest.TestCase):
    def run_mock_video(self, send_error=None, **overrides):
        capture = Mock()
        frames = [np.full((24, 40, 3), color, np.uint8) for color in (10, 40, 80, 120, 160)]
        capture.read.side_effect = [(True, frame) for frame in frames]
        segmenter, recognizer = Mock(), Mock()
        camera = Mock(width=40, height=24)
        camera.send.side_effect = send_error
        displayed = []
        self.banners = []
        with ExitStack() as mocks:
            mocks.enter_context(patch.object(app, "parse_args", return_value=arguments(**overrides)))
            # Record the setup banners but leave the frames untouched.
            mocks.enter_context(patch.object(
                app, "draw_banner", side_effect=lambda frame, lines: self.banners.append(lines) or frame))
            mocks.enter_context(patch.object(app, "download_models"))
            mocks.enter_context(patch.object(app, "open_camera", return_value=capture))
            mocks.enter_context(patch.object(app, "create_tasks", return_value=(segmenter, recognizer)))
            opening = mocks.enter_context(patch.object(app, "open_virtual_camera", return_value=camera))
            mocks.enter_context(patch.object(app, "person_mask", return_value=np.ones((24, 40), np.float32)))
            mocks.enter_context(patch.object(app, "person_fraction", return_value=0.0))
            # The teleport fires on the first check once the background is ready.
            mocks.enter_context(patch.object(app, "update_goku", side_effect=[True, False, False]))
            mocks.enter_context(patch.object(app, "update_saiyan"))
            times = (0, 0, 0.1, 4.1, 4.3, 4.35, 5.2, 5.22, 8.2, 8.23)
            mocks.enter_context(patch.object(app.time, "perf_counter", side_effect=times))
            mocks.enter_context(patch.object(app.cv2, "namedWindow"))
            mocks.enter_context(patch.object(app.cv2, "resizeWindow"))
            destroy = mocks.enter_context(patch.object(app.cv2, "destroyAllWindows"))
            mocks.enter_context(patch.object(app.cv2, "imshow", side_effect=lambda name, frame: displayed.append(frame.copy())))
            keys = (0, 0, 0, 0, ord("q"))
            mocks.enter_context(patch.object(app.cv2, "waitKey", side_effect=keys))
            mocks.enter_context(patch.object(app.cv2, "putText", side_effect=AssertionError("Only the setup banner draws text")))
            mocks.enter_context(patch("builtins.print"))
            if send_error is None:
                self.assertEqual(app.run(), 0)
            else:
                with self.assertRaisesRegex(RuntimeError, "send failed"):
                    app.run()
            opening.assert_called_once()
            self.assertEqual(opening.call_args.args[1:], (40, 24))
            capture.release.assert_called_once()
            segmenter.close.assert_called_once()
            recognizer.close.assert_called_once()
            camera.close.assert_called_once()
            destroy.assert_called_once()
        return frames, camera, displayed

    def test_preview_and_virtual_output_are_clean_in_all_phases(self):
        frames, camera, displayed = self.run_mock_video()
        self.assertEqual(len(displayed), 5)
        for sent, preview in zip(camera.send.call_args_list, displayed):
            np.testing.assert_array_equal(sent.args[0], preview)
        np.testing.assert_array_equal(displayed[0], frames[0])  # Calibrating.
        np.testing.assert_array_equal(displayed[2], frames[2])  # Ready: the gesture fires.
        np.testing.assert_array_equal(displayed[4], frames[1])  # Vanished: only the background.
        # The room is empty from the first frame, so it is captured straight
        # away; then the banner explains the gestures.
        self.assertTrue(self.banners[0][0].startswith("Capturing the empty room"))
        self.assertTrue(self.banners[-1][0].startswith("Ready!"))

    def test_virtual_output_is_not_mirrored(self):
        _, camera, displayed = self.run_mock_video(no_mirror=False)
        for sent, preview in zip(camera.send.call_args_list, displayed):
            np.testing.assert_array_equal(sent.args[0], preview[:, ::-1])

    def test_all_devices_close_on_send_error(self):
        self.run_mock_video(send_error=RuntimeError("send failed"))


class BackgroundCheckTests(unittest.TestCase):
    def test_background_with_a_person_is_rejected(self):
        background = np.zeros((24, 40, 3), np.uint8)
        person = np.zeros((24, 40), np.float32)
        with patch.object(app, "person_mask", return_value=person):
            self.assertEqual(app.person_fraction(Mock(), background, 0), 0.0)
        person[4:20, 10:25] = 1.0
        with patch.object(app, "person_mask", return_value=person):
            self.assertAlmostEqual(app.person_fraction(Mock(), background, 0), 16 * 15 / (24 * 40))


class CalibrationTests(unittest.TestCase):
    def test_background_is_captured_once_the_frame_stays_empty(self):
        frames = [np.full((24, 40, 3), color, np.uint8) for color in (10, 40, 80, 120)]
        capture = Mock()
        capture.read.side_effect = [(True, frame) for frame in frames]
        with ExitStack() as mocks:
            mocks.enter_context(patch.object(app, "parse_args", return_value=arguments(virtual_camera=False)))
            mocks.enter_context(patch.object(app, "download_models"))
            mocks.enter_context(patch.object(app, "open_camera", return_value=capture))
            mocks.enter_context(patch.object(app, "create_tasks", return_value=(Mock(), Mock())))
            mocks.enter_context(patch.object(app, "update_goku", return_value=False))
            mocks.enter_context(patch.object(app, "update_saiyan"))
            # A person in the first frame, then the room stays empty.
            checks = mocks.enter_context(patch.object(app, "person_fraction", side_effect=[0.3, 0.0, 0.0]))
            times = (0, 0, 0.1, 0.2, 0.9, 1.0)
            mocks.enter_context(patch.object(app.time, "perf_counter", side_effect=times))
            for name in ("namedWindow", "resizeWindow", "destroyAllWindows", "imshow"):
                mocks.enter_context(patch.object(app.cv2, name))
            mocks.enter_context(patch.object(app.cv2, "waitKey", side_effect=(0, 0, 0, ord("q"))))
            printed = mocks.enter_context(patch("builtins.print"))
            self.assertEqual(app.run(), 0)
        self.assertEqual(checks.call_count, 3)  # No checks once the background is ready.
        messages = [str(call.args[0]) for call in printed.call_args_list]
        self.assertEqual(sum("still in the frame" in message for message in messages), 1)
        self.assertTrue(any("Background ready" in message for message in messages))

    def run_with_background_file(self, path, person_checks):
        frames = [np.full((24, 40, 3), color, np.uint8) for color in (10, 40, 80, 120)]
        capture = Mock()
        capture.read.side_effect = [(True, frame) for frame in frames]
        with ExitStack() as mocks:
            mocks.enter_context(patch.object(app, "parse_args", return_value=arguments(
                virtual_camera=False, background_file=path)))
            mocks.enter_context(patch.object(app, "download_models"))
            mocks.enter_context(patch.object(app, "open_camera", return_value=capture))
            mocks.enter_context(patch.object(app, "create_tasks", return_value=(Mock(), Mock())))
            mocks.enter_context(patch.object(app, "update_goku", return_value=False))
            mocks.enter_context(patch.object(app, "update_saiyan"))
            checks = mocks.enter_context(patch.object(app, "person_fraction", side_effect=person_checks))
            mocks.enter_context(patch.object(app.time, "perf_counter", side_effect=(0, 0, 0.1, 0.2, 0.9, 1.0)))
            for name in ("namedWindow", "resizeWindow", "destroyAllWindows", "imshow"):
                mocks.enter_context(patch.object(app.cv2, name))
            mocks.enter_context(patch.object(app.cv2, "waitKey", side_effect=(0, 0, 0, ord("q"))))
            mocks.enter_context(patch("builtins.print"))
            self.assertEqual(app.run(), 0)
        return frames, checks

    def test_clean_background_is_saved_and_reused_next_time(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "background.png"
            frames, checks = self.run_with_background_file(path, [0.0, 0.0, 0.0])
            self.assertEqual(checks.call_count, 3)
            np.testing.assert_array_equal(app.cv2.imread(str(path)), frames[2])
            # Next start: the saved plate is used straight away, no person checks.
            _, checks = self.run_with_background_file(path, [])
            self.assertEqual(checks.call_count, 0)


class BannerTests(unittest.TestCase):
    def test_banner_darkens_only_the_bottom_and_keeps_size(self):
        frame = np.full((720, 1280, 3), 200, np.uint8)
        output = app.draw_banner(frame, ["Leave the camera view", "for a few seconds"])
        self.assertEqual(output.shape, frame.shape)
        np.testing.assert_array_equal(output[:500], frame[:500])
        self.assertLess(output[-5:].mean(), frame[-5:].mean())
        np.testing.assert_array_equal(frame, 200)  # The input frame is not modified.


class BlurBackgroundTests(unittest.TestCase):
    def test_blur_softens_detail_but_keeps_size(self):
        background = np.zeros((60, 80, 3), np.uint8)
        background[:, ::2] = 255
        blurred = app.blur_background(background)
        self.assertEqual(blurred.shape, background.shape)
        self.assertLess(blurred.std(), background.std() * 0.2)


if __name__ == "__main__":
    unittest.main()
