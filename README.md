# GokuCam

Turn Super Saiyan and use Instant Transmission on your webcam. It runs on macOS, Windows, and Linux, and works in Google Meet, Microsoft Teams, Zoom, or any app that can use a camera, through the OBS Virtual Camera.

Video is processed locally on your computer. Nothing is recorded or sent anywhere.

## Where the idea came from

GokuCam started as a fork of [Nicovic17/vanish_webcam](https://github.com/Nicovic17/vanish_webcam), a Windows app that makes you vanish from your webcam with a hand gesture. The idea of triggering effects with gestures, and the background capture this project builds on, come from there.

GokuCam focuses on Dragon Ball. It adds Instant Transmission, the Super Saiyan transformation, a blurred background for video calls, a saved background, and launchers for every OS. The original's V-sign and Naruto modes, its audio, and its Italian interface were removed.

## Gestures

**Super Saiyan:** raise both fists in front of you and hold them for 1 second.
- Your real hair turns gold, and a golden aura flares up around you.
- Do it again to power down.

**Instant Transmission:** hold your index and middle fingers straight and together on your forehead for 1 second.
- Speed lines and a flash appear, you collapse into a beam of light, and you're gone.
- To come back, lower your hand, then do the gesture again.
- If you break the pose before the second is up, the countdown restarts.
- If you're Super Saiyan, the golden hair teleports with you.

The terminal shows when a gesture is recognized. No text is drawn on the video.

## Install and run

You need Python 3.12; 3.11 and 3.13 also work. MediaPipe does not support newer versions yet. The first run creates a `.venv` folder, installs the dependencies, and downloads the MediaPipe models.

**macOS**

```bash
brew install python@3.12
./start.sh
```

When macOS asks, allow your terminal app to use the camera. If you missed the prompt, turn it on in **System Settings → Privacy & Security → Camera**, then restart the terminal.

**Linux**

```bash
./start.sh
```

**Windows**

1. Install Python 3.12 from [python.org](https://www.python.org/downloads/), with the `py` launcher.
2. Double-click `start.bat`, or run it from a terminal with options:

```bat
start.bat --blur-background
```

## First start: capture the background

1. Step out of the camera frame.
2. As soon as the frame has been empty for half a second, the background is captured. The terminal tells you when you can step back in.

This background is what's left when you teleport away. It's saved to `background.png` and reused on later starts, so you only do this once. If the room changes or the camera moves, press `SPACE` to capture a new one.

## Use it in Google Meet or Microsoft Teams

GokuCam sends its video to the **OBS Virtual Camera**. Meet, Teams, and Zoom see it as a normal webcam.

### 1. Install the OBS Virtual Camera (once)

- **macOS:**
  1. Install [OBS Studio](https://obsproject.com/) and open it.
  2. Click **Start Virtual Camera** once.
  3. Allow the extension in **System Settings → General → Login Items & Extensions → Camera Extensions**.
  4. Stop the virtual camera and quit OBS.
- **Windows:** install [OBS Studio](https://obsproject.com/). The virtual camera comes with it.
- **Linux:** install `v4l2loopback`, then load it:

  ```bash
  sudo modprobe v4l2loopback devices=1 exclusive_caps=1 card_label="GokuCam"
  ```

  The `exclusive_caps=1` option lets Chrome-based browsers see the device.

### 2. Start GokuCam

```bash
./start.sh --blur-background --virtual-camera
```

On Windows, run the same options with `start.bat`. The terminal prints `Virtual camera on: OBS Virtual Camera` when it's working.

**OBS itself must be closed** (or at least its own virtual camera stopped) while GokuCam is running. Only one app can drive the virtual camera at a time.

### 3. Pick the camera in your call

- **Google Meet:**
  1. Go to **Settings → Video → Camera**.
  2. Choose **OBS Virtual Camera**.
  3. Turn off **Backgrounds and effects**.
- **Microsoft Teams:**
  1. Go to **Settings → Devices → Camera** (or the camera menu before joining).
  2. Choose **OBS Virtual Camera**.
  3. Set **Background effects** to none.
- **Zoom:** go to **Settings → Video → Camera**, choose **OBS Virtual Camera**, and set the virtual background to none.

Turn the call app's own blur and background effects off. They would blur or cut out the aura and the teleport. Use `--blur-background` instead: GokuCam blurs the room itself, so you teleport out of the blurred background and the effects stay sharp.

The preview window is mirrored like a selfie view. The virtual camera output is not, so the other people in the call see you the right way round. Keyboard controls only work while the preview window has focus.

## Controls

- `SPACE`: capture a new background.
- `F`: toggle fullscreen.
- `Q` or `ESC`: quit.

## Options

| Option | What it does |
| --- | --- |
| `--blur-background` | Blur the room. You teleport out of the blurred background. |
| `--virtual-camera` / `--no-virtual-camera` | Require the virtual camera, or show only the preview window. By default, the virtual camera is used when it's available. |
| `--virtual-backend obs\|unitycapture\|v4l2loopback` | Choose the virtual camera driver. |
| `--background-file PATH` | Where the background is saved (default `background.png`). |
| `--camera N` | Use another webcam (default `0`). |
| `--width`, `--height`, `--fps` | Capture size and maximum output FPS. |
| `--no-mirror` | Don't mirror the preview. |
| `--gesture-debug` | Print gesture recognition details to the terminal. |

## Development

```bash
.venv/bin/python -m pip install pytest
.venv/bin/python -m pytest -q
```

All effects are drawn in code, from MediaPipe models:
- **Person segmentation:** finds your silhouette.
- **Hand and face landmarks:** recognize the gestures.
- **Hair segmentation:** finds your hair for the golden recolouring.

No anime frames or images are included.
