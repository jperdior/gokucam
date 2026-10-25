# VanishCam

Webcam effects for your video calls: vanish with a V sign, or teleport and go Super Saiyan like Goku. It runs on macOS, Windows, and Linux, and works in Google Meet, Microsoft Teams, Zoom, or any app that can use a camera, through the OBS Virtual Camera.

Video is processed locally on your computer. Nothing is recorded or sent anywhere.

## Where the idea came from

This project started as a fork of [Nicovic17/vanish_webcam](https://github.com/Nicovic17/vanish_webcam), a Windows app that makes you vanish from your webcam with a hand gesture. The background capture and the V-sign dissolve come from there. This version adds:

- **Goku mode:** Instant Transmission with two fingers on the forehead, and a Super Saiyan transformation with golden hair and aura.
- **Cross-platform launchers:** one script for macOS and Linux, one for Windows.
- **Video-call output:** a blurred background, and virtual camera output that is not mirrored.
- **A saved background:** you don't have to leave the frame every time you start.

The Naruto mode, the bundled audio, and the Italian interface from the original were removed.

## Effects

**Normal Vanish**
- Raise your index and middle fingers in a V sign.
- You dissolve into the background over about 3.6 seconds.
- Press `R` to reappear.

**Goku: Instant Transmission**
- Hold your index and middle fingers straight and together on your forehead for 1 second.
- Speed lines and a flash appear, you collapse into a beam of light, and you vanish.
- To come back, lower your hand, then repeat the gesture.
- If you break the pose before the second is up, the countdown restarts.

**Goku: Super Saiyan**
- In Goku mode, raise both fists in front of you and hold them for 1 second.
- Your real hair turns gold, and a golden aura flares up around you.
- Do it again to power down.
- The golden hair stays on while you teleport.

The terminal shows when a gesture is recognized and the countdown. No text is drawn on the video.

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
2. Double-click `start.bat`, or run it from a terminal:

```bat
start.bat --mode goku
```

A menu lets you pick the effect: click it, or press `1` (Normal Vanish) or `2` (Goku). To skip the menu, pass `--mode normal` or `--mode goku`.

## First start: capture the background

1. Step out of the camera frame.
2. As soon as the frame has been empty for half a second, the background is captured. The terminal tells you when you can step back in.

The background is saved to `background.png` and reused on later starts, so you only do this once. If the room changes or the camera moves, press `SPACE` to capture a new one.

## Use it in Google Meet or Microsoft Teams

VanishCam sends its video to the **OBS Virtual Camera**. Meet, Teams, and Zoom see it as a normal webcam.

### 1. Install the OBS Virtual Camera (once)

- **macOS:**
  1. Install [OBS Studio](https://obsproject.com/) and open it.
  2. Click **Start Virtual Camera** once.
  3. Allow the extension in **System Settings → General → Login Items & Extensions → Camera Extensions**.
  4. Stop the virtual camera and quit OBS.
- **Windows:** install [OBS Studio](https://obsproject.com/). The virtual camera comes with it.
- **Linux:** install `v4l2loopback`, then load it:

  ```bash
  sudo modprobe v4l2loopback devices=1 exclusive_caps=1 card_label="VanishCam"
  ```

  The `exclusive_caps=1` option lets Chrome-based browsers see the device.

### 2. Start VanishCam

```bash
./start.sh --mode goku --blur-background --virtual-camera
```

On Windows, run the same options with `start.bat`. The terminal prints `Virtual camera on: OBS Virtual Camera` when it's working.

**OBS itself must be closed** (or at least its own virtual camera stopped) while VanishCam is running. Only one app can drive the virtual camera at a time.

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

Turn the call app's own blur and background effects off. They would blur or cut out the effect itself. Use `--blur-background` instead: VanishCam blurs the room itself, so you vanish into the blurred background and the effect stays sharp.

The preview window is mirrored like a selfie view. The virtual camera output is not, so the other people in the call see you the right way round. Keyboard controls only work while the preview window has focus.

## Controls

- `SPACE`: capture a new background.
- `V`: start the vanish manually (Normal mode).
- `R`: reappear (Normal mode).
- `F`: toggle fullscreen.
- `Q` or `ESC`: quit.

In Goku mode, only the gestures make you vanish and reappear.

## Options

| Option | What it does |
| --- | --- |
| `--mode normal\|goku` | Skip the start menu. |
| `--blur-background` | Blur the room. You vanish into the blurred background. |
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

The effects are drawn in code, from MediaPipe models:
- **Person segmentation:** finds your silhouette.
- **Hand and face landmarks:** recognize the gestures.
- **Hair segmentation:** finds your hair for the golden recolouring.

No anime frames or images are included.
