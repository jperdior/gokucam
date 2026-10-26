# GokuCam

Turn Super Saiyan, use Instant Transmission, and throw a Spirit Bomb on your webcam. It runs on macOS, Windows, and Linux, and works in Google Meet, Microsoft Teams, Zoom, or any app that can use a camera, through the OBS Virtual Camera.

Video is processed locally on your computer. Nothing is recorded or sent anywhere.

## Gestures

**Super Saiyan:** raise both fists in front of you and hold them for 1 second.
- Your real hair turns gold, and a golden aura flares up around you.
- Do it again to power down.

**Instant Transmission:** hold your index and middle fingers straight and together on your forehead for 1 second.
- Speed lines and a flash appear, you collapse into a beam of light, and you're gone.
- To come back, lower your hand, then do the gesture again.
- If you're Super Saiyan, the golden hair teleports with you.

**Spirit Bomb:** step back from the camera and hold both hands high above your head for 1 second.
- A glowing ball gathers energy above you and grows while your hands stay up.
- Lower your hands to throw it at the camera.
- From close up it won't start, so the terminal will tell you to step back.

If you break a pose before the second is up, the countdown restarts. The terminal shows when a gesture is recognized.

## Install

1. **Python 3.12.** 3.11 and 3.13 also work. MediaPipe does not support newer versions yet.
   - **macOS:** `brew install python@3.12`
   - **Windows:** install it from [python.org](https://www.python.org/downloads/), with the `py` launcher.
2. **Virtual camera.** Only needed for video calls.
   - **macOS:**
     1. Install [OBS Studio](https://obsproject.com/), open it, and click **Start Virtual Camera** once.
     2. Allow the extension in **System Settings → General → Login Items & Extensions → Camera Extensions**.
     3. Quit OBS.
   - **Windows:** install [OBS Studio](https://obsproject.com/). The virtual camera comes with it.
   - **Linux:** install `v4l2loopback` and load it. The `exclusive_caps=1` option lets Chrome-based browsers see the device:

     ```bash
     sudo modprobe v4l2loopback devices=1 exclusive_caps=1 card_label="GokuCam"
     ```

## Run

```bash
./start.sh --blur-background        # macOS / Linux
start.bat --blur-background         # Windows
```

The first run creates a `.venv` folder, installs the dependencies, and downloads the MediaPipe models. On macOS, allow your terminal app to use the camera when asked. If you missed the prompt, turn it on in **System Settings → Privacy & Security → Camera**, then restart the terminal.

### First run: leave the frame

**Nothing works until GokuCam has a picture of the empty room.** The blur, the aura, and the teleport are all drawn over it.

1. When the video says **Leave the camera view**, step completely out, including hands, shoulders, and the top of your head.
2. After a few seconds it says **Capturing the empty room...**. If part of you is still visible, the terminal says how much of the image you're covering.
3. When it says **Ready! Come back in**, step back in. The gestures now work.

The picture is saved to `background.png` and reused on later starts, so you only do this once. If the room changes or the camera moves, press `SPACE` and leave the frame again.

## Use it in Google Meet, Microsoft Teams, or Zoom

1. **Quit OBS** if it's open. Only one app can drive the virtual camera at a time.
2. **Start GokuCam.** When the virtual camera is installed, GokuCam uses it automatically; the terminal prints `Virtual camera on: OBS Virtual Camera`. Pass `--virtual-camera` to stop with an error if it isn't available.
3. **In the call, choose OBS Virtual Camera as your camera.** Turn off the call app's own background effects; they would blur or cut out the aura and the teleport. Use `--blur-background` instead.
   - **Meet:** **Settings → Video → Camera**, then turn off **Backgrounds and effects**.
   - **Teams:** **Settings → Devices → Camera**, then set **Background effects** to none.
   - **Zoom:** **Settings → Video → Camera**, then set the virtual background to none.

The preview window is mirrored like a selfie view. The video sent to the call is not mirrored, so people see you the right way round.

## Controls

These keys only work while the preview window has focus.

- `SPACE`: capture a new background.
- `F`: toggle fullscreen.
- `Q` or `ESC`: quit. On macOS the window's close button is disabled, so use these keys, or `Ctrl+C` in the terminal.

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

All effects are drawn in code, with no anime frames or images. They rely on these MediaPipe models:
- **Person segmentation:** finds your silhouette.
- **Hand and face landmarks:** recognize the gestures.
- **Hair segmentation:** finds your hair for the golden recolouring.

## Credits

Inspired by [Nicovic17/vanish_webcam](https://github.com/Nicovic17/vanish_webcam), a webcam app that makes you vanish with a hand gesture.
