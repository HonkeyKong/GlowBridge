# GlowBridge

**GlowBridge** is a lightweight, low-latency Python ambilight-style system that captures video from a V4L2 device (e.g., HDMI-to-USB), samples edge colors, and streams them to one or more **WLED** controllers via UDP Realtime (DRGB).

It's designed to be simple, hackable, low-CPU (Raspberry Pi friendly), and free of proprietary software or cloud dependencies. Supports multiple independent LED strips and WLED devices for whole-room ambient lighting.

---

Features

- V4L2 capture with MJPEG for low CPU usage
- **Multi-strip support**: Send different edge sections to multiple WLED devices simultaneously
- **Flexible side mapping**: Enable/disable individual sides per strip (e.g., only use bottom+left LEDs)
- Edge sampling with configurable patch averaging and margins
- Black-threshold suppression to prevent shimmer in dark scenes
- Scene-cut detection with faster transitions on large luminance changes
- EMA smoothing and per-frame step limiting for natural motion
- Saturation and brightness boosting to avoid washed-out colors
- Automatic image cropping to avoid sampling black bars
- Built-in calibration modes: `--test-chase`, `--test-sides`, `--test-sides-all`

---

## Requirements

### Software

- Linux with a working **V4L2** capture device (e.g., `/dev/video0`)
- Python **3.11+** recommended
- Python packages: `opencv-python`, `numpy`

### Hardware

- HDMI → USB capture device (UVC compliant)
- Addressable LED strip (WS2812B / SK6812 / similar)
- WLED controller (ESP8266 or ESP32)
- Local network connectivity between host and WLED device
- OpenCV (on Raspberry Pi: `sudo apt install python3-opencv`)

---

## Bill of Materials

| Component           | Example                                                    | Approx. Price (USD) | Notes                          |
| ------------------- | ---------------------------------------------------------- | ------------------- | ------------------------------ |
| Raspberry Pi        | [Pi 3B+ / Pi 4 / Pi Zero 2 W](https://a.co/d/8MqKKTY)      | \$15–\$45           | Pi 3+ recommended for Ethernet |
| HDMI → USB Capture  | [Generic UVC adapter](https://a.co/d/9cMBmpn)              | \$5–\$25            | Must support MJPEG             |
| LED Strip           | [WS2812B](https://a.co/d/83mTGrY)                          | \$5–\$20            | ~150–300 LEDs typical          |
| WLED Controller     | [GLEDOPTO ESP8266 WLED Controller](https://a.co/d/hd0Iq79) | \$3–$25             | ESP8266 also works             |
| Power Supply (LEDs) | [5V 5A+](https://a.co/d/gNvGSdp)                           | \$10–\$30           | Depends on LED count           |

- Note: For your power needs, you can use this [WLED Calculator](https://wled-calculator.github.io/).

---

## Installation

### 1. Install system dependencies

On **Raspberry Pi / Debian / Ubuntu**:

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv v4l-utils libatlas-base-dev libjasper-dev libharfbuzz0b libwebp6
```

On **Fedora / RHEL**:

```bash
sudo dnf install python3 python3-pip v4l-utils
```

### 2. Clone or download GlowBridge

```bash
cd /path/to/glowbridge
```

### 3. Create and activate a Python virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 5. Verify your V4L2 capture device

```bash
v4l2-ctl --list-devices
```

Ensure your device (e.g., `/dev/video0`) is present and accessible.

### 6. Configure GlowBridge

Edit or create a `config.toml` in the project directory (see [Configuration](#configuration) below for details), copy the `config.example.toml`, or use environment variables or CLI flags.

---

## Quick Start

1. Connect and verify your capture device (e.g., HDMI dongle):
   
   ```bash
   v4l2-ctl --list-devices
   ```

2. Configure WLED IP/port, LED counts, and video device (see Configuration below).
   
   - Lots of cheap LED strips group multiple LEDs per chip (often 3). Count those groups as one LED. The default 69 LEDs in the script represent 207 physical LEDs on such strips.

3. Run a test to confirm mapping:
   
   ```bash
   python3 glowbridge.py --test-sides
   ```
   
   - It cycles RIGHT → TOP → LEFT → BOTTOM so you can confirm orientation. You can also run with `--test-sides-all` to light up all sides at once.

4. Start streaming:
   
   ```bash
   python3 glowbridge.py
   ```

> The script keeps WLED in Realtime mode by sending DRGB packets at `out_fps` until you stop it (Ctrl+C). A lock file prevents multiple instances.

---

## Configuration

`settings.py` centralizes configuration. Precedence (later wins): defaults → config file → environment variables → CLI flags. Use `python3 glowbridge.py --print-config` to see the resolved settings.

### Config file (TOML)

GlowBridge loads the first file it finds in this order:

- `./config.toml` (alongside the script)
- `~/.config/glowbridge/config.toml`
- `/etc/glowbridge/config.toml`

Example (legacy single-strip format, still supported):

```toml
[wled]
ip = "192.168.1.50"
port = 21324
timeout_seconds = 2

[video]
device = "/dev/video0"
cap_w = 640
cap_h = 480
cap_fps = 30
crop_to_16_9 = true

[layout]
right = 17
top = 26
left = 15
bottom = 11

[sampling]
sample_w = 160
sample_h = 90
edge_margin = 3
patch_r = 2

[effects]
out_fps = 30
smooth_alpha = 0.88
max_step_per_frame = 48
black_luma_threshold = 12
black_chroma_threshold = 10
scene_cut_luma_jump = 35
scene_cut_boost_mult = 2.0
sat_boost = 1.35
val_boost = 1.05

lock_path = "/tmp/glowbridge.lock"
```

### Multi-strip configuration (recommended)

For multiple LED strips or WLED devices, use the `strips` array. Each strip can:

- Output to one or more WLED targets (mirror the same colors to multiple devices)
- Enable only specific sides by setting unwanted sides to `0`
- Have independent timeout settings

```toml
[video]
device = "/dev/video0"
cap_w = 640
cap_h = 480
cap_fps = 30

[sampling]
sample_w = 160
sample_h = 90
edge_margin = 6
patch_r = 2

[effects]
out_fps = 30
smooth_alpha = 0.70
max_step_per_frame = 128
black_luma_threshold = 12
black_chroma_threshold = 10
scene_cut_luma_jump = 20
scene_cut_boost_mult = 3.0
sat_boost = 1.35
val_boost = 1.05

# Define multiple strips with different layouts and targets
[layout]
strips = [
  # Main TV backlight with all four sides
  { 
    name = "tv",
    timeout_seconds = 2,
    right = 12,
    top = 22,
    left = 12,
    bottom = 22,
    targets = [
      { ip = "192.168.1.50", port = 21324 }
    ]
  },

  # Room ambience using only bottom and left edges, sent to a second WLED
  { 
    name = "shelf",
    timeout_seconds = 2,
    right = 0,      # disabled
    top = 0,        # disabled
    left = 15,
    bottom = 25,
    targets = [
      { ip = "192.168.1.51", port = 21324 }
    ]
  },

  # Mirror the same output to multiple WLED devices
  { 
    name = "mirror",
    timeout_seconds = 2,
    right = 10,
    top = 15,
    left = 10,
    bottom = 15,
    targets = [
      { ip = "192.168.1.52", port = 21324 },
      { ip = "192.168.1.53", port = 21324 }
    ]
  }
]

lock_path = "/tmp/glowbridge.lock"
```

### Environment variables

Prefix keys with `GLOWBRIDGE_`. Examples:

```bash
export GLOWBRIDGE_WLED_IP=192.168.1.50
export GLOWBRIDGE_RIGHT=20
export GLOWBRIDGE_OUT_FPS=25
```

### Command-line overrides

CLI flags mirror the config keys and take highest priority. Examples:

```bash
# Temporary LED count change
python3 glowbridge.py --right 20 --top 30 --left 20 --bottom 10

# Override capture device and print the resolved config
python3 glowbridge.py --video-dev /dev/video1 --print-config
```

---

## Customization

Tune these keys (via config file, env vars, or CLI) to fit your hardware:

- **Multi-strip mode**: Define `strips = [...]` array in config (see examples above)
  - Each strip has: `name`, `timeout_seconds`, `right`, `top`, `left`, `bottom`, `targets`
  - Set any side to `0` to disable it for that strip
  - `targets` is an array of `{ip, port}` objects (can send to multiple WLED devices)
- **Legacy single-strip mode**: `wled.ip`, `wled.port`, `wled.timeout_seconds` (still supported)
- Capture: `video.device`, `video.cap_w`, `video.cap_h`, `video.cap_fps`, `video.crop_to_16_9`

### LED Layout (counter-clockwise)

**Legacy mode**:

- `layout.right`, `layout.top`, `layout.left`, `layout.bottom`: number of LEDs on each side
- `layout.led_count`: derived total, must match strip length in WLED

**Multi-strip mode**:

- Each strip in the `strips` array has its own `right`, `top`, `left`, `bottom` counts
- Set any side to `0` to disable it (e.g., `right = 0, top = 0` for only left+bottom LEDs)
- Each strip independently calculates `led_count` from its enabled sides

Mapping assumptions in `build_led_colors_from_frame()`:

- Order is counter-clockwise
- Start at bottom-right
- RIGHT side is sampled bottom → top
- TOP side right → left
- LEFT side top → bottom
- BOTTOM side left → right
- **Disabled sides** (count = 0) are skipped entirely in the output

If your physical start corner or direction differs:

- Adjust the side order or directions in `build_led_colors_from_frame()`; or
- Use WLED’s segment/reverse settings to match this order without changing code.

### Capture & Sampling

- `video.cap_w`, `video.cap_h`, `video.cap_fps`: requested capture format (MJPEG is set by default)
- `video.crop_to_16_9`: crop vertically to 16:9 before downsampling (recommended for HDMI sources)
- `sampling.sample_w`, `sampling.sample_h`: downsample size used for edge sampling (smaller = faster)
- `sampling.edge_margin`: how far inward to sample from each edge (in sample-space pixels)
- `sampling.patch_r`: patch radius (2 → 5×5); larger values smooth noisy sources

### Output & Smoothing

- `effects.out_fps`: packet rate to WLED (15–30 is ample)
- `effects.smooth_alpha`: EMA smoothing factor (0 = off, 0.85–0.95 typical)
- `effects.max_step_per_frame`: per-channel clamp limit (lower = gentler changes)
- `effects.scene_cut_luma_jump`: luminance jump threshold to detect a cut
- `effects.scene_cut_boost_mult`: multiplier to allow faster change on scene cuts

### Color Boost & Black Suppression

- `effects.sat_boost`: saturation multiplier (e.g., 1.35)
- `effects.val_boost`: brightness multiplier (e.g., 1.05)
- `effects.black_luma_threshold`: suppress near-black shimmer (higher = more aggressive)
- `effects.black_chroma_threshold`: suppress noise when colors are basically gray

---

## Test Modes

Use these to validate LED order and side counts:

```bash
# Single bright pixel chasing through all LEDs
python3 glowbridge.py --test-chase

# One side at a time (best for calibration)
python3 glowbridge.py --test-sides

# Color all sides simultaneously
python3 glowbridge.py --test-sides-all
```

While a test is running, the script continues sending packets to keep WLED in Realtime mode.

---

## How It Works

- Captures frames from `video.device`, optionally center-crops to 16:9, then downsamples to `sampling.sample_w × sampling.sample_h`.
- Samples edge patches per LED with `sample_patch_rgb()` (keeps the brightest 25% of the patch by luma to avoid dull averages).
- Applies black-threshold masking to reduce flicker in dark scenes.
- Boosts saturation/brightness in HSV.
- Computes scene-cut detection via average luma; uses step limiting and EMA smoothing for stability.
- **For each configured strip**: builds LED color array based on enabled sides, then sends to all targets for that strip
- Packs colors into WLED DRGB UDP payload: `byte0=2 (DRGB)`, `byte1=timeout seconds`, followed by `RGB` triples for each LED.
- Supports simultaneous output to multiple WLED devices (each strip can have multiple targets).

---

## Troubleshooting

- "Could not open /dev/video0": Check permissions; add your user to the `video` group, or run with appropriate udev rules.
- No Realtime in WLED: Ensure UDP Notifier is enabled and not blocked by firewall; confirm IP/port for each target.
- **Multiple devices not responding**: Verify each WLED device IP/port in `targets` array; check network connectivity to all devices.
- **Wrong sides lighting up**: Verify LED counts in strip config; use `--test-sides --strip <name>` to test a specific strip.
- Colors look washed out: Increase `effects.sat_boost` (try 1.5) or lower `effects.smooth_alpha`.
- Motion feels sluggish: Increase `effects.max_step_per_frame` or reduce `effects.smooth_alpha`.
- Performance issues: Lower `sampling.sample_w/h`, keep MJPEG, reduce `effects.out_fps`.
- Wrong LED order/direction: Use `--test-sides` to observe mapping, then adjust `build_led_colors_from_frame()` or WLED segment settings.
- LED color order (RGB vs GRB): Configure color order in WLED; if needed, remap channels before sending in `send_wled()`.
- **Testing specific strips**: Use `--strip <name>` with test modes to verify individual strip configurations.

---

## Notes

- The script uses a lock file (default `/tmp/glowbridge.lock`) to prevent multiple instances; override with `lock_path`.
- On exit, WLED will time out of Realtime mode after `wled.timeout_seconds`.
- Signal handling for `SIGINT`/`SIGTERM` stops the loop cleanly.

---

## Optional: Systemd Service

For auto-start, create a systemd service that runs this script on boot once your capture device is ready. Ensure you export your venv or use system Python with required packages installed.

```ini
[Unit]
Description=GlowBridge Ambient Lighting Service
After=network-online.target
Wants=network-online.target

# If your capture device is USB, this helps avoid race conditions
After=dev-video0.device
Requires=dev-video0.device

[Service]
Type=simple
User=youruser
Group=video

ExecStart=/usr/bin/python3 /path/to/glowbridge_folder/glowbridge.py
WorkingDirectory=/path/to/glowbridge_folder

# Restart if the capture device or script hiccups
Restart=on-failure
RestartSec=2

# Clean shutdown
KillSignal=SIGTERM
TimeoutStopSec=5

# Basic hardening (safe defaults)
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```
