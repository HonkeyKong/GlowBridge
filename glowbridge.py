#!/usr/bin/env python3

import os
import cv2
import sys
import time
import fcntl
import signal
import socket
import argparse
import numpy as np
from html import parser
from pathlib import Path
from types import SimpleNamespace
from settings import load_settings

_lock_fd = None
running = True

LOCK_PATH = "/tmp/glowbridge.lock"

import os
from pathlib import Path

def _read_text(path: Path) -> str:
    try:
        return path.read_text(errors="ignore")
    except Exception:
        return ""

def _is_usb_video_node(video: str) -> bool:
    # video like "video0"
    base = Path("/sys/class/video4linux") / video
    uevent = _read_text(base / "device" / "uevent")
    if "ID_BUS=usb" in uevent:
        return True

    # Fallback: resolve sysfs path and look for "/usb"
    try:
        dev_path = (base / "device").resolve()
        return "/usb" in str(dev_path)
    except Exception:
        return False

def _pick_v4l_by_id(prefer_usb: bool = True) -> str | None:
    """
    Returns a stable symlink path like /dev/v4l/by-id/usb-...-video-index0
    or None if not available.
    """
    by_id = Path("/dev/v4l/by-id")
    if not by_id.exists():
        return None

    # Only consider entries that point to a video node
    entries = []
    for p in sorted(by_id.iterdir()):
        try:
            target = p.resolve()  # -> /dev/videoX
        except Exception:
            continue
        if target.name.startswith("video"):
            entries.append((p, target))

    if not entries:
        return None

    # If requested, filter to USB-backed video nodes using sysfs
    if prefer_usb:
        usb_entries = []
        for link, target in entries:
            if _is_usb_video_node(target.name):
                usb_entries.append((link, target))
        entries = usb_entries or entries

    # Heuristic: prefer "video-index0" (usually the capture stream)
    def score(item):
        link, target = item
        name = link.name.lower()
        s = 0
        if "video-index0" in name:
            s -= 100
        if "index0" in name:
            s -= 25
        # lower /dev/video number is a mild preference
        try:
            s += int(target.name.replace("video", "")) * 0.1
        except Exception:
            pass
        return s

    best_link, _ = sorted(entries, key=score)[0]
    return str(best_link)

def autodetect_video_device(prefer_usb: bool = True, verify_frames: bool = True) -> str | None:
    """
    Returns a device path to open with OpenCV:
      - Prefer /dev/v4l/by-id/* (stable)
      - Fallback to /dev/videoX detected as USB via sysfs
      - Optionally verify capture works by reading a frame
    """
    # 1) Stable by-id path (best UX)
    by_id = _pick_v4l_by_id(prefer_usb=prefer_usb)
    if by_id:
        if not verify_frames:
            return by_id
        cap = cv2.VideoCapture(by_id, cv2.CAP_V4L2)
        if cap.isOpened():
            ok, frame = cap.read()
            cap.release()
            if ok and frame is not None:
                return by_id
        else:
            cap.release()

    # 2) Fallback: enumerate /dev/video* and pick USB
    sys_base = Path("/sys/class/video4linux")
    if not sys_base.exists():
        return None

    vids = sorted([p.name for p in sys_base.iterdir() if p.name.startswith("video")],
                  key=lambda s: int(s.replace("video", "") or "9999"))

    candidates = [v for v in vids if (not prefer_usb or _is_usb_video_node(v))] or vids

    if verify_frames:
        for v in candidates:
            dev = f"/dev/{v}"
            cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
            if not cap.isOpened():
                cap.release()
                continue
            ok, frame = cap.read()
            cap.release()
            if ok and frame is not None:
                return dev
        return None

    return f"/dev/{candidates[0]}" if candidates else None

def acquire_lock_or_die():
    global _lock_fd
    _lock_fd = open(LOCK_PATH, "w")
    try:
        fcntl.flock(_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Another instance is already running (lock held). Exiting.", file=sys.stderr)
        sys.exit(2)

def _ansi_bg(r: int, g: int, b: int) -> str:
    # 24-bit background color, then reset later
    return f"\x1b[48;2;{r};{g};{b}m"

def _ansi_reset() -> str:
    return "\x1b[0m"

def build_preview_grid(colors: np.ndarray, right: int, top: int, left: int, bottom: int):
    """
    Build a 2D perimeter grid of indices into `colors` matching your CCW mapping:
      idx 0..right-1     = RIGHT bottom->top
      next top           = TOP right->left
      next left          = LEFT top->bottom
      last bottom        = BOTTOM left->right

    Returns:
      grid_idx: (H, W) int array, -1 for non-perimeter cells, else LED index
      H, W chosen to fit the longest sides.
    """
    W = max(top, bottom)
    H = max(left, right)

    grid = -np.ones((H, W), dtype=np.int32)

    idx = 0

    # RIGHT column: bottom->top
    x = W - 1
    for i in range(right):
        t = (i + 0.5) / right
        y = int((H - 1) * (1.0 - t))  # bottom->top
        grid[y, x] = idx
        idx += 1

    # TOP row: right->left
    y = 0
    for i in range(top):
        t = (i + 0.5) / top
        x = int((W - 1) * (1.0 - t))  # right->left
        grid[y, x] = idx
        idx += 1

    # LEFT column: top->bottom
    x = 0
    for i in range(left):
        t = (i + 0.5) / left
        y = int((H - 1) * t)  # top->bottom
        grid[y, x] = idx
        idx += 1

    # BOTTOM row: left->right
    y = H - 1
    for i in range(bottom):
        t = (i + 0.5) / bottom
        x = int((W - 1) * t)  # left->right
        grid[y, x] = idx
        idx += 1

    return grid

def render_preview_ansi(colors: np.ndarray, grid_idx: np.ndarray, stats: str = ""):
    """
    Draws a rectangular perimeter preview using ANSI background colors.
    Uses two spaces per cell for a more square look.
    """
    # Move cursor to top-left and clear screen
    out = ["\x1b[H\x1b[2J"]

    H, W = grid_idx.shape
    for y in range(H):
        line = []
        for x in range(W):
            li = int(grid_idx[y, x])
            if li < 0:
                line.append(_ansi_reset() + "  ")
            else:
                r, g, b = (int(colors[li, 0]), int(colors[li, 1]), int(colors[li, 2]))
                line.append(_ansi_bg(r, g, b) + "  ")
        line.append(_ansi_reset())
        out.append("".join(line))

    if stats:
        out.append(_ansi_reset() + stats)

    sys.stdout.write("\n".join(out) + "\n")
    sys.stdout.flush()


def auto_crop_content_rgb(rgb_small: np.ndarray,
                          luma_thresh: int,
                          min_percent: float,
                          pad: int,
                          min_w: int,
                          min_h: int) -> np.ndarray:
    """
    Crops out black bars/padding from an RGB image by finding the bounding box
    of "bright enough" pixels based on luma.
    Operates on already-downsampled frames (fast).
    """
    h, w = rgb_small.shape[:2]
    # luma approx (uint16 to avoid overflow)
    r = rgb_small[..., 0].astype(np.uint16)
    g = rgb_small[..., 1].astype(np.uint16)
    b = rgb_small[..., 2].astype(np.uint16)
    luma = ((54*r + 183*g + 19*b) >> 8).astype(np.uint8)

    # A pixel counts as "content" if luma >= threshold
    bright = (luma >= luma_thresh)

    # Row/col is "content" if enough pixels are bright
    row_frac = bright.mean(axis=1)  # 0..1
    col_frac = bright.mean(axis=0)

    rows = np.where(row_frac >= min_percent)[0]
    cols = np.where(col_frac >= min_percent)[0]

    # If we can't find content reliably, don't crop
    if rows.size == 0 or cols.size == 0:
        return rgb_small

    y0, y1 = int(rows[0]), int(rows[-1])
    x0, x1 = int(cols[0]), int(cols[-1])

    # Pad + clamp
    y0 = max(0, y0 - pad)
    y1 = min(h - 1, y1 + pad)
    x0 = max(0, x0 - pad)
    x1 = min(w - 1, x1 + pad)

    # Ensure not too tiny (avoid weird crops on dark scenes)
    if (x1 - x0 + 1) < min_w or (y1 - y0 + 1) < min_h:
        return rgb_small

    return rgb_small[y0:y1+1, x0:x1+1]


def boost_saturation_rgb(colors: np.ndarray, sat_boost: float, val_boost: float) -> np.ndarray:
    # colors: (N,3) uint8 RGB
    if sat_boost == 1.0 and val_boost == 1.0:
        return colors

    # OpenCV expects HSV in 0..179 for H and 0..255 for S,V
    rgb = colors.reshape(1, -1, 3).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    hsv[..., 1] *= sat_boost   # S
    hsv[..., 2] *= val_boost   # V

    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2], 0, 255)

    bgr2 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    rgb2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB)
    return rgb2.reshape(-1, 3)


def apply_effects(colors_raw: np.ndarray, prev: "np.ndarray|None", effects: "EffectsCfg") -> tuple[np.ndarray, np.ndarray]:
    """Apply black thresholding, saturation/value boost, and scene-cut aware smoothing.

    Returns (colors_out_uint8, prev_float32_for_next_frame).
    """
    # 1) remove near-black noise
    colors = apply_black_threshold(
        colors_raw,
        black_luma=effects.black_luma_threshold,
        black_chroma=effects.black_chroma_threshold,
    )

    # 2) boost saturation/value to avoid washed look
    colors = boost_saturation_rgb(colors, effects.sat_boost, effects.val_boost)

    colors_f = colors.astype(np.float32)

    if prev is None:
        prev_f = colors_f
    else:
        prev_f = prev.astype(np.float32)

        # Detect scene cut via average luma jump
        l_prev = avg_luma(prev_f)
        l_new = avg_luma(colors_f)
        l_jump = abs(l_new - l_prev)

        max_step = float(max(1.0, effects.max_step_per_frame))
        if l_jump > float(effects.scene_cut_luma_jump):
            # allow faster response on cuts
            max_step *= float(effects.scene_cut_boost_mult)

        # Clamp change per frame (prevents harsh flicker)
        delta = colors_f - prev_f
        delta = np.clip(delta, -max_step, max_step)
        colors_f = prev_f + delta

        # EMA smoothing
        a = float(effects.smooth_alpha)
        colors_f = (a * prev_f) + ((1.0 - a) * colors_f)

        prev_f = colors_f

    colors_out = np.clip(colors_f, 0, 255).astype(np.uint8)
    return colors_out, prev_f
def avg_luma(colors_f: np.ndarray) -> float:
    r = colors_f[:,0]; g = colors_f[:,1]; b = colors_f[:,2]
    return float((54*r + 183*g + 19*b).mean() / 256.0)

def limit_step(prev_colors: np.ndarray, new_colors: np.ndarray, max_step: int) -> np.ndarray:
    """
    Clamp per-channel change per LED to [-max_step, +max_step].
    Inputs: float32 arrays shape (N,3)
    """
    delta = new_colors - prev_colors
    delta = np.clip(delta, -max_step, max_step)
    return prev_colors + delta

def apply_black_threshold(colors: np.ndarray, black_luma: int, black_chroma: int) -> np.ndarray:
    # Use int32 to avoid overflow (183*255 > int16 max)
    c = colors.astype(np.int32)
    r, g, b = c[:, 0], c[:, 1], c[:, 2]

    luma = (54*r + 183*g + 19*b) >> 8
    chroma = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)

    out = colors.copy()
    mask = (luma < black_luma) | ((luma < (black_luma + 8)) & (chroma < black_chroma))
    out[mask] = 0
    return out

def handle_sigint(sig, frame):
    global running
    running = False

signal.signal(signal.SIGINT, handle_sigint)
signal.signal(signal.SIGTERM, handle_sigint)

# def crop_center_16_9(frame_bgr: np.ndarray) -> np.ndarray:
#     h, w = frame_bgr.shape[:2]
#     target_h = int(w * 9 / 16)
#     if target_h <= 0 or target_h > h:
#         return frame_bgr  # can't crop sensibly, return as-is
#     y0 = (h - target_h) // 2
#     return frame_bgr[y0:y0 + target_h, :]

def sample_patch_rgb(img_rgb: np.ndarray, x: int, y: int, r: int) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    x0 = max(0, x - r)
    x1 = min(w, x + r + 1)
    y0 = max(0, y - r)
    y1 = min(h, y + r + 1)
    
    patch = img_rgb[y0:y1, x0:x1].reshape(-1, 3).astype(np.int32)
    # sort by luma, keep the top fraction 
    luma = (54*patch[:,0] + 183*patch[:,1] + 19*patch[:,2])  # int32 safe
    # scaled 
    keep = max(1, int(len(luma) * 0.25)) 
    # top 25% 
    idx = np.argpartition(luma, -keep)[-keep:] 
    mean = patch[idx].mean(axis=0) 
    return np.clip(mean, 0, 255).astype(np.uint8)


def build_led_colors_from_frame(sample_rgb: np.ndarray,
                                right: int, top: int, left: int, bottom: int,
                                edge_margin: int, patch_r: int) -> np.ndarray:
    h, w = sample_rgb.shape[:2]
    m = max(0, min(edge_margin, w - 1, h - 1))
    r = patch_r

    led_count = max(0, right) + max(0, top) + max(0, left) + max(0, bottom)
    colors = np.zeros((led_count, 3), dtype=np.uint8)
    idx = 0

    if right > 0:
        x = w - 1 - m
        for i in range(right):
            t = (i + 0.5) / right
            y = int((h - 1 - m) * (1.0 - t) + m * t)
            colors[idx] = sample_patch_rgb(sample_rgb, x, y, r)
            idx += 1

    if top > 0:
        y = m
        for i in range(top):
            t = (i + 0.5) / top
            x = int((w - 1 - m) * (1.0 - t) + m * t)
            colors[idx] = sample_patch_rgb(sample_rgb, x, y, r)
            idx += 1

    if left > 0:
        x = m
        for i in range(left):
            t = (i + 0.5) / left
            y = int(m * (1.0 - t) + (h - 1 - m) * t)
            colors[idx] = sample_patch_rgb(sample_rgb, x, y, r)
            idx += 1

    if bottom > 0:
        y = h - 1 - m
        for i in range(bottom):
            t = (i + 0.5) / bottom
            x = int(m * (1.0 - t) + (w - 1 - m) * t)
            colors[idx] = sample_patch_rgb(sample_rgb, x, y, r)
            idx += 1

    return colors


def run_test_chase(sock, strip):
    """Simple single-pixel chase to verify realtime UDP output."""
    idx = 0
    delay = 0.05  # seconds

    led_count = int(getattr(strip.layout, "led_count", 0))
    if led_count <= 0:
        led_count = int(getattr(strip.layout, "right", 0) + getattr(strip.layout, "top", 0) +
                        getattr(strip.layout, "left", 0) + getattr(strip.layout, "bottom", 0))
    if led_count <= 0:
        print(f"Strip '{getattr(strip, 'name', 'strip')}' has 0 LEDs configured; nothing to test.")
        return

    targets = getattr(strip, "targets", None)
    if not targets:
        # legacy single-target support
        w = getattr(strip, "wled", None)
        if w:
            targets = [w]
    if not targets:
        print(f"Strip '{getattr(strip, 'name', 'strip')}' has no WLED targets configured.")
        return

    timeout_seconds = int(getattr(strip, "timeout_seconds", 2))

    print("Running TEST CHASE (Ctrl+C to stop)")
    while running:
        colors = np.zeros((led_count, 3), dtype=np.uint8)
        colors[idx] = [255, 255, 255]  # bright white pixel
        send_wled(sock, colors, targets, timeout_seconds)

        idx = (idx + 1) % led_count
        time.sleep(delay)


def run_test_sides(sock, strip):
    """
    Cycles one side at a time so it's obvious what you're looking at.
    Order: RIGHT -> TOP -> LEFT -> BOTTOM (skips disabled sides).
    """
    targets = getattr(strip, "targets", None)
    if not targets:
        w = getattr(strip, "wled", None)
        if w:
            targets = [w]
    if not targets:
        print(f"Strip '{getattr(strip, 'name', 'strip')}' has no WLED targets configured.")
        return

    r = int(getattr(strip.layout, "right", 0))
    t = int(getattr(strip.layout, "top", 0))
    l = int(getattr(strip.layout, "left", 0))
    b = int(getattr(strip.layout, "bottom", 0))
    led_count = int(getattr(strip.layout, "led_count", r + t + l + b))
    if led_count <= 0:
        print(f"Strip '{getattr(strip, 'name', 'strip')}' has 0 LEDs configured; nothing to test.")
        return

    # Build enabled side ranges dynamically
    sides = []
    off = 0
    if r > 0:
        sides.append(("RIGHT (should be right side)", off, off + r, [255, 0, 0]))  # red
        off += r
    if t > 0:
        sides.append(("TOP (should be top side)", off, off + t, [0, 255, 0]))  # green
        off += t
    if l > 0:
        sides.append(("LEFT (should be left side)", off, off + l, [0, 0, 255]))  # blue
        off += l
    if b > 0:
        sides.append(("BOTTOM (should be bottom)", off, off + b, [255, 255, 0]))  # yellow
        off += b

    timeout_seconds = int(getattr(strip, "timeout_seconds", 2))
    on_time = 1.0
    off_time = 0.25

    print("Running TEST SIDES (cycling one side at a time). Ctrl+C to stop.")
    while running:
        for name, start, end, color in sides:
            if not running:
                break
            print(name)
            colors = np.zeros((led_count, 3), dtype=np.uint8)
            if end > start:
                colors[start:end] = color

            # keep realtime asserted, but don't spam
            t_end = time.perf_counter() + on_time
            while running and time.perf_counter() < t_end:
                send_wled(sock, colors, targets, timeout_seconds)
                time.sleep(0.05)

            # brief off gap
            colors[:] = 0
            t_end = time.perf_counter() + off_time
            while running and time.perf_counter() < t_end:
                send_wled(sock, colors, targets, timeout_seconds)
                time.sleep(0.05)


def run_test_sides_all(sock, strip):
    """Colors all enabled sides at once."""
    targets = getattr(strip, "targets", None)
    if not targets:
        w = getattr(strip, "wled", None)
        if w:
            targets = [w]
    if not targets:
        print(f"Strip '{getattr(strip, 'name', 'strip')}' has no WLED targets configured.")
        return

    r = int(getattr(strip.layout, "right", 0))
    t = int(getattr(strip.layout, "top", 0))
    l = int(getattr(strip.layout, "left", 0))
    b = int(getattr(strip.layout, "bottom", 0))
    led_count = int(getattr(strip.layout, "led_count", r + t + l + b))
    if led_count <= 0:
        print(f"Strip '{getattr(strip, 'name', 'strip')}' has 0 LEDs configured; nothing to test.")
        return

    print("Running TEST SIDES ALL (all sides at once). Ctrl+C to stop.")
    colors = np.zeros((led_count, 3), dtype=np.uint8)
    i = 0
    if r > 0:
        colors[i:i+r] = [255, 0, 0]; i += r        # RIGHT red
    if t > 0:
        colors[i:i+t] = [0, 255, 0]; i += t        # TOP green
    if l > 0:
        colors[i:i+l] = [0, 0, 255]; i += l        # LEFT blue
    if b > 0:
        colors[i:i+b] = [255, 255, 0]; i += b      # BOTTOM yellow

    timeout_seconds = int(getattr(strip, "timeout_seconds", 2))
    while running:
        send_wled(sock, colors, targets, timeout_seconds)
        time.sleep(0.05)

# ----------------------------
# WLED DRGB packet:
# byte0 = 2 (DRGB)
# byte1 = timeout seconds (1-2 recommended)
# bytes 2.. = RGBRGBRGB... for every LED in order
# ----------------------------

PROTO_DRGB = 2

def send_wled(sock, colors: np.ndarray, targets, timeout_seconds: int):
    payload = bytearray(2 + colors.shape[0] * 3)
    payload[0] = PROTO_DRGB
    payload[1] = max(1, min(255, int(timeout_seconds)))
    payload[2:] = colors.reshape(-1).tobytes()
    for t in targets:
        sock.sendto(payload, (t.ip, t.port))


def main():
    # Parse GlowBridge-only flags, then let settings.py parse the rest.
    test_parser = argparse.ArgumentParser(add_help=False)
    test_parser.add_argument("--test-chase", action="store_true", help="Run LED chase test (first/selected strip)")
    test_parser.add_argument("--test-sides", action="store_true", help="Cycle sides one at a time (first/selected strip)")
    test_parser.add_argument("--test-sides-all", action="store_true", help="Color all sides at once (first/selected strip)")
    test_parser.add_argument("--strip", type=str, default=None, help="Select strip by name (defaults to first)")
    test_args, remaining = test_parser.parse_known_args(sys.argv[1:])

    settings, args = load_settings(remaining, app_name="glowbridge")

    # Lock after loading settings so lock_path can be configured.
    global LOCK_PATH
    LOCK_PATH = getattr(settings, "lock_path", LOCK_PATH)
    acquire_lock_or_die()

    last_dbg = 0.0


    # UDP socket to WLED
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Build strip list (new multi-strip or legacy single-strip config)
    strips = list(getattr(settings, "strips", ()) or [])
    if not strips:
        strips = [SimpleNamespace(
            name="default",
            timeout_seconds=settings.wled.timeout_seconds,
            layout=settings.layout,
            targets=(SimpleNamespace(ip=settings.wled.ip, port=settings.wled.port),),
        )]

    # Choose strip for tests / preview
    strip = strips[0]
    if test_args.strip:
        for st in strips:
            if getattr(st, "name", "") == test_args.strip:
                strip = st
                break

    if test_args.test_chase:
        run_test_chase(sock, strip)
        return 0

    if test_args.test_sides:
        run_test_sides(sock, strip)
        return 0

    if test_args.test_sides_all:
        run_test_sides_all(sock, strip)
        return 0

    # Open capture
    video_dev = settings.video.device
    if not video_dev or str(video_dev).lower() == "auto":
        detected = autodetect_video_device(prefer_usb=True, verify_frames=True)
        if not detected:
            print("ERROR: Could not auto-detect a working capture device.", file=sys.stderr)
            return 1
        video_dev = detected
        print(f"Auto-detected capture device: {video_dev}")

    cap = cv2.VideoCapture(video_dev, cv2.CAP_V4L2)

    if not cap.isOpened():
        print(f"ERROR: Could not open {settings.video.device}", file=sys.stderr)
        return 1

    # Request MJPEG + size/fps
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.video.cap_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.video.cap_h)
    cap.set(cv2.CAP_PROP_FPS, settings.video.cap_fps)

    frame_interval = 1.0 / max(1, settings.effects.out_fps)
    next_t = time.perf_counter()

    print("Streaming to WLED via DRGB realtime...")
    prev_colors: dict[str, np.ndarray] = {}

    while running:
        # pace output
        now = time.perf_counter()
        if now < next_t:
            time.sleep(max(0.0, next_t - now))
        next_t += frame_interval

        ok, frame = cap.read()
        if not ok or frame is None:
            # If capture hiccups, just skip this frame
            continue

        # Downsample for cheap sampling
        frame_small = cv2.resize(frame, (settings.sampling.sample_w, settings.sampling.sample_h), interpolation=cv2.INTER_AREA)
        # Convert to RGB
        rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        
        if settings.sampling.auto_crop_black_bars:
            rgb = auto_crop_content_rgb(
                rgb,
                luma_thresh=settings.sampling.crop_luma_threshold,
                min_percent=settings.sampling.crop_min_percent,
                pad=settings.sampling.crop_pad,
                min_w=settings.sampling.crop_min_w,
                min_h=settings.sampling.crop_min_h,
            )
        # Build DRGB packet

        # Send to all configured strips/targets
        for st in strips:
            lay = st.layout
            colors_raw = build_led_colors_from_frame(
                rgb,
                lay.right, lay.top, lay.left, lay.bottom,
                settings.sampling.edge_margin, settings.sampling.patch_r
            )

            prev = prev_colors.get(st.name)
            if prev is None or prev.shape != colors_raw.shape:
                prev = colors_raw.copy()

            colors_out, prev_f = apply_effects(colors_raw, prev, settings.effects)
            prev_colors[st.name] = prev_f

            # --- Preview (selected strip only) ---
            if args.preview and st is strip:
                if 'preview_grid' not in locals():
                    preview_grid = build_preview_grid(
                        colors_out,
                        lay.right, lay.top, lay.left, lay.bottom
                    )
                    # Hide cursor once
                    sys.stdout.write("\x1b[?25l")
                    sys.stdout.flush()
                    preview_next = 0.0

                nowp = time.perf_counter()
                if nowp >= preview_next:
                    preview_next = nowp + (1.0 / max(1e-3, args.preview_fps))
                    mn = int(colors_out.min())
                    mx = int(colors_out.max())
                    mean = int(colors_out.mean())
                    stats = f"{st.name}: mean={mean} min={mn} max={mx}  (Ctrl+C to quit)"
                    render_preview_ansi(colors_out, preview_grid, stats=stats)
            # --- Send to WLED ---

            send_wled(sock, colors_out, st.targets, st.timeout_seconds)


    cap.release()

    # On exit, let WLED timeout quickly back to normal mode.
    # (We could also send a final packet with timeout=1 and black, but timeout is fine.)
    sys.stdout.write("\x1b[?25h")
    sys.stdout.flush()

    print("Stopped.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())