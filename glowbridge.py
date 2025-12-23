#!/usr/bin/env python3

import cv2
import sys
import time
import fcntl
import signal
import socket
import argparse
import numpy as np

from settings import load_settings

LOCK_PATH = "/tmp/glowbridge.lock"
_lock_fd = None

# ----------------------------
# WLED DRGB packet:
# byte0 = 2 (DRGB)
# byte1 = timeout seconds (1-2 recommended)
# bytes 2.. = RGBRGBRGB... for every LED in order  :contentReference[oaicite:4]{index=4}
# ----------------------------
PROTO_DRGB = 2
TIMEOUT_SECONDS = 2

running = True

def acquire_lock_or_die():
    global _lock_fd
    _lock_fd = open(LOCK_PATH, "w")
    try:
        fcntl.flock(_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Another instance is already running (lock held). Exiting.", file=sys.stderr)
        sys.exit(2)

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

def crop_center_16_9(frame_bgr: np.ndarray) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    target_h = int(w * 9 / 16)
    if target_h <= 0 or target_h > h:
        return frame_bgr  # can't crop sensibly, return as-is
    y0 = (h - target_h) // 2
    return frame_bgr[y0:y0 + target_h, :]


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

    led_count = right + top + left + bottom
    colors = np.zeros((led_count, 3), dtype=np.uint8)
    idx = 0

    # RIGHT (bottom->top)
    x = w - 1 - m
    for i in range(right):
        t = (i + 0.5) / right
        y = int((h - 1 - m) * (1.0 - t) + m * t)
        colors[idx] = sample_patch_rgb(sample_rgb, x, y, r)
        idx += 1

    # TOP (right->left)
    y = m
    for i in range(top):
        t = (i + 0.5) / top
        x = int((w - 1 - m) * (1.0 - t) + m * t)
        colors[idx] = sample_patch_rgb(sample_rgb, x, y, r)
        idx += 1

    # LEFT (top->bottom)
    x = m
    for i in range(left):
        t = (i + 0.5) / left
        y = int(m * (1.0 - t) + (h - 1 - m) * t)
        colors[idx] = sample_patch_rgb(sample_rgb, x, y, r)
        idx += 1

    # BOTTOM (left->right)
    y = h - 1 - m
    for i in range(bottom):
        t = (i + 0.5) / bottom
        x = int(m * (1.0 - t) + (w - 1 - m) * t)
        colors[idx] = sample_patch_rgb(sample_rgb, x, y, r)
        idx += 1

    return colors

def run_test_chase(sock, settings):
    idx = 0
    delay = 0.05  # seconds

    print("Running TEST CHASE (Ctrl+C to stop)")
    while running:
        colors = np.zeros((settings.layout.led_count, 3), dtype=np.uint8)
        colors[idx] = [255, 255, 255]  # bright white pixel

        send_wled(sock, colors, settings)

        idx = (idx + 1) % settings.layout.led_count
        time.sleep(delay)


def run_test_sides(sock, settings):
    """
    Cycles one side at a time so it's obvious what you're looking at.
    RIGHT -> TOP -> LEFT -> BOTTOM, repeating.
    """
    print("Running TEST SIDES (cycling one side at a time). Ctrl+C to stop.")
    sides = [
        ("RIGHT (should be right side)", 0, settings.layout.right, [255, 0, 0]),                        # red
        ("TOP (should be top side)", settings.layout.right, settings.layout.right + settings.layout.top, [0, 255, 0]),                  # green
        ("LEFT (should be left side)", settings.layout.right + settings.layout.top, settings.layout.right + settings.layout.top + settings.layout.left, [0, 0, 255]),   # blue
        ("BOTTOM (should be bottom)", settings.layout.right + settings.layout.top + settings.layout.left, settings.layout.led_count, [255, 255, 0]),    # yellow
    ]

    on_time = 1.0
    off_time = 0.25

    while running:
        for name, start, end, color in sides:
            if not running:
                break
            print(name)
            colors = np.zeros((settings.layout.led_count, 3), dtype=np.uint8)
            colors[start:end] = color

            # keep realtime asserted, but don't spam
            t_end = time.perf_counter() + on_time
            while running and time.perf_counter() < t_end:
                send_wled(sock, colors, settings)
                time.sleep(0.05)

            # brief off gap
            colors[:] = 0
            t_end = time.perf_counter() + off_time
            while running and time.perf_counter() < t_end:
                send_wled(sock, colors, settings)
                time.sleep(0.05)


def run_test_sides_all(sock, settings):
    """Colors all sides at once (original behavior)."""
    print("Running TEST SIDES ALL (all sides at once). Ctrl+C to stop.")
    colors = np.zeros((settings.layout.led_count, 3), dtype=np.uint8)
    i = 0

    colors[i:i+settings.layout.right] = [255, 0, 0]; i += settings.layout.right       # RIGHT red
    colors[i:i+settings.layout.top]   = [0, 255, 0]; i += settings.layout.top         # TOP green
    colors[i:i+settings.layout.left]  = [0, 0, 255]; i += settings.layout.left        # LEFT blue
    colors[i:i+settings.layout.bottom]= [255, 255, 0]; i += settings.layout.bottom    # BOTTOM yellow

    while running:
        send_wled(sock, colors, settings)
        time.sleep(0.05)

def send_wled(sock, colors, settings):
    payload = bytearray(2 + settings.layout.led_count * 3)
    payload[0] = PROTO_DRGB
    payload[1] = TIMEOUT_SECONDS
    payload[2:] = colors.reshape(-1).tobytes()
    sock.sendto(payload, (settings.wled.ip, settings.wled.port))


def main():
    acquire_lock_or_die()
    last_dbg = 0.0
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-chase", action="store_true", help="Run LED chase test")
    parser.add_argument("--test-sides", action="store_true", help="Cycle sides one at a time (best for calibration)")
    parser.add_argument("--test-sides-all", action="store_true", help="Color all sides at once")
    args = parser.parse_args()

    settings, args = load_settings(sys.argv[1:], app_name="glowbridge")

    # UDP socket to WLED
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    if args.test_chase:
        run_test_chase(sock, settings)
        return 0

    if args.test_sides:
        run_test_sides(sock, settings)
        return 0

    if args.test_sides_all:
        run_test_sides_all(sock, settings)
        return 0

    # Open capture
    cap = cv2.VideoCapture(settings.video.device, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"ERROR: Could not open {settings.video.device}", file=sys.stderr)
        return 1

    # Request MJPEG + size/fps
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.video.cap_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.video.cap_h)
    cap.set(cv2.CAP_PROP_FPS, settings.video.cap_fps)

    # Smoothing buffer
    prev = None

    frame_interval = 1.0 / max(1, settings.effects.out_fps)
    next_t = time.perf_counter()

    print(f"Streaming {settings.layout.led_count} LEDs to WLED {settings.wled.ip}:{settings.wled.port} via DRGB realtime...")

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

        if settings.video.crop_to_16_9:
            frame = crop_center_16_9(frame)

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

        colors = build_led_colors_from_frame(
            rgb,
            settings.layout.right, settings.layout.top, settings.layout.left, settings.layout.bottom,
            settings.sampling.edge_margin, settings.sampling.patch_r
        )

        # 1) Apply black threshold to reduce noise on blacks
        colors = apply_black_threshold(
            colors,
            black_luma=settings.effects.black_luma_threshold,
            black_chroma=settings.effects.black_chroma_threshold
        )

        # 2) Make colors less washed
        colors = boost_saturation_rgb(colors, settings.effects.sat_boost, settings.effects.val_boost)

        # 3) Scene-cut aware clamp + EMA smoothing
        colors_f = colors.astype(np.float32)

        if prev is None:
            prev = colors_f
        else:
            # detect cut by luma jump
            l_prev = avg_luma(prev)
            l_new  = avg_luma(colors_f)
            is_cut = abs(l_new - l_prev) >= settings.effects.scene_cut_luma_jump

            step = settings.effects.max_step_per_frame * (settings.effects.scene_cut_boost_mult if is_cut else 1.0)
            colors_f = limit_step(prev, colors_f, int(step))

            if settings.effects.smooth_alpha > 0.0:
                colors_f = (settings.effects.smooth_alpha * prev) + ((1.0 - settings.effects.smooth_alpha) * colors_f)

            prev = colors_f

        colors_out = np.clip(colors_f, 0, 255).astype(np.uint8)

        if colors_out.shape != (settings.layout.right + settings.layout.top + settings.layout.left + settings.layout.bottom, 3):
            print("BAD SHAPE:", colors_out.shape)

        # Build DRGB packet
        payload = bytearray(2 + settings.layout.led_count * 3)
        payload[0] = PROTO_DRGB
        payload[1] = settings.wled.timeout_seconds

        # Flatten RGB
        payload[2:] = colors_out.reshape(-1).tobytes()

        sock.sendto(payload, (settings.wled.ip, settings.wled.port))

    cap.release()

    # On exit, let WLED timeout quickly back to normal mode.
    # (We could also send a final packet with timeout=1 and black, but timeout is fine.)
    print("Stopped.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
