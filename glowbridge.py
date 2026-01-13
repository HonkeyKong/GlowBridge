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

# Optional web UI dependencies (FastAPI + Uvicorn)
try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import HTMLResponse, JSONResponse, Response
    import uvicorn
except Exception:  # pragma: no cover
    FastAPI = None  # type: ignore
    HTTPException = None  # type: ignore
    Request = None  # type: ignore
    HTMLResponse = None  # type: ignore
    JSONResponse = None  # type: ignore
    Response = None  # type: ignore
    uvicorn = None  # type: ignore
from pathlib import Path
from types import SimpleNamespace
from dataclasses import asdict, replace
import threading
from typing import Any, Dict, Optional

from settings import load_settings, Settings, SamplingCfg, EffectsCfg, LayoutCfg, StripCfg, WledCfg, WledTargetCfg

_lock_fd = None
running = True

# -------- Web UI runtime state --------
_runtime_lock = threading.Lock()
_runtime_settings: Optional[Settings] = None
_last_rgb_preview: Optional[np.ndarray] = None  # RGB uint8
_last_frame_jpeg: Optional[bytes] = None
_last_led_png: Dict[str, bytes] = {}
_last_led_colors: Dict[str, np.ndarray] = {}
_last_stats: Dict[str, Any] = {}
_web_token: Optional[str] = None

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



# -----------------------------
# Web UI helpers
# -----------------------------

def _require_auth(request: "Request"):
    if _web_token:
        auth = request.headers.get("authorization") or ""
        if not auth.lower().startswith("bearer "):
            raise HTTPException(status_code=401, detail="Missing bearer token")
        tok = auth.split(" ", 1)[1].strip()
        if tok != _web_token:
            raise HTTPException(status_code=403, detail="Invalid token")

def _settings_to_dict(s: Settings) -> dict:
    d = asdict(s)
    # asdict converts tuples to lists; that's fine for JSON.
    return d

def _apply_patch_settings(cur: Settings, patch: dict) -> Settings:
    """
    Applies a JSON patch-like dict onto Settings using dataclasses.replace.
    Supported keys: sampling.*, effects.* (and a few legacy layout/wled keys).
    """
    s = cur
    if not isinstance(patch, dict):
        return s

    if "sampling" in patch and isinstance(patch["sampling"], dict):
        samp = s.sampling
        sd = patch["sampling"]
        # only update known fields if present
        for k in list(sd.keys()):
            if not hasattr(samp, k):
                sd.pop(k, None)
        if sd:
            samp = replace(samp, **sd)
            s = replace(s, sampling=samp)

    if "effects" in patch and isinstance(patch["effects"], dict):
        eff = s.effects
        ed = patch["effects"]
        for k in list(ed.keys()):
            if not hasattr(eff, k):
                ed.pop(k, None)
        if ed:
            eff = replace(eff, **ed)
            s = replace(s, effects=eff)

    # Legacy single-strip fields
    if "layout" in patch and isinstance(patch["layout"], dict):
        lay = s.layout
        ld = patch["layout"]
        for k in list(ld.keys()):
            if not hasattr(lay, k):
                ld.pop(k, None)
        if ld:
            lay = replace(lay, **ld)
            s = replace(s, layout=lay)

    if "wled" in patch and isinstance(patch["wled"], dict):
        w = s.wled
        wd = patch["wled"]
        for k in list(wd.keys()):
            if not hasattr(w, k):
                wd.pop(k, None)
        if wd:
            w = replace(w, **wd)
            s = replace(s, wled=w)

    return s

def _render_led_preview_png(colors: np.ndarray, grid_idx: np.ndarray, scale: int = 16) -> bytes:
    """
    Render a perimeter grid preview to a PNG. Uses OpenCV for speed and avoids extra deps.
    """
    H, W = grid_idx.shape
    scale = max(4, int(scale))
    img = np.zeros((H * scale, W * scale, 3), dtype=np.uint8)

    for y in range(H):
        for x in range(W):
            li = int(grid_idx[y, x])
            if li < 0:
                continue
            r, g, b = (int(colors[li, 0]), int(colors[li, 1]), int(colors[li, 2]))
            # OpenCV uses BGR
            y0, y1 = y * scale, (y + 1) * scale
            x0, x1 = x * scale, (x + 1) * scale
            img[y0:y1, x0:x1] = (b, g, r)

    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return b""
    return bytes(buf)

def _build_app() -> "FastAPI":
    app = FastAPI(title="GlowBridge Web UI")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        # Single-file UI: lightweight controls + previews.
        html = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>GlowBridge</title>
  <style>
    body{font-family:system-ui,Segoe UI,Roboto,Arial,sans-serif;margin:0;background:#0b0f14;color:#e6edf3}
    header{padding:12px 16px;background:#111826;border-bottom:1px solid #223}
    h1{margin:0;font-size:18px}
    .wrap{display:grid;grid-template-columns: 380px 1fr;gap:16px;padding:16px}
    .card{background:#111826;border:1px solid #223;border-radius:10px;padding:12px}
    label{display:block;font-size:12px;color:#9aa7b2;margin-top:10px}
    input[type=range]{width:100%}
    .row{display:flex;gap:10px;align-items:center}
    .row span{min-width:70px;color:#9aa7b2;font-size:12px}
    img{max-width:100%;border-radius:10px;border:1px solid #223;background:#000}
    select,button,input{background:#0b0f14;color:#e6edf3;border:1px solid #223;border-radius:8px;padding:6px}
    button{cursor:pointer}
    small{color:#9aa7b2}
  </style>
</head>
<body>
<header><h1>GlowBridge Web UI</h1></header>
<div class="wrap">
  <div class="card">
    <div class="row">
      <span>Strip</span>
      <select id="stripSel"></select>
      <button id="refreshBtn">Refresh</button>
    </div>

    <label>Smoothing (smooth_alpha) <span id="smoothVal"></span></label>
    <input id="smooth" type="range" min="0" max="0.98" step="0.01"/>

    <label>Max step/frame (max_step_per_frame) <span id="stepVal"></span></label>
    <input id="step" type="range" min="1" max="128" step="1"/>

    <label>Saturation boost (sat_boost) <span id="satVal"></span></label>
    <input id="sat" type="range" min="1.0" max="2.5" step="0.01"/>

    <label>Value boost (val_boost) <span id="valVal"></span></label>
    <input id="val" type="range" min="1.0" max="2.0" step="0.01"/>

    <label>Edge margin (edge_margin) <span id="mVal"></span></label>
    <input id="margin" type="range" min="0" max="20" step="1"/>

    <label>Patch radius (patch_r) <span id="pVal"></span></label>
    <input id="patch" type="range" min="0" max="8" step="1"/>

    <div style="margin-top:12px">
      <button id="applyBtn">Apply</button>
      <small id="status"></small>
    </div>
  </div>

  <div class="card">
    <div class="row" style="justify-content:space-between">
      <div><b>Preview</b> <small id="stats"></small></div>
    </div>
    <div style="display:grid;grid-template-columns: 1fr 1fr; gap:12px; margin-top:10px">
      <div>
        <div style="margin-bottom:6px"><small>Sampled/cropped frame</small></div>
        <img id="frameImg" src="/api/preview/frame.jpg" />
      </div>
      <div>
        <div style="margin-bottom:6px"><small>LED perimeter</small></div>
        <img id="ledImg" src="/api/preview/led.png" />
      </div>
    </div>
  </div>
</div>

<script>
const $ = (id)=>document.getElementById(id);
let state = null;

function setText(id, v){ $(id).textContent = v; }

function loadState(){
  return fetch('/api/state').then(r=>r.json()).then(s=>{
    state = s;
    const strips = s.strips || [];
    const sel = $('stripSel');
    sel.innerHTML = '';
    for(const st of strips){
      const opt = document.createElement('option');
      opt.value = st.name;
      opt.textContent = st.name;
      sel.appendChild(opt);
    }
    if(strips.length){
      sel.value = strips[0].name;
    }
    // populate controls
    $('smooth').value = s.settings.effects.smooth_alpha;
    $('step').value = s.settings.effects.max_step_per_frame;
    $('sat').value = s.settings.effects.sat_boost;
    $('val').value = s.settings.effects.val_boost;
    $('margin').value = s.settings.sampling.edge_margin;
    $('patch').value = s.settings.sampling.patch_r;
    updateLabels();
  });
}

function updateLabels(){
  setText('smoothVal', Number($('smooth').value).toFixed(2));
  setText('stepVal', $('step').value);
  setText('satVal', Number($('sat').value).toFixed(2));
  setText('valVal', Number($('val').value).toFixed(2));
  setText('mVal', $('margin').value);
  setText('pVal', $('patch').value);
}

function refreshImages(){
  const strip = $('stripSel').value || '';
  $('frameImg').src = '/api/preview/frame.jpg?ts=' + Date.now();
  $('ledImg').src = '/api/preview/led.png?strip=' + encodeURIComponent(strip) + '&ts=' + Date.now();
}

async function apply(){
  const payload = {
    effects: {
      smooth_alpha: Number($('smooth').value),
      max_step_per_frame: Number($('step').value),
      sat_boost: Number($('sat').value),
      val_boost: Number($('val').value),
    },
    sampling: {
      edge_margin: Number($('margin').value),
      patch_r: Number($('patch').value),
    }
  };
  $('status').textContent = ' applying...';
  const r = await fetch('/api/settings', {
    method: 'PATCH',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify(payload)
  });
  if(!r.ok){
    $('status').textContent = ' error';
    return;
  }
  $('status').textContent = ' applied';
  setTimeout(()=> $('status').textContent='', 1200);
  await loadState();
}

$('refreshBtn').onclick = ()=>{ loadState().then(refreshImages); };
$('applyBtn').onclick = ()=>apply();
$('stripSel').onchange = ()=>refreshImages();

['smooth','step','sat','val','margin','patch'].forEach(id=>{
  $(id).addEventListener('input', updateLabels);
});

loadState().then(()=>{
  refreshImages();
  setInterval(()=>{
    refreshImages();
    fetch('/api/state').then(r=>r.json()).then(s=>{
      setText('stats', s.stats_line || '');
    }).catch(()=>{});
  }, 500);
});
</script>
</body>
</html>
"""
        return HTMLResponse(html)

    @app.get("/api/state")
    async def state(request: Request):
        _require_auth(request)
        with _runtime_lock:
            s = _runtime_settings
            stats = dict(_last_stats)
        if s is None:
            raise HTTPException(status_code=503, detail="Engine not running")
        strips = []
        for st in (s.strips or ()):
            strips.append({"name": st.name, "led_count": st.led_count})
        if not strips:
            # legacy "default" strip
            strips = [{"name": "default", "led_count": s.layout.led_count}]
        # Compose a short human line
        stats_line = ""
        if stats:
            stats_line = " | ".join([f"{k}={v}" for k,v in stats.items() if isinstance(v,(int,float,str))])
        return JSONResponse({
            "settings": _settings_to_dict(s),
            "strips": strips,
            "stats": stats,
            "stats_line": stats_line,
        })

    @app.patch("/api/settings")
    async def patch_settings(request: Request):
        _require_auth(request)
        patch = await request.json()
        with _runtime_lock:
            cur = _runtime_settings
            if cur is None:
                raise HTTPException(status_code=503, detail="Engine not running")
            new_s = _apply_patch_settings(cur, patch)
            # basic sanity
            if not (0.0 <= new_s.effects.smooth_alpha < 1.0):
                raise HTTPException(status_code=400, detail="smooth_alpha must be in [0.0,1.0)")
            if new_s.effects.out_fps <= 0:
                raise HTTPException(status_code=400, detail="out_fps must be > 0")
            if new_s.sampling.sample_w <= 0 or new_s.sampling.sample_h <= 0:
                raise HTTPException(status_code=400, detail="sample_w/sample_h must be > 0")
            _runtime_settings = new_s
        return JSONResponse({"ok": True})

    @app.get("/api/preview/frame.jpg")
    async def preview_frame(request: Request):
        _require_auth(request)
        with _runtime_lock:
            jpg = _last_frame_jpeg
        if not jpg:
            raise HTTPException(status_code=503, detail="No preview yet")
        return Response(content=jpg, media_type="image/jpeg")

    @app.get("/api/preview/led.png")
    async def preview_led(request: Request, strip: str = "default", scale: int = 18):
        _require_auth(request)
        with _runtime_lock:
            png = _last_led_png.get(strip)
            if png is None and strip != "default":
                png = _last_led_png.get("default")
        if not png:
            raise HTTPException(status_code=503, detail="No LED preview yet")
        return Response(content=png, media_type="image/png")

    return app

def start_webui(bind: str, port: int):
    if FastAPI is None or uvicorn is None:
        raise RuntimeError("FastAPI/Uvicorn not installed. Install requirements_webui.txt to use --web.")
    app = _build_app()
    config = uvicorn.Config(app, host=bind, port=int(port), log_level="warning")
    server = uvicorn.Server(config)
    server.run()


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

def sample_patch_rgb(img_rgb: np.ndarray, x: int, y: int, r: int) -> np.ndarray:
    h, w = img_rgb.shape[:2]

    x = max(0, min(w - 1, int(x)))
    y = max(0, min(h - 1, int(y)))

    if r <= 0:
        return img_rgb[y, x]

    x0 = max(0, x - r)
    x1 = min(w, x + r + 1)
    y0 = max(0, y - r)
    y1 = min(h, y + r + 1)

    # OpenCV mean is implemented in C and is much faster than per-LED NumPy work
    roi = img_rgb[y0:y1, x0:x1]
    m = cv2.mean(roi)  # returns (R, G, B, A?) as floats
    return np.array([int(m[0]), int(m[1]), int(m[2])], dtype=np.uint8)

def build_led_colors_from_frame(sample_rgb: np.ndarray,
                                right: int, top: int, left: int, bottom: int,
                                edge_margin: int, patch_r: int) -> np.ndarray:
    """
    Fast vectorized sampler:
    - Assumes sample_rgb is already blurred if you want patch averaging.
    - Ignores patch_r (kept in signature for compatibility).
    """
    h, w = sample_rgb.shape[:2]
    m = int(max(0, min(edge_margin, w - 1, h - 1)))

    rN = max(0, int(right))
    tN = max(0, int(top))
    lN = max(0, int(left))
    bN = max(0, int(bottom))

    led_count = rN + tN + lN + bN
    colors = np.zeros((led_count, 3), dtype=np.uint8)
    if led_count == 0:
        return colors

    idx = 0

    # RIGHT column: bottom->top
    if rN:
        x = (w - 1 - m)
        t = (np.arange(rN, dtype=np.float32) + 0.5) / float(rN)
        y = ((h - 1 - m) * (1.0 - t) + m * t).astype(np.int32)
        y = np.clip(y, 0, h - 1)
        colors[idx:idx+rN] = sample_rgb[y, x]
        idx += rN

    # TOP row: right->left
    if tN:
        y = m
        t = (np.arange(tN, dtype=np.float32) + 0.5) / float(tN)
        x = ((w - 1 - m) * (1.0 - t) + m * t).astype(np.int32)
        x = np.clip(x, 0, w - 1)
        colors[idx:idx+tN] = sample_rgb[y, x]
        idx += tN

    # LEFT column: top->bottom
    if lN:
        x = m
        t = (np.arange(lN, dtype=np.float32) + 0.5) / float(lN)
        y = (m * (1.0 - t) + (h - 1 - m) * t).astype(np.int32)
        y = np.clip(y, 0, h - 1)
        colors[idx:idx+lN] = sample_rgb[y, x]
        idx += lN

    # BOTTOM row: left->right
    if bN:
        y = (h - 1 - m)
        t = (np.arange(bN, dtype=np.float32) + 0.5) / float(bN)
        x = (m * (1.0 - t) + (w - 1 - m) * t).astype(np.int32)
        x = np.clip(x, 0, w - 1)
        colors[idx:idx+bN] = sample_rgb[y, x]
        idx += bN

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
    # Web UI
    test_parser.add_argument("--web", action="store_true", help="Enable FastAPI web UI")
    test_parser.add_argument("--web-bind", type=str, default="0.0.0.0", help="Bind address for web UI")
    test_parser.add_argument("--web-port", type=int, default=8787, help="Port for web UI")
    test_parser.add_argument("--web-token", type=str, default=None, help="Optional bearer token for web UI")
    test_args, remaining = test_parser.parse_known_args(sys.argv[1:])

    settings, args = load_settings(remaining, app_name="glowbridge")

    # Initialize runtime settings for live web tweaks
    global _runtime_settings, _web_token
    with _runtime_lock:
        _runtime_settings = settings
    _web_token = test_args.web_token

    if test_args.web:
        if FastAPI is None or uvicorn is None:
            print('Web UI requested, but optional dependencies are not installed.')
            print('Install them with:  pip install -r requirements_webui.txt')
            print('or, on Raspbian:   sudo apt install python3-fastapi python3-uvicorn')
            print('Or:                 pip install fastapi uvicorn[standard]')
            return 2
        t = threading.Thread(target=start_webui, args=(test_args.web_bind, test_args.web_port), daemon=True)
        t.start()
        print(f"Web UI: http://{test_args.web_bind}:{test_args.web_port} (token={'on' if _web_token else 'off'})")


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

    with _runtime_lock:
        s0 = _runtime_settings or settings
    frame_interval = 1.0 / max(1, int(s0.effects.out_fps))
    last_out_fps = int(s0.effects.out_fps)
    next_t = time.perf_counter()

    # Throttle expensive web preview encodes so they don't steal render time.
    preview_interval = 1.0 / 5.0  # 5 Hz web UI updates
    next_preview_t = 0.0

    print("Streaming to WLED via DRGB realtime...")
    prev_colors: dict[str, np.ndarray] = {}

    while running:
        # Snapshot runtime settings (for live web tweaks)
        with _runtime_lock:
            s = _runtime_settings or settings

        # If out_fps changed, adjust pacing immediately
        cur_fps = int(s.effects.out_fps)
        if cur_fps != last_out_fps and cur_fps > 0:
            last_out_fps = cur_fps
            frame_interval = 1.0 / max(1, cur_fps)
            next_t = time.perf_counter()

        ok, frame = cap.read()
        if not ok or frame is None:
            # If capture hiccups, just skip this frame
            continue

        # Gate processing/sending rate to out_fps (avoid "double pacing")
        now = time.perf_counter()
        if now < next_t:
            continue
        next_t = now + frame_interval

        # Downsample for cheap sampling
        frame_small = cv2.resize(frame, (s.sampling.sample_w, s.sampling.sample_h), interpolation=cv2.INTER_AREA)
        # Convert to RGB
        rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        
        if s.sampling.auto_crop_black_bars:
            rgb = auto_crop_content_rgb(
                rgb,
                luma_thresh=s.sampling.crop_luma_threshold,
                min_percent=s.sampling.crop_min_percent,
                pad=s.sampling.crop_pad,
                min_w=s.sampling.crop_min_w,
                min_h=s.sampling.crop_min_h,
            )

        # If patch_r > 0, blur once and do single-pixel sampling from the blurred image.
        # This replaces per-LED patch averaging with one fast C blur per frame.
        if s.sampling.patch_r > 0:
            k = (2 * int(s.sampling.patch_r) + 1)
            rgb_sample = cv2.blur(rgb, (k, k))
        else:
            rgb_sample = rgb

        # rgb = cv2.blur(rgb, (5, 5))

        do_preview = (FastAPI is not None) and (now >= next_preview_t)
        if do_preview:
            next_preview_t = now + preview_interval

        # Update web preview of sampled/cropped frame
        # if FastAPI is not None:
        if do_preview:
            try:
                ok_j, buf_j = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok_j:
                    with _runtime_lock:
                        global _last_rgb_preview, _last_frame_jpeg
                        _last_rgb_preview = rgb
                        _last_frame_jpeg = bytes(buf_j)
            except Exception:
                pass

        # Build DRGB packet

        # Send to all configured strips/targets
        for st in strips:
            lay = st.layout
            colors_raw = build_led_colors_from_frame(
                rgb_sample,
                lay.right, lay.top, lay.left, lay.bottom,
                s.sampling.edge_margin, s.sampling.patch_r
            )

            prev = prev_colors.get(st.name)
            if prev is None or prev.shape != colors_raw.shape:
                prev = colors_raw.copy()

            colors_out, prev_f = apply_effects(colors_raw, prev, s.effects)
            prev_colors[st.name] = prev_f

            # Update web preview of LED colors (per strip)
            # if FastAPI is not None:
            if do_preview:
                try:
                    grid = build_preview_grid(colors_out, lay.right, lay.top, lay.left, lay.bottom)
                    png = _render_led_preview_png(colors_out, grid, scale=18)
                    with _runtime_lock:
                        _last_led_colors[st.name] = colors_out.copy()
                        _last_led_png[st.name] = png
                        _last_stats.update({
                            'fps': last_out_fps,
                            'frame_w': int(rgb.shape[1]),
                            'frame_h': int(rgb.shape[0]),
                        })
                except Exception:
                    pass

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