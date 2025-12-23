#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import argparse
import os
import sys

# Python 3.11+:
try:
    import tomllib  # type: ignore
except Exception:
    tomllib = None  # noqa


def _as_int(v: str) -> int:
    return int(v.strip())


def _as_float(v: str) -> float:
    return float(v.strip())


def _as_bool(v: str) -> bool:
    s = v.strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    raise ValueError(f"Not a boolean: {v!r}")


def _expand_path(p: str | None) -> Path | None:
    if not p:
        return None
    return Path(os.path.expanduser(p)).resolve()


@dataclass(frozen=True)
class WledCfg:
    ip: str = "192.168.1.150"
    port: int = 21324
    timeout_seconds: int = 2


@dataclass(frozen=True)
class VideoCfg:
    device: str = "/dev/video0"
    cap_w: int = 640
    cap_h: int = 480
    cap_fps: int = 30
    crop_to_16_9: bool = True


@dataclass(frozen=True)
class SamplingCfg:
    sample_w: int = 160
    sample_h: int = 90
    edge_margin: int = 3
    patch_r: int = 2
    auto_crop_black_bars: bool = True
    crop_luma_threshold: int = 16
    crop_min_percent: float = 0.08
    crop_pad: int = 2
    crop_min_w: int = 80
    crop_min_h: int = 45


@dataclass(frozen=True)
class LayoutCfg:
    # Counter-clockwise, start at bottom-right by default.
    right: int = 17
    top: int = 26
    left: int = 15
    bottom: int = 11
    start_corner: str = "bottom_right"  # reserved for future use
    direction: str = "ccw"              # reserved for future use

    @property
    def led_count(self) -> int:
        return self.right + self.top + self.left + self.bottom


@dataclass(frozen=True)
class EffectsCfg:
    out_fps: int = 30
    smooth_alpha: float = 0.88
    max_step_per_frame: int = 48

    black_luma_threshold: int = 12
    black_chroma_threshold: int = 10

    scene_cut_luma_jump: int = 35
    scene_cut_boost_mult: float = 2.0

    sat_boost: float = 1.35
    val_boost: float = 1.05


@dataclass(frozen=True)
class Settings:
    wled: WledCfg = WledCfg()
    video: VideoCfg = VideoCfg()
    sampling: SamplingCfg = SamplingCfg()
    layout: LayoutCfg = LayoutCfg()
    effects: EffectsCfg = EffectsCfg()

    # Optional: lock path etc. live here too if you want
    lock_path: str = "/tmp/glowbridge.lock"

    def summary(self) -> str:
        return (
            "Settings:\n"
            f"  WLED: {self.wled.ip}:{self.wled.port} timeout={self.wled.timeout_seconds}s\n"
            f"  Video: dev={self.video.device} {self.video.cap_w}x{self.video.cap_h}@{self.video.cap_fps} crop16:9={self.video.crop_to_16_9}\n"
            f"  Layout: R={self.layout.right} T={self.layout.top} L={self.layout.left} B={self.layout.bottom} total={self.layout.led_count}\n"
            f"  Sampling: {self.sampling.sample_w}x{self.sampling.sample_h} margin={self.sampling.edge_margin} patch_r={self.sampling.patch_r}\n"
            f"  Effects: out_fps={self.effects.out_fps} smooth={self.effects.smooth_alpha} step={self.effects.max_step_per_frame} "
            f"black(luma={self.effects.black_luma_threshold}, chroma={self.effects.black_chroma_threshold}) "
            f"cut(jump={self.effects.scene_cut_luma_jump}, mult={self.effects.scene_cut_boost_mult}) "
            f"boost(sat={self.effects.sat_boost}, val={self.effects.val_boost})\n"
            f"  Lock: {self.lock_path}\n"
        )


# -----------------------------
# Config loading
# -----------------------------

def default_config_paths(app_name: str = "glowbridge") -> list[Path]:
    return [
        Path.cwd() / "config.toml",
        Path.home() / ".config" / app_name / "config.toml",
        Path("/etc") / app_name / "config.toml",
    ]


def load_toml(path: Path) -> dict:
    if tomllib is None:
        raise RuntimeError(
            "tomllib not available. Use Python 3.11+ or `pip install tomli` and import tomli as tomllib."
        )
    data = path.read_bytes()
    return tomllib.loads(data.decode("utf-8"))


def merge_from_dict(s: Settings, d: dict) -> Settings:
    # Nested helpers
    def get(section: str, key: str, cast=None):
        sec = d.get(section, {})
        if not isinstance(sec, dict):
            return None
        if key not in sec:
            return None
        val = sec[key]
        return cast(val) if cast else val

    # WLED
    w = s.wled
    w = replace(w,
        ip=get("wled", "ip") or w.ip,
        port=get("wled", "port", int) or w.port,
        timeout_seconds=get("wled", "timeout_seconds", int) or w.timeout_seconds,
    )

    # Video
    v = s.video
    v = replace(v,
        device=get("video", "device") or v.device,
        cap_w=get("video", "cap_w", int) or v.cap_w,
        cap_h=get("video", "cap_h", int) or v.cap_h,
        cap_fps=get("video", "cap_fps", int) or v.cap_fps,
        crop_to_16_9=get("video", "crop_to_16_9", bool) if get("video", "crop_to_16_9") is not None else v.crop_to_16_9,
    )

    # Sampling
    samp = s.sampling
    samp = replace(samp,
        sample_w=get("sampling", "sample_w", int) or samp.sample_w,
        sample_h=get("sampling", "sample_h", int) or samp.sample_h,
        edge_margin=get("sampling", "edge_margin", int) or samp.edge_margin,
        patch_r=get("sampling", "patch_r", int) or samp.patch_r,
    )

    # Layout
    lay = s.layout
    lay = replace(lay,
        right=get("layout", "right", int) or lay.right,
        top=get("layout", "top", int) or lay.top,
        left=get("layout", "left", int) or lay.left,
        bottom=get("layout", "bottom", int) or lay.bottom,
        start_corner=get("layout", "start_corner") or lay.start_corner,
        direction=get("layout", "direction") or lay.direction,
    )

    # Effects
    e = s.effects
    def _flt(section, key, default):
        v = get(section, key)
        return float(v) if v is not None else default

    e = replace(e,
        out_fps=get("effects", "out_fps", int) or e.out_fps,
        smooth_alpha=_flt("effects", "smooth_alpha", e.smooth_alpha),
        max_step_per_frame=get("effects", "max_step_per_frame", int) or e.max_step_per_frame,
        black_luma_threshold=get("effects", "black_luma_threshold", int) or e.black_luma_threshold,
        black_chroma_threshold=get("effects", "black_chroma_threshold", int) or e.black_chroma_threshold,
        scene_cut_luma_jump=get("effects", "scene_cut_luma_jump", int) or e.scene_cut_luma_jump,
        scene_cut_boost_mult=_flt("effects", "scene_cut_boost_mult", e.scene_cut_boost_mult),
        sat_boost=_flt("effects", "sat_boost", e.sat_boost),
        val_boost=_flt("effects", "val_boost", e.val_boost),
    )

    lock_path = d.get("lock_path", s.lock_path)

    return Settings(wled=w, video=v, sampling=samp, layout=lay, effects=e, lock_path=lock_path)


# -----------------------------
# Env + CLI
# -----------------------------

def apply_env(s: Settings) -> Settings:
    """
    Env vars are optional. Prefix: GLOWBRIDGE_
      GLOWBRIDGE_WLED_IP
      GLOWBRIDGE_WLED_PORT
      GLOWBRIDGE_WLED_TIMEOUT
      GLOWBRIDGE_VIDEO_DEV
      GLOWBRIDGE_OUT_FPS
      ...etc
    """
    env = os.environ

    w = s.wled
    v = s.video
    lay = s.layout
    e = s.effects
    samp = s.sampling

    def E(name: str) -> str | None:
        return env.get(f"GLOWBRIDGE_{name}")

    # WLED
    if E("WLED_IP"):
        w = replace(w, ip=E("WLED_IP") or w.ip)
    if E("WLED_PORT"):
        w = replace(w, port=_as_int(E("WLED_PORT")))
    if E("WLED_TIMEOUT"):
        w = replace(w, timeout_seconds=_as_int(E("WLED_TIMEOUT")))

    # Video
    if E("VIDEO_DEV"):
        v = replace(v, device=E("VIDEO_DEV") or v.device)

    # Layout
    if E("RIGHT"):
        lay = replace(lay, right=_as_int(E("RIGHT")))
    if E("TOP"):
        lay = replace(lay, top=_as_int(E("TOP")))
    if E("LEFT"):
        lay = replace(lay, left=_as_int(E("LEFT")))
    if E("BOTTOM"):
        lay = replace(lay, bottom=_as_int(E("BOTTOM")))

    # Sampling
    if E("SAMPLE_W"):
        samp = replace(samp, sample_w=_as_int(E("SAMPLE_W")))
    if E("SAMPLE_H"):
        samp = replace(samp, sample_h=_as_int(E("SAMPLE_H")))
    if E("EDGE_MARGIN"):
        samp = replace(samp, edge_margin=_as_int(E("EDGE_MARGIN")))
    if E("PATCH_R"):
        samp = replace(samp, patch_r=_as_int(E("PATCH_R")))

    # Effects
    if E("OUT_FPS"):
        e = replace(e, out_fps=_as_int(E("OUT_FPS")))
    if E("SMOOTH_ALPHA"):
        e = replace(e, smooth_alpha=_as_float(E("SMOOTH_ALPHA")))
    if E("MAX_STEP"):
        e = replace(e, max_step_per_frame=_as_int(E("MAX_STEP")))
    if E("SAT_BOOST"):
        e = replace(e, sat_boost=_as_float(E("SAT_BOOST")))
    if E("VAL_BOOST"):
        e = replace(e, val_boost=_as_float(E("VAL_BOOST")))
    if E("BLACK_LUMA"):
        e = replace(e, black_luma_threshold=_as_int(E("BLACK_LUMA")))
    if E("BLACK_CHROMA"):
        e = replace(e, black_chroma_threshold=_as_int(E("BLACK_CHROMA")))

    lock_path = s.lock_path
    if E("LOCK_PATH"):
        lock_path = E("LOCK_PATH") or lock_path

    return Settings(wled=w, video=v, sampling=samp, layout=lay, effects=e, lock_path=lock_path)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=True)

    p.add_argument("--config", help="Path to config.toml (overrides auto-search)")
    p.add_argument("--print-config", action="store_true", help="Print effective config and exit")

    # Runtime modes you already have
    p.add_argument("--test-chase", action="store_true")
    p.add_argument("--test-sides", action="store_true")
    p.add_argument("--test-sides-all", action="store_true")

    # Common overrides
    p.add_argument("--wled-ip")
    p.add_argument("--wled-port", type=int)
    p.add_argument("--wled-timeout", type=int)

    p.add_argument("--video-dev")

    p.add_argument("--right", type=int)
    p.add_argument("--top", type=int)
    p.add_argument("--left", type=int)
    p.add_argument("--bottom", type=int)

    p.add_argument("--out-fps", type=int)
    p.add_argument("--smooth-alpha", type=float)
    p.add_argument("--max-step", type=int)

    p.add_argument("--sat-boost", type=float)
    p.add_argument("--val-boost", type=float)

    p.add_argument("--black-luma", type=int)
    p.add_argument("--black-chroma", type=int)

    p.add_argument("--sample-w", type=int)
    p.add_argument("--sample-h", type=int)
    p.add_argument("--edge-margin", type=int)
    p.add_argument("--patch-r", type=int)

    return p


def apply_cli(s: Settings, args: argparse.Namespace) -> Settings:
    w = s.wled
    v = s.video
    lay = s.layout
    e = s.effects
    samp = s.sampling

    if args.wled_ip:
        w = replace(w, ip=args.wled_ip)
    if args.wled_port is not None:
        w = replace(w, port=args.wled_port)
    if args.wled_timeout is not None:
        w = replace(w, timeout_seconds=args.wled_timeout)

    if args.video_dev:
        v = replace(v, device=args.video_dev)

    if args.right is not None:
        lay = replace(lay, right=args.right)
    if args.top is not None:
        lay = replace(lay, top=args.top)
    if args.left is not None:
        lay = replace(lay, left=args.left)
    if args.bottom is not None:
        lay = replace(lay, bottom=args.bottom)

    if args.out_fps is not None:
        e = replace(e, out_fps=args.out_fps)
    if args.smooth_alpha is not None:
        e = replace(e, smooth_alpha=args.smooth_alpha)
    if args.max_step is not None:
        e = replace(e, max_step_per_frame=args.max_step)

    if args.sat_boost is not None:
        e = replace(e, sat_boost=args.sat_boost)
    if args.val_boost is not None:
        e = replace(e, val_boost=args.val_boost)

    if args.black_luma is not None:
        e = replace(e, black_luma_threshold=args.black_luma)
    if args.black_chroma is not None:
        e = replace(e, black_chroma_threshold=args.black_chroma)

    if args.sample_w is not None:
        samp = replace(samp, sample_w=args.sample_w)
    if args.sample_h is not None:
        samp = replace(samp, sample_h=args.sample_h)
    if args.edge_margin is not None:
        samp = replace(samp, edge_margin=args.edge_margin)
    if args.patch_r is not None:
        samp = replace(samp, patch_r=args.patch_r)

    return Settings(wled=w, video=v, sampling=samp, layout=lay, effects=e, lock_path=s.lock_path)


def load_settings(argv: list[str] | None = None, app_name: str = "glowbridge") -> tuple[Settings, argparse.Namespace]:
    """
    Returns (settings, parsed_args).
    You can pass sys.argv[1:] as argv, or None to default.
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    s = Settings()  # defaults

    # config path selection
    cfg_path = _expand_path(args.config)
    if cfg_path is None:
        for p in default_config_paths(app_name):
            if p.exists():
                cfg_path = p
                break

    if cfg_path is not None and cfg_path.exists():
        d = load_toml(cfg_path)
        s = merge_from_dict(s, d)

    # env overrides
    s = apply_env(s)

    # cli overrides
    s = apply_cli(s, args)

    # sanity checks
    if s.layout.led_count <= 0:
        raise ValueError("LED count is zero/negative, check layout config.")
    if s.effects.out_fps <= 0:
        raise ValueError("out_fps must be > 0.")
    if s.wled.port <= 0 or s.wled.port > 65535:
        raise ValueError("wled.port must be 1..65535.")
    if s.video.cap_w <= 0 or s.video.cap_h <= 0:
        raise ValueError("Capture size must be > 0.")
    if s.sampling.sample_w <= 0 or s.sampling.sample_h <= 0:
        raise ValueError("Sample size must be > 0.")
    if not (0.0 <= s.effects.smooth_alpha < 1.0):
        raise ValueError("smooth_alpha should be in [0.0, 1.0).")

    if args.print_config:
        print(s.summary())
        sys.exit(0)

    return s, args
