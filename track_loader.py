"""
track_loader.py — Extracts Interlagos (São Paulo GP) qualifying track geometry
(X, Y, Z, distance s, curvature kappa(s), and vertical slope sin(theta) = dz/ds)
resampled to 1 m intervals (0 to 4309 m) directly from real telemetry data.
No hand-placed Gaussian corners or typed elevation waypoints.
Fails loudly if telemetry data is unavailable.
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


@dataclass
class TrackData:
    s: np.ndarray            # Distance array [0, 1, ..., length_m] m (shape: N)
    x: np.ndarray            # X coordinates [m]
    y: np.ndarray            # Y coordinates [m]
    z: np.ndarray            # Elevation Z coordinates [m above sea level]
    kappa: np.ndarray        # Signed curvature [1/m] (+ left, - right)
    sin_theta: np.ndarray    # Vertical slope dz/ds
    heading_rad: np.ndarray  # Track tangent heading [rad]
    corner_labels: Dict[str, float]  # Key corner apex distances [m]
    length_m: int = 4309


def _compute_track_derivatives(s: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple:
    """
    Computes tangent heading, signed curvature kappa(s), and vertical slope sin(theta) = dz/ds
    directly from 3D coordinates using consistent analytical derivatives.
    """
    dx = np.gradient(x, s)
    dy = np.gradient(y, s)
    ddx = np.gradient(dx, s)
    ddy = np.gradient(dy, s)

    # Tangent heading
    heading_rad = np.unwrap(np.arctan2(dy, dx))

    # Signed 2D curvature: kappa = (dx*ddy - dy*ddx) / (dx^2 + dy^2)^(3/2)
    speed_sq = dx**2 + dy**2
    speed_denom = np.maximum(1e-6, speed_sq ** 1.5)
    kappa = (dx * ddy - dy * ddx) / speed_denom

    # Vertical slope sin(theta) = dz/ds
    sin_theta = np.clip(np.gradient(z, s), -0.15, 0.15)

    return heading_rad, kappa, sin_theta


def _extract_corner_labels(s: np.ndarray, kappa: np.ndarray) -> Dict[str, float]:
    """
    Identifies true physical corner apexes from curvature extrema along the real driven line.
    """
    peaks_pos, _ = find_peaks(kappa, height=0.007, distance=40)
    peaks_neg, _ = find_peaks(-kappa, height=0.007, distance=40)
    all_peaks = sorted(np.concatenate([peaks_pos, peaks_neg]))

    corner_labels = {}
    for i, p in enumerate(all_peaks, start=1):
        name = f"Apex_{i}_{'L' if kappa[p] > 0 else 'R'}"
        corner_labels[name] = float(s[p])
    return corner_labels


def _load_or_fetch_real_telemetry(cache_file: str, length_m: int = 4309) -> tuple:
    """
    Loads cached real qualifying telemetry X/Y/Z or fetches from OpenF1 / FastF1.
    Fails loudly if data cannot be retrieved.
    """
    if os.path.exists(cache_file):
        data = np.load(cache_file)
        return data["s"], data["x"], data["y"], data["z"]

    # Attempt to fetch real telemetry from OpenF1 for the 2025 Sao Paulo qualifying fastest lap
    try:
        import requests
        resp = requests.get(
            "https://api.openf1.org/v1/location?session_key=9865&driver_number=4"
            "&date>=2025-11-08T19:02:47&date<=2025-11-08T19:03:56.6",
            timeout=15,
        )
        if resp.status_code != 200 or not resp.json():
            raise RuntimeError(f"OpenF1 API returned status {resp.status_code}")
        loc = resp.json()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to fetch real telemetry track data and cache '{cache_file}' is missing! "
            f"Cannot build track geometry from real telemetry: {exc}"
        ) from exc

    x_raw = np.array([p["x"] for p in loc], dtype=np.float64) / 10.0
    y_raw = np.array([p["y"] for p in loc], dtype=np.float64) / 10.0
    z_raw = np.array([p["z"] for p in loc], dtype=np.float64) / 10.0

    x_raw -= x_raw[0]
    y_raw -= y_raw[0]

    ds_raw = np.hypot(np.diff(x_raw), np.diff(y_raw))
    s_raw = np.concatenate([[0.0], np.cumsum(ds_raw)])
    s_norm = s_raw * (float(length_m) / s_raw[-1])

    cs_x = CubicSpline(s_norm, np.append(x_raw[:-1], x_raw[0]), bc_type="periodic")
    cs_y = CubicSpline(s_norm, np.append(y_raw[:-1], y_raw[0]), bc_type="periodic")
    cs_z = CubicSpline(s_norm, np.append(z_raw[:-1], z_raw[0]), bc_type="periodic")

    s_grid = np.arange(0, length_m + 1, 1, dtype=np.float64)
    x_smooth = gaussian_filter1d(cs_x(s_grid), sigma=7.0, mode="wrap")
    y_smooth = gaussian_filter1d(cs_y(s_grid), sigma=7.0, mode="wrap")
    z_smooth = gaussian_filter1d(cs_z(s_grid), sigma=12.0, mode="wrap")

    os.makedirs(os.path.dirname(os.path.abspath(cache_file)), exist_ok=True)
    np.savez_compressed(cache_file, s=s_grid, x=x_smooth, y=y_smooth, z=z_smooth)
    return s_grid, x_smooth, y_smooth, z_smooth


def load_interlagos_track(
    config_path: str = "config_brazil_2026.json",
    use_fastf1: bool = True,
    cache_file: str = "data/interlagos_real_telemetry.npz",
) -> TrackData:
    """
    Loads the Interlagos track geometry computed directly from real telemetry X/Y/Z.
    Fails loudly if telemetry data is unavailable.
    """
    length_m = 4309
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            length_m = int(cfg.get("event", {}).get("circuit_length_m", 4309))

    if not use_fastf1 and not os.path.exists(cache_file):
        raise RuntimeError(
            "use_fastf1 is False but no telemetry cache file exists. "
            "Real telemetry data is required to compute track geometry."
        )

    s, x, y, z = _load_or_fetch_real_telemetry(cache_file=cache_file, length_m=length_m)
    heading_rad, kappa, sin_theta = _compute_track_derivatives(s, x, y, z)
    corner_labels = _extract_corner_labels(s, kappa)

    return TrackData(
        s=s,
        x=x,
        y=y,
        z=z,
        kappa=kappa,
        sin_theta=sin_theta,
        heading_rad=heading_rad,
        corner_labels=corner_labels,
        length_m=length_m,
    )


if __name__ == "__main__":
    track = load_interlagos_track()
    print(f"Loaded Interlagos track: {len(track.s)} points, length={track.length_m} m")
    print(f"Elevation range: {track.z.min():.1f} m to {track.z.max():.1f} m")
    print(f"Slope dz/ds range: {track.sin_theta.min():.4f} to {track.sin_theta.max():.4f}")
    print(f"Curvature range: {track.kappa.min():.4f} to {track.kappa.max():.4f}")
    print(f"Detected {len(track.corner_labels)} corner apexes:")
    for name, s_val in list(track.corner_labels.items())[:8]:
        print(f"  {name}: {s_val:.1f} m")
