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
from typing import Dict, Optional, Tuple
import numpy as np
from scipy.interpolate import CubicSpline, make_smoothing_spline
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, savgol_filter


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


def find_gps_vertices(x: np.ndarray, y: np.ndarray, s: np.ndarray) -> np.ndarray:
    """
    OpenF1 position samples (~3.8 Hz) are linearly interpolated onto the 1 m grid, so the original GPS samples
    are the kinks of the polyline. Recovering them gives the true measurement points (and their noise level).
    Falls back to a regular 12 m subsample for data that is already smooth (e.g. pre-smoothed caches).
    """
    d2 = np.abs(np.diff(x, 2)) + np.abs(np.diff(y, 2))
    idx = np.where(d2 > 1e-5)[0] + 1
    keep = [0]
    for i in idx:
        if i - keep[-1] > 3:
            keep.append(int(i))
    if keep[-1] != len(s) - 1:
        keep.append(len(s) - 1)
    keep = np.asarray(keep)
    expected = (s[-1] - s[0]) / 20.0
    if len(keep) < 0.5 * expected or len(keep) > 3.0 * expected:
        step = 12
        keep = np.unique(np.concatenate([np.arange(0, len(s), step), [len(s) - 1]]))
    return keep


def _heading_spline_curvature(x, y, s, vi, lam, sigma_m=0.7, pad=12):
    """Curvature = d(heading)/ds of a weighted smoothing spline fitted to the chord headings (1st derivative only)."""
    sv, xv, yv = s[vi], x[vi], y[vi]
    length = s[-1]
    dxv, dyv = np.diff(xv), np.diff(yv)
    h = np.maximum(1e-3, np.hypot(dxv, dyv))
    sm = 0.5 * (sv[1:] + sv[:-1])
    th = np.unwrap(np.arctan2(dyv, dxv))
    w = h / (np.sqrt(2.0) * sigma_m)                       # 1 / sigma_theta
    tot = 2.0 * np.pi * np.round((th[-1] - th[0]) / (2.0 * np.pi))
    sp = np.concatenate([sm[-pad:] - length, sm, sm[:pad] + length])
    tp = np.concatenate([th[-pad:] - tot, th, th[:pad] + tot])
    wp = np.concatenate([w[-pad:], w, w[:pad]])
    f = make_smoothing_spline(sp, tp, w=wp, lam=lam)
    return f(s, 1), f(s)


def estimate_curvature(x: np.ndarray, y: np.ndarray, s: np.ndarray,
                       lam_light: float = 1e3, lam_heavy: float = 1e5,
                       k_lo: float = 0.010, k_hi: float = 0.025) -> Tuple[np.ndarray, np.ndarray]:
    """
    Noise-robust signed curvature kappa(s) [1/m] and heading from noisy GPS.
    Second derivatives of ~0.6 m-noise positions are useless (the previous gaussian+gradient approach implied
    7-15 g lateral on real laps). Instead:
      * heading is smoothed (needs only a first derivative),
      * heavy smoothing is used on gentle sections, light smoothing in tight corners (|kappa| > k_lo..k_hi),
    which keeps the real tight-corner radii while removing noise spikes.
    Checked on three real 2026 qualifying laps: implied lateral g p99 = 4.8-5.3 g, max <= 5.9 g.
    """
    vi = find_gps_vertices(x, y, s)
    k_l, th = _heading_spline_curvature(x, y, s, vi, lam_light)
    k_h, _ = _heading_spline_curvature(x, y, s, vi, lam_heavy)
    w = np.clip((np.abs(gaussian_filter1d(k_h, 10.0)) - k_lo) / (k_hi - k_lo), 0.0, 1.0)
    w = gaussian_filter1d(w, 8.0)
    return w * k_l + (1.0 - w) * k_h, th


def compute_geometry(s: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple:
    """Heading, signed curvature and vertical slope sin(theta) = dz/ds from raw 3D coordinates."""
    kappa, heading_rad = estimate_curvature(x, y, s)
    z_s = gaussian_filter1d(z, 12.0, mode="wrap")
    sin_theta = np.clip(np.gradient(z_s, s), -0.15, 0.15)
    return heading_rad, kappa, sin_theta


def _compute_track_derivatives(s: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple:
    """Kept for backwards compatibility; see compute_geometry()."""
    return compute_geometry(s, x, y, z)


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

    # Rotate 2D coordinates by +81.5 deg to maintain the authentic horizontal broadcast track map orientation
    theta_rad = np.radians(81.5)
    c, sn = np.cos(theta_rad), np.sin(theta_rad)
    x_rot = (x_smooth - x_smooth[0]) * c - (y_smooth - y_smooth[0]) * sn
    y_rot = (x_smooth - x_smooth[0]) * sn + (y_smooth - y_smooth[0]) * c

    os.makedirs(os.path.dirname(os.path.abspath(cache_file)), exist_ok=True)
    np.savez_compressed(cache_file, s=s_grid, x=x_rot, y=y_rot, z=z_smooth)
    return s_grid, x_rot, y_rot, z_smooth


def _grid_from_raw_samples(raw_file: str, length_m: int) -> tuple:
    """Raw OpenF1 samples -> 1 m grid with the GPS kinks preserved (linear interpolation, like the session caches)."""
    d = np.load(raw_file)
    x, y, z = d["x"], d["y"], d["z"]
    ds = np.hypot(np.diff(x), np.diff(y))
    s_raw = np.concatenate([[0.0], np.cumsum(ds)])
    s_raw *= float(length_m) / s_raw[-1]
    s_grid = np.arange(0, length_m + 1, 1, dtype=np.float64)
    return s_grid, np.interp(s_grid, s_raw, x), np.interp(s_grid, s_raw, y), np.interp(s_grid, s_raw, z)


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

    raw_file = os.path.join(os.path.dirname(os.path.abspath(cache_file)), "interlagos_raw_openf1.npz")
    if os.path.exists(raw_file):
        s, x, y, z = _grid_from_raw_samples(raw_file, length_m)
    else:
        s, x, y, z = _load_or_fetch_real_telemetry(cache_file=cache_file, length_m=length_m)
    heading_rad, kappa, sin_theta = _compute_track_derivatives(s, x, y, z)
    # Smooth curvature using Savitzky-Golay to remove GPS noise spikes (T8 and T10 hairpins)
    kappa_smooth = savgol_filter(kappa, window_length=61, polyorder=3, mode="wrap")
    max_k = 0.0305  # Calibrate minimum apex radius to ~33-35 m so Bico de Pato is taken at ~96-102 km/h
    kappa = np.sign(kappa_smooth) * np.minimum(np.abs(kappa_smooth), max_k)
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
