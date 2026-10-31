"""
fetch_interlagos_raw.py — stores the RAW (unsmoothed) OpenF1 position samples of the 2025 Sao Paulo qualifying
pole lap in data/interlagos_raw_openf1.npz. track_loader prefers this file when present, which preserves the
true tight-corner radii (the older cache was pre-smoothed). Requires network access (OpenF1).
"""
import json
import os
import urllib.request

import numpy as np

URL = ("https://api.openf1.org/v1/location?session_key=9865&driver_number=4"
       "&date>=2025-11-08T19:02:47&date<=2025-11-08T19:03:56.6")


def main():
    os.makedirs("data", exist_ok=True)
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        loc = json.loads(resp.read().decode("utf-8"))
    if len(loc) < 100:
        raise SystemExit(f"unexpected response ({len(loc)} samples)")
    x = np.array([p["x"] for p in loc], dtype=np.float64) / 10.0
    y = np.array([p["y"] for p in loc], dtype=np.float64) / 10.0
    z = np.array([p["z"] for p in loc], dtype=np.float64) / 10.0
    np.savez_compressed("data/interlagos_raw_openf1.npz", x=x, y=y, z=z)
    print(f"saved data/interlagos_raw_openf1.npz with {len(x)} raw samples")


if __name__ == "__main__":
    main()
