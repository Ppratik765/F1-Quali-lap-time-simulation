import os
import urllib.request
import json
import numpy as np
from datetime import datetime, timedelta
from scipy.interpolate import interp1d

os.makedirs("data", exist_ok=True)

def fetch_json(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode('utf-8'))

targets = [
    {
        "circuit": "Austria",
        "session_key": 11311,
        "driver_number": 63,
        "lap_number": 19,
        "filename": "data/austria_2026_telemetry.npz"
    },
    {
        "circuit": "Spain",
        "session_key": 11303,
        "driver_number": 63,
        "lap_number": 12,
        "filename": "data/spain_2026_telemetry.npz"
    },
    {
        "circuit": "Silverstone",
        "session_key": 11322,
        "driver_number": 12,
        "lap_number": 18,
        "filename": "data/silverstone_2026_telemetry.npz"
    }
]

for t in targets:
    if os.path.exists(t["filename"]):
        print(f"{t['circuit']} already cached at {t['filename']}")
        continue

    print(f"Fetching {t['circuit']} (session {t['session_key']}, driver {t['driver_number']}, lap {t['lap_number']})...")
    laps = fetch_json(f"https://api.openf1.org/v1/laps?session_key={t['session_key']}&driver_number={t['driver_number']}&lap_number={t['lap_number']}")
    lap = laps[0]
    t_start_str = lap['date_start']
    lap_dur = float(lap['lap_duration'])
    s1 = float(lap.get('duration_sector_1', 0.0) or 0.0)
    s2 = float(lap.get('duration_sector_2', 0.0) or 0.0)
    s3 = float(lap.get('duration_sector_3', 0.0) or 0.0)

    dt_start = datetime.fromisoformat(t_start_str.replace('Z', '+00:00'))
    dt_end = dt_start + timedelta(seconds=lap_dur + 0.2)
    t_end_str = dt_end.isoformat()

    cars = fetch_json(f"https://api.openf1.org/v1/car_data?session_key={t['session_key']}&driver_number={t['driver_number']}&date>={t_start_str}&date<={t_end_str}")
    locs = fetch_json(f"https://api.openf1.org/v1/location?session_key={t['session_key']}&driver_number={t['driver_number']}&date>={t_start_str}&date<={t_end_str}")

    print(f"  {len(cars)} car records, {len(locs)} location records")

    # Parse timestamps to elapsed seconds
    t_car = np.array([(datetime.fromisoformat(c['date'].replace('Z', '+00:00')) - dt_start).total_seconds() for c in cars])
    speed = np.array([float(c['speed']) for c in cars])
    throttle = np.array([float(c['throttle']) for c in cars])
    brake = np.array([float(c['brake']) for c in cars])

    t_loc = np.array([(datetime.fromisoformat(l['date'].replace('Z', '+00:00')) - dt_start).total_seconds() for l in locs])
    x_raw = np.array([float(l['x']) for l in locs])
    y_raw = np.array([float(l['y']) for l in locs])
    z_raw = np.array([float(l['z']) for l in locs])

    # Sort and remove duplicate timestamps
    _, u_car = np.unique(t_car, return_index=True)
    t_car = t_car[u_car]
    speed = speed[u_car]
    throttle = throttle[u_car]
    brake = brake[u_car]

    _, u_loc = np.unique(t_loc, return_index=True)
    t_loc = t_loc[u_loc]
    x_raw = x_raw[u_loc]
    y_raw = y_raw[u_loc]
    z_raw = z_raw[u_loc]

    # Integrate distance along car data: s(t) = trapz(v_ms, t)
    v_ms = speed / 3.6
    dt_c = np.diff(t_car)
    s_car = np.concatenate([[0.0], np.cumsum(0.5 * (v_ms[:-1] + v_ms[1:]) * dt_c)])
    lap_length = s_car[-1]

    # Uniform 1 m distance grid
    s_grid = np.arange(0.0, lap_length, 1.0)
    speed_grid = interp1d(s_car, speed, kind='linear', fill_value='extrapolate')(s_grid)
    throttle_grid = interp1d(s_car, throttle, kind='linear', fill_value='extrapolate')(s_grid)
    brake_grid = interp1d(s_car, brake, kind='linear', fill_value='extrapolate')(s_grid)

    # Interpolate x, y, z onto t_car, then s_grid
    x_interp = interp1d(t_loc, x_raw, kind='linear', fill_value='extrapolate')(t_car)
    y_interp = interp1d(t_loc, y_raw, kind='linear', fill_value='extrapolate')(t_car)
    z_interp = interp1d(t_loc, z_raw, kind='linear', fill_value='extrapolate')(t_car)

    x_grid = interp1d(s_car, x_interp, kind='linear', fill_value='extrapolate')(s_grid)
    y_grid = interp1d(s_car, y_interp, kind='linear', fill_value='extrapolate')(s_grid)
    z_grid = interp1d(s_car, z_interp, kind='linear', fill_value='extrapolate')(s_grid)

    np.savez_compressed(
        t["filename"],
        s=s_grid,
        speed_kmh=speed_grid,
        throttle=throttle_grid,
        brake=brake_grid,
        x=x_grid,
        y=y_grid,
        z=z_grid,
        lap_time=lap_dur,
        sector_times=np.array([s1, s2, s3]),
        circuit_name=t["circuit"],
        lap_length=lap_length
    )
    print(f"Saved {t['circuit']}: {len(s_grid)} points, lap time {lap_dur:.3f} s, length {lap_length:.1f} m")

print("All session telemetry cached successfully!")

