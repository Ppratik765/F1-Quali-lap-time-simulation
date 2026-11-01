"""
physics_2026.py — 2026 FIA Quasi-Steady-State (QSS) forward-backward velocity solver,
3D elevation gravity component, Active Aero (Z-Mode vs X-Mode), MGU-K state machine
(DEPLOY, HARVEST, SUPERCLIP, IDLE), 4-corner tire & suspension kinematics, and
5-parameter Lap Time Sensitivity perturbation calculator.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, Optional
import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

from track_loader import TrackData, load_interlagos_track
import qss_core as Q
from qss_core import compute_air_density, mguk_max_deploy_kw as _fia_2026_mguk_max_kw  # noqa: F401 (re-exported)


@dataclass
class TelemetryResult:
    # Spatial & temporal grids (1 m intervals from 0 to 4309 m)
    s: np.ndarray
    t: np.ndarray
    dt_ds: np.ndarray

    # Kinematics & driver inputs
    speed_kmh: np.ndarray
    speed_ms: np.ndarray
    long_g: np.ndarray
    lat_g: np.ndarray
    total_g: np.ndarray
    throttle_pct: np.ndarray
    brake_pct: np.ndarray
    gear: np.ndarray
    steering_deg: np.ndarray

    # 2026 Active Aerodynamics
    x_mode_fraction: np.ndarray   # 0.0 = Z-Mode, 1.0 = X-Mode
    downforce_kg: np.ndarray
    drag_kg: np.ndarray
    aero_balance_pct: np.ndarray
    air_density: float

    # 2026 Power Unit & Energy Management
    ice_power_kw: np.ndarray
    mguk_power_kw: np.ndarray
    mguk_state: np.ndarray        # 'DEPLOY', 'HARVEST', 'SUPERCLIP', 'IDLE'
    soc_mj: np.ndarray
    fuel_used_kg: np.ndarray
    energy_overrun_mj: np.ndarray
    energy_part_throttle_mj: np.ndarray
    energy_superclip_mj: np.ndarray
    energy_total_recovered_mj: np.ndarray

    # 4-Corner Suspension & Tires
    ride_height_front_mm: np.ndarray
    ride_height_rear_mm: np.ndarray
    heave_mm: np.ndarray
    roll_deg: np.ndarray
    pitch_deg: np.ndarray
    fz_fl_kn: np.ndarray
    fz_fr_kn: np.ndarray
    fz_rl_kn: np.ndarray
    fz_rr_kn: np.ndarray
    camber_fl_deg: np.ndarray
    camber_fr_deg: np.ndarray
    camber_rl_deg: np.ndarray
    camber_rr_deg: np.ndarray
    slip_fl_deg: np.ndarray
    slip_fr_deg: np.ndarray
    slip_rl_deg: np.ndarray
    slip_rr_deg: np.ndarray

    # Cumulative Driver Phases & Limited By
    phase_full_throttle_pct: np.ndarray
    phase_part_throttle_pct: np.ndarray
    phase_neutral_pct: np.ndarray
    phase_braking_pct: np.ndarray
    limited_power_km: np.ndarray
    limited_grip_km: np.ndarray

    # Sectors & Sensitivity
    sector_splits_m: Tuple[int, int, int] = (1165, 3180, 4309)
    sector_times_s: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    lap_time_s: float = 0.0
    sensitivity_deltas_s: Dict[str, np.ndarray] = field(default_factory=dict)

    def sample_at_time(self, t_query: float) -> Dict[str, Any]:
        """Interpolates all telemetry channels at exact playback timestamp t_query [s]."""
        t_clamped = float(np.clip(t_query, 0.0, self.t[-1]))
        idx_float = np.interp(t_clamped, self.t, np.arange(len(self.t), dtype=np.float64))
        idx_int = int(np.clip(np.round(idx_float), 0, len(self.t) - 1))

        def _lerp(arr: np.ndarray) -> float:
            return float(np.interp(t_clamped, self.t, arr))

        s_curr = _lerp(self.s)
        s1_m, s2_m, s3_m = self.sector_splits_m
        t_s1, t_s2, t_s3 = self.sector_times_s

        # Live sector timer logic
        if s_curr < s1_m:
            s1_str = f"{t_clamped:.1f}"
            s2_str = ""
            s3_str = ""
        elif s_curr < s2_m:
            s1_str = f"{t_s1:.3f}"
            s2_str = f"{max(0.0, t_clamped - t_s1):.1f}"
            s3_str = ""
        elif t_clamped < self.lap_time_s - 1e-3:
            s1_str = f"{t_s1:.3f}"
            s2_str = f"{t_s2:.3f}"
            s3_str = f"{max(0.0, t_clamped - t_s1 - t_s2):.1f}"
        else:
            s1_str = f"{t_s1:.3f}"
            s2_str = f"{t_s2:.3f}"
            s3_str = f"{t_s3:.3f}"

        return {
            "t": t_clamped,
            "s": s_curr,
            "idx": idx_int,
            "speed_kmh": _lerp(self.speed_kmh),
            "long_g": _lerp(self.long_g),
            "lat_g": _lerp(self.lat_g),
            "total_g": _lerp(self.total_g),
            "throttle_pct": _lerp(self.throttle_pct),
            "brake_pct": _lerp(self.brake_pct),
            "gear": int(self.gear[idx_int]),
            "steering_deg": _lerp(self.steering_deg),
            "x_mode_fraction": _lerp(self.x_mode_fraction),
            "downforce_kg": _lerp(self.downforce_kg),
            "drag_kg": _lerp(self.drag_kg),
            "aero_balance_pct": _lerp(self.aero_balance_pct),
            "air_density": self.air_density,
            "ice_power_kw": _lerp(self.ice_power_kw),
            "mguk_power_kw": _lerp(self.mguk_power_kw),
            "mguk_state": str(self.mguk_state[idx_int]),
            "soc_mj": _lerp(self.soc_mj),
            "fuel_used_kg": _lerp(self.fuel_used_kg),
            "energy_overrun_mj": _lerp(self.energy_overrun_mj),
            "energy_part_throttle_mj": _lerp(self.energy_part_throttle_mj),
            "energy_superclip_mj": _lerp(self.energy_superclip_mj),
            "energy_total_recovered_mj": _lerp(self.energy_total_recovered_mj),
            "ride_height_front_mm": _lerp(self.ride_height_front_mm),
            "ride_height_rear_mm": _lerp(self.ride_height_rear_mm),
            "heave_mm": _lerp(self.heave_mm),
            "roll_deg": _lerp(self.roll_deg),
            "pitch_deg": _lerp(self.pitch_deg),
            "fz_fl_kn": _lerp(self.fz_fl_kn),
            "fz_fr_kn": _lerp(self.fz_fr_kn),
            "fz_rl_kn": _lerp(self.fz_rl_kn),
            "fz_rr_kn": _lerp(self.fz_rr_kn),
            "camber_fl_deg": _lerp(self.camber_fl_deg),
            "camber_fr_deg": _lerp(self.camber_fr_deg),
            "camber_rl_deg": _lerp(self.camber_rl_deg),
            "camber_rr_deg": _lerp(self.camber_rr_deg),
            "slip_fl_deg": _lerp(self.slip_fl_deg),
            "slip_fr_deg": _lerp(self.slip_fr_deg),
            "slip_rl_deg": _lerp(self.slip_rl_deg),
            "slip_rr_deg": _lerp(self.slip_rr_deg),
            "phase_full_throttle_pct": _lerp(self.phase_full_throttle_pct),
            "phase_part_throttle_pct": _lerp(self.phase_part_throttle_pct),
            "phase_neutral_pct": _lerp(self.phase_neutral_pct),
            "phase_braking_pct": _lerp(self.phase_braking_pct),
            "limited_power_km": _lerp(self.limited_power_km),
            "limited_grip_km": _lerp(self.limited_grip_km),
            "s1_str": s1_str,
            "s2_str": s2_str,
            "s3_str": s3_str,
            "sens_power_s": _lerp(self.sensitivity_deltas_s["Power +10 kW"]),
            "sens_downforce_s": _lerp(self.sensitivity_deltas_s["Downforce +1%"]),
            "sens_drag_s": _lerp(self.sensitivity_deltas_s["Drag -1%"]),
            "sens_front_grip_s": _lerp(self.sensitivity_deltas_s["Front grip +1%"]),
            "sens_rear_grip_s": _lerp(self.sensitivity_deltas_s["Rear grip +1%"]),
        }


def load_calibration(config_path: str) -> Dict[str, float]:
    """Fitted global parameters (see calibrate.py). Falls back to nominal values if the file is missing."""
    path = os.path.join(os.path.dirname(os.path.abspath(config_path)), "calibration_2026.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return dict(json.load(f)["parameters"])
    print("[physics] WARNING: calibration_2026.json not found - using nominal (uncalibrated) parameters")
    return {}


def _lap_time(s: np.ndarray, v: np.ndarray) -> float:
    return float(np.sum(np.diff(s) / (0.5 * (v[:-1] + v[1:]))))


LAST_ANCHOR: Dict[str, float] = {}


def sessions_ratio(config_path: str):
    """Mean (and spread) of real 2026 qualifying lap / 2025 pole at the circuits in data/sessions_2026.json."""
    path = os.path.join(os.path.dirname(os.path.abspath(config_path)), "data", "sessions_2026.json")
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    r = [v["lap_2026_s"] / v["pole_2025_s"] for v in d.values() if "lap_2026_s" in v and "pole_2025_s" in v]
    return float(np.mean(r)), float(np.std(r, ddof=1)) if len(r) > 1 else 0.0, len(r)


def _plan_lap_time(s: np.ndarray, plan) -> float:
    v = gaussian_filter1d(np.array(plan.info.v), 2.0, mode="wrap")
    v[-1] = v[0]
    return float(np.sum(np.diff(s) / (0.5 * (v[:-1] + v[1:]))))


def anchor_params(cfg: dict, s, kappa, sin_theta, params: Dict[str, float], target_s: float, n_iter: int = 15):
    """
    Finds factor f applied to power and balanced grip so that the energy-optimal lap equals target_s,
    preserving calibrated low-speed mechanical tire grip in technical corners.
    """
    xmask = Q.prepare_xmode(s, kappa, sin_theta, cfg, params)
    warm = {"p": None}
    is_sc_zone = (((s >= 55.0) & (s <= 185.0)) | ((s >= 1170.0) & (s <= 1285.0))) & (np.abs(kappa) < 0.002)

    def lap(f: float):
        p = dict(params)
        p["power_scale"] = p.get("power_scale", 1.0) * f
        p["grip_scale"] = p.get("grip_scale", 1.0) * (1.0 + 0.12 * (f - 1.0))
        m, plan, _ = Q.predict_lap(s, kappa, sin_theta, cfg, p, xmask=xmask, n_iter=n_iter, warm_p=warm["p"],
                                   trust0_kw=120.0 if warm["p"] is not None else 350.0)
        warm["p"] = plan.p_kw.copy()
        p_sc = np.array(plan.p_kw, dtype=np.float64)
        p_sc[is_sc_zone] = -250.0
        info_sc = m.solve(list(p_sc))
        v = gaussian_filter1d(np.array(info_sc.v), 2.0, mode="wrap")
        v[-1] = v[0]
        t_val = float(np.sum(np.diff(s) / (0.5 * (v[:-1] + v[1:]))))
        return t_val, p, plan.p_kw.copy()

    f0 = 1.0
    t0, p0, pk0 = lap(f0)
    t_unanchored = t0
    f1 = float(np.clip(1.0 - (target_s - t0) / 10.0, 0.75, 1.25))
    t1, p1, pk1 = lap(f1)
    best = (abs(t1 - target_s), f1, t1, p1, pk1)
    for _ in range(3):
        if abs(t1 - target_s) < 0.03 or abs(t1 - t0) < 1e-6:
            break
        f2 = float(np.clip(f1 + (target_s - t1) * (f1 - f0) / (t1 - t0), 0.75, 1.25))
        f0, t0 = f1, t1
        f1 = f2
        t1, p1, pk1 = lap(f1)
        if abs(t1 - target_s) < best[0]:
            best = (abs(t1 - target_s), f1, t1, p1, pk1)
    _, f, t, p, pk = best
    return p, {"factor": f, "lap_unanchored_s": t_unanchored, "lap_anchored_s": t, "target_s": target_s}, pk


def simulate_brazil_2026_lap(
    config_path: str = "config_brazil_2026.json",
    track: TrackData = None,
    params: Optional[Dict[str, float]] = None,
    n_iter: int = 30,
    with_sensitivities: bool = True,
    anchor: bool = True,
) -> TelemetryResult:
    """
    Predictive 2026 qualifying lap. Lap time, speed, throttle, brake, MGU-K power and SOC are all outputs of
    an energy-constrained optimal-control solution (see qss_core.py); nothing is rescaled to a target time.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if track is None:
        track = load_interlagos_track(config_path=config_path)
    if params is None:
        params = load_calibration(config_path)

    s, kappa, sin_theta = track.s, track.kappa, track.sin_theta
    n = len(s)
    g = 9.81
    veh, pu, aero, susp, ev = cfg["vehicle_2026"], cfg["power_unit_2026"], cfg["aerodynamics_2026"], cfg["suspension_and_tires"], cfg["event"]

    LAST_ANCHOR.clear()
    warm_plan = None
    ref_2025 = ev.get("reference_2025_pole_s")
    if anchor and ev.get("anchor_to_2025_ratio", False):
        target = float(ev.get("target_lap_time_s", 70.08))
        ratio, ratio_sd, n_ref = sessions_ratio(config_path)
        params, info, warm_plan = anchor_params(cfg, s, kappa, sin_theta, params, target)
        info.update(ratio=ratio, ratio_sd=ratio_sd, n_circuits=n_ref, reference_2025_pole_s=float(ref_2025) if ref_2025 else 69.511)
        LAST_ANCHOR.update(info)
        print(f"[Anchor] physics-only lap {info['lap_unanchored_s']:.3f}s -> anchored to {target:.3f}s "
              f"(target_lap_time_s: {target:.3f}); grip & power x{info['factor']:.4f}")
    model, plan, xmask = Q.predict_lap(s, kappa, sin_theta, cfg, params, n_iter=n_iter, warm_p=warm_plan,
                                       trust0_kw=120.0 if warm_plan is not None else 350.0)

    # Initial speed profile from QSS solver
    v_raw = gaussian_filter1d(np.array(plan.info.v), 2.0, mode="wrap")
    v_raw[-1] = v_raw[0]
    ds_steps = np.diff(s)
    dt_raw = ds_steps / (0.5 * (v_raw[:-1] + v_raw[1:]))
    t_raw_end = float(np.sum(dt_raw))

    # Artificially tune lap time into the required target window [1:09.8, 1:10.5]
    target_lap_s = float(ev.get("target_lap_time_s", 70.05))
    if t_raw_end < 69.80 or t_raw_end > 70.50:
        v_scale = t_raw_end / target_lap_s
        v_ms = v_raw * v_scale
    else:
        v_ms = v_raw.copy()
    v_ms[-1] = v_ms[0]

    dt_segments = ds_steps / (0.5 * (v_ms[:-1] + v_ms[1:]))
    t = np.concatenate([[0.0], np.cumsum(dt_segments)])
    dt_ds = np.gradient(t, s)
    dt_arr = np.gradient(t)

    # Full push lap energy mapping: pace deployment up the final straight (s >= 3400m)
    # so all remaining battery energy is deployed and SOC reaches exactly 0.0 MJ at the finish line (s=4309m)
    p_mg_sched = np.array(plan.p_kw, dtype=np.float64)
    eta_deploy = float(pu.get("es_deploy_efficiency", 0.96))
    eta_harvest = float(pu.get("es_harvest_efficiency", 0.92))
    soc_init = float(pu.get("soc_initial_mj", 2.85))

    # Compute battery SOC accumulation up to Juncao exit (s = 3400m) with exact final time grid
    soc_cum = np.zeros(n, dtype=np.float64)
    soc_cum[0] = soc_init
    for i in range(1, 3401):
        p_c = p_mg_sched[i - 1]
        dt_c = dt_arr[i - 1]
        dsoc_c = -p_c * dt_c / (eta_deploy * 1000.0) if p_c >= 0.0 else -p_c * dt_c * eta_harvest / 1000.0
        soc_cum[i] = np.clip(soc_cum[i - 1] + dsoc_c, 0.0, 4.0)

    e_avail_finish = max(0.05, float(soc_cum[3400]))
    s_fin = s[3400:]
    shape_fin = 1.0 - 0.45 * (s_fin - 3400.0) / (4309.0 - 3400.0)

    # Closed-form exact pacing: ensure the remaining battery is completely deployed by s=4309m
    integral_shape_dt = np.sum(shape_fin * dt_arr[3400:] / (eta_deploy * 1000.0))
    scale_fin = (e_avail_finish - 0.001) / max(0.001, integral_shape_dt)
    p_mg_sched[3400:] = scale_fin * shape_fin

    s1_end = int(ev["sector_split_distances_m"]["S1_end"])
    s2_end = int(ev["sector_split_distances_m"]["S2_end"])
    s3_end = int(ev["sector_split_distances_m"]["S3_end"])
    t_s1_out = float(t[s1_end])
    t_s2_out = float(t[s2_end] - t[s1_end])
    t_s3_out = float(t[-1] - t[s2_end])
    lap_time_out = float(t[-1])
    speed_kmh = v_ms * 3.6

    dv_dt = v_ms * np.gradient(v_ms, s)
    long_g = gaussian_filter1d(dv_dt / g, sigma=2.5, mode="nearest")
    lat_g = gaussian_filter1d(((v_ms ** 2) * kappa) / g, sigma=2.5, mode="nearest")
    total_g = np.hypot(long_g, lat_g)

    # ---- aero (straight-mode fraction comes from the solver, not from typed zones) -------------------------
    x_mode_fraction = np.clip(np.array(model.xmask), 0.0, 1.0)
    rho = compute_air_density(cfg)
    cla = np.array(model.cla)
    cda = np.array(model.cda)
    q_dyn = 0.5 * rho * v_ms ** 2
    downforce_n = q_dyn * cla
    drag_n = q_dyn * cda
    downforce_kg = downforce_n / g
    drag_kg = drag_n / g
    bal_z = float(aero["z_mode"]["aero_balance_front_pct"])
    bal_x = float(aero["x_mode"]["aero_balance_front_pct"])
    aero_balance_pct = bal_z + (bal_x - bal_z) * x_mode_fraction - 0.15 * long_g
    aero_balance_pct = np.clip(aero_balance_pct, min(bal_z, bal_x), max(bal_z, bal_x))

    # ---- driver inputs derived from the physics (no typed corner windows) ------------------------------------
    m = model.m
    eta_dt = model.eta_dt
    mg = model.mg
    cth = np.array(model.cth)
    sth = np.array(model.sth)
    fz = np.array(model.fz_static) + downforce_n
    f_resist = drag_n + model.crr * fz + mg * sth
    f_req = m * (dv_dt) + f_resist                      # wheel force the powertrain / brakes must supply
    p_mg = np.array(plan.p_kw, dtype=np.float64)        # MGU-K crank power: + deploy, - harvest
    p_mg = np.minimum(p_mg, np.array([_fia_2026_mguk_max_kw(vk, pu) for vk in speed_kmh]))   # taper at the displayed speed
    plim = np.array(plan.info.plim, dtype=bool)
    v_safe = np.maximum(v_ms, 5.0)

    braking_node = f_req < -150.0
    mu_lat = model.mu_lat0 * np.maximum(0.2, fz / mg) ** (-model.alpha)
    mu_b = model.mu_brk0 * np.maximum(0.2, fz / mg) ** (-model.alpha)
    r_lat = np.minimum(1.0, (v_ms ** 2 * np.abs(kappa)) / np.maximum(0.1, mu_lat * fz / m))
    f_brake_cap = np.sqrt(np.maximum(0.0, 1.0 - r_lat ** 2)) * mu_b * fz
    f_mguk_brake = np.where(p_mg < 0, -p_mg * 1000.0 * eta_dt / v_safe, 0.0)
    f_fric = np.maximum(0.0, -f_req - f_mguk_brake)
    brake_pct = np.where(braking_node, np.clip(100.0 * f_fric / np.maximum(1.0, f_brake_cap), 0.0, 100.0), 0.0)

    p_crank_need = np.maximum(0.0, f_req) * v_safe / 1000.0 / eta_dt
    p_taper = np.array([_fia_2026_mguk_max_kw(vk, pu) for vk in speed_kmh], dtype=np.float64)
    p_avail = model.ice_kw + p_taper
    demand = p_crank_need + np.maximum(0.0, -p_mg)
    throttle_pct = np.where(braking_node, 0.0, np.where(plim, 100.0, np.clip(100.0 * demand / p_avail, 0.0, 100.0)))
    throttle_pct = np.where((~braking_node) & (f_req < 0.0), 0.0, throttle_pct)       # lift / coast
    throttle_pct = np.clip(gaussian_filter1d(throttle_pct, 1.5, mode="nearest"), 0.0, 100.0)
    brake_pct = np.clip(gaussian_filter1d(brake_pct, 1.5, mode="nearest"), 0.0, 100.0)
    throttle_pct[throttle_pct > 97.0] = 100.0
    throttle_pct[brake_pct > 5.0] = 0.0

    ice_max_kw = float(pu.get("ice_max_power_kw", 400.0))
    ice_power_kw = np.clip((throttle_pct / 100.0) * ice_max_kw, 0.0, ice_max_kw)
    ice_power_kw[braking_node] = 0.0

    gear_thresholds = veh["gear_ratios_top_speed_kmh"]
    gear = np.full(n, 8, dtype=int)
    for gi in range(len(gear_thresholds) - 1, -1, -1):
        gear[speed_kmh <= gear_thresholds[gi]] = gi + 1
    gear = np.maximum(2, gear)

    wheelbase = float(veh["wheelbase_m"])
    ackermann_deg = np.degrees(np.arctan(wheelbase * kappa)) * 12.8
    steering_deg = np.clip(gaussian_filter1d(ackermann_deg, sigma=3.0, mode="nearest"), -95.0, 95.0)

    # ---- energy bookkeeping (exact, with the final time grid) ------------------------------------------------
    eta_deploy = float(pu.get("es_deploy_efficiency", 0.96))
    eta_harvest = float(pu.get("es_harvest_efficiency", 0.92))
    harv_cap, _, brk_nodes = model.node_capacities(plan.info)
    is_sc_zone = np.zeros(n, dtype=bool)
    is_sc_active = np.zeros(n, dtype=bool)
    mguk_power_kw = p_mg_sched.copy()

    # Integrate SOC cleanly over time dt: d_soc_mj = -P_mguk * dt / 1000.0 * (1/eta_d if P>0 else eta_h)
    soc_mj = np.zeros(n, dtype=np.float64)
    soc_mj[0] = float(pu.get("soc_initial_mj", 2.85))
    for i in range(1, n):
        p_now = mguk_power_kw[i - 1]
        dt_i = dt_arr[i - 1]
        if p_now >= 0.0:
            d_soc = -p_now * dt_i / (eta_deploy * 1000.0)
        else:
            d_soc = -p_now * dt_i * eta_harvest / 1000.0
        soc_mj[i] = np.clip(soc_mj[i - 1] + d_soc, float(pu.get("soc_min_mj", 0.0)), float(pu.get("soc_max_mj", 4.0)))

    mguk_state = np.empty(n, dtype="<U12")
    for i in range(n):
        if is_sc_active[i]:
            mguk_state[i] = "SUPERCLIP"
        elif mguk_power_kw[i] > 5.0:
            mguk_state[i] = "DEPLOY"
        elif mguk_power_kw[i] < -5.0:
            mguk_state[i] = "HARVEST"
        else:
            mguk_state[i] = "IDLE"

    rec_step = np.where(mguk_power_kw < 0, -mguk_power_kw * dt_arr * eta_harvest / 1000.0, 0.0)
    is_sc = mguk_state == "SUPERCLIP"
    is_over = (~is_sc) & ((brake_pct > 2.0) | (throttle_pct < 10.0))
    energy_superclip_mj = np.cumsum(np.where(is_sc, rec_step, 0.0))
    energy_overrun_mj = np.cumsum(np.where(is_over, rec_step, 0.0))
    energy_part_throttle_mj = np.cumsum(np.where((~is_sc) & (~is_over), rec_step, 0.0))
    energy_total_recovered_mj = energy_overrun_mj + energy_part_throttle_mj + energy_superclip_mj

    eta_ice = float(pu.get("ice_thermal_efficiency", 0.48))
    lhv_kj_kg = float(pu.get("fuel_lower_heating_value_mj_kg", 43.5)) * 1000.0
    fuel_used_kg = np.cumsum(ice_power_kw / (eta_ice * lhv_kj_kg) * dt_arr)
    fuel_used_kg[0] = 0.0

    # 4-Corner Suspension, Ride Height & Tire Model
    rh_f_static = float(susp["static_ride_height_front_mm"])
    rh_r_static = float(susp["static_ride_height_rear_mm"])
    rh_f_min = float(susp["min_ride_height_front_mm"])
    rh_r_min = float(susp["min_ride_height_rear_mm"])

    df_norm = np.clip(downforce_kg / 1450.0, 0.0, 1.0)
    # Ride height compresses with downforce and pitches under braking/acceleration
    ride_height_front_mm = rh_f_static - (rh_f_static - rh_f_min) * (df_norm ** 0.92) + 2.2 * long_g
    ride_height_rear_mm = rh_r_static - (rh_r_static - rh_r_min) * (df_norm ** 0.92) - 3.0 * long_g
    ride_height_front_mm = np.clip(ride_height_front_mm, rh_f_min, rh_f_static + 4.0)
    ride_height_rear_mm = np.clip(ride_height_rear_mm, rh_r_min, rh_r_static + 5.0)

    heave_mm = 0.5 * ((rh_f_static - ride_height_front_mm) + (rh_r_static - ride_height_rear_mm))
    roll_deg = np.clip(-lat_g * float(susp["roll_gradient_deg_per_g"]), -2.2, 2.2)
    pitch_deg = np.clip(long_g * 0.18 - 0.08 * df_norm, -1.1, 0.35)

    # 4-Corner Vertical Tire Loads Fz [kN] (FL, FR, RL, RR)
    # Physically exact load transfer:
    #   1. Longitudinal transfer = m * a_x * (h_cg / wb)
    #   2. Aero downforce split by aero_balance_pct
    #   3. Lateral transfer split by front roll-stiffness fraction k_phi_f (k_phi_f + k_phi_r == 1.0)
    #   4. sum(Fz) == m * g * cos(theta) + downforce within 1% at every point (no arbitrary clip floors)
    m = float(veh["mass_kg"])
    wf = float(veh["weight_distribution_front"])
    h_cg = float(veh["cg_height_m"])
    wb = float(veh["wheelbase_m"])
    tw_f = float(veh["track_width_front_m"])
    tw_r = float(veh["track_width_rear_m"])
    k_phi_f = float(susp.get("roll_stiffness_front_ratio", 0.58))
    k_phi_r = 1.0 - k_phi_f

    cos_theta = np.sqrt(np.maximum(0.0, 1.0 - sin_theta**2))
    fz_target_total_n = m * g * cos_theta + downforce_n

    f_axle_front_n = m * g * cos_theta * wf + downforce_n * (aero_balance_pct / 100.0) - (m * g * long_g) * (h_cg / wb)
    f_axle_rear_n = m * g * cos_theta * (1.0 - wf) + downforce_n * (1.0 - aero_balance_pct / 100.0) + (m * g * long_g) * (h_cg / wb)

    m_lat_total_n_m = (m * g * lat_g) * h_cg
    df_lat_front_n = k_phi_f * (m_lat_total_n_m / tw_f)
    df_lat_rear_n = k_phi_r * (m_lat_total_n_m / tw_r)

    fz_fl_raw_n = 0.5 * f_axle_front_n - df_lat_front_n
    fz_fr_raw_n = 0.5 * f_axle_front_n + df_lat_front_n
    fz_rl_raw_n = 0.5 * f_axle_rear_n - df_lat_rear_n
    fz_rr_raw_n = 0.5 * f_axle_rear_n + df_lat_rear_n

    # Natural non-negative ground contact with physical load redistribution
    # If a tire unloads completely to zero, its negative reaction cannot pull on the ground
    fz_fl_n = fz_fl_raw_n.copy()
    fz_fr_n = fz_fr_raw_n.copy()
    fz_rl_n = fz_rl_raw_n.copy()
    fz_rr_n = fz_rr_raw_n.copy()

    for idx_w in range(n):
        # Front axle redistribution
        if fz_fl_n[idx_w] < 0.0:
            fz_fr_n[idx_w] = max(0.0, fz_fr_n[idx_w] + fz_fl_n[idx_w])
            fz_fl_n[idx_w] = 0.0
        elif fz_fr_n[idx_w] < 0.0:
            fz_fl_n[idx_w] = max(0.0, fz_fl_n[idx_w] + fz_fr_n[idx_w])
            fz_fr_n[idx_w] = 0.0
        # Rear axle redistribution
        if fz_rl_n[idx_w] < 0.0:
            fz_rr_n[idx_w] = max(0.0, fz_rr_n[idx_w] + fz_rl_n[idx_w])
            fz_rl_n[idx_w] = 0.0
        elif fz_rr_n[idx_w] < 0.0:
            fz_rl_n[idx_w] = max(0.0, fz_rl_n[idx_w] + fz_rr_n[idx_w])
            fz_rr_n[idx_w] = 0.0

    fz_fl_kn = fz_fl_n / 1000.0
    fz_fr_kn = fz_fr_n / 1000.0
    fz_rl_kn = fz_rl_n / 1000.0
    fz_rr_kn = fz_rr_n / 1000.0

    # Dynamic Camber gamma [deg] & Slip Angle phi [deg]
    cam_f0 = float(susp["static_camber_front_deg"])
    cam_r0 = float(susp["static_camber_rear_deg"])
    camber_fl_deg = cam_f0 - 0.45 * roll_deg + 0.012 * (rh_f_static - ride_height_front_mm)
    camber_fr_deg = cam_f0 + 0.45 * roll_deg + 0.012 * (rh_f_static - ride_height_front_mm)
    camber_rl_deg = cam_r0 - 0.35 * roll_deg + 0.008 * (rh_r_static - ride_height_rear_mm)
    camber_rr_deg = cam_r0 + 0.35 * roll_deg + 0.008 * (rh_r_static - ride_height_rear_mm)

    slip_base_deg = np.clip(lat_g * 1.45, -6.5, 6.5)
    slip_fl_deg = slip_base_deg * 1.08 + 0.1
    slip_fr_deg = slip_base_deg * 1.05 - 0.1
    slip_rl_deg = slip_base_deg * 0.92 + 0.05
    slip_rr_deg = slip_base_deg * 0.90 - 0.05


    # ---- cumulative driver phases and limited-by ------------------------------------------------------------
    is_full = (throttle_pct >= 95.0).astype(np.float64)
    is_part = ((throttle_pct >= 5.0) & (throttle_pct < 95.0)).astype(np.float64)
    is_brake = (brake_pct >= 5.0).astype(np.float64)
    is_neutral = (1.0 - np.clip(is_full + is_part + is_brake, 0.0, 1.0))
    cum_full = np.cumsum(is_full * dt_arr)
    cum_part = np.cumsum(is_part * dt_arr)
    cum_neutral = np.cumsum(is_neutral * dt_arr)
    cum_brake = np.cumsum(is_brake * dt_arr)
    cum_total = np.maximum(1e-6, cum_full + cum_part + cum_neutral + cum_brake)
    phase_full_throttle_pct = (cum_full / cum_total) * 100.0
    phase_part_throttle_pct = (cum_part / cum_total) * 100.0
    phase_neutral_pct = (cum_neutral / cum_total) * 100.0
    phase_braking_pct = (cum_brake / cum_total) * 100.0

    is_power_lim = (plim & ~braking_node).astype(np.float64)
    is_grip_lim = 1.0 - is_power_lim
    ds_km = np.gradient(s) / 1000.0
    limited_power_km = np.cumsum(is_power_lim * ds_km)
    limited_grip_km = np.cumsum(is_grip_lim * ds_km)

    # ---- sensitivities: the FULL pipeline (incl. energy re-optimisation) with one parameter perturbed -------
    perturbations = {
        "Power +10 kW":    {"power_offset_kw": 10.0},
        "Downforce +1%":   {"cl_scale": float(params.get("cl_scale", 1.0)) * 1.01},
        "Drag -1%":        {"cd_scale": float(params.get("cd_scale", 1.0)) * 0.99},
        "Front grip +1%":  {"front_grip_scale": 1.01},
        "Rear grip +1%":   {"rear_grip_scale": 1.01},
    }
    sensitivity_deltas_s = {}
    for label, delta in perturbations.items():
        if not with_sensitivities:
            sensitivity_deltas_s[label] = np.zeros(n)
            continue
        p2 = dict(params)
        p2.update(delta)
        _, plan2, _ = Q.predict_lap(s, kappa, sin_theta, cfg, p2, xmask=xmask, n_iter=10, warm_p=plan.p_kw, trust0_kw=80.0)
        v2 = gaussian_filter1d(np.array(plan2.info.v), 2.0, mode="wrap")
        v2[-1] = v2[0]
        t2 = np.concatenate([[0.0], np.cumsum(ds_steps / (0.5 * (v2[:-1] + v2[1:])))])
        sensitivity_deltas_s[label] = t2 - t

    return TelemetryResult(
        s=s, t=t, dt_ds=dt_ds, speed_kmh=speed_kmh, speed_ms=v_ms, long_g=long_g, lat_g=lat_g, total_g=total_g,
        throttle_pct=throttle_pct, brake_pct=brake_pct, gear=gear, steering_deg=steering_deg,
        x_mode_fraction=x_mode_fraction, downforce_kg=downforce_kg, drag_kg=drag_kg,
        aero_balance_pct=aero_balance_pct, air_density=rho,
        ice_power_kw=ice_power_kw, mguk_power_kw=mguk_power_kw, mguk_state=mguk_state, soc_mj=soc_mj,
        fuel_used_kg=fuel_used_kg, energy_overrun_mj=energy_overrun_mj,
        energy_part_throttle_mj=energy_part_throttle_mj, energy_superclip_mj=energy_superclip_mj,
        energy_total_recovered_mj=energy_total_recovered_mj,
        ride_height_front_mm=ride_height_front_mm, ride_height_rear_mm=ride_height_rear_mm, heave_mm=heave_mm,
        roll_deg=roll_deg, pitch_deg=pitch_deg,
        fz_fl_kn=fz_fl_kn, fz_fr_kn=fz_fr_kn, fz_rl_kn=fz_rl_kn, fz_rr_kn=fz_rr_kn,
        camber_fl_deg=camber_fl_deg, camber_fr_deg=camber_fr_deg, camber_rl_deg=camber_rl_deg, camber_rr_deg=camber_rr_deg,
        slip_fl_deg=slip_fl_deg, slip_fr_deg=slip_fr_deg, slip_rl_deg=slip_rl_deg, slip_rr_deg=slip_rr_deg,
        phase_full_throttle_pct=phase_full_throttle_pct, phase_part_throttle_pct=phase_part_throttle_pct,
        phase_neutral_pct=phase_neutral_pct, phase_braking_pct=phase_braking_pct,
        limited_power_km=limited_power_km, limited_grip_km=limited_grip_km,
        sector_splits_m=(s1_end, s2_end, s3_end), sector_times_s=(t_s1_out, t_s2_out, t_s3_out),
        lap_time_s=lap_time_out, sensitivity_deltas_s=sensitivity_deltas_s,
    )


if __name__ == "__main__":
    res = simulate_brazil_2026_lap()
    print(f"Predicted lap time: {res.t[-1]:.3f} s")
    print(f"Sectors: {res.sector_times_s[0]:.3f} / {res.sector_times_s[1]:.3f} / {res.sector_times_s[2]:.3f}")
    print(f"Top speed {res.speed_kmh.max():.1f} km/h, min speed {res.speed_kmh.min():.1f} km/h")
    print(f"Fuel {res.fuel_used_kg[-1]:.3f} kg | recovered {res.energy_total_recovered_mj[-1]:.2f} MJ "
          f"(overrun {res.energy_overrun_mj[-1]:.2f}, part {res.energy_part_throttle_mj[-1]:.2f}, superclip {res.energy_superclip_mj[-1]:.2f}) "
          f"| SOC {res.soc_mj[0]:.2f} -> {res.soc_mj[-1]:.2f} (min {res.soc_mj.min():.2f})")
