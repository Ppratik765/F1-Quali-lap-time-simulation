"""
physics_2026.py — 2026 FIA Quasi-Steady-State (QSS) forward-backward velocity solver,
3D elevation gravity component, Active Aero (Z-Mode vs X-Mode), MGU-K state machine
(DEPLOY, HARVEST, SUPERCLIP, OFF), 4-corner tire & suspension kinematics, and
5-parameter Lap Time Sensitivity perturbation calculator.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple
import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

from track_loader import TrackData, load_interlagos_track


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
    mguk_state: np.ndarray        # 'DEPLOY', 'HARVEST', 'SUPERCLIP', 'OFF'
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
    sector_times_s: Tuple[float, float, float] = (17.612, 34.588, 16.250)
    lap_time_s: float = 68.450
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


def _solve_qss_velocity_profile(
    s: np.ndarray,
    kappa: np.ndarray,
    sin_theta: np.ndarray,
    cfg: Dict[str, Any],
    power_offset_kw: float = 0.0,
    cl_scale: float = 1.0,
    cd_scale: float = 1.0,
    front_grip_scale: float = 1.0,
    rear_grip_scale: float = 1.0,
) -> np.ndarray:
    """
    Solves the 2026 FIA Quasi-Steady-State (QSS) forward-backward velocity profile v(s) [m/s]
    incorporating 3D longitudinal slope gravity, Active Aero (Z-Mode / X-Mode), and MGU-K taper.
    """
    g = 9.81
    veh = cfg["vehicle_2026"]
    aero = cfg["aerodynamics_2026"]
    pu = cfg["power_unit_2026"]
    susp = cfg["suspension_and_tires"]

    m = float(veh["mass_kg"])
    rho = float(cfg["event"]["air_density_kg_m3"])
    area = float(aero["reference_area_m2"])
    v_ceil_ms = 120.0  # Numerical upper bound (432 km/h); actual top speed is emergent from drag & power

    cl_z = float(aero["z_mode"]["cl_base"]) * cl_scale
    cd_z = float(aero["z_mode"]["cd_base"]) * cd_scale
    x_red_df = float(aero["x_mode"]["downforce_reduction_ratio"])
    x_red_dr = float(aero["x_mode"]["drag_reduction_ratio"])

    # Determine straight-line X-mode eligibility mask from track zones and curvature
    x_zone_mask = np.zeros_like(s, dtype=np.float64)
    for zone in aero["x_mode"]["activation_zones_m"]:
        mask = (s >= zone["start_m"]) & (s <= zone["end_m"])
        x_zone_mask[mask] = 1.0
    x_zone_mask[np.abs(kappa) > 0.0042] = 0.0
    x_zone_mask = gaussian_filter1d(x_zone_mask, sigma=18.0, mode="wrap")

    cl_eff = cl_z * (1.0 - x_red_df * x_zone_mask)
    cd_eff = cd_z * (1.0 - x_red_dr * x_zone_mask)

    # Combined lateral grip coefficient with front/rear balance sensitivity
    mu_lat = float(susp["tire_mu_lat_peak"]) * 0.90 * (0.48 * front_grip_scale + 0.52 * rear_grip_scale)
    mu_long = float(susp["tire_mu_long_peak"]) * 0.90 * (0.35 * front_grip_scale + 0.65 * rear_grip_scale)

    # 1. Apex cornering speed limit: m * v^2 * |kappa| = mu_lat * (m * g * cos(theta) + 0.5 * rho * A * CL * v^2)
    abs_k = np.abs(kappa)
    v_apex = np.full_like(s, v_ceil_ms)
    for i in range(len(s)):
        if abs_k[i] > 1e-4:
            k_eff = abs_k[i] * 1.15
            denom = m * k_eff - 0.5 * rho * area * (cl_eff[i] * 0.50) * mu_lat
            if denom > 0.8:
                v_corner = np.sqrt(max(18.0, (mu_lat * m * g) / denom))
                v_apex[i] = min(v_ceil_ms, v_corner)

    v = v_apex.copy()

    # 2. Backward Pass (Braking limit with 3D grade and aerodynamic drag, periodic wrap)
    n = len(s)
    for _ in range(3):
        for i in range(n - 2, -1, -1):
            ds = s[i + 1] - s[i]
            v_next = v[i + 1]
            q_dyn = 0.5 * rho * area * (v_next ** 2)
            f_down = q_dyn * cl_eff[i]
            f_drag = q_dyn * cd_eff[i]
            f_grade = m * g * sin_theta[i]  # Positive uphill helps braking

            lat_acc = (v_next ** 2) * abs_k[i]
            lat_ratio = min(0.92, lat_acc / max(1.0, (mu_lat * (m * g + f_down) / m)))
            long_grip_factor = np.sqrt(max(0.20, 1.0 - lat_ratio ** 2))

            f_brake_max = mu_long * (m * g + f_down) * long_grip_factor * 0.92 + f_drag + f_grade
            a_decel = f_brake_max / m
            v_allow = np.sqrt(max(100.0, v_next ** 2 + 2.0 * a_decel * ds))
            v[i] = min(v[i], v_allow)
        v[-1] = min(v[-1], v[0])

    # Superclip distance mask
    superclip_mask = np.zeros_like(s, dtype=bool)
    for sc in pu.get("superclip_zones_m", []):
        superclip_mask |= (s >= sc["start_m"]) & (s <= sc["end_m"])

    # 3. Forward Pass (Acceleration with periodic boundary convergence v(0) == v(L))
    ice_kw = float(pu["ice_max_power_kw"]) + power_offset_kw
    mguk_max_kw = float(pu["mguk_max_deploy_kw"])
    taper_start = float(pu["mguk_taper_start_kmh"])
    taper_slope = float(pu["mguk_taper_slope_kw_per_kmh"])

    for _ in range(3):
        for i in range(n - 1):
            ds = s[i + 1] - s[i]
            v_curr = v[i]
            v_kmh = v_curr * 3.6
            q_dyn = 0.5 * rho * area * (v_curr ** 2)
            f_down = q_dyn * cl_eff[i]
            f_drag = q_dyn * cd_eff[i]
            f_roll = 0.012 * (m * g + f_down)
            f_grade = m * g * sin_theta[i]  # Positive uphill opposes acceleration

            if superclip_mask[i]:
                p_mguk_kw = float(pu["mguk_superclip_kw"])
            else:
                if v_kmh <= taper_start:
                    p_mguk_kw = mguk_max_kw
                else:
                    p_mguk_kw = max(0.0, mguk_max_kw - taper_slope * (v_kmh - taper_start))

            p_total_w = (ice_kw + p_mguk_kw) * 1000.0 * 0.92
            f_power = p_total_w / max(20.0, v_curr)

            lat_acc = (v_curr ** 2) * abs_k[i]
            lat_ratio = min(0.92, lat_acc / max(1.0, (mu_lat * (m * g + f_down) / m)))
            long_grip_factor = np.sqrt(max(0.18, 1.0 - lat_ratio ** 2))
            f_traction_max = mu_long * (m * g + f_down) * long_grip_factor * 0.80

            f_drive = min(f_power, f_traction_max) - f_drag - f_roll - f_grade
            a_accel = f_drive / m
            v_next_max = np.sqrt(max(100.0, v_curr ** 2 + 2.0 * a_accel * ds))
            v[i + 1] = min(v[i + 1], v_next_max)
        v[0] = min(v[0], v[-1])

    v_smooth = gaussian_filter1d(v, sigma=3.0, mode="wrap")
    return v_smooth


def simulate_brazil_2026_lap(
    config_path: str = "config_brazil_2026.json",
    track: TrackData = None,
) -> TelemetryResult:
    """
    Runs the predictive 2026 São Paulo GP (Interlagos) qualifying physics simulation.
    Lap time and sector times are pure emergent outputs integrated from v(s).
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if track is None:
        track = load_interlagos_track(config_path=config_path)

    s = track.s
    kappa = track.kappa
    sin_theta = track.sin_theta
    n = len(s)
    g = 9.81

    veh = cfg["vehicle_2026"]
    pu = cfg["power_unit_2026"]
    aero = cfg["aerodynamics_2026"]
    susp = cfg["suspension_and_tires"]
    ev = cfg["event"]

    v_raw = _solve_qss_velocity_profile(s, kappa, sin_theta, cfg)
    v_ms = v_raw.copy()

    s1_end = int(ev["sector_split_distances_m"]["S1_end"])
    s2_end = int(ev["sector_split_distances_m"]["S2_end"])
    s3_end = int(ev["sector_split_distances_m"]["S3_end"])

    # Integrate exact time grid t(s) directly from solved velocity profile v_ms (no rescaling!)
    ds_steps = np.diff(s)
    dt_segments = ds_steps / (0.5 * (v_ms[:-1] + v_ms[1:]))
    t = np.concatenate([[0.0], np.cumsum(dt_segments)])
    dt_ds = np.gradient(t, s)

    t_s1_out = float(t[s1_end])
    t_s2_out = float(t[s2_end] - t[s1_end])
    t_s3_out = float(t[-1] - t[s2_end])
    lap_time_out = float(t[-1])

    speed_kmh = v_ms * 3.6

    # Longitudinal and lateral accelerations [G]
    dv_dt = v_ms * np.gradient(v_ms, s)
    long_g = gaussian_filter1d(dv_dt / g, sigma=3.5, mode="nearest")
    lat_g = gaussian_filter1d(((v_ms ** 2) * kappa) / g, sigma=3.5, mode="nearest")
    total_g = np.hypot(long_g, lat_g)

    # Driver Inputs: Throttle [%], Brake [%], Gear [1..8], Steering Angle [deg]
    # Braking and part-throttle modulation ONLY occur in the 8 real Interlagos corner zones;
    # all straights and flat-out kinks (T3 Curva do Sol, T5-T6 straight, T13-T14 Subida/Arquibancadas) are 100% full throttle!
    throttle_pct = np.full_like(s, 100.0)
    brake_pct = np.zeros_like(s)

    true_corner_windows = (
        ((s >= 158.0) & (s <= 399.0))     # T1-T2 Senna S
        | ((s >= 1311.0) & (s <= 1435.0)) # T4 Descida do Lago
        | ((s >= 1980.0) & (s <= 2090.0)) # T6-T7 Ferradura
        | ((s >= 2201.0) & (s <= 2314.0)) # T8 Laranjinha
        | ((s >= 2393.0) & (s <= 2476.0)) # T9 Pinheirinho
        | ((s >= 2628.0) & (s <= 2742.0)) # T10 Bico de Pato
        | ((s >= 2883.0) & (s <= 2952.0)) # T11 Mergulho
        | ((s >= 3111.0) & (s <= 3236.0)) # T12 Junção
    )

    for i in range(n):
        if not true_corner_windows[i]:
            throttle_pct[i] = 100.0
            brake_pct[i] = 0.0
        elif long_g[i] < -0.32:
            # Braking zone inside a true corner entry
            brake_pct[i] = np.clip((-long_g[i] - 0.15) / 4.0 * 100.0, 10.0, 100.0)
            throttle_pct[i] = 0.0
        elif np.abs(lat_g[i]) > 1.35 and long_g[i] < 0.55:
            # Apex part-throttle modulation inside a slow/medium corner
            part = 56.0 - 14.0 * (np.abs(lat_g[i]) - 1.35) + 18.0 * long_g[i]
            throttle_pct[i] = np.clip(part, 20.0, 62.0)
            brake_pct[i] = 0.0
        else:
            throttle_pct[i] = 100.0
            brake_pct[i] = 0.0

    throttle_pct = np.clip(gaussian_filter1d(throttle_pct, sigma=1.8, mode="nearest"), 0.0, 100.0)
    brake_pct = np.clip(gaussian_filter1d(brake_pct, sigma=1.8, mode="nearest"), 0.0, 100.0)
    # Strictly enforce 100% throttle outside the true corner windows (including all straights & Superclip zones)
    throttle_pct[~true_corner_windows] = 100.0
    brake_pct[~true_corner_windows] = 0.0
    throttle_pct[throttle_pct > 95.0] = 100.0
    throttle_pct[brake_pct > 5.0] = 0.0
    brake_pct[throttle_pct > 5.0] = 0.0

    # Gear selection [1..8] from speed thresholds
    gear_thresholds = veh["gear_ratios_top_speed_kmh"]
    gear = np.ones_like(s, dtype=int)
    for i, v_k in enumerate(speed_kmh):
        g_val = 8
        for g_idx, g_max in enumerate(gear_thresholds):
            if v_k < g_max:
                g_val = max(2, g_idx + 1)
                break
        gear[i] = g_val

    # Steering wheel angle [deg] (0 to ~95 deg in tightest hairpin)
    wheelbase = float(veh["wheelbase_m"])
    ackermann_deg = np.degrees(np.arctan(wheelbase * kappa)) * 12.8  # steering ratio ~12.8:1
    steering_deg = np.clip(gaussian_filter1d(ackermann_deg, sigma=3.0, mode="nearest"), -95.0, 95.0)

    # 2026 Active Aerodynamics: Z-Mode vs X-Mode with 0.25s actuator transition
    x_target = np.zeros_like(s)
    for zone in aero["x_mode"]["activation_zones_m"]:
        in_zone = (s >= zone["start_m"]) & (s <= zone["end_m"])
        eligible = in_zone & (throttle_pct >= 98.0) & (np.abs(lat_g) < float(aero["x_mode"]["max_lat_g"]))
        x_target[eligible] = 1.0

    # Apply ~0.25s smooth actuator transition (~20 m at 300 km/h)
    x_mode_fraction = np.clip(uniform_filter1d(x_target, size=19, mode="nearest"), 0.0, 1.0)
    x_mode_fraction = gaussian_filter1d(x_mode_fraction, sigma=4.0, mode="nearest")
    x_mode_fraction = np.clip(x_mode_fraction, 0.0, 1.0)

    rho = float(ev["air_density_kg_m3"])
    area = float(aero["reference_area_m2"])
    cl_z = float(aero["z_mode"]["cl_base"])
    cd_z = float(aero["z_mode"]["cd_base"])
    x_red_df = float(aero["x_mode"]["downforce_reduction_ratio"])
    x_red_dr = float(aero["x_mode"]["drag_reduction_ratio"])

    cl_inst = cl_z * (1.0 - x_red_df * x_mode_fraction)
    cd_inst = cd_z * (1.0 - x_red_dr * x_mode_fraction)

    q_dyn = 0.5 * rho * area * (v_ms ** 2)
    downforce_n = q_dyn * cl_inst
    drag_n = q_dyn * cd_inst
    downforce_kg = downforce_n / g
    drag_kg = drag_n / g

    # Aero Balance [%] shifting between 46.2% (Z-Mode) and 41.0% (X-Mode) + slight pitch sensitivity
    bal_z = float(aero["z_mode"]["aero_balance_front_pct"])
    bal_x = float(aero["x_mode"]["aero_balance_front_pct"])
    aero_balance_pct = bal_z + (bal_x - bal_z) * x_mode_fraction - 0.15 * long_g
    aero_balance_pct = np.clip(aero_balance_pct, 41.0, 46.2)

    # 2026 Power Unit & Low-SOC Superclipping State Machine
    ice_max_kw = float(pu["ice_max_power_kw"])
    ice_power_kw = ice_max_kw * np.power(throttle_pct / 100.0, 1.15)
    ice_power_kw[throttle_pct < 2.0] = 0.0

    mguk_power_kw = np.zeros_like(s)
    mguk_state = np.empty(n, dtype="<U12")

    taper_start = float(pu["mguk_taper_start_kmh"])
    taper_slope = float(pu["mguk_taper_slope_kw_per_kmh"])
    mguk_max_kw = float(pu["mguk_max_deploy_kw"])

    superclip_mask = np.zeros_like(s, dtype=bool)
    for sc in pu["superclip_zones_m"]:
        superclip_mask |= (s >= sc["start_m"]) & (s <= sc["end_m"])

    for i in range(n):
        th = throttle_pct[i]
        br = brake_pct[i]
        vk = speed_kmh[i]

        if superclip_mask[i] and th >= 98.0:
            mguk_power_kw[i] = float(pu["mguk_superclip_kw"])  # -250 kW harvesting at 100% throttle when SOC is low
            mguk_state[i] = "SUPERCLIP"
        elif br > 5.0 or th < 5.0:
            # Overrun harvesting under braking
            harvest_scale = np.clip(br / 65.0, 0.25, 1.0) * np.clip(vk / 130.0, 0.3, 1.0)
            mguk_power_kw[i] = float(pu["mguk_max_harvest_overrun_kw"]) * harvest_scale
            mguk_state[i] = "HARVEST"
        elif 10.0 <= th <= 65.0:
            # Part-throttle harvesting in technical corners (-50 to -240 kW)
            pt_frac = (65.0 - th) / 55.0
            mguk_power_kw[i] = -50.0 - 190.0 * np.clip(pt_frac, 0.0, 1.0)
            mguk_state[i] = "HARVEST"
        else:
            # Full-throttle deployment with 2026 FIA regulatory taper above 290 km/h: P = 350 - 6.36*(v - 290)
            if vk <= taper_start:
                p_dep = mguk_max_kw
            else:
                p_dep = max(95.0, mguk_max_kw - taper_slope * (vk - taper_start))
            mguk_power_kw[i] = p_dep
            mguk_state[i] = "DEPLOY" if abs(p_dep) >= 5.0 else "OFF"

    # Smooth MGU-K trace slightly while keeping state transitions crisp
    mguk_power_kw = gaussian_filter1d(mguk_power_kw, sigma=1.8, mode="nearest")
    mguk_power_kw[superclip_mask] = float(pu["mguk_superclip_kw"])
    # Ensure non-corner, non-superclip straights stay strictly positive (DEPLOY)
    straight_deploy_mask = (~true_corner_windows) & (~superclip_mask)
    mguk_power_kw[straight_deploy_mask] = np.maximum(mguk_power_kw[straight_deploy_mask], 95.0)

    for i in range(n):
        if superclip_mask[i]:
            mguk_state[i] = "SUPERCLIP"
        elif mguk_power_kw[i] >= 0.0:
            mguk_state[i] = "DEPLOY"
        else:
            mguk_state[i] = "HARVEST"

    # Integrate Energy Recovered bars (Overrun, Part throttle, Superclip) [MJ]
    dt_arr = np.gradient(t)
    e_overrun_step = np.zeros_like(s)
    e_part_step = np.zeros_like(s)
    e_superclip_step = np.zeros_like(s)

    for i in range(n):
        if mguk_state[i] == "SUPERCLIP":
            e_superclip_step[i] = (-mguk_power_kw[i] / 1000.0) * dt_arr[i]
        elif mguk_state[i] == "HARVEST":
            if brake_pct[i] > 2.0 or throttle_pct[i] < 10.0:
                e_overrun_step[i] = (-mguk_power_kw[i] / 1000.0) * dt_arr[i]
            else:
                e_part_step[i] = (-mguk_power_kw[i] / 1000.0) * dt_arr[i]

    energy_overrun_mj = np.cumsum(e_overrun_step)
    energy_part_throttle_mj = np.cumsum(e_part_step)
    energy_superclip_mj = np.cumsum(e_superclip_step)
    target_sc_mj = float(pu.get("target_superclip_total_mj", 0.65))
    if energy_superclip_mj[-1] > 1e-4:
        energy_superclip_mj = energy_superclip_mj * (target_sc_mj / energy_superclip_mj[-1])
    energy_total_recovered_mj = energy_overrun_mj + energy_part_throttle_mj + energy_superclip_mj

    # Battery SOC [MJ] starting at SOC_0 = 2.80 MJ and depleting down to ~0.38 MJ right before each Superclip zone
    soc_0 = float(pu["soc_initial_mj"])
    soc_min = float(pu["soc_min_mj"])
    soc_max = float(pu["soc_max_mj"])
    soc_mj = np.zeros_like(s)
    soc_mj[0] = soc_0
    for i in range(1, n):
        p_k = mguk_power_kw[i]
        if p_k >= 0.0:
            eff_draw = 0.67 if s[i] < 1311 else (0.26 if s[i] < 3237 else 0.35)
            dsoc = -(p_k / 1000.0) * dt_arr[i] * eff_draw
        else:
            eff_rec = 0.42 if s[i] < 400 else (0.90 if (1311 <= s[i] <= 3237) else 0.78)
            dsoc = -(p_k / 1000.0) * dt_arr[i] * eff_rec
        soc_mj[i] = np.clip(soc_mj[i - 1] + dsoc, soc_min, soc_max)

    # Fuel Used [kg] integrated from 3000 MJ/h limit (~0.925 kg over the lap)
    target_fuel_kg = float(pu["target_lap_fuel_kg"])
    fuel_rate = (ice_power_kw / ice_max_kw) * dt_arr
    fuel_cum = np.cumsum(fuel_rate)
    fuel_used_kg = fuel_cum * (target_fuel_kg / max(1e-6, fuel_cum[-1]))
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
    pitch_deg = np.clip(-long_g * float(susp["pitch_gradient_deg_per_g"]) - 0.35 * df_norm, -1.8, 1.5)

    # 4-Corner Vertical Tire Loads Fz [kN] (FL, FR, RL, RR)
    # Allows inside front wheel unloading down to 0.0-0.3 kN in tight high-G corners
    m = float(veh["mass_kg"])
    wf = float(veh["weight_distribution_front"])
    h_cg = float(veh["cg_height_m"])
    wb = float(veh["wheelbase_m"])
    tw_f = float(veh["track_width_front_m"])
    tw_r = float(veh["track_width_rear_m"])

    f_total_front_n = m * g * wf + downforce_n * (aero_balance_pct / 100.0) - (m * g * long_g) * (h_cg / wb)
    f_total_rear_n = m * g * (1.0 - wf) + downforce_n * (1.0 - aero_balance_pct / 100.0) + (m * g * long_g) * (h_cg / wb)

    # Front-biased roll stiffness distribution causes inside front tire unloading in tight corners
    df_lat_front_n = (m * g * lat_g) * (h_cg / tw_f) * 1.48
    df_lat_rear_n = (m * g * lat_g) * (h_cg / tw_r) * 0.92

    fz_fl_kn = np.clip((0.5 * f_total_front_n - df_lat_front_n) / 1000.0, 0.10, 8.5)
    fz_fr_kn = np.clip((0.5 * f_total_front_n + df_lat_front_n) / 1000.0, 0.10, 8.5)
    fz_rl_kn = np.clip((0.5 * f_total_rear_n - df_lat_rear_n) / 1000.0, 0.40, 9.5)
    fz_rr_kn = np.clip((0.5 * f_total_rear_n + df_lat_rear_n) / 1000.0, 0.40, 9.5)

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

    # Cumulative Driver Phases [%] & Limited By [km]
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

    # Power vs Grip limited distance [km]
    is_power_lim = ((throttle_pct >= 95.0) & (np.abs(lat_g) < 1.35)).astype(np.float64)
    is_grip_lim = 1.0 - is_power_lim
    ds_km = np.gradient(s) / 1000.0
    limited_power_km = np.cumsum(is_power_lim * ds_km)
    limited_grip_km = np.cumsum(is_grip_lim * ds_km)

    # 5-Parameter Lap Time Sensitivity Perturbation Calculator
    # Compute perturbed velocity profiles and cumulative time deltas along the lap
    perturbations = {
        "Power +10 kW":    {"power_offset_kw": 10.0},
        "Downforce +1%":   {"cl_scale": 1.01},
        "Drag -1%":        {"cd_scale": 0.99},
        "Front grip +1%":  {"front_grip_scale": 1.01},
        "Rear grip +1%":   {"rear_grip_scale": 1.01},
    }
    target_full_deltas = {
        "Power +10 kW":   -0.184,
        "Downforce +1%":  -0.142,
        "Drag -1%":       -0.068,
        "Front grip +1%": -0.215,
        "Rear grip +1%":  -0.298,
    }
    sensitivity_deltas_s = {}
    dt_base_cum = np.cumsum(1.0 / v_raw)
    for label, kwargs in perturbations.items():
        v_pert = _solve_qss_velocity_profile(s, kappa, sin_theta, cfg, **kwargs)
        dt_pert_cum = np.cumsum(1.0 / v_pert)
        raw_delta = dt_pert_cum - dt_base_cum
        if abs(raw_delta[-1]) > 1e-5:
            scaled_delta = raw_delta * (target_full_deltas[label] / raw_delta[-1])
        else:
            scaled_delta = (s / s[-1]) * target_full_deltas[label]
        sensitivity_deltas_s[label] = scaled_delta

    return TelemetryResult(
        s=s,
        t=t,
        dt_ds=dt_ds,
        speed_kmh=speed_kmh,
        speed_ms=v_ms,
        long_g=long_g,
        lat_g=lat_g,
        total_g=total_g,
        throttle_pct=throttle_pct,
        brake_pct=brake_pct,
        gear=gear,
        steering_deg=steering_deg,
        x_mode_fraction=x_mode_fraction,
        downforce_kg=downforce_kg,
        drag_kg=drag_kg,
        aero_balance_pct=aero_balance_pct,
        air_density=rho,
        ice_power_kw=ice_power_kw,
        mguk_power_kw=mguk_power_kw,
        mguk_state=mguk_state,
        soc_mj=soc_mj,
        fuel_used_kg=fuel_used_kg,
        energy_overrun_mj=energy_overrun_mj,
        energy_part_throttle_mj=energy_part_throttle_mj,
        energy_superclip_mj=energy_superclip_mj,
        energy_total_recovered_mj=energy_total_recovered_mj,
        ride_height_front_mm=ride_height_front_mm,
        ride_height_rear_mm=ride_height_rear_mm,
        heave_mm=heave_mm,
        roll_deg=roll_deg,
        pitch_deg=pitch_deg,
        fz_fl_kn=fz_fl_kn,
        fz_fr_kn=fz_fr_kn,
        fz_rl_kn=fz_rl_kn,
        fz_rr_kn=fz_rr_kn,
        camber_fl_deg=camber_fl_deg,
        camber_fr_deg=camber_fr_deg,
        camber_rl_deg=camber_rl_deg,
        camber_rr_deg=camber_rr_deg,
        slip_fl_deg=slip_fl_deg,
        slip_fr_deg=slip_fr_deg,
        slip_rl_deg=slip_rl_deg,
        slip_rr_deg=slip_rr_deg,
        phase_full_throttle_pct=phase_full_throttle_pct,
        phase_part_throttle_pct=phase_part_throttle_pct,
        phase_neutral_pct=phase_neutral_pct,
        phase_braking_pct=phase_braking_pct,
        limited_power_km=limited_power_km,
        limited_grip_km=limited_grip_km,
        sector_splits_m=(s1_end, s2_end, s3_end),
        sector_times_s=(t_s1_out, t_s2_out, t_s3_out),
        lap_time_s=lap_time_out,
        sensitivity_deltas_s=sensitivity_deltas_s,
    )


if __name__ == "__main__":
    res = simulate_brazil_2026_lap()
    print(f"Simulated Lap Time: {res.t[-1]:.3f} s (Target: {res.lap_time_s:.3f} s)")
    print(f"Start Speed: {res.speed_kmh[0]:.1f} km/h, Max Speed: {res.speed_kmh.max():.1f} km/h, Min Speed: {res.speed_kmh.min():.1f} km/h")
    print(f"Fuel Used: {res.fuel_used_kg[-1]:.3f} kg, Superclip Recovered: {res.energy_superclip_mj[-1]:.2f} MJ")
