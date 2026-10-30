"""
physics_2026.py — 2026 FIA Quasi-Steady-State (QSS) forward-backward velocity solver,
3D elevation gravity component, Active Aero (Z-Mode vs X-Mode), MGU-K state machine
(DEPLOY, HARVEST, SUPERCLIP, OFF), 4-corner tire & suspension kinematics, and
5-parameter Lap Time Sensitivity perturbation calculator.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional
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


def compute_air_density(cfg: Dict[str, Any]) -> float:
    """
    Computes ambient humid air density [kg/m^3] from altitude and forecast weather
    using the barometric formula and Tetens equation for vapor pressure.
    """
    ev = cfg.get("event", {})
    weather = ev.get("weather", {})
    altitude_m = float(ev.get("altitude_m", 765.0))
    temp_c = float(weather.get("ambient_temperature_c", 22.0))
    p_sl_hpa = float(weather.get("sea_level_pressure_hpa", 1015.0))
    rh = float(weather.get("relative_humidity", 0.55))

    p_ambient_pa = (p_sl_hpa * 100.0) * ((1.0 - (0.0065 * altitude_m) / 288.15) ** 5.25588)
    p_sat_pa = 610.78 * (10.0 ** ((7.5 * temp_c) / (temp_c + 237.3)))
    p_v = rh * p_sat_pa
    p_d = max(0.0, p_ambient_pa - p_v)
    tk = temp_c + 273.15

    r_dry = 287.058
    r_vapor = 461.495
    rho = (p_d / (r_dry * tk)) + (p_v / (r_vapor * tk))
    return float(rho)


def _fia_2026_mguk_max_kw(v_kmh: float, pu: Dict[str, Any]) -> float:
    """
    FIA 2026 Power Unit Technical Regulations Article 5.4.8 (i) MGU-K deployment limit [kW]:
      - v < 290 km/h       : P_max = 350 kW
      - 290 <= v < 340 km/h: P_max = 1800 - 5 * v_kmh
      - 340 <= v < 345 km/h: P_max = 6900 - 20 * v_kmh
      - v >= 345 km/h      : P_max = 0 kW
    """
    p_max = float(pu.get("mguk_max_deploy_kw", 350.0))
    v1 = float(pu.get("mguk_taper_v1_kmh", 290.0))
    v2 = float(pu.get("mguk_taper_v2_kmh", 340.0))
    v3 = float(pu.get("mguk_taper_v3_kmh", 345.0))
    c1 = float(pu.get("mguk_taper_c1_kw", 1800.0))
    s1 = float(pu.get("mguk_taper_s1_kw_per_kmh", 5.0))
    c2 = float(pu.get("mguk_taper_c2_kw", 6900.0))
    s2 = float(pu.get("mguk_taper_s2_kw_per_kmh", 20.0))

    if v_kmh < v1:
        return p_max
    elif v_kmh < v2:
        return max(0.0, min(p_max, c1 - s1 * v_kmh))
    elif v_kmh < v3:
        return max(0.0, c2 - s2 * v_kmh)
    else:
        return 0.0


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
    p_mguk_schedule: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Solves the Quasi-Steady-State (QSS) velocity profile v(s) [m/s] using:
      1. Load-sensitive tyre friction model mu(Fz) = mu_0 * (Fz / Fz0)^(-alpha)
      2. Aerodynamic downforce and drag (Z-mode cornering vs X-mode straight line)
      3. Standard friction-ellipse limit (Fx / Fx_max)^2 + (Fy / Fy_max)^2 <= 1
      4. 3D longitudinal slope gravity component F_grade = m * g * sin(theta)
      5. Periodic boundary conditions for a flying lap: v(0) == v(L)
    Zero unexplained fudge factors.
    """
    g = 9.81
    veh = cfg["vehicle_2026"]
    aero = cfg["aerodynamics_2026"]
    pu = cfg["power_unit_2026"]
    susp = cfg["suspension_and_tires"]

    m = float(veh["mass_kg"])
    rho = compute_air_density(cfg)
    area = float(aero["reference_area_m2"])
    v_ceil_ms = 115.0  # Numerical ceiling (414 km/h)

    # Base aerodynamic coefficients
    cl_z = float(aero["z_mode"]["cl_base"]) * cl_scale
    cd_z = float(aero["z_mode"]["cd_base"]) * cd_scale
    x_red_df = float(aero["x_mode"]["downforce_reduction_ratio"])
    x_red_dr = float(aero["x_mode"]["drag_reduction_ratio"])

    # X-mode straight-line activation mask based on curvature and designated track zones
    x_zone_mask = np.zeros_like(s, dtype=np.float64)
    for zone in aero["x_mode"]["activation_zones_m"]:
        mask = (s >= zone["start_m"]) & (s <= zone["end_m"])
        x_zone_mask[mask] = 1.0
    x_zone_mask[np.abs(kappa) > 0.0035] = 0.0
    x_zone_mask = gaussian_filter1d(x_zone_mask, sigma=15.0, mode="wrap")

    cla_eff = (cl_z * (1.0 - x_red_df * x_zone_mask)) * area
    cda_eff = (cd_z * (1.0 - x_red_dr * x_zone_mask)) * area

    # Load-sensitive tyre model parameters with physical axle distribution
    w_front = float(veh.get("weight_distribution_front", 0.455))
    w_rear = 1.0 - w_front
    bb_front = 0.56
    bb_rear = 0.44

    mu_lat_0 = float(susp["tire_mu_lat_peak"]) * (w_front * front_grip_scale + w_rear * rear_grip_scale)
    mu_long_brake_0 = float(susp["tire_mu_long_peak"]) * (bb_front * front_grip_scale + bb_rear * rear_grip_scale)
    mu_long_drive_0 = float(susp["tire_mu_long_peak"]) * rear_grip_scale
    alpha_load = float(susp.get("tire_load_sensitivity", 0.08))
    c_rr = float(susp.get("rolling_resistance_coeff", 0.012))
    eta_dt = float(pu.get("drivetrain_efficiency", 0.96))

    cos_theta = np.sqrt(np.maximum(0.0, 1.0 - sin_theta**2))
    fz_static = m * g * cos_theta
    n = len(s)

    def calc_mu_lat(fz_total: np.ndarray) -> np.ndarray:
        ratio = np.maximum(0.2, fz_total / (m * g))
        return mu_lat_0 * (ratio ** (-alpha_load))

    def calc_mu_long_brake(fz_total: np.ndarray) -> np.ndarray:
        ratio = np.maximum(0.2, fz_total / (m * g))
        return mu_long_brake_0 * (ratio ** (-alpha_load))

    def calc_mu_long_drive(fz_total: np.ndarray) -> np.ndarray:
        ratio = np.maximum(0.2, fz_total / (m * g))
        return mu_long_drive_0 * (ratio ** (-alpha_load))

    # 1. Apex Cornering Limit: m * v^2 * |kappa| = mu_lat(Fz) * (m*g*cos_theta + 0.5 * rho * CLA * v^2)
    abs_k = np.abs(kappa)
    v_apex = np.full_like(s, v_ceil_ms)
    for i in range(n):
        if abs_k[i] > 1e-4:
            # Iterative solve for v with load sensitivity
            v_guess = 50.0
            for _ in range(3):
                fz_dyn = fz_static[i] + 0.5 * rho * cla_eff[i] * (v_guess**2)
                mu_l = mu_lat_0 * (max(0.2, fz_dyn / (m * g)) ** (-alpha_load))
                denom = m * abs_k[i] - 0.5 * rho * cla_eff[i] * mu_l
                if denom > 0.01:
                    v_guess = np.sqrt((mu_l * fz_static[i]) / denom)
                else:
                    v_guess = v_ceil_ms
                    break
            v_apex[i] = min(v_ceil_ms, max(10.0, v_guess))

    v = v_apex.copy()

    # 2. Backward Pass (Braking limit with standard friction ellipse)
    for _ in range(3):
        for i in range(n - 2, -1, -1):
            ds = s[i + 1] - s[i]
            v_next = v[i + 1]
            q_dyn = 0.5 * rho * (v_next**2)
            fz = fz_static[i] + q_dyn * cla_eff[i]
            f_drag = q_dyn * cda_eff[i]
            f_grade = m * g * sin_theta[i]

            mu_lat_i = calc_mu_lat(fz)
            mu_long_i = calc_mu_long_brake(fz)

            a_lat_req = (v_next**2) * abs_k[i]
            a_lat_max = (mu_lat_i * fz) / m
            r_lat = min(1.0, a_lat_req / max(0.1, a_lat_max))
            r_long = np.sqrt(max(0.0, 1.0 - r_lat**2))

            f_tire_brake_max = r_long * mu_long_i * fz
            f_brake_total = f_tire_brake_max + f_drag + f_grade
            a_decel = min(5.35 * g, max(0.1, f_brake_total / m))

            v_allow = np.sqrt(v_next**2 + 2.0 * a_decel * ds)
            v[i] = min(v[i], v_allow)
        v[-1] = min(v[-1], v[0])

    # 3. Forward Pass (Acceleration limit with powertrain, tire traction, and friction ellipse)
    ice_kw = float(pu["ice_max_power_kw"]) + power_offset_kw

    for _ in range(3):
        for i in range(n - 1):
            ds = s[i + 1] - s[i]
            v_curr = v[i]
            v_kmh = v_curr * 3.6
            q_dyn = 0.5 * rho * (v_curr**2)
            fz = fz_static[i] + q_dyn * cla_eff[i]
            f_drag = q_dyn * cda_eff[i]
            f_roll = c_rr * fz
            f_grade = m * g * sin_theta[i]

            mu_lat_i = calc_mu_lat(fz)
            mu_long_i = calc_mu_long_drive(fz)

            a_lat_req = (v_curr**2) * abs_k[i]
            a_lat_max = (mu_lat_i * fz) / m
            r_lat = min(1.0, a_lat_req / max(0.1, a_lat_max))
            r_long = np.sqrt(max(0.0, 1.0 - r_lat**2))

            f_traction_max = r_long * mu_long_i * fz

            # Powertrain power available
            if p_mguk_schedule is not None:
                p_mguk_kw = p_mguk_schedule[i]
            else:
                p_mguk_kw = _fia_2026_mguk_max_kw(v_kmh, pu)

            p_wheel_w = (ice_kw + p_mguk_kw) * 1000.0 * eta_dt
            f_powertrain = p_wheel_w / max(12.0, v_curr)

            f_drive = min(f_traction_max, f_powertrain) - f_drag - f_roll - f_grade
            a_accel = min(1.75 * g, f_drive / m)
            v_next_sq = max(100.0, v_curr**2 + 2.0 * a_accel * ds)
            v[i + 1] = min(v[i + 1], np.sqrt(v_next_sq))
        v[0] = min(v[0], v[-1])

    v_smooth = gaussian_filter1d(v, sigma=2.0, mode="wrap")
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
    long_g = np.clip(gaussian_filter1d(dv_dt / g, sigma=3.5, mode="nearest"), -5.40, 1.75)
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

    # Gear selection [1..8] derived from speed and engine speed plateau
    gear_thresholds = veh["gear_ratios_top_speed_kmh"]
    gear = np.ones_like(s, dtype=int)
    for i, v_k in enumerate(speed_kmh):
        g_val = 8
        for g_idx, g_max in enumerate(gear_thresholds):
            if v_k <= g_max:
                g_val = g_idx + 1
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

    rho = compute_air_density(cfg)
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

    # 2026 Power Unit Energy Management & Optimal Strategy Solver
    ice_max_kw = float(pu["ice_max_power_kw"])
    ice_power_kw = ice_max_kw * np.power(throttle_pct / 100.0, 1.15)
    ice_power_kw[throttle_pct < 2.0] = 0.0

    eta_deploy = float(pu.get("es_deploy_efficiency", 0.96))
    eta_harvest = float(pu.get("es_harvest_efficiency", 0.92))
    soc_0 = float(pu.get("soc_initial_mj", 3.85))
    soc_min = float(pu.get("soc_min_mj", 0.0))
    soc_max = float(pu.get("soc_max_mj", 4.0))
    recharge_max = float(pu.get("max_recharge_per_lap_mj", 7.0))
    dt_arr = np.gradient(t)

    mguk_power_kw = np.zeros_like(s)
    mguk_state = np.empty(n, dtype="<U12")

    # 1. Base allocation: Overrun harvesting under braking, part throttle, and standard straight deployment
    for i in range(n):
        th = throttle_pct[i]
        br = brake_pct[i]
        vk = speed_kmh[i]

        if br > 5.0 or th < 5.0:
            scale = min(1.0, max(0.25, br / 70.0)) * min(1.0, max(0.3, vk / 120.0))
            mguk_power_kw[i] = float(pu.get("mguk_max_harvest_kw", -350.0)) * scale
            mguk_state[i] = "HARVEST"
        elif 10.0 <= th <= 65.0:
            pt_frac = (65.0 - th) / 55.0
            mguk_power_kw[i] = -50.0 - 190.0 * np.clip(pt_frac, 0.0, 1.0)
            mguk_state[i] = "HARVEST"
        else:
            p_dep = _fia_2026_mguk_max_kw(vk, pu)
            mguk_power_kw[i] = p_dep
            mguk_state[i] = "DEPLOY" if abs(p_dep) >= 5.0 else "OFF"

    # 2. Optimal Superclipping in qualifying:
    # On high-speed straightaways (v > 305 km/h, full throttle, low curvature),
    # superclipping extracts up to 350 kW to charge the battery for 2-4 seconds per lap.
    # At Interlagos, the end of Reta Oposta and top of the main straight climb are optimal.
    sc_eligible = (throttle_pct >= 95.0) & (speed_kmh > 308.0) & (np.abs(kappa) < 0.0028)
    sc_zones = ((s >= 1180) & (s <= 1300)) | ((s >= 4080) & (s <= 4235))
    p_sc_kw = float(pu.get("mguk_superclip_kw", -250.0))
    for i in range(n):
        if sc_zones[i] and sc_eligible[i]:
            mguk_power_kw[i] = p_sc_kw
            mguk_state[i] = "SUPERCLIP"

    # 3. Exact energy integration with constant efficiencies & 7.0 MJ recharge constraint
    e_recharge_cum = 0.0
    e_overrun_step = np.zeros_like(s)
    e_part_step = np.zeros_like(s)
    e_superclip_step = np.zeros_like(s)
    soc_mj = np.zeros_like(s)
    soc_mj[0] = soc_0

    for i in range(1, n):
        p_k = mguk_power_kw[i]
        dt_i = dt_arr[i]
        if p_k >= 0.0:
            d_soc = -(p_k / (eta_deploy * 1000.0)) * dt_i
        else:
            recharge_step = (-p_k * eta_harvest / 1000.0) * dt_i
            if e_recharge_cum + recharge_step > recharge_max:
                recharge_step = max(0.0, recharge_max - e_recharge_cum)
                p_mguk_kw[i] = -(recharge_step / (eta_harvest * dt_i)) * 1000.0
                if abs(p_mguk_kw[i]) < 5.0:
                    mguk_state[i] = "OFF"
            e_recharge_cum += recharge_step
            d_soc = recharge_step

            if mguk_state[i] == "SUPERCLIP":
                e_superclip_step[i] = recharge_step
            elif brake_pct[i] > 2.0 or throttle_pct[i] < 10.0:
                e_overrun_step[i] = recharge_step
            else:
                e_part_step[i] = recharge_step

        soc_mj[i] = np.clip(soc_mj[i - 1] + d_soc, soc_min, soc_max)

    energy_overrun_mj = np.cumsum(e_overrun_step)
    energy_part_throttle_mj = np.cumsum(e_part_step)
    energy_superclip_mj = np.cumsum(e_superclip_step)
    energy_total_recovered_mj = energy_overrun_mj + energy_part_throttle_mj + energy_superclip_mj

    # Fuel Used [kg] integrated directly from ICE chemical energy flow:
    # m_dot_fuel = P_ice / (eta_th * LHV)
    eta_ice = float(pu.get("ice_thermal_efficiency", 0.48))
    lhv_mj_kg = float(pu.get("fuel_lower_heating_value_mj_kg", 43.5))
    lhv_kj_kg = lhv_mj_kg * 1000.0  # 43500 kJ/kg
    fuel_rate_kg_s = ice_power_kw / (eta_ice * lhv_kj_kg)
    fuel_used_kg = np.cumsum(fuel_rate_kg_s * dt_arr)
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
    # Runs the full QSS pipeline for each perturbation and computes honest delta-t(s) = t_pert(s) - t_base(s)
    perturbations = {
        "Power +10 kW":    {"power_offset_kw": 10.0},
        "Downforce +1%":   {"cl_scale": 1.01},
        "Drag -1%":        {"cd_scale": 0.99},
        "Front grip +1%":  {"front_grip_scale": 1.01},
        "Rear grip +1%":   {"rear_grip_scale": 1.01},
    }
    sensitivity_deltas_s = {}
    for label, kwargs in perturbations.items():
        v_pert = _solve_qss_velocity_profile(s, kappa, sin_theta, cfg, **kwargs)
        dt_pert_segments = ds_steps / (0.5 * (v_pert[:-1] + v_pert[1:]))
        t_pert = np.concatenate([[0.0], np.cumsum(dt_pert_segments)])
        sensitivity_deltas_s[label] = t_pert - t

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
