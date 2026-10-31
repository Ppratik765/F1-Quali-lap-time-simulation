"""
qss_core.py — Fast, energy-aware quasi-steady-state (QSS) lap solver for the 2026 F1 rules.

What is new compared with the earlier solver
--------------------------------------------
* The velocity profile is no longer solved with "MGU-K at full deployment everywhere".
  Deployment/harvest is a decision variable p(s) [kW at the crank, + deploy, - harvest].
* The battery is a hard constraint: 0 <= SOC <= 4 MJ window, recharge <= 7 MJ/lap, MGU-K <= 350 kW with the
  FIA speed taper, ICE <= 400 kW (3000 MJ/h fuel-energy flow at 48 % efficiency).
* p(s) is found by a sequential linear programme (SLP):
      - exact lap-time gradient dT/dp(s) from an adjoint pass through the forward solver,
      - one LP per iteration that allocates the limited electrical energy where it saves the most time,
      - a line search on the true (non-linear) lap time.
  Harvest is free where it replaces friction braking or uses spare ICE power (corner apex / traction limited),
  and has a real time cost where the car is power limited (that is "superclipping").
* Pure-python float maths (no numpy scalars in the hot loops): ~0.1 s per solve.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import linprog

G = 9.81
BRAKE_BIAS_FRONT = 0.56


# --------------------------------------------------------------------------------------
# Regulation helpers
# --------------------------------------------------------------------------------------
def compute_air_density(cfg: Dict[str, Any]) -> float:
    """Moist-air density [kg/m^3] from altitude and weather (barometric formula + Tetens)."""
    ev = cfg.get("event", {})
    weather = ev.get("weather", {})
    altitude_m = float(ev.get("altitude_m", 765.0))
    temp_c = float(weather.get("ambient_temperature_c", 22.0))
    p_sl_hpa = float(weather.get("sea_level_pressure_hpa", 1015.0))
    rh = float(weather.get("relative_humidity", 0.55))

    p_amb = (p_sl_hpa * 100.0) * ((1.0 - (0.0065 * altitude_m) / 288.15) ** 5.25588)
    p_sat = 610.78 * (10.0 ** ((7.5 * temp_c) / (temp_c + 237.3)))
    p_v = rh * p_sat
    p_d = max(0.0, p_amb - p_v)
    tk = temp_c + 273.15
    return float(p_d / (287.058 * tk) + p_v / (461.495 * tk))


def mguk_max_deploy_kw(v_kmh: float, pu: Dict[str, Any]) -> float:
    """MGU-K deployment limit vs speed (350 kW up to 290 km/h, then the FIA linear taper to 0 at 345 km/h)."""
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
    if v_kmh < v2:
        return max(0.0, min(p_max, c1 - s1 * v_kmh))
    if v_kmh < v3:
        return max(0.0, c2 - s2 * v_kmh)
    return 0.0


def auto_straight_mask(s: np.ndarray, kappa: np.ndarray, k_thr: float = 0.0022, min_len_m: float = 280.0) -> np.ndarray:
    """
    Geometry-based X-mode (straight-mode) eligibility: contiguous stretches with very low curvature that are
    long enough to be a designated straight. Returns 0/1 array (not yet restricted by braking zones).
    """
    k = gaussian_filter1d(np.abs(kappa), 6.0, mode="wrap")
    flat = k < k_thr
    mask = np.zeros_like(s, dtype=np.float64)
    n = len(s)
    i = 0
    while i < n:
        if flat[i]:
            j = i
            while j < n and flat[j]:
                j += 1
            if (s[min(j, n) - 1] - s[i]) >= min_len_m:
                mask[i:j] = 1.0
            i = j
        else:
            i += 1
    return mask


# --------------------------------------------------------------------------------------
# Lap model
# --------------------------------------------------------------------------------------
@dataclass
class SolveInfo:
    v: List[float]              # speed profile [m/s]
    vcap: List[float]           # backward-pass (corner / braking) speed envelope [m/s]
    t: float                    # lap time [s]
    plim: List[bool]            # node is power-limited on the forward pass
    capped: List[bool]          # node i+1 is bound by the braking/corner envelope (not by the forward pass)
    a_x: List[float]            # realised longitudinal acceleration [m/s^2] (segment i -> i+1)
    f_req: List[float]          # wheel force needed [N] (negative -> braking)
    dt: List[float]             # segment time [s]


class LapModel:
    """Vehicle + track model producing the velocity profile for a given MGU-K power schedule."""

    def __init__(
        self,
        s: np.ndarray,
        kappa: np.ndarray,
        sin_theta: np.ndarray,
        cfg: Dict[str, Any],
        cl_scale: float = 1.0,
        cd_scale: float = 1.0,
        front_grip_scale: float = 1.0,
        rear_grip_scale: float = 1.0,
        power_offset_kw: float = 0.0,
        power_scale: float = 1.0,
        xmask: Optional[np.ndarray] = None,
    ):
        self.cfg = cfg
        veh = cfg["vehicle_2026"]
        aero = cfg["aerodynamics_2026"]
        pu = cfg["power_unit_2026"]
        tyr = cfg["suspension_and_tires"]
        self.pu = pu

        self.s = np.asarray(s, dtype=np.float64)
        self.n = len(self.s)
        self.ds = [float(d) for d in np.diff(self.s)]
        self.absk = [float(abs(k)) for k in kappa]
        self.sth = [float(x) for x in sin_theta]
        self.cth = [math.sqrt(max(0.0, 1.0 - x * x)) for x in self.sth]

        self.m = float(veh["mass_kg"])
        self.wf = float(veh.get("weight_distribution_front", 0.455))
        self.h = float(veh.get("cg_height_m", 0.28))
        self.wb = float(veh.get("wheelbase_m", 3.4))
        self.rho = compute_air_density(cfg)
        area = float(aero["reference_area_m2"])
        self.crr = float(tyr.get("rolling_resistance_coeff", 0.012))
        self.alpha = float(tyr.get("tire_load_sensitivity", 0.08))
        self.eta_dt = float(pu.get("drivetrain_efficiency", 0.96))
        self.ice_kw = float(pu["ice_max_power_kw"]) * power_scale + power_offset_kw
        self.v_ceil = 115.0

        # aero maps (CLA / CDA in m^2) with straight-mode blend
        self.xmask = np.zeros(self.n) if xmask is None else np.asarray(xmask, dtype=np.float64)
        self._set_aero(aero, area, cl_scale, cd_scale)

        mg = self.m * G
        self.mg = mg
        self.mu_lat0 = float(tyr["tire_mu_lat_peak"]) * (self.wf * front_grip_scale + (1 - self.wf) * rear_grip_scale)
        self.mu_brk0 = float(tyr["tire_mu_long_peak"]) * (
            BRAKE_BIAS_FRONT * front_grip_scale + (1 - BRAKE_BIAS_FRONT) * rear_grip_scale
        )
        self.mu_drv0 = float(tyr["tire_mu_long_peak"]) * rear_grip_scale
        self.fz_static = [mg * c for c in self.cth]
        self.fzr_ref = (1.0 - self.wf) * mg
        self.h_over_l = self.h / self.wb
        self._apex_cache: Optional[List[float]] = None

    # ------------------------------------------------------------------
    def _set_aero(self, aero, area, cl_scale, cd_scale):
        cl_z = float(aero["z_mode"]["cl_base"]) * cl_scale
        cd_z = float(aero["z_mode"]["cd_base"]) * cd_scale
        x_df = float(aero["x_mode"]["downforce_reduction_ratio"])
        x_dr = float(aero["x_mode"]["drag_reduction_ratio"])
        bal_z = float(aero["z_mode"]["aero_balance_front_pct"]) / 100.0
        bal_x = float(aero["x_mode"]["aero_balance_front_pct"]) / 100.0
        xm = self.xmask
        self.cla = [cl_z * area * (1.0 - x_df * x) for x in xm]
        self.cda = [cd_z * area * (1.0 - x_dr * x) for x in xm]
        self.bal_front = [bal_z + (bal_x - bal_z) * x for x in xm]
        self._apex_cache = None

    def rebuild_aero(self, xmask: np.ndarray, cl_scale: float, cd_scale: float):
        self.xmask = np.asarray(xmask, dtype=np.float64)
        aero = self.cfg["aerodynamics_2026"]
        self._set_aero(aero, float(aero["reference_area_m2"]), cl_scale, cd_scale)

    # ------------------------------------------------------------------
    def _apex_speeds(self) -> List[float]:
        if self._apex_cache is not None:
            return list(self._apex_cache)
        rho, m, alpha, mg = self.rho, self.m, self.alpha, self.mg
        out = [self.v_ceil] * self.n
        for i in range(self.n):
            k = self.absk[i]
            if k <= 1e-4:
                continue
            fzs = self.fz_static[i]
            cla = self.cla[i]
            v = 50.0
            for _ in range(4):
                fz = fzs + 0.5 * rho * cla * v * v
                mu = self.mu_lat0 * max(0.2, fz / mg) ** (-alpha)
                den = m * k - 0.5 * rho * cla * mu
                if den > 0.01:
                    v = math.sqrt(mu * fzs / den)
                else:
                    v = self.v_ceil
                    break
            out[i] = min(self.v_ceil, max(10.0, v))
        self._apex_cache = out
        return list(out)

    # ------------------------------------------------------------------
    def _fwd_accel(self, i: int, u: float, pk_kw: float) -> Tuple[float, bool, float, float]:
        """Longitudinal acceleration available at node i for speed^2 = u and MGU-K power pk_kw.
        Returns (a, power_limited, f_drive_available, f_resist)."""
        rho, m, mg, alpha = self.rho, self.m, self.mg, self.alpha
        v = math.sqrt(u)
        q = 0.5 * rho * u
        fz = self.fz_static[i] + q * self.cla[i]
        f_drag = q * self.cda[i]
        f_roll = self.crr * fz
        f_grade = mg * self.sth[i]

        # lateral usage -> friction ellipse
        mu_lat = self.mu_lat0 * max(0.2, fz / mg) ** (-alpha)
        a_lat_req = u * self.absk[i]
        a_lat_max = mu_lat * fz / m
        r_lat = min(1.0, a_lat_req / max(0.1, a_lat_max))
        r_long = math.sqrt(max(0.0, 1.0 - r_lat * r_lat))

        # traction: driven (rear) axle only, incl. longitudinal load transfer, load-sensitive friction
        fzr0 = (1.0 - self.wf) * self.fz_static[i] + q * self.cla[i] * (1.0 - self.bal_front[i])
        mu_d = self.mu_drv0 * max(0.2, fzr0 / self.fzr_ref) ** (-alpha)
        f_trac = r_long * mu_d * fzr0 / max(0.3, 1.0 - mu_d * self.h_over_l * r_long)

        f_pow = (self.ice_kw + pk_kw) * 1000.0 * self.eta_dt / max(12.0, v)
        plim = f_pow < f_trac
        f_drive = (f_pow if plim else f_trac) - f_drag - f_roll - f_grade
        return f_drive / m, plim, f_drive, f_drag + f_roll + f_grade

    def _brake_decel(self, i: int, vn: float) -> float:
        rho, m, mg, alpha = self.rho, self.m, self.mg, self.alpha
        q = 0.5 * rho * vn * vn
        fz = self.fz_static[i] + q * self.cla[i]
        f_drag = q * self.cda[i]
        f_grade = mg * self.sth[i]
        mu_lat = self.mu_lat0 * max(0.2, fz / mg) ** (-alpha)
        mu_b = self.mu_brk0 * max(0.2, fz / mg) ** (-alpha)
        a_lat_req = vn * vn * self.absk[i]
        a_lat_max = mu_lat * fz / m
        r_lat = min(1.0, a_lat_req / max(0.1, a_lat_max))
        r_long = math.sqrt(max(0.0, 1.0 - r_lat * r_lat))
        f_b = r_long * mu_b * fz + f_drag + f_grade
        return max(0.1, f_b / m)

    # ------------------------------------------------------------------
    def solve(self, p_sched: Optional[List[float]] = None, apex: Optional[List[float]] = None) -> SolveInfo:
        n, ds = self.n, self.ds
        pu = self.pu
        v = apex if apex is not None else self._apex_speeds()

        # backward (braking) envelope
        for _ in range(3):
            for i in range(n - 2, -1, -1):
                a = self._brake_decel(i, v[i + 1])
                va = math.sqrt(v[i + 1] * v[i + 1] + 2.0 * a * ds[i])
                if va < v[i]:
                    v[i] = va
            if v[n - 1] > v[0]:
                v[n - 1] = v[0]
        vcap = list(v)

        plim = [False] * n
        capped = [True] * n
        a_x = [0.0] * (n - 1)
        f_req = [0.0] * (n - 1)
        taper = mguk_max_deploy_kw
        for _ in range(3):
            if v[0] > v[n - 1]:
                v[0] = v[n - 1]
            for i in range(n - 1):
                vc = v[i]
                pk = taper(vc * 3.6, pu) if p_sched is None else p_sched[i]
                if pk > 0.0:
                    cap = taper(vc * 3.6, pu)
                    if pk > cap:
                        pk = cap
                a, pl, _, _ = self._fwd_accel(i, vc * vc, pk)
                un = vc * vc + 2.0 * a * ds[i]
                if un < 100.0:
                    un = 100.0
                vn = math.sqrt(un)
                plim[i] = pl
                if vn < vcap[i + 1]:
                    v[i + 1] = vn
                    capped[i + 1] = False
                else:
                    v[i + 1] = vcap[i + 1]
                    capped[i + 1] = True

        dt = [0.0] * (n - 1)
        t_tot = 0.0
        for i in range(n - 1):
            vm = 0.5 * (v[i] + v[i + 1])
            dt[i] = ds[i] / vm
            t_tot += dt[i]
            a_x[i] = (v[i + 1] ** 2 - v[i] ** 2) / (2.0 * ds[i])
        return SolveInfo(v=v, vcap=vcap, t=t_tot, plim=plim, capped=capped, a_x=a_x, f_req=f_req, dt=dt)

    # ------------------------------------------------------------------
    def gradient(self, info: SolveInfo, p_sched: List[float]) -> List[float]:
        """Adjoint gradient g[i] = dT/dp_i [s per kW] through the forward pass (exact for the piecewise model)."""
        n, ds = self.n, self.ds
        v = info.v
        taper = mguk_max_deploy_kw
        # derivative of the segment times wrt u_k = v_k^2
        tu = [0.0] * n
        for i in range(n - 1):
            vm = 0.5 * (v[i] + v[i + 1])
            c = -ds[i] / (vm * vm) * 0.25
            tu[i] += c / v[i]
            tu[i + 1] += c / v[i + 1]
        lam = [0.0] * n
        g = [0.0] * (n - 1)
        lam[n - 1] = tu[n - 1]
        for i in range(n - 2, -1, -1):
            vc = v[i]
            u = vc * vc
            pk = p_sched[i]
            if pk > 0.0:
                cap = taper(vc * 3.6, self.pu)
                if pk > cap:
                    pk = cap
            if info.capped[i + 1]:
                lam[i] = tu[i]
                g[i] = 0.0
                continue
            a0, pl, _, _ = self._fwd_accel(i, u, pk)
            du = 0.5 + 1e-3 * u
            a1, _, _, _ = self._fwd_accel(i, u + du, pk)
            a_u = (a1 - a0) / du
            if pl and pk <= taper(vc * 3.6, self.pu) + 1e-9:
                a_p = self.eta_dt * 1000.0 / (self.m * max(12.0, vc))
            else:
                a_p = 0.0
            if (u + 2.0 * a0 * ds[i]) < 100.0:      # clamp active
                lam[i] = tu[i]
                g[i] = 0.0
                continue
            g[i] = lam[i + 1] * 2.0 * ds[i] * a_p
            lam[i] = tu[i] + (1.0 + 2.0 * ds[i] * a_u) * lam[i + 1]
        return g

    # ------------------------------------------------------------------
    def node_capacities(self, info: SolveInfo) -> Tuple[List[float], List[float], List[bool]]:
        """
        Per-node physical bounds from the current solution.
        Returns (harvest_cap_kw, deploy_cap_kw, is_braking):
          * braking node  : harvest limited by the MGU-K (350 kW) and by the braking power available (free)
          * drive node    : harvest limited by spare ICE power when not power-limited (free, e.g. corner apex),
                            350 kW when power-limited (costly: this is superclipping)
        """
        n = self.n
        v = info.v
        pu = self.pu
        hmax = abs(float(pu.get("mguk_max_harvest_kw", -350.0)))
        harv = [0.0] * n
        dep = [0.0] * n
        brk = [False] * n
        for i in range(n - 1):
            vc = 0.5 * (v[i] + v[i + 1])
            u = vc * vc
            q = 0.5 * self.rho * u
            f_resist = q * self.cda[i] + self.crr * (self.fz_static[i] + q * self.cla[i]) + self.mg * self.sth[i]
            a = (v[i + 1] ** 2 - v[i] ** 2) / (2.0 * self.ds[i])
            f_req = self.m * a + f_resist           # wheel force the powertrain/brakes must supply
            if f_req < -50.0:
                brk[i] = True
                p_brk = (-f_req) * vc / 1000.0 * self.eta_dt
                harv[i] = min(hmax, p_brk)
                dep[i] = 0.0
            else:
                p_crank = max(0.0, f_req) * vc / 1000.0 / self.eta_dt
                if info.plim[i]:
                    harv[i] = min(hmax, float(pu.get("superclip_max_kw", hmax)))   # harvest at full throttle (0 = no superclip)
                else:
                    harv[i] = max(0.0, min(hmax, self.ice_kw - p_crank))
                dep[i] = mguk_max_deploy_kw(vc * 3.6, pu)
        harv[n - 1], dep[n - 1], brk[n - 1] = harv[n - 2], dep[n - 2], brk[n - 2]
        return harv, dep, brk


# --------------------------------------------------------------------------------------
# Energy optimiser (sequential LP)
# --------------------------------------------------------------------------------------
@dataclass
class EnergyPlan:
    p_kw: np.ndarray            # MGU-K crank power per node (+ deploy / - harvest)
    info: SolveInfo
    soc: np.ndarray             # battery state of charge per node [MJ]
    recharge_mj: float
    deploy_elec_mj: float
    history: List[float]


def soc_trace(model: LapModel, p_kw: List[float], dt: List[float], harvest_cap: List[float], soc0: float,
              soc_max: float, soc_min: float, eta_d: float, eta_h: float, recharge_max: float,
              sc_mask: Optional[List[bool]] = None, sc_max_mj: Optional[float] = None):
    """Exact SOC integration with node-level physical limits. Returns (soc, recharged_mj, deployed_elec_mj, p_eff)."""
    n = model.n
    soc = [0.0] * n
    soc[0] = soc0
    rech = 0.0
    dep_e = 0.0
    sc_e = 0.0                      # energy harvested at full throttle (superclip) so far
    p_eff = [0.0] * n
    for i in range(1, n):
        d = dt[i - 1]
        p = p_kw[i - 1]
        if p >= 0.0:
            e = p * d / (eta_d * 1000.0)
            if soc[i - 1] - e < soc_min:
                e = max(0.0, soc[i - 1] - soc_min)
                p = e * eta_d * 1000.0 / d if d > 0 else 0.0
            soc[i] = soc[i - 1] - e
            dep_e += e
        else:
            ph = min(-p, harvest_cap[i - 1])
            e = ph * d * eta_h / 1000.0
            is_sc = sc_mask is not None and sc_mask[i - 1]
            if is_sc and sc_max_mj is not None and sc_e + e > sc_max_mj:
                e = max(0.0, sc_max_mj - sc_e)
                ph = e * 1000.0 / (eta_h * d) if d > 0 else 0.0
            if rech + e > recharge_max:
                e = max(0.0, recharge_max - rech)
                ph = e * 1000.0 / (eta_h * d) if d > 0 else 0.0
            if soc[i - 1] + e > soc_max:
                e = max(0.0, soc_max - soc[i - 1])
                ph = e * 1000.0 / (eta_h * d) if d > 0 else 0.0
            soc[i] = soc[i - 1] + e
            rech += e
            if is_sc:
                sc_e += e
            p = -ph
        p_eff[i - 1] = p
    p_eff[n - 1] = p_eff[n - 2]
    return soc, rech, dep_e, p_eff


def optimise_energy(
    model: LapModel,
    n_iter: int = 14,
    block_m: float = 20.0,
    warm_p: Optional[np.ndarray] = None,
    trust0_kw: float = 350.0,
    verbose: bool = False,
) -> EnergyPlan:
    """Sequential LP that minimises lap time subject to the battery and regulation constraints."""
    pu = model.pu
    n = model.n
    eta_d = float(pu.get("es_deploy_efficiency", 0.96))
    eta_h = float(pu.get("es_harvest_efficiency", 0.92))
    soc0 = float(pu.get("soc_initial_mj", 4.0))
    soc_max = float(pu.get("soc_max_mj", 4.0))
    soc_min = float(pu.get("soc_min_mj", 0.0))
    soc_end_min = float(pu.get("soc_end_min_mj", 0.0))
    rech_max = float(pu.get("max_recharge_per_lap_mj", 7.0))
    sc_max = pu.get("superclip_max_mj_per_lap")            # energy that may be harvested at full throttle per lap
    sc_max = None if sc_max is None else float(sc_max)
    mdf = float(pu.get("mguk_min_deploy_fraction", 0.0))   # phase 2: idle full-throttle blocks must deploy >= this fraction
    forced = None
    forced_nodes = None             # nodes (power-limited, not braking) where phase 2 enforces the minimum deploy
    phase = 0
    margin = 0.02

    nb = int(math.ceil((n - 1) / block_m))
    blk = np.minimum(np.arange(n) // int(block_m), nb - 1)
    J = nb

    p_node = np.zeros(n) if warm_p is None else np.array(warm_p, dtype=np.float64)

    def evaluate(p_vec: np.ndarray):
        # The lap must be timed with the power the battery can really supply: iterate solve -> SOC clamp -> re-solve
        # until the speed profile and the energy-limited power agree (clamping only ever removes power).
        p_cur = np.array(p_vec, dtype=np.float64)
        consistent = False
        for _ in range(8):
            info = model.solve(list(p_cur))
            harv, dep, brk = model.node_capacities(info)
            sc_mask = [bool(a) and not b for a, b in zip(list(info.plim) + [info.plim[-1]], brk)]
            soc, rech, dep_e, p_eff = soc_trace(model, list(p_cur), info.dt + [info.dt[-1]], harv, soc0, soc_max, soc_min,
                                                eta_d, eta_h, rech_max, sc_mask, sc_max)
            if float(np.max(np.abs(np.array(p_eff) - p_cur))) < 0.5:
                consistent = True
                break
            p_cur = np.array(p_eff, dtype=np.float64)
        feasible = consistent and (min(soc) >= soc_min - 1e-6) and (soc[-1] >= soc_end_min - 1e-6)
        return info, harv, dep, brk, soc, rech, dep_e, np.array(p_eff), feasible

    # nodes -> blocks helpers
    counts = np.bincount(blk, minlength=J).astype(float)

    def block_mean(x: np.ndarray) -> np.ndarray:
        return np.bincount(blk, weights=x, minlength=J) / counts

    def block_sum(x: np.ndarray) -> np.ndarray:
        return np.bincount(blk, weights=x, minlength=J)

    info, harv, dep, brk, soc, rech, dep_e, p_eff, feas = evaluate(p_node)
    p_node = p_eff
    best_t = info.t
    history = [best_t]
    trust = trust0_kw
    small_gain = 0

    s_j = np.arange(J) * block_m
    dts_init = np.array(info.dt + [info.dt[-1]])
    harv_brk = np.array([h if b else 0.0 for h, b in zip(harv, brk)])
    rech_phys_est = float(np.sum(harv_brk * dts_init * eta_h / 1000.0)) + (sc_max if sc_max is not None else 0.0) + 0.35
    rech_max = min(rech_max, rech_phys_est)
    if mdf > 0.0:
        forced = (s_j >= 3300.0) & (block_mean(np.array(dep)) > 30.0)

    converged = False
    phase_it = 0
    for it in range(n_iter * (2 if mdf > 0.0 else 1)):
        g = np.array(model.gradient(info, list(p_node)) + [0.0])
        dts = np.array(info.dt + [info.dt[-1]])
        G_j = block_sum(g)                                  # s per kW of block-wide power
        dt_j = block_sum(dts)
        hbound = block_mean(np.array(harv))
        dbound = block_mean(np.array([0.0 if b else d for b, d in zip(brk, dep)]))
        p0_j = block_mean(p_node)

        # variables x = [p_plus (J), p_minus (J)]
        c = np.concatenate([G_j, -G_j])
        # SOC(j) = soc0 - cumsum(pplus*dt/(eta_d*1000)) + cumsum(pminus*dt*eta_h/1000)
        cum = np.tril(np.ones((J, J)))
        e_dep = np.diag(dt_j / (eta_d * 1000.0))
        e_har = np.diag(dt_j * eta_h / 1000.0)
        S_plus = -cum @ e_dep
        S_minus = cum @ e_har
        S = np.hstack([S_plus, S_minus])                      # SOC_j - soc0 = S x
        # trust-region rows
        I = np.eye(J)
        tr_A = [np.hstack([I, -I]), np.hstack([-I, I])]
        tr_b = [p0_j + trust, -(p0_j - trust)]
        bounds = [(0.0, float(max(0.0, d))) for d in dbound] + [(0.0, float(max(0.0, h))) for h in hbound]
        if forced is not None:
            for j in np.where(forced)[0]:
                dj = float(max(0.0, dbound[j]))
                bounds[j] = (mdf * dj, dj)
                bounds[J + j] = (0.0, 0.0)
        rech_row = np.concatenate([np.zeros(J), dt_j * eta_h / 1000.0])[None, :]
        extra_A, extra_b = [], []
        if sc_max is not None:
            sc_nodes = np.array([(1.0 if (pl and not bk) else 0.0) for pl, bk in zip(list(info.plim) + [info.plim[-1]], brk)])
            sc_dt_j = block_sum(dts * sc_nodes)
            extra_A.append(np.concatenate([np.zeros(J), sc_dt_j * eta_h / 1000.0])[None, :])
            extra_b.append(np.array([sc_max]))
        res = None
        # the current plan can sit exactly on the SOC limits (clamped), so the safety margin is relaxed if the LP
        # has no feasible point inside the trust region
        for mrg in (margin, 0.005, 0.0, -0.03):
            A_ub = [S, -S, S[-1:] * -1.0, rech_row] + extra_A + tr_A
            b_ub = [np.full(J, soc_max - mrg - soc0), np.full(J, soc0 - (soc_min + mrg)),
                    np.array([soc0 - soc_end_min - mrg]), np.array([rech_max - 0.02])] + extra_b + tr_b
            res = linprog(c, A_ub=np.vstack(A_ub), b_ub=np.concatenate(b_ub), bounds=bounds, method="highs")
            if res.status == 0:
                break
        if res.status != 0:
            trust *= 0.5
            continue
        p_new_j = res.x[:J] - res.x[J:]
        accepted = False
        for alpha in (1.0, 0.5, 0.25, 0.1, 0.05, 0.02):
            p_try = p_node + alpha * (p_new_j - p0_j)[blk]        # delta on the node-level plan (keeps its fine structure)
            inf2, h2, d2, b2, soc2, rech2, de2, peff2, feas2 = evaluate(p_try)
            if feas2 and inf2.t < best_t - 1e-5:
                accepted = True
                break
        if accepted:
            info, harv, dep, brk, soc, rech, dep_e, p_node = inf2, h2, d2, b2, soc2, rech2, de2, peff2
            gain = best_t - info.t
            best_t = info.t
            history.append(best_t)
            if verbose:
                print(f"   iter {it:2d}: T={best_t:.4f}  trust={trust:.0f}  alpha={alpha}  recharge={rech:.2f}  dep_el={dep_e:.2f}  socmin={min(soc):.2f}")
            if alpha >= 1.0:
                trust = min(trust0_kw, trust * 1.5)          # full step worked: let the trust region recover
            if gain < 2e-4:
                small_gain += 1
                if small_gain >= 3:
                    converged = True
            else:
                small_gain = 0
        else:
            if verbose:
                print(f"   iter {it:2d}: rejected (LP status {res.status}, pred. step {np.abs(p_new_j - p0_j).max():.0f} kW, last trial T={inf2.t:.4f} vs {best_t:.4f}, feasible={feas2}) trust={trust:.0f}")
            trust *= 0.5
            if trust < 3.0:
                converged = True

        phase_it += 1
        if phase == 0 and mdf > 0.0 and phase_it >= n_iter:
            converged = True                       # phase 1 used its whole budget: move on to phase 2
        if converged:
            if phase == 0 and mdf > 0.0:
                # Phase 2: an MGU-K that sits at exactly zero torque on a power-limited straight while charge remains is
                # not something a real energy controller does. Force a minimum deployment there and re-optimise.
                phase = 1
                sc_like = np.array([(1.0 if (pl and not bk) else 0.0)
                                    for pl, bk in zip(list(info.plim) + [info.plim[-1]], brk)])
                idle = (np.abs(p_node) < 6.0) & (sc_like > 0) & (np.array(dep) > 50.0)
                new_forced = block_mean(idle.astype(float)) >= 0.15
                forced = new_forced if forced is None else (forced | new_forced | ((s_j >= 3300.0) & (dbound > 30.0)))
                forced_nodes = forced[blk] & (sc_like > 0)
                p_proj = p_node.copy()
                info, harv, dep, brk, soc, rech, dep_e, p_node, _ = evaluate(p_proj)
                best_t = info.t
                history.append(best_t)
                trust = trust0_kw
                small_gain = 0
                converged = False
                phase_it = 0
                if verbose:
                    print(f"   phase 2: {int(forced.sum())} idle blocks forced to >= {mdf:.0%} deploy, T={best_t:.4f}")
            else:
                break

    return EnergyPlan(p_kw=np.array(p_node), info=info, soc=np.array(soc), recharge_mj=rech, deploy_elec_mj=dep_e, history=history)


def update_xmode_mask(model: LapModel, straight_mask: np.ndarray, cl_scale: float, cd_scale: float, passes: int = 2,
                      lead_m: float = 25.0, taper_sigma_m: float = 12.0) -> np.ndarray:
    """
    Straight-mode (X-mode) is only allowed on straights AND not while braking: the car returns to Z-mode
    `lead_m` before the braking point (actuator time ~0.25 s). Iterates with the solved speed profile.
    """
    n = model.n
    mask = straight_mask.copy()
    for _ in range(passes):
        smooth = gaussian_filter1d(mask, taper_sigma_m, mode="wrap")
        model.rebuild_aero(smooth, cl_scale, cd_scale)
        info = model.solve(None)
        v = info.v
        braking = np.array([(v[i] - v[i + 1]) > 0.0 and info.capped[i + 1] for i in range(n - 1)] + [False])
        new = straight_mask.copy()
        i = 0
        lead = int(lead_m)
        while i < n:
            if braking[i] and straight_mask[i] > 0:
                j = i
                while j < n and straight_mask[j] > 0:
                    j += 1
                lo = max(0, i - lead)
                new[lo:j] = 0.0
                i = j
            else:
                i += 1
        mask = new
    model.rebuild_aero(gaussian_filter1d(mask, taper_sigma_m, mode="wrap"), cl_scale, cd_scale)
    return gaussian_filter1d(mask, taper_sigma_m, mode="wrap")


# --------------------------------------------------------------------------------------
# Pipeline helpers
# --------------------------------------------------------------------------------------
DEFAULT_PARAMS: Dict[str, float] = {
    "cl_scale": 1.0,          # downforce multiplier (global)
    "cd_scale": 1.0,          # drag multiplier (global)
    "grip_scale": 1.0,        # tyre friction multiplier (global)
    "power_scale": 1.0,       # ICE power multiplier
    "front_grip_scale": 1.0,  # sensitivity perturbations only
    "rear_grip_scale": 1.0,
    "power_offset_kw": 0.0,
}


def make_model(s, kappa, sin_theta, cfg, params: Optional[Dict[str, float]] = None,
               xmask: Optional[np.ndarray] = None) -> LapModel:
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    model = LapModel(
        s, kappa, sin_theta, cfg,
        cl_scale=p["cl_scale"], cd_scale=p["cd_scale"],
        front_grip_scale=p["grip_scale"] * p["front_grip_scale"],
        rear_grip_scale=p["grip_scale"] * p["rear_grip_scale"],
        power_offset_kw=p["power_offset_kw"], power_scale=p["power_scale"],
        xmask=xmask,
    )
    return model


def prepare_xmode(s, kappa, sin_theta, cfg, params: Optional[Dict[str, float]] = None) -> np.ndarray:
    """Computes the (smoothed) straight-mode mask once for a track: geometry straights minus braking zones."""
    xm_cfg = cfg["aerodynamics_2026"]["x_mode"]
    zones = xm_cfg.get("activation_zones_m") if xm_cfg.get("zone_mode", "auto") == "manual" else None
    if zones:
        straight = np.zeros(len(s))
        for z in zones:
            straight[(s >= z["start_m"]) & (s <= z["end_m"])] = 1.0
        straight[np.abs(kappa) > float(xm_cfg.get("max_kappa", 0.0035))] = 0.0
    else:
        straight = auto_straight_mask(s, kappa)
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    model = make_model(s, kappa, sin_theta, cfg, p, xmask=straight)
    return update_xmode_mask(model, straight, p["cl_scale"], p["cd_scale"])


def predict_lap(s, kappa, sin_theta, cfg, params=None, xmask=None, n_iter: int = 20,
                warm_p: Optional[np.ndarray] = None, trust0_kw: float = 350.0, verbose: bool = False):
    """Full pipeline: aero mode mask -> energy-optimal deployment -> velocity profile. Returns (model, plan, xmask)."""
    if xmask is None:
        xmask = prepare_xmode(s, kappa, sin_theta, cfg, params)
    model = make_model(s, kappa, sin_theta, cfg, params, xmask=xmask)
    plan = optimise_energy(model, n_iter=n_iter, warm_p=warm_p, trust0_kw=trust0_kw, verbose=verbose)
    return model, plan, xmask
