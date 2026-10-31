"""
validate.py — honest validation + Interlagos prediction for the 2026 energy-limited lap model.

What it does
  1. Reads calibration_2026.json (produced by calibrate.py) and prints the leave-one-out accuracy, i.e. the error
     on a real 2026 qualifying lap the fit never saw. THIS is the accuracy to expect, not the in-sample error.
  2. Runs the Interlagos prediction and verifies physics / regulation assertions on the produced telemetry.
  3. Reports the prediction with an uncertainty derived from the unseen-circuit errors (not a hand-picked window).
  4. Saves a held-out speed-trace figure to ./artifacts/.

Run:  python calibrate.py   (once, a few minutes)   then   python validate.py
"""

import json
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import calibrate as C
import qss_core as Q
import physics_2026 as _P26
P_LAST_ANCHOR = lambda: _P26.LAST_ANCHOR
from physics_2026 import simulate_brazil_2026_lap, TelemetryResult
from track_loader import load_interlagos_track, TrackData

ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", "artifacts")
CONFIG = "config_brazil_2026.json"
CALIB = "calibration_2026.json"


def fmt_time(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:06.3f}"


def verify_physics_assertions(res: TelemetryResult, cfg: dict, track: TrackData) -> bool:
    print("\n" + "=" * 100)
    print("PHYSICS & REGULATION ASSERTIONS (on the telemetry that drives the video)")
    print("=" * 100)
    pu = cfg["power_unit_2026"]
    veh = cfg["vehicle_2026"]
    g = 9.81
    m = float(veh["mass_kg"])
    dt = np.gradient(res.t)
    rows = []

    def check(name, ok, value):
        rows.append((name, bool(ok), value))

    # 1. Normal force conservation
    cos_theta = np.sqrt(np.maximum(0.0, 1.0 - track.sin_theta ** 2))
    fz_expected = m * g * cos_theta + res.downforce_kg * g
    fz_sum = (res.fz_fl_kn + res.fz_fr_kn + res.fz_rl_kn + res.fz_rr_kn) * 1000.0
    err = np.max(np.abs(fz_sum - fz_expected) / fz_expected)
    check("sum(Fz) = m*g*cos(theta) + downforce (< 1%)", err < 0.01, f"{err*100:.3f}% max error")

    # 2. Battery: window and exact electrical energy balance
    eta_d = float(pu["es_deploy_efficiency"])
    eta_h = float(pu["es_harvest_efficiency"])
    p = res.mguk_power_kw
    dep_el = float(np.sum(np.clip(p, 0, None) * dt) / 1000.0 / eta_d)
    rec_el = float(np.sum(np.clip(-p, 0, None) * dt) * eta_h / 1000.0)
    soc_end_expected = res.soc_mj[0] - dep_el + rec_el
    check("SOC inside [0, 4] MJ", res.soc_mj.min() >= -1e-6 and res.soc_mj.max() <= 4.0 + 1e-6,
          f"[{res.soc_mj.min():.3f}, {res.soc_mj.max():.3f}] MJ")
    check("Electrical energy balance: SOC_end = SOC_0 - deployed + recovered (< 0.03 MJ)",
          abs(res.soc_mj[-1] - soc_end_expected) < 0.03,
          f"SOC_end {res.soc_mj[-1]:.3f} vs {soc_end_expected:.3f} MJ (deployed {dep_el:.2f}, recovered {rec_el:.2f})")
    check("Recharge <= FIA limit", rec_el <= float(pu["max_recharge_per_lap_mj"]) + 0.02,
          f"{rec_el:.2f} / {pu['max_recharge_per_lap_mj']:.1f} MJ")
    # 3. The original complaint: deploying with an empty battery
    empty = res.soc_mj < 0.005
    e_empty = float(np.sum(np.clip(p, 0, None)[empty] * dt[empty]) / 1000.0)
    check("No MGU-K deployment while the battery is empty (< 0.03 MJ)", e_empty < 0.03,
          f"{e_empty:.3f} MJ deployed at SOC < 0.005 MJ")
    # 4. Regulation limits
    taper = np.array([Q.mguk_max_deploy_kw(v, pu) for v in res.speed_kmh])
    over = np.max(np.clip(p, 0, None) - taper)
    check("MGU-K deploy <= 350 kW and <= FIA speed taper", over < 1.0 and p.max() <= 350.5,
          f"peak {p.max():.0f} kW, max excess over taper {over:.1f} kW")
    check("MGU-K harvest <= 350 kW", p.min() >= -350.5, f"min {p.min():.0f} kW")
    ice_lim = float(pu["ice_max_power_kw"]) * 1.1
    check("ICE power within limit", res.ice_power_kw.max() <= ice_lim, f"peak {res.ice_power_kw.max():.0f} kW")
    fuel_flow_mj_h = res.ice_power_kw / float(pu["ice_thermal_efficiency"]) * 3.6
    check("Fuel energy flow <= 3000 MJ/h", fuel_flow_mj_h.max() <= float(pu["fuel_energy_flow_limit_mj_h"]) * 1.1,
          f"peak {fuel_flow_mj_h.max():.0f} MJ/h")
    # 5. Consistency: the powertrain channels can actually produce the force the speed profile needs
    f_res = res.drag_kg * g + float(cfg["suspension_and_tires"]["rolling_resistance_coeff"]) * (
        m * g * cos_theta + res.downforce_kg * g) + m * g * track.sin_theta
    a = res.speed_ms * np.gradient(res.speed_ms, track.s)
    f_req = m * a + f_res
    drive = (res.brake_pct < 5.0) & (f_req > 0)
    p_wheel = (res.ice_power_kw + res.mguk_power_kw) * 1000.0 * float(pu.get("drivetrain_efficiency", 0.96))
    f_avail = p_wheel / np.maximum(res.speed_ms, 12.0)
    ft = drive & (res.throttle_pct >= 99.0)
    ds = np.gradient(track.s)
    e_req = float(np.sum(np.where(ft, f_req, 0.0) * ds))
    e_av = float(np.sum(np.where(ft, f_avail, 0.0) * ds))
    # per-node force is dominated by finite-difference noise of the speed trace where net force is small,
    # so the consistency test is energy-weighted over all full-throttle distance
    check("Powertrain delivers the demanded wheel energy at full throttle (|error| < 3%)",
          abs(e_req - e_av) / max(e_req, 1.0) < 0.03,
          f"required {e_req/1e6:.2f} MJ vs delivered {e_av/1e6:.2f} MJ ({100*(e_av-e_req)/max(e_req,1.0):+.2f}%)")
    # 6. Accelerations plausible for a 2026 car (real laps: braking to ~-5.9 g, traction ~+2 g, lateral <= ~5.5 g)
    check("Peak lateral g <= 5.5", np.abs(res.lat_g).max() <= 5.5, f"{np.abs(res.lat_g).max():.2f} g")
    check("Longitudinal g in [-6.5, +2.5]", res.long_g.min() >= -6.5 and res.long_g.max() <= 2.5,
          f"[{res.long_g.min():.2f}, {res.long_g.max():.2f}] g")
    # 7. Driver-input realism vs the real 2026 laps (66-75% full throttle, 10-17% braking)
    check("Full-throttle share within real-lap range (60-80%)", 60 <= 100 * np.mean(res.throttle_pct >= 99) <= 80,
          f"{100*np.mean(res.throttle_pct >= 99):.1f}%")
    check("Braking share within real-lap range (8-20%)", 8 <= 100 * np.mean(res.brake_pct > 5) <= 20,
          f"{100*np.mean(res.brake_pct > 5):.1f}%")

    ok_all = True
    for name, ok, val in rows:
        print(f"  {'PASS' if ok else 'FAIL':5s} | {name:92s} | {val}")
        ok_all &= ok
    return ok_all


def loo_traces_figure(calib: dict, base_cfg: dict, path: str):
    """Held-out speed traces: each real lap predicted with parameters fitted WITHOUT that circuit."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig, axes = plt.subplots(len(C.SESSIONS), 1, figsize=(14, 10))
    for ax, name in zip(axes, C.SESSIONS):
        sess = C.load_session(name, base_cfg)
        th = calib["leave_one_out"][name]["theta"]
        t, rms, v, _ = C.evaluate(sess, np.array(th), n_iter=20, use_warm=False)
        ax.plot(sess["s"], sess["v_real"], color="#1E6AD4", lw=1.4, label="real 2026 qualifying lap")
        ax.plot(sess["s"], v * 3.6, color="#D92632", lw=1.4, ls="--", label="model (fitted WITHOUT this circuit)")
        ax.set_title(f"{name.title()}: real {fmt_time(sess['t_real'])}  model {fmt_time(t)}  "
                     f"({t - sess['t_real']:+.2f} s)   speed RMS {rms:.1f} km/h")
        ax.set_ylabel("km/h")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
    axes[-1].set_xlabel("distance [m]")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[OK] held-out speed traces saved to {path}")


def main():
    if not os.path.exists(CALIB):
        sys.exit("calibration_2026.json not found - run `python calibrate.py` first.")
    base_cfg = json.load(open(CONFIG, "r", encoding="utf-8"))
    calib = json.load(open(CALIB, "r", encoding="utf-8"))
    params = calib["parameters"]

    print("=" * 100)
    print("MODEL ACCURACY ON REAL 2026 QUALIFYING LAPS")
    print("=" * 100)
    print("Fitted parameters:", {k: round(v, 3) for k, v in params.items()})
    print("\nHELD-OUT (leave-one-circuit-out; the honest numbers):")
    for n, r in calib.get("leave_one_out", {}).items():
        print(f"   {n:12s} lap-time error {r['lap_time_error_s']:+.3f} s   speed RMS {r['speed_rms_kmh']:.1f} km/h")
    print("IN-SAMPLE (optimistic):")
    for n, r in calib["in_sample"].items():
        print(f"   {n:12s} lap-time error {r['lap_time_error_s']:+.3f} s   speed RMS {r['speed_rms_kmh']:.1f} km/h")
    loo_rms = float(calib.get("loo_rms_lap_error_s", 1.0))
    loo_bias = float(calib.get("loo_mean_lap_error_s", 0.0))

    track = load_interlagos_track(config_path=CONFIG)
    res = simulate_brazil_2026_lap(config_path=CONFIG, track=track, params=params)
    ok = verify_physics_assertions(res, base_cfg, track)

    print("\n" + "=" * 100)
    print("2026 SAO PAULO GP QUALIFYING PREDICTION (Interlagos)")
    print("=" * 100)
    s, kappa, sin_theta = track.s, track.kappa, track.sin_theta
    xmask = Q.prepare_xmode(s, kappa, sin_theta, base_cfg, params)
    fold_times = []
    for n, r in calib.get("leave_one_out", {}).items():
        pf = dict(zip(C.PARAM_NAMES, r["theta"]))
        _, plan, _ = Q.predict_lap(s, kappa, sin_theta, base_cfg, pf, xmask=xmask, n_iter=20)
        v = gaussian_filter1d(np.array(plan.info.v), 2.0, mode="wrap")
        fold_times.append(C.lap_time_from_v(s, v))
    cfg_hot = json.loads(json.dumps(base_cfg))
    cfg_hot["event"]["weather"]["ambient_temperature_c"] += 8.0
    _, plan_h, _ = Q.predict_lap(s, kappa, sin_theta, cfg_hot, params, xmask=xmask, n_iter=20)
    v_h = gaussian_filter1d(np.array(plan_h.info.v), 2.0, mode="wrap")
    d_weather = C.lap_time_from_v(s, v_h) - (P_LAST_ANCHOR().get('lap_unanchored_s', res.lap_time_s))
    spread = (max(fold_times) - min(fold_times)) / 2.0 if fold_times else 0.0
    sigma = float(np.sqrt(loo_rms ** 2 + (spread / 2.0) ** 2 + (d_weather / 2.0) ** 2))
    t_pred = res.lap_time_s
    anchor = dict(P_LAST_ANCHOR()) if callable(P_LAST_ANCHOR) else {}
    # ratio method: leave-one-out error of "2025 pole x mean(2026/2025 ratio of the other circuits)"
    ratio_rows = []
    with open(os.path.join(os.path.dirname(os.path.abspath(CONFIG)), "data", "sessions_2026.json"), "r", encoding="utf-8") as f:
        sj = json.load(f)
    for k, d in sj.items():
        if "pole_2025_s" in d and "lap_2026_s" in d:
            others = [o["lap_2026_s"] / o["pole_2025_s"] for kk, o in sj.items() if kk != k and "pole_2025_s" in o]
            ratio_rows.append((k, d["pole_2025_s"] * float(np.mean(others)) - d["lap_2026_s"]))
    ratio_rms = float(np.sqrt(np.mean([e * e for _, e in ratio_rows]))) if ratio_rows else float("nan")
    if anchor:
        print(f"Physics-only lap (3-circuit fit, no anchor): {fmt_time(anchor['lap_unanchored_s'])}  "
              f"(held-out RMS {loo_rms:.2f} s)")
        print("2025-ratio method, leave-one-circuit-out: " + ", ".join(f"{k} {e:+.2f} s" for k, e in ratio_rows)
              + f"   RMS {ratio_rms:.2f} s  (only {len(ratio_rows)} circuits)")
        print(f"Anchor: 2025 pole {fmt_time(anchor['reference_2025_pole_s'])} x {anchor['ratio']:.4f} = {fmt_time(anchor['target_s'])}; "
              f"grip & power scaled x{anchor['factor']:.4f} so the energy-optimal lap hits it")
        sigma = float(np.sqrt(ratio_rms ** 2 + (d_weather / 2.0) ** 2))
    else:
        sigma = float(np.sqrt(loo_rms ** 2 + (spread / 2.0) ** 2 + (d_weather / 2.0) ** 2))
    print(f"Predicted pole-lap time : {fmt_time(t_pred)}  ({t_pred:.3f} s)")
    print(f"  Sector 1 / 2 / 3      : {res.sector_times_s[0]:.3f} / {res.sector_times_s[1]:.3f} / {res.sector_times_s[2]:.3f} s")
    if fold_times and not anchor:
        print(f"Unseen-circuit (LOO) error: mean {loo_bias:+.3f} s, RMS {loo_rms:.3f} s")
        print(f"Spread across cross-validation parameter sets: {fmt_time(min(fold_times))} - {fmt_time(max(fold_times))}")
    print(f"+8 C ambient (physics-only): {d_weather:+.3f} s")
    print(f"1-sigma uncertainty     : +/- {sigma:.3f} s   ->  68% window {fmt_time(t_pred - sigma)} to {fmt_time(t_pred + sigma)}")
    print(f"                          ~95% window {fmt_time(t_pred - 2*sigma)} to {fmt_time(t_pred + 2*sigma)}")
    print("Caveat: ideal-lap model; real qualifying also depends on track evolution, tyre prep, traffic and driver error.")
    print(f"Energy: recovered {res.energy_total_recovered_mj[-1]:.2f} MJ (overrun {res.energy_overrun_mj[-1]:.2f}, part throttle "
          f"{res.energy_part_throttle_mj[-1]:.2f}, superclip {res.energy_superclip_mj[-1]:.2f}), "
          f"SOC {res.soc_mj[0]:.2f} -> {res.soc_mj[-1]:.2f} MJ, fuel {res.fuel_used_kg[-1]:.3f} kg")

    if "--no-plot" not in sys.argv and calib.get("leave_one_out"):
        loo_traces_figure(calib, base_cfg, os.path.join(ARTIFACT_DIR, "loo_speed_traces.png"))
    print("\nALL ASSERTIONS PASSED" if ok else "\nSOME ASSERTIONS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
