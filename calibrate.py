"""
calibrate.py — fits a handful of *physical* global parameters (downforce, drag, tyre grip, ICE power)
of the energy-limited lap model to real 2026 qualifying laps and reports honest, leave-one-out accuracy.

Usage:
    python calibrate.py            # LOO cross-validation + final fit on all sessions -> calibration_2026.json
    python calibrate.py --quick    # skip cross-validation (final fit only)

Method
------
* Sessions: real 2026 qualifying laps cached in data/*.npz (OpenF1): Austria, Spain, Silverstone.
* Every evaluation runs the FULL pipeline (energy-optimal deployment under the 4 MJ / 7 MJ limits).
* Loss = sum over laps of [ (lap-time error)^2 + (0.04 * speed-trace RMS)^2 ] + weak ridge to the nominal values.
* Leave-one-out: fit on two circuits, predict the third. The spread of those *unseen-circuit* errors is the
  honest uncertainty used for the Interlagos prediction (no fudge, no hard-coded target).
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize

import qss_core as Q
from track_loader import compute_geometry

_DEFAULT_SESSIONS = {
    "austria": {"file": "data/austria_2026_telemetry.npz", "altitude_m": 660.0},
    "spain": {"file": "data/spain_2026_telemetry.npz", "altitude_m": 120.0},
    "silverstone": {"file": "data/silverstone_2026_telemetry.npz", "altitude_m": 150.0},
}


def _load_sessions(path: str = "data/sessions_2026.json") -> dict:
    """Calibration sessions. Add more real 2026 laps (python cache_telemetry.py) by listing them in
    data/sessions_2026.json: {"mexico": {"file": "data/mexico_2026_telemetry.npz", "altitude_m": 2240}, ...}"""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            extra = json.load(f)
        return {k: v for k, v in extra.items() if os.path.exists(v["file"])}
    return dict(_DEFAULT_SESSIONS)


SESSIONS = _load_sessions()
PARAM_NAMES = ["cl_scale", "cd_scale", "grip_scale", "power_scale"]
BOUNDS = [(0.80, 1.35), (0.80, 1.35), (0.85, 1.25), (0.92, 1.08)]
RIDGE_SIGMA = 0.15
SPEED_WEIGHT = 0.04


def load_session(name: str, base_cfg: dict) -> dict:
    meta = SESSIONS[name]
    d = np.load(meta["file"])
    s, x, y, z = d["s"], d["x"], d["y"], d["z"]
    _, kappa, sin_theta = compute_geometry(s, x, y, z)
    cfg = json.loads(json.dumps(base_cfg))
    cfg["event"]["altitude_m"] = meta["altitude_m"]
    return {
        "name": name, "s": s, "kappa": kappa, "sin_theta": sin_theta, "cfg": cfg,
        "t_real": float(d["lap_time"]), "v_real": d["speed_kmh"], "throttle_real": d["throttle"],
        "brake_real": d["brake"], "xmask": None, "warm": None,
    }


def lap_time_from_v(s, v):
    v = np.asarray(v)
    return float(np.sum(np.diff(s) / (0.5 * (v[:-1] + v[1:]))))


def evaluate(sess: dict, theta, n_iter=3, use_warm=True):
    params = dict(zip(PARAM_NAMES, theta))
    if sess["xmask"] is None:
        sess["xmask"] = Q.prepare_xmode(sess["s"], sess["kappa"], sess["sin_theta"], sess["cfg"], params)
    model, plan, _ = Q.predict_lap(
        sess["s"], sess["kappa"], sess["sin_theta"], sess["cfg"], params, xmask=sess["xmask"],
        n_iter=n_iter, warm_p=sess["warm"] if use_warm else None, trust0_kw=120.0 if (use_warm and sess["warm"] is not None) else 350.0,
    )
    sess["warm"] = plan.p_kw.copy()
    v = gaussian_filter1d(np.array(plan.info.v), 2.0, mode="wrap")
    t = lap_time_from_v(sess["s"], v)
    rms = float(np.sqrt(np.mean((v * 3.6 - sess["v_real"]) ** 2)))
    return t, rms, v, plan


def loss(theta, sessions):
    th = np.asarray(theta)
    tot = 0.0
    for sess in sessions:
        t, rms, _, _ = evaluate(sess, th)
        tot += (t - sess["t_real"]) ** 2 + (SPEED_WEIGHT * rms) ** 2
    tot += 0.5 * float(np.sum(((th - 1.0) / RIDGE_SIGMA) ** 2))
    return tot


def fit(sessions, x0=None, maxiter=45, verbose=True):
    for sess in sessions:
        sess["warm"] = None
    x0 = np.ones(len(PARAM_NAMES)) if x0 is None else np.asarray(x0)
    t0 = time.time()
    res = minimize(loss, x0, args=(sessions,), method="Nelder-Mead", bounds=BOUNDS,
                   options={"maxiter": maxiter, "xatol": 4e-3, "fatol": 1e-3, "initial_simplex": _simplex(x0)})
    if verbose:
        print(f"   fit on {[s['name'] for s in sessions]}: theta={np.round(res.x, 3)} loss={res.fun:.3f} nfev={res.nfev} ({time.time()-t0:.0f}s)")
    return res.x


def _simplex(x0, step=0.06):
    pts = [x0]
    for i in range(len(x0)):
        p = np.array(x0, dtype=float)
        p[i] += step if p[i] + step <= BOUNDS[i][1] else -step
        pts.append(p)
    return np.array(pts)


def report(sess, theta, tag):
    sess["warm"] = None
    t, rms, v, plan = evaluate(sess, theta, n_iter=20, use_warm=False)
    err = t - sess["t_real"]
    print(f"   [{tag}] {sess['name']:12s} real {sess['t_real']:.3f}s  sim {t:.3f}s  err {err:+.3f}s   speed RMS {rms:.1f} km/h   "
          f"top speed sim {v.max()*3.6:.0f} / real {sess['v_real'].max():.0f}")
    return err, rms


def _task(name: str, sessions: dict, names: list, out_dir: str):
    """One independent unit of work (so folds can run in parallel processes)."""
    if name.startswith("loo:"):
        held = name.split(":", 1)[1]
        th = fit([sessions[n] for n in names if n != held])
        err, rms = report(sessions[held], th, "HELD-OUT")
        res = {"theta": [float(a) for a in th], "lap_time_error_s": float(err), "speed_rms_kmh": float(rms)}
    elif name == "final":
        th = fit([sessions[n] for n in names])
        res = {"theta": [float(a) for a in th], "in_sample": {}}
        for n in names:
            e, r = report(sessions[n], th, "in-sample")
            res["in_sample"][n] = {"lap_time_error_s": float(e), "speed_rms_kmh": float(r)}
    elif name == "nominal":
        res = {"in_sample": {}}
        for n in names:
            e, r = report(sessions[n], np.ones(4), "nominal")
            res["in_sample"][n] = {"lap_time_error_s": float(e), "speed_rms_kmh": float(r)}
    else:
        raise SystemExit(f"unknown task {name}")
    json.dump(res, open(os.path.join(out_dir, name.replace(":", "_") + ".json"), "w"), indent=2)


def merge(out_dir: str, out_path: str):
    final = json.load(open(os.path.join(out_dir, "final.json")))
    loo = {}
    for n in SESSIONS:
        p = os.path.join(out_dir, f"loo_{n}.json")
        if os.path.exists(p):
            loo[n] = json.load(open(p))
    out = {
        "parameters": dict(zip(PARAM_NAMES, final["theta"])),
        "in_sample": final["in_sample"],
        "leave_one_out": loo,
        "notes": "Fitted with the full energy-limited model on real 2026 qualifying laps (OpenF1). "
                 "Use leave_one_out errors, not in_sample, as the honest accuracy estimate.",
    }
    if loo:
        errs = np.array([v["lap_time_error_s"] for v in loo.values()])
        out["loo_rms_lap_error_s"] = float(np.sqrt(np.mean(errs ** 2)))
        out["loo_mean_lap_error_s"] = float(np.mean(errs))
        print(f"LOO lap-time error: mean {np.mean(errs):+.3f}s, RMS {np.sqrt(np.mean(errs**2)):.3f}s")
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="final fit only (no cross-validation)")
    ap.add_argument("--task", default=None, help="run one task: nominal | final | loo:<session>  (for parallel runs)")
    ap.add_argument("--merge", action="store_true", help="merge task outputs into the calibration file")
    ap.add_argument("--work-dir", default="calibration_work")
    ap.add_argument("--config", default="config_brazil_2026.json")
    ap.add_argument("--out", default="calibration_2026.json")
    args = ap.parse_args()
    os.makedirs(args.work_dir, exist_ok=True)

    if args.merge:
        merge(args.work_dir, args.out)
        return
    base_cfg = json.load(open(args.config, "r", encoding="utf-8"))
    sessions = {n: load_session(n, base_cfg) for n in SESSIONS}
    names = list(sessions)
    if args.task:
        _task(args.task, sessions, names, args.work_dir)
        return
    tasks = ["nominal", "final"] + ([] if args.quick else [f"loo:{n}" for n in names])
    for tname in tasks:
        _task(tname, sessions, names, args.work_dir)
    merge(args.work_dir, args.out)


if __name__ == "__main__":
    main()
