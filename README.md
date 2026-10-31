# 2026 F1 São Paulo GP (Interlagos) Qualifying Telemetry & 3D Vehicle Dynamics Video Generator

Broadcast-grade `1920x1080` telemetry compositor and off-screen 3D translucent X-Ray vehicle dynamics video generator simulating a predictive qualifying flying lap for the **2026 Formula 1 São Paulo Grand Prix** at **Autódromo José Carlos Pace (Interlagos — 4309 m)**.

Designed to replicate official Formula 1 broadcast graphics with high tabular typography standards, an honest predictive Quasi-Steady-State (QSS) vehicle physics engine conforming strictly to the 2026 FIA Technical Regulations, and articulated 3D multi-body suspension kinematics.

---

## 1. Predictive Physics & Regulation Engine (`qss_core.py`, `physics_2026.py`)

Lap time, speed, throttle, brake, MGU-K power and battery SOC are **outputs** of an energy-constrained
optimal-control solution. Nothing is rescaled to a target time (`event.reference_comparison` in the config is
only printed for comparison).

### Vehicle model
- Forward/backward quasi-steady-state solver (pure-python, ~0.06 s per solve): corner limit, braking limit,
  acceleration limit, 3D grade, rolling resistance, drag, downforce.
- Load-sensitive tyre friction with a friction ellipse. Traction is limited by the **driven (rear) axle load**
  including longitudinal load transfer; braking uses all four tyres. No hidden acceleration caps.
- Active aero: straight-mode (low drag / low downforce) on straights detected from the track geometry, switched
  back to cornering mode shortly before each braking zone. No typed distance zones (`zone_mode: "manual"` to override).
- Moist-air density from altitude + weather inputs.

### Power unit and energy (the important part)
- ICE <= 400 kW (3000 MJ/h fuel-energy flow at 48 % efficiency). MGU-K <= 350 kW with the speed taper
  (350 kW to 290 km/h, linear to 0 at 345 km/h), harvest <= 350 kW.
- Battery: 4 MJ usable window, constant efficiencies, recharge <= 7 MJ per lap (config).
- MGU-K power p(s) is a decision variable. A sequential LP (exact adjoint gradient of lap time + line search)
  allocates the limited electrical energy to where it saves the most time. Harvest is free under braking and at
  corner apexes (spare ICE power) and costs lap time where the car is power limited (superclip).
- Because deployment is energy-limited, the SOC curve can no longer sit at zero while the car deploys at full power.

### Calibration and honest accuracy (`calibrate.py`, `validate.py`)
- Four physical global parameters (downforce, drag, tyre grip, ICE power) are fitted on **real 2026 qualifying
  laps** (Austria, Spain, Silverstone) using the full energy-limited model.
- Leave-one-out cross-validation: fit on two circuits, predict the third. Those unseen-circuit errors are the
  accuracy to expect and define the uncertainty window reported for Interlagos.
- The Interlagos prediction uses the fitted parameters stored in `calibration_2026.json`.

### Sensitivity analysis
Each of the five perturbations re-runs the full pipeline, including the energy re-optimisation.

---

## 2. Track Data & Geometry (`track_loader.py`)

- Source: OpenF1 position samples (X, Y, Z), cached in `data/`.
- Curvature from noisy GPS: second derivatives of 0.6 m-noise positions are unusable (the old approach implied
  7-15 g lateral on real laps). The raw GPS samples are recovered, **heading** is smoothed with a weighted
  smoothing spline (first derivative only), with heavy smoothing on gentle sections and light smoothing in
  tight corners. On three real laps the implied lateral g is p99 ~ 5 g, max <= 5.9 g.
- Elevation slope from smoothed Z.
- `data/interlagos_real_telemetry.npz` was already smoothed when cached. For better tight-corner radii
  regenerate raw samples with `python fetch_interlagos_raw.py` (needs network).
- Fails loudly if telemetry is unavailable.

---

## 3. Broadcast HUD & Off-Screen 3D Articulated Vehicle

### Broadcast HUD (`hud_renderer.py`)
- Precision-graded `#F3EFE6` warm beige background with deep charcoal typography.
- Tabular typography (`Bahnschrift` / `Chakra Petch`) preventing horizontal digit jitter.
- Progressive live line graphs: Speed (0-350 km/h), Power (-400 to +450 kW with ICE/MGU-K breakdown), and SOC (0-4.0 MJ).
- Dynamic 2D track map color-coded by powertrain mode (`DEPLOY`, `HARVEST`, `SUPERCLIP`, `OFF`).
- Driver telemetry cluster: throttle/brake bars, steering angle, 8-speed gear readout, G-force vector, and dynamic tyre loads ($F_z, \gamma, \varphi$).

### Off-Screen 3D X-Ray Renderer (`car_3d_xray.py`)
- 235,000-triangle 2026 F1 carbon chassis rendered off-screen with `PyVista` using a translucent smoked-glass shader.
- Custom 128-segment revolved 18-inch Pirelli low-profile slick tires with BBS wheel rim geometry.
- Articulated front wheel steering synchronized with Ackermann angles.
- Dynamic ride height heave (front compresses $42\text{ mm} \rightarrow 10\text{ mm}$, rear $90\text{ mm} \rightarrow 30\text{ mm}$), roll ($\phi$), and pitch ($\theta$).
- 3D vertical tyre load vectors ($F_z$) and aerodynamic downforce vectors.

---

## 4. Repository Structure

```
├── config_brazil_2026.json    # FIA 2026 + circuit parameters (nominal values)
├── calibration_2026.json      # fitted global parameters + leave-one-out accuracy (from calibrate.py)
├── qss_core.py                # energy-aware QSS solver, adjoint gradient, SLP energy optimiser
├── physics_2026.py            # builds the full telemetry (TelemetryResult) from the solution
├── track_loader.py            # real-telemetry track geometry (robust curvature)
├── calibrate.py               # fit on real 2026 laps + leave-one-out cross-validation
├── validate.py                # accuracy report, physics/regulation assertions, Interlagos prediction
├── cache_telemetry.py         # fetch real 2026 sessions from OpenF1
├── fetch_interlagos_raw.py    # optional: raw (unsmoothed) Interlagos GPS samples
├── car_3d_xray.py / hud_renderer.py / generate_video.py   # rendering pipeline (unchanged interface)
├── data/                      # cached real telemetry
└── assets/                    # 3D meshes
```

---

## 5. Usage

```bash
pip install -r requirements.txt
python calibrate.py          # a few minutes: fit + leave-one-out; writes calibration_2026.json
python validate.py           # accuracy, assertions, Interlagos prediction (+ artifacts/loo_speed_traces.png)
python generate_video.py --preview
python generate_video.py --full
```
