# 2026 F1 São Paulo GP (Interlagos) Qualifying Telemetry & 3D Vehicle Dynamics Video Generator

Broadcast-grade `1920x1080` telemetry and 3D translucent X-Ray vehicle dynamics simulation for the **2026 Formula 1 São Paulo Grand Prix** at **Autódromo José Carlos Pace (Interlagos — 4309 m — Qualifying)**.

---

## Architecture

| Module | Description |
| :--- | :--- |
| [`config_brazil_2026.json`](./config_brazil_2026.json) | Externalized 2026 FIA Technical Regulations & Interlagos circuit parameters (altitude `765 m`, air density `1.110 kg/m³`, Active Aero Z-Mode/X-Mode zones, MGU-K `350 kW` deployment taper & `-250 kW` Superclip zones, vehicle mass `768 kg`, sector split distances). |
| [`track_loader.py`](./track_loader.py) | Extracts Interlagos (`4309 m`) 3D track geometry (`X, Y, Z`, distance $s$, signed curvature $\kappa(s)$, and vertical slope $\sin\theta = dz/ds$) resampled to $1\text{ m}$ intervals with built-in GPS survey closed-loop spline & elevation profile. |
| [`physics_2026.py`](./physics_2026.py) | 2026 FIA Quasi-Steady-State (QSS) forward-backward velocity solver, 3D elevation gravity component ($F_{\text{grade}} = mg\frac{dz}{ds}$), Active Aero (`Z-Mode` vs `X-Mode`), MGU-K state machine (`DEPLOY`, `HARVEST`, `SUPERCLIP`, `OFF`), 4-corner suspension & tire loads, and 5-parameter Lap Time Sensitivity perturbation calculator. |
| [`car_3d_xray.py`](./car_3d_xray.py) | Off-screen `PyVista` (`710x560`) translucent smoked-glass 2026 F1 car renderer with articulated chassis heave, roll, pitch ($2.5\times$ visual scale), steering front wheels, glowing internal double-wishbone/pushrod/rocker suspension linkages, and 6 dynamic white 3D force vectors. |
| [`hud_renderer.py`](./hud_renderer.py) | Pixel-exact `1920x1080` 2D broadcast compositor using `PyCairo` and `Pillow` with condensed geometric tabular typography (`Bahnschrift`). |
| [`generate_video.py`](./generate_video.py) | Master CLI pipeline supporting `--preview` (10s 30 FPS test clip + 3 diagnostic PNGs) and `--full` (full qualifying lap + 2s outro at 60 FPS 1080p via an `ffmpeg` pipe). |

---

## Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Render 10-Second Preview & Diagnostic Frames
```bash
python generate_video.py --preview
```
Outputs:
- `preview_brazil_2026.mp4` (`10 s`, `30 FPS`, `1920x1080`)
- `frame_01_superclip.png` (`SUPERCLIP` `-250 kW` at `100%` throttle on *Reta Oposta*)
- `frame_02_braking.png` (`HARVEST` `-350 kW` Overrun under heavy braking into *Descida do Lago*)
- `frame_03_cornering.png` (`DEPLOY` `+350 kW` high-G cornering with inside front wheel unloading)

### 3. Render Full 1080p60 Broadcast Video
```bash
python generate_video.py --full
```
Outputs:
- `brazil_2026_quali_1080p60.mp4` (`1:08.450` qualifying lap + `2.0 s` outro hold at `60 FPS`)
