# 2026 F1 São Paulo GP (Interlagos) Qualifying Telemetry & 3D Vehicle Dynamics Video Generator

Broadcast-grade `1920x1080` telemetry compositor and off-screen 3D translucent X-Ray vehicle dynamics video generator simulating a qualifying flying lap for the **2026 Formula 1 São Paulo Grand Prix** at **Autódromo José Carlos Pace (Interlagos — 4309 m)**.

Designed to replicate official Formula 1 broadcast graphics with high tabular typography standards, quasi-steady-state (QSS) vehicle physics conforming to 2026 FIA technical regulations, and articulated 3D multi-body suspension kinematics.

---

## Key Features & Visual Architecture

### 1. Broadcast HUD Aesthetics & Pixel-Exact Compositor
- **Warm Beige Theme**: Precision-graded `#F3EFE6` canvas background paired with deep charcoal (`#181A1E`) metrics and tabular geometric typography (`Bahnschrift` / `Chakra Petch`) to eliminate horizontal digit jitter.
- **Top Header**: Centered Brazilian flag vector roundel + bold header **`SÃO PAULO GRAND PRIX 2026`** with circuit metadata **`Autódromo José Carlos Pace - 4309 m - qualifying`**.
- **Top-Right Watermark**: Right-aligned clean tabular text **`Priyanshu Pratik`**.
- **Telemetry Line Graphs**: 3 progressive live graphs tracking distance $s \in [0, 4309\text{ m}]$:
  - **SPEED [km/h]**: 0 to 350 km/h with progressive crimson trace (`#D92632`) and moving cursor.
  - **POWER [kW]**: -400 to +450 kW overlaying `ICE demanded` (400 kW max) and signed `MGU-K` power.
  - **SOC [MJ]**: 0 to 4.0 MJ battery state of charge with progressive emerald gradient fill (`#27AE60`).
- **Driver Inputs Cluster**: Progressive throttle bar (green), brake bar (red), active gear indicator (max 7th gear), and vector steering wheel rotation icon with degree readout ($0^\circ$ to $95^\circ$).
- **2D Interlagos Track Map**: High-accuracy GPS survey circuit path dynamically painted by powertrain state:
  - Royal Blue (`#1E6AD4`): **DEPLOY** ($+5\text{ kW}$ to $+350\text{ kW}$)
  - Crimson Red (`#B82525`): **HARVEST** (Overrun braking down to $-350\text{ kW}$ & part-throttle recovery)
  - Dark Charcoal (`#2C2F33`): **OFF** (Neutral / 0 kW)
  - White Dot: Instantaneous leading vehicle position with $-350\text{ to }+350\text{ kW}$ gradient legend.
- **Digital Readouts & Metrics**:
  - Live lap time clock, instant speed, dynamic MGU-K pill badge (`DEPLOY`, `HARVEST`, `OFF`).
  - Total fuel consumed (integrated from $3000\text{ MJ/h}$ energy flow limit $\approx 0.908\text{ kg}$).
  - Energy recovered breakdown bars: `Overrun` (braking), `Part throttle` (corner apexes), and `Superclip` ($0.00\text{ MJ}$ in qualifying).
  - 5-parameter Lap Time Sensitivity perturbation table (`Power +10 kW`, `Downforce +1%`, `Drag -1%`, `Front grip +1%`, `Rear grip +1%`).
  - Sector splits table (`S1`, `S2`, `S3`) with live elapsed time locking in at sector boundaries.
  - Real-time driver phase pie chart (`Full throttle`, `Part throttle`, `Braking`, `Neutral`) and limiting factor telemetry (`Power-limited` vs `Grip-limited`).

---

### 2. 2026 FIA Technical Regulations & Energy Strategy Engine

- **Realistic 2026 Qualifying Pace**: Calibrated to an authentic 2026 pole qualifying lap of **`1:10.840`**:
  - **Sector 1**: `18.285 s` (Start/Finish straight, Senna S $T1\text{–}T2$, Curva do Sol $T3$, through mid *Reta Oposta*)
  - **Sector 2**: `35.915 s` (*Descida do Lago* $T4\text{–}T5$, *Ferradura* $T6\text{–}T7$, *Laranjinha* $T8$, *Pinheirinho* $T9$, *Bico de Pato* $T10$, *Mergulho* $T11$, *Junção* $T12$)
  - **Sector 3**: `16.640 s` (Steep uphill acceleration through *Subida dos Boxes* $T13$ and *Arquibancadas* $T14$ back to the line)
- **Authentic Gear Usage**: 2026 gear ratios cap maximum speed at **7th gear** (~$322\text{ km/h}$ peak straight-line speed).
- **2026 Power Unit & Energy Recovery**:
  - Internal Combustion Engine (ICE): Up to $400\text{ kW}$ ($536\text{ hp}$) limited by $3000\text{ MJ/h}$ fuel energy flow.
  - Electric Motor Generator Unit - Kinetic (MGU-K): Up to $350\text{ kW}$ ($469\text{ hp}$) deployment with mandatory 2026 FIA high-speed taper above $290\text{ km/h}$:
    $$P_{\text{MGU-K}}(v) = \max\left(0,\, 350 - 6.36(v - 290)\right)$$
  - **Qualifying Energy Strategy (Zero Superclipping)**: In qualifying ("hot lap"), drivers never harvest against 100% full throttle on straights (`Superclip = 0.00 MJ`); straights are pure, unbroken `DEPLOY` with natural regulatory high-speed taper. Braking and harvesting occur exclusively in the 8 genuine corner entries (`HARVEST` up to $-350\text{ kW}$) and slow technical apexes (`-50 to -240 kW`).
- **2026 Active Aerodynamics (Z-Mode vs. X-Mode)**:
  - **Z-Mode (High Downforce)**: Active in all cornering and braking zones ($C_L = 2.68$, $C_D = 0.82$, aero balance $46.2\%$).
  - **X-Mode (Low Drag Straight Mode)**: Smoothly engages over $0.25\text{ s}$ on *Reta Oposta* ($s = 725\text{–}1310\text{ m}$) and the Main Straight ($s = 3460\text{–}180\text{ m}$) when throttle $\ge 98\%$ and lateral acceleration $< 0.5\text{ G}$, reducing drag by $52\%$ and downforce by $28\%$ with aero balance shifting to $41.0\%$.
- **Atmospheric & 3D Topography Physics**:
  - Altitude: $\sim 765\text{ m}$ above sea level $\rightarrow$ air density $\rho = 1.110\text{ kg/m}^3$.
  - 3D Longitudinal Slope Gravity: Incorporates $F_{\text{grade}} = mg\sin\theta = mg\frac{dz}{ds}$ across the $44\text{ m}$ elevation variation (steep downhill plunge through Senna S & Descida do Lago; grueling uphill climb from Junção to the finish line).

---

### 3. Off-Screen 3D Translucent X-Ray Car Dynamics

- **High-Poly CAD Model**: 235,000-triangle 2026 F1 carbon chassis rendered off-screen with `PyVista` using a translucent smoked-glass shader (`alpha = 0.52`).
- **100% Perfectly Round Wheels**: Custom 128-segment revolved 18-inch Pirelli low-profile slick tires with BBS wheel rim geometry and zero polygon faceting or mesh artifacts.
- **Kinematic Articulation**:
  - Articulated front wheel steering synchronized with rack-and-pinion Ackermann angles.
  - Dynamic ride height heave: front ride height compresses from $42\text{ mm}$ static to $10\text{ mm}$ under full aerodynamic load; rear compresses from $90\text{ mm}$ to $30\text{ mm}$.
  - Pitch ($\theta$) and Roll ($\phi$) responding dynamically to longitudinal and lateral G-forces.
- **4-Corner Telemetry Overlay**: FL, FR, RL, RR vertical tire loads $F_z\text{ [kN]}$, dynamic camber $\gamma\text{ [}^\circ\text{]}$, and slip angles $\varphi\text{ [}^\circ\text{]}$, capturing realistic inside-front tire unloading down to $0.1\text{ kN}$ in tight hairpins.
- **Dynamic 3D Force Vectors**:
  - Crimson vertical tire load vectors ($F_z$) at each contact patch.
  - Royal blue aerodynamic and chassis load vectors positioned along the floor and diffuser.

---

## Repository Structure

```
├── config_brazil_2026.json    # Externalized FIA 2026 & Interlagos simulation parameters
├── track_loader.py            # High-accuracy closed-loop 3D spline & elevation track loader
├── physics_2026.py            # 2026 FIA QSS forward-backward velocity & energy solver
├── car_3d_xray.py             # Off-screen PyVista 3D translucent X-Ray F1 car renderer
├── hud_renderer.py            # Pixel-exact 1920x1080 2D broadcast compositor (PyCairo + PIL)
├── generate_video.py          # Master CLI pipeline supporting --preview and --full
├── requirements.txt           # Python dependency specifications
└── assets/                    # 3D meshes (chassis, wishbones, and 128-segment round wheels)
```

---

## Installation & Requirements

Ensure you have Python 3.10+ and `ffmpeg` installed on your system.

```bash
pip install -r requirements.txt
```

*Required packages:* `numpy`, `scipy`, `fastf1`, `pyvista`, `vtk`, `Pillow`, `matplotlib`, `pycairo`, `tqdm`.

---

## Usage

### 1. Render Preview Video & Diagnostic Frames
Renders a 10-second 30 FPS test video (`preview_brazil_2026.mp4`) and saves 3 high-resolution 1080p diagnostic PNG frames:
```bash
python generate_video.py --preview
```

**Diagnostic Outputs:**
- `frame_01_superclip.png`: High-speed straight deployment ($322\text{ km/h}$, 7th gear, full track map visible)
- `frame_02_braking.png`: Heavy braking zone into *Descida do Lago* (Overrun harvest $-350\text{ kW}$, $85\%$ brake, nose dive)
- `frame_03_cornering.png`: Technical cornering apex exit ($2.27\text{ G}$ lateral load, inside front wheel unloaded)
- `preview_brazil_2026.mp4`: 10-second 30 FPS preview clip

### 2. Render Full Broadcast Lap Video
Renders the complete 70.840s qualifying lap + 2.0s outro at 60 FPS 1080p via direct FFmpeg pipe:
```bash
python generate_video.py --full
```

Output:
- `brazil_2026_quali_1080p60.mp4` (`1920x1080`, `60 FPS`, H.264 / AAC)
