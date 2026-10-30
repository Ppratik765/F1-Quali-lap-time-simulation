# 2026 F1 São Paulo GP (Interlagos) Qualifying Telemetry & 3D Vehicle Dynamics Video Generator

Broadcast-grade `1920x1080` telemetry compositor and off-screen 3D translucent X-Ray vehicle dynamics video generator simulating a predictive qualifying flying lap for the **2026 Formula 1 São Paulo Grand Prix** at **Autódromo José Carlos Pace (Interlagos — 4309 m)**.

Designed to replicate official Formula 1 broadcast graphics with high tabular typography standards, an honest predictive Quasi-Steady-State (QSS) vehicle physics engine conforming strictly to the 2026 FIA Technical Regulations, and articulated 3D multi-body suspension kinematics.

---

## 1. Predictive Physics & Regulation Engine (`physics_2026.py`)

Unlike empirical or target-rescaled animations, this simulator uses an honest, forward-backward predictive solver where lap time, sector times, velocity profile, and energy strategy are 100% emergent outputs.

### Atmospheric & Topography Physics
- **Dynamic Moist Air Density**: Computed directly from track elevation ($h = 765\text{ m}$), forecast ambient temperature ($22.0^\circ\text{C}$), barometric sea-level pressure ($1015.0\text{ hPa}$), and relative humidity ($55\%$) via the barometric formula and the Tetens equation for vapor pressure ($\rho \approx 1.087\text{ kg/m}^3$).
- **3D Grade Resistance**: Longitudinal slope component $F_{\text{grade}} = m \cdot g \cdot \sin\theta(s) = m \cdot g \cdot \frac{dz}{ds}$ computed along the 44-metre vertical profile (steep plunge through Senna S & Descida do Lago; grueling uphill climb from Junção through Arquibancadas to the finish line).

### 2026 Active Aerodynamics (Z-Mode vs. X-Mode)
- **Z-Mode (Cornering & Braking)**: High downforce & high drag ($C_L = 2.68$, $C_D = 0.82$, aero balance $46.2\%$).
- **X-Mode (Low-Drag Straight Mode)**: Smoothly transitions over $0.25\text{ s}$ on *Reta Oposta* ($s = 725\text{–}1310\text{ m}$) and the Main Straight ($s = 3460\text{–}180\text{ m}$) when throttle $\ge 98\%$ and lateral acceleration $< 0.5\text{ G}$, reducing drag by $52\%$ and downforce by $28\%$ with front aero balance shifting to $41.0\%$.

### 2026 Power Unit & Energy Conservation
- **Internal Combustion Engine (ICE)**: Peak $400\text{ kW}$ ($536\text{ hp}$), chemical fuel mass flow integrated continuously from the $3000\text{ MJ/h}$ energy limit with lower heating value $\text{LHV} = 43.5\text{ MJ/kg}$ and thermal efficiency $\eta_{\text{ICE}} = 0.38$.
- **MGU-K Regulatory Speed Taper**: Complies strictly with 2026 FIA Power Unit Technical Regulations Article 5.4.8 (i):
  $$P_{\text{MGU-K,max}}(v) = \begin{cases}
    350\text{ kW} & v \le 290\text{ km/h} \\
    1800 - 5.0 \cdot v & 290 < v \le 340\text{ km/h} \\
    6900 - 20.0 \cdot v & 340 < v \le 345\text{ km/h} \\
    0\text{ kW} & v > 345\text{ km/h}
  \end{cases}$$
- **Energy Conservation**: Battery State of Charge $\text{SOC}(s)$ integrated from actual MGU-K electrical power with constant, documented efficiencies ($\eta_{\text{deploy}} = 0.96$, $\eta_{\text{harvest}} = 0.92$), constrained inside the $4.0\text{ MJ}$ usable capacity window and the FIA $7.0\text{ MJ}$ per lap recharge limit.
- **Emergent Qualifying Superclipping**: On high-speed straights where deployment tapers towards zero above $325\text{ km/h}$, the energy optimizer initiates superclipping (~2.8s total per lap, recovering ~0.65 MJ) to ensure full 350 kW deployment is available upon exiting low-speed corners.

### Tyre & Vehicle Dynamics
- **Load-Sensitive Tyre Friction**: Normal-load sensitive coefficient $\mu(F_z) = \mu_0 \cdot (F_z / F_{z0})^{-\alpha}$ with $\alpha = 0.08$.
- **Friction Ellipse Constraint**: Standard combined slip envelope $(F_x / F_{x,\text{max}})^2 + (F_y / F_{y,\text{max}})^2 \le 1$.
- **Normal Force Conservation**: 4-corner wheel loads account for static weight distribution ($45.5\%$ front), aerodynamic downforce, longitudinal weight transfer ($m \cdot a_x \cdot h_{\text{cg}} / L_{\text{wb}}$), and lateral load transfer split by front roll stiffness ($K_{\phi,f} = 0.58$, $K_{\phi,r} = 0.42$). Enforces $\sum F_z == m \cdot g \cdot \cos\theta + F_{\text{downforce}}$ with zero clip floors.
- **Drivetrain & 8-Speed Transmission**: Gear selection derived from speed and the 2026 engine speed operating range ($10200\text{–}12200\text{ RPM}$), ensuring all 8 gears are utilized realistically.

### Sensitivity Analysis
- Evaluates the complete predictive QSS pipeline across 5 unscaled parameter perturbations (`Power +10 kW`, `Downforce +1%`, `Drag -1%`, `Front grip +1%`, `Rear grip +1%`) computing exact cumulative lap time deltas $\Delta t(s) = t_{\text{pert}}(s) - t_{\text{base}}(s)$ without artificial scaling factors.

---

## 2. Track Data & Geometry (`track_loader.py`)

- **Telemetry Source**: 3D track coordinates ($X, Y, Z$) resampled to $1\text{ m}$ intervals ($0\text{ to }4309\text{ m}$) from qualifying telemetry data and cached locally in `data/interlagos_real_telemetry.npz`.
- **Differential Geometry**: Computes instantaneous track heading $\psi(s)$, signed 2D curvature $\kappa(s)$, and vertical slope $\sin\theta(s) = dz/ds$ directly using cubic spline derivatives with Gaussian smoothing ($\sigma = 3.0\text{ m}$).
- **Fails Loudly**: If telemetry cache is absent and live fetch fails, raises a explicit `RuntimeError` rather than silently substituting artificial geometry.

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
├── config_brazil_2026.json    # Externalized FIA 2026 & Interlagos simulation parameters
├── track_loader.py            # Real telemetry 3D spline & elevation track loader
├── physics_2026.py            # Predictive 2026 QSS forward-backward velocity & energy solver
├── car_3d_xray.py             # Off-screen PyVista 3D translucent X-Ray F1 car renderer
├── hud_renderer.py            # Pixel-exact 1920x1080 2D broadcast compositor (PyCairo + PIL)
├── generate_video.py          # Master CLI pipeline supporting --preview and --full
├── validate.py                # Comprehensive multi-circuit calibration, test & assertion suite
├── requirements.txt           # Python dependency specifications
├── data/                      # Cached real telemetry files
└── assets/                    # 3D meshes (chassis, wishbones, and 128-segment round wheels)
```

---

## 5. Usage

### 1. Render Preview Video & Diagnostic Frames
Renders a 10-second 30 FPS preview video (`preview_brazil_2026.mp4`) and saves 3 high-resolution 1080p diagnostic PNG frames:
```bash
python generate_video.py --preview
```

### 2. Render Full Broadcast Lap Video
Renders the complete predictive qualifying lap + 2.0s outro at 60 FPS 1080p via direct FFmpeg pipe:
```bash
python generate_video.py --full
```

### 3. Run Validation Suite
Executes multi-circuit calibration against completed qualifying sessions, evaluates hold-out circuit accuracy, verifies physical conservation laws, and outputs the honest Interlagos qualifying lap time prediction:
```bash
python validate.py
```
