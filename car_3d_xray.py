"""
car_3d_xray.py — Off-screen PyVista 3D translucent X-Ray F1 car renderer with
articulated chassis pitch, roll, vertical heave (ride height), steering front wheels,
glowing internal suspension wishbones, and 6 dynamic white 3D force arrows.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pyvista as pv
from PIL import Image


def _make_tube(p1: np.ndarray, p2: np.ndarray, radius: float = 0.015, n_sides: int = 14) -> pv.PolyData:
    """Creates a 3D cylindrical linkage tube between two 3D points."""
    line = pv.Line(p1, p2, resolution=1)
    return line.tube(radius=radius, n_sides=n_sides)


def _make_lofted_body(sections: List[Tuple[float, float, float, float, float]], n_pts: int = 28) -> pv.PolyData:
    """
    Creates a smooth aerodynamic lofted surface from cross-sections:
    each tuple is (x, y_center, z_center, half_width_y, half_height_z).
    """
    theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    n_sec = len(sections)
    pts = []
    for x, yc, zc, wy, hz in sections:
        for t in theta:
            # Superellipse cross-section for authentic F1 monocoque/sidepod sculpting
            ct, st = np.cos(t), np.sin(t)
            y = yc + wy * np.sign(ct) * (np.abs(ct) ** 0.75)
            z = zc + hz * np.sign(st) * (np.abs(st) ** 0.80)
            pts.append([x, y, z])
    pts_arr = np.array(pts, dtype=np.float64)

    faces = []
    for i in range(n_sec - 1):
        row1 = i * n_pts
        row2 = (i + 1) * n_pts
        for j in range(n_pts):
            jn = (j + 1) % n_pts
            faces.extend([4, row1 + j, row2 + j, row2 + jn, row1 + jn])

    mesh = pv.PolyData(pts_arr, np.array(faces))
    return mesh.triangulate().compute_normals(auto_orient_normals=True, inplace=False)


def _make_curved_wing(
    x_front: float,
    x_rear: float,
    y_span: float,
    z_base: float,
    thickness: float,
    arch_z: float = 0.03,
    sweep_x: float = 0.08,
    ny: int = 25,
) -> pv.PolyData:
    """Creates a swept, arched 2026 aerodynamic wing element."""
    ys = np.linspace(-y_span, y_span, ny)
    sections = []
    for y in ys:
        u = abs(y) / max(1e-6, y_span)
        x_mid = 0.5 * (x_front + x_rear) - sweep_x * (u ** 1.6)
        chord = (x_front - x_rear) * (1.0 - 0.18 * (u ** 2))
        z_mid = z_base + arch_z * np.sin(np.pi * u)
        sections.append((x_mid, y, z_mid, 0.5 * chord, 0.5 * thickness))

    # Build wing surface along span
    n_prof = 16
    alpha = np.linspace(0, 2 * np.pi, n_prof, endpoint=False)
    pts = []
    for x_mid, y, z_mid, hc, ht in sections:
        for a in alpha:
            px = x_mid + hc * np.cos(a)
            pz = z_mid + ht * np.sin(a) + 0.012 * np.cos(a)
            pts.append([px, y, pz])
    pts_arr = np.array(pts, dtype=np.float64)
    faces = []
    for i in range(len(sections) - 1):
        r1 = i * n_prof
        r2 = (i + 1) * n_prof
        for j in range(n_prof):
            jn = (j + 1) % n_prof
            faces.extend([4, r1 + j, r2 + j, r2 + jn, r1 + jn])
    return pv.PolyData(pts_arr, np.array(faces)).triangulate().compute_normals(auto_orient_normals=True, inplace=False)


def _merge_meshes(meshes: List[pv.DataSet]) -> pv.PolyData:
    poly_list = [m.extract_surface() if not isinstance(m, pv.PolyData) else m for m in meshes]
    merged = pv.merge(poly_list)
    return merged.compute_normals(auto_orient_normals=True, inplace=False)


class F1XRayCarRenderer:
    """
    Off-screen PyVista 3D renderer for the 2026 F1 translucent X-Ray car rig.
    Uses in-place VTK actor matrix transformations per frame for high-speed video rendering.
    """

    def __init__(
        self,
        window_size: Tuple[int, int] = (710, 560),
        motion_scale: float = 2.5,
    ):
        self.window_size = list(window_size)
        self.motion_scale = motion_scale

        # Base yaw (-153.5 deg) + scale (0.82) so that with the exact camera specification
        # camera.position = [(-3.8, -2.6, 2.3), (0.0, 0.0, 0.25), (0.0, 0.0, 1.0)],
        # the car and its 6 force arrows sit cleanly between the RR/RL and FR/FL telemetry blocks.
        self.base_yaw_rad = np.radians(-153.5)
        self.model_scale = 0.80
        c, s = np.cos(self.base_yaw_rad), np.sin(self.base_yaw_rad)
        sc = self.model_scale
        self.R_base = np.array([
            [c * sc, -s * sc, 0.0, 0.0],
            [s * sc,  c * sc, 0.0, 0.0],
            [0.0,     0.0,    sc,  0.24],
            [0.0,     0.0,    0.0, 1.0],
        ], dtype=np.float64)

        self.plotter = pv.Plotter(off_screen=True, window_size=self.window_size)
        self.plotter.set_background("#08080A")
        try:
            self.plotter.enable_anti_aliasing("ssaa")
        except Exception:
            try:
                self.plotter.enable_anti_aliasing("msaa")
            except Exception:
                pass

        # Exact camera specification from Section 4
        self.plotter.camera.position = [(-3.8, -2.6, 2.3), (0.0, 0.0, 0.25), (0.0, 0.0, 1.0)][0]
        self.plotter.camera.focal_point = [(-3.8, -2.6, 2.3), (0.0, 0.0, 0.25), (0.0, 0.0, 1.0)][1]
        self.plotter.camera.up = [(-3.8, -2.6, 2.3), (0.0, 0.0, 0.25), (0.0, 0.0, 1.0)][2]
        self.plotter.camera.view_angle = 36.5

        self._build_scene()

    def _local_to_world_matrix(
        self,
        dx: float = 0.0,
        dy: float = 0.0,
        dz: float = 0.0,
        roll_rad: float = 0.0,
        pitch_rad: float = 0.0,
        yaw_rad: float = 0.0,
        pivot: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        scale_xyz: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> np.ndarray:
        """Builds a 4x4 homogeneous transformation matrix in world space."""
        px, py, pz = pivot
        sx, sy, sz = scale_xyz

        T_neg = np.eye(4)
        T_neg[:3, 3] = [-px, -py, -pz]

        S = np.diag([sx, sy, sz, 1.0])

        cr, sr = np.cos(roll_rad), np.sin(roll_rad)
        cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
        cy, sy_a = np.cos(yaw_rad), np.sin(yaw_rad)

        Rx = np.array([[1, 0, 0, 0], [0, cr, -sr, 0], [0, sr, cr, 0], [0, 0, 0, 1]], dtype=np.float64)
        Ry = np.array([[cp, 0, sp, 0], [0, 1, 0, 0], [-sp, 0, cp, 0], [0, 0, 0, 1]], dtype=np.float64)
        Rz = np.array([[cy, -sy_a, 0, 0], [sy_a, cy, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)

        T_pos = np.eye(4)
        T_pos[:3, 3] = [px + dx, py + dy, pz + dz]

        M_local = T_pos @ Rz @ Ry @ Rx @ S @ T_neg
        return self.R_base @ M_local

    def _build_scene(self) -> None:
        """Constructs all hierarchical 3D meshes in local car coordinates (+X Front, +Y Left, +Z Up)."""
        # ---------------------------------------------------------------------
        # 1. SMOKED-GLASS BODYWORK (#B0B5BC, opacity=0.26)
        # ---------------------------------------------------------------------
        body_parts = []

        # Sculpted Nose Cone -> Monocoque -> Engine Cover loft
        fuselage_sections = [
            (2.08, 0.0, 0.15, 0.08, 0.035),
            (1.75, 0.0, 0.19, 0.14, 0.065),
            (1.35, 0.0, 0.25, 0.20, 0.115),
            (0.85, 0.0, 0.30, 0.25, 0.165),
            (0.35, 0.0, 0.32, 0.28, 0.190),
            (-0.15, 0.0, 0.36, 0.29, 0.240),
            (-0.75, 0.0, 0.35, 0.24, 0.215),
            (-1.35, 0.0, 0.29, 0.15, 0.140),
            (-1.75, 0.0, 0.24, 0.08, 0.085),
        ]
        body_parts.append(_make_lofted_body(fuselage_sections))

        # Sculpted Sidepods (Left & Right) with Coke-bottle undercut
        for sign in (-1.0, 1.0):
            sp_sections = [
                (0.55, sign * 0.42, 0.27, 0.18, 0.13),
                (0.25, sign * 0.46, 0.28, 0.22, 0.15),
                (-0.25, sign * 0.43, 0.26, 0.20, 0.14),
                (-0.75, sign * 0.34, 0.23, 0.15, 0.11),
                (-1.25, sign * 0.22, 0.20, 0.10, 0.08),
            ]
            body_parts.append(_make_lofted_body(sp_sections))

        # Sculpted Ground-Effect Floor & Rear Diffuser Tunnels
        floor_sections = [
            (1.05, 0.0, 0.08, 0.56, 0.022),
            (0.45, 0.0, 0.07, 0.68, 0.025),
            (-0.35, 0.0, 0.07, 0.68, 0.025),
            (-1.15, 0.0, 0.08, 0.64, 0.028),
            (-1.65, 0.0, 0.14, 0.56, 0.055),
            (-1.92, 0.0, 0.21, 0.52, 0.075),
        ]
        body_parts.append(_make_lofted_body(floor_sections))

        # Airbox intake & Roll Hoop + Cockpit Halo
        airbox_sections = [
            (0.02, 0.0, 0.58, 0.11, 0.10),
            (-0.35, 0.0, 0.57, 0.10, 0.11),
            (-0.95, 0.0, 0.49, 0.06, 0.08),
            (-1.45, 0.0, 0.39, 0.03, 0.05),
        ]
        body_parts.append(_make_lofted_body(airbox_sections))

        halo_r = _make_tube(np.array([-0.05, -0.22, 0.48]), np.array([0.46, 0.0, 0.52]), radius=0.020)
        halo_l = _make_tube(np.array([-0.05,  0.22, 0.48]), np.array([0.46, 0.0, 0.52]), radius=0.020)
        halo_c = _make_tube(np.array([0.46, 0.0, 0.52]), np.array([0.56, 0.0, 0.38]), radius=0.018)
        body_parts.extend([halo_r, halo_l, halo_c])

        # 2026 Swept Multi-Element Front Wing + Curved Endplates
        body_parts.append(_make_curved_wing(2.14, 1.84, 0.82, 0.09, 0.028, arch_z=0.025, sweep_x=0.10))
        body_parts.append(_make_curved_wing(1.98, 1.72, 0.80, 0.13, 0.024, arch_z=0.030, sweep_x=0.09))
        body_parts.append(_make_curved_wing(1.86, 1.62, 0.76, 0.17, 0.020, arch_z=0.032, sweep_x=0.08))
        for sign in (-1.0, 1.0):
            ep_sections = [
                (2.06, sign * 0.82, 0.16, 0.016, 0.09),
                (1.82, sign * 0.82, 0.20, 0.018, 0.12),
                (1.60, sign * 0.80, 0.23, 0.016, 0.11),
            ]
            body_parts.append(_make_lofted_body(ep_sections))

        # 2026 Active Rear Wing + Beam Wing + Curved Endplates
        body_parts.append(_make_curved_wing(-1.68, -1.94, 0.52, 0.58, 0.032, arch_z=-0.015, sweep_x=-0.04))
        body_parts.append(_make_curved_wing(-1.76, -1.98, 0.50, 0.70, 0.036, arch_z=-0.010, sweep_x=-0.03))
        body_parts.append(_make_curved_wing(-1.66, -1.88, 0.48, 0.33, 0.024, arch_z=0.010, sweep_x=-0.02))
        for sign in (-1.0, 1.0):
            rw_ep = [
                (-1.66, sign * 0.51, 0.56, 0.015, 0.22),
                (-1.84, sign * 0.51, 0.58, 0.016, 0.24),
                (-2.00, sign * 0.49, 0.58, 0.014, 0.22),
            ]
            body_parts.append(_make_lofted_body(rw_ep))
            body_parts.append(_make_tube(np.array([-1.52, sign * 0.09, 0.34]), np.array([-1.76, sign * 0.09, 0.60]), radius=0.016))

        body_mesh = _merge_meshes(body_parts)
        self.actor_body = self.plotter.add_mesh(
            body_mesh,
            color="#B0B5BC",
            opacity=0.26,
            smooth_shading=True,
            specular=0.70,
            specular_power=35,
        )
        # Subtle structural rib lines for CAD X-Ray translucency
        feature_edges = body_mesh.extract_feature_edges(
            boundary_edges=True, non_manifold_edges=False, feature_edges=True, feature_angle=28.0
        )
        self.actor_body_wire = self.plotter.add_mesh(
            feature_edges,
            color="#D8DEE9",
            opacity=0.14,
            line_width=1.1,
        )

        # ---------------------------------------------------------------------
        # 2. GLOWING INTERNAL X-RAY LINKAGES (opacity=0.95)
        # ---------------------------------------------------------------------
        # 2A. Magenta/Pink (#D95B96) Front & Rear Double Wishbones + Central Keel
        wishbone_parts = []
        x_f = 1.28
        for sign in (-1.0, 1.0):
            y_in = sign * 0.16
            y_out = sign * 0.72
            # Upper wishbone A-arm
            wishbone_parts.append(_make_tube(np.array([x_f + 0.18, y_in, 0.35]), np.array([x_f, y_out, 0.37]), 0.014))
            wishbone_parts.append(_make_tube(np.array([x_f - 0.18, y_in, 0.34]), np.array([x_f, y_out, 0.37]), 0.014))
            # Lower wishbone A-arm
            wishbone_parts.append(_make_tube(np.array([x_f + 0.20, y_in, 0.17]), np.array([x_f, y_out, 0.19]), 0.015))
            wishbone_parts.append(_make_tube(np.array([x_f - 0.20, y_in, 0.17]), np.array([x_f, y_out, 0.19]), 0.015))
            # Steering track rod
            wishbone_parts.append(_make_tube(np.array([x_f + 0.06, y_in, 0.26]), np.array([x_f + 0.05, y_out, 0.27]), 0.012))

        x_r = -1.32
        for sign in (-1.0, 1.0):
            y_in = sign * 0.15
            y_out = sign * 0.70
            # Upper wishbone
            wishbone_parts.append(_make_tube(np.array([x_r + 0.22, y_in, 0.37]), np.array([x_r, y_out, 0.39]), 0.014))
            wishbone_parts.append(_make_tube(np.array([x_r - 0.16, y_in, 0.36]), np.array([x_r, y_out, 0.39]), 0.014))
            # Lower wishbone + driveshaft
            wishbone_parts.append(_make_tube(np.array([x_r + 0.22, y_in, 0.18]), np.array([x_r, y_out, 0.20]), 0.015))
            wishbone_parts.append(_make_tube(np.array([x_r - 0.18, y_in, 0.18]), np.array([x_r, y_out, 0.20]), 0.015))
            wishbone_parts.append(_make_tube(np.array([x_r, y_in, 0.28]), np.array([x_r, y_out, 0.29]), 0.016))

        # Central floor longitudinal keel beam in magenta
        wishbone_parts.append(_make_tube(np.array([-1.42, 0.0, 0.13]), np.array([1.65, 0.0, 0.15]), 0.014))

        wishbone_mesh = _merge_meshes(wishbone_parts)
        self.actor_wishbones = self.plotter.add_mesh(
            wishbone_mesh,
            color="#D95B96",
            opacity=0.95,
            smooth_shading=True,
            ambient=0.50,
        )

        # 2B. Cyan (#4A90E2) Pushrods & Central Hybrid Power Unit Spine
        cyan_parts = []
        for sign in (-1.0, 1.0):
            cyan_parts.append(_make_tube(np.array([x_f + 0.04, sign * 0.14, 0.43]), np.array([x_f, sign * 0.70, 0.21]), 0.015))
            cyan_parts.append(_make_tube(np.array([x_r + 0.08, sign * 0.14, 0.44]), np.array([x_r, sign * 0.68, 0.21]), 0.015))

        # Central hybrid power unit spine
        cyan_parts.append(_make_tube(np.array([-1.32, 0.0, 0.27]), np.array([1.28, 0.0, 0.31]), 0.022))
        cyan_parts.append(_make_tube(np.array([-0.85, -0.10, 0.22]), np.array([0.45, -0.10, 0.24]), 0.014))
        cyan_parts.append(_make_tube(np.array([-0.85,  0.10, 0.22]), np.array([0.45,  0.10, 0.24]), 0.014))

        cyan_mesh = _merge_meshes(cyan_parts)
        self.actor_cyan = self.plotter.add_mesh(
            cyan_mesh,
            color="#4A90E2",
            opacity=0.95,
            smooth_shading=True,
            ambient=0.50,
        )

        # 2C. Orange (#E67E22) Inboard Rockers, Torsion Bars & Heave Dampers
        orange_parts = []
        orange_parts.append(_make_tube(np.array([x_f + 0.04, -0.15, 0.43]), np.array([x_f + 0.04, 0.15, 0.43]), 0.019))
        orange_parts.append(_make_tube(np.array([x_f - 0.14, -0.11, 0.37]), np.array([x_f + 0.14, -0.11, 0.43]), 0.015))
        orange_parts.append(_make_tube(np.array([x_f - 0.14,  0.11, 0.37]), np.array([x_f + 0.14,  0.11, 0.43]), 0.015))
        orange_parts.append(_make_tube(np.array([x_f + 0.04, 0.0, 0.19]), np.array([x_f + 0.04, 0.0, 0.43]), 0.017))

        orange_parts.append(_make_tube(np.array([x_r + 0.08, -0.14, 0.44]), np.array([x_r + 0.08, 0.14, 0.44]), 0.019))
        orange_parts.append(_make_tube(np.array([x_r - 0.14, -0.11, 0.37]), np.array([x_r + 0.15, -0.11, 0.44]), 0.015))
        orange_parts.append(_make_tube(np.array([x_r - 0.14,  0.11, 0.37]), np.array([x_r + 0.15,  0.11, 0.44]), 0.015))

        orange_mesh = _merge_meshes(orange_parts)
        self.actor_orange = self.plotter.add_mesh(
            orange_mesh,
            color="#E67E22",
            opacity=0.95,
            smooth_shading=True,
            ambient=0.55,
        )

        # Center of Gravity / Aero Center white sphere marker inside chassis
        cg_sphere = pv.Sphere(radius=0.055, center=(0.05, 0.05, 0.27))
        self.actor_cg = self.plotter.add_mesh(
            cg_sphere,
            color="#EAECEF",
            opacity=0.95,
            smooth_shading=True,
            ambient=0.65,
        )

        # ---------------------------------------------------------------------
        # 3. 4 WHEEL ASSEMBLIES (#222428, Front wheels yaw by steering_angle_deg)
        # ---------------------------------------------------------------------
        self.wheel_centers = {
            "FL": (1.28,  0.77, 0.33),
            "FR": (1.28, -0.77, 0.33),
            "RL": (-1.32,  0.75, 0.33),
            "RR": (-1.32, -0.75, 0.33),
        }
        self.wheel_actors = {}
        self.rim_actors = {}

        for corner, (wx, wy, wz) in self.wheel_centers.items():
            is_front = corner.startswith("F")
            width = 0.33 if is_front else 0.39
            tire = pv.Cylinder(center=(wx, wy, wz), direction=(0.0, 1.0, 0.0), radius=0.325, height=width, resolution=36)
            tire = tire.compute_normals(auto_orient_normals=True, inplace=False)
            rim = pv.Cylinder(center=(wx, wy, wz), direction=(0.0, 1.0, 0.0), radius=0.195, height=width + 0.01, resolution=28)
            rim = rim.compute_normals(auto_orient_normals=True, inplace=False)

            self.wheel_actors[corner] = self.plotter.add_mesh(
                tire,
                color="#222428",
                opacity=0.92,
                smooth_shading=True,
                specular=0.22,
                ambient=0.22,
            )
            self.rim_actors[corner] = self.plotter.add_mesh(
                rim,
                color="#141619",
                opacity=0.96,
                smooth_shading=True,
                specular=0.40,
            )

        # ---------------------------------------------------------------------
        # 4. 6 WHITE 3D FORCE VECTORS (#FFFFFF)
        # ---------------------------------------------------------------------
        unit_arrow_up = pv.Arrow(
            start=(0.0, 0.0, 0.0),
            direction=(0.0, 0.0, 1.0),
            tip_length=0.30,
            tip_radius=0.095,
            shaft_radius=0.038,
            scale=1.0,
        )
        unit_arrow_down = pv.Arrow(
            start=(0.0, 0.0, 0.0),
            direction=(0.0, 0.0, -1.0),
            tip_length=0.30,
            tip_radius=0.095,
            shaft_radius=0.038,
            scale=1.0,
        )
        unit_arrow_drag = pv.Arrow(
            start=(0.0, 0.0, 0.0),
            direction=(-0.72, 0.0, -0.69),
            tip_length=0.30,
            tip_radius=0.095,
            shaft_radius=0.038,
            scale=1.0,
        )

        self.arrow_tire_actors = {}
        for corner in ("FL", "FR", "RL", "RR"):
            self.arrow_tire_actors[corner] = self.plotter.add_mesh(
                unit_arrow_up.copy(),
                color="#FFFFFF",
                opacity=1.0,
                smooth_shading=True,
                ambient=0.90,
            )

        self.actor_arrow_downforce = self.plotter.add_mesh(
            unit_arrow_down.copy(),
            color="#FFFFFF",
            opacity=1.0,
            smooth_shading=True,
            ambient=0.90,
        )
        self.actor_arrow_drag = self.plotter.add_mesh(
            unit_arrow_drag.copy(),
            color="#FFFFFF",
            opacity=1.0,
            smooth_shading=True,
            ambient=0.90,
        )

    def render_frame(self, sample: Dict[str, Any]) -> Image.Image:
        """
        Updates the 3D car rig's pitch, roll, heave, front wheel steering, and 6 force
        vectors for the given telemetry sample, and returns a 710x560 RGBA PIL Image.
        """
        # 1. Dynamic Chassis Pitch, Roll & Heave (scaled by 2.5x visual multiplier)
        heave_m = -(float(sample["heave_mm"]) / 1000.0) * self.motion_scale
        roll_rad = np.radians(float(sample["roll_deg"]) * self.motion_scale)
        pitch_rad = np.radians(float(sample["pitch_deg"]) * self.motion_scale)

        M_chassis = self._local_to_world_matrix(
            dz=heave_m,
            roll_rad=roll_rad,
            pitch_rad=pitch_rad,
            pivot=(0.0, 0.0, 0.25),
        )
        for actor in (
            self.actor_body,
            self.actor_body_wire,
            self.actor_wishbones,
            self.actor_cyan,
            self.actor_orange,
            self.actor_cg,
        ):
            actor.user_matrix = M_chassis

        # 2. Update 4 Wheels (Front wheels yaw dynamically by steering_angle_deg)
        road_steer_rad = np.radians(np.clip(float(sample["steering_deg"]) * 0.28, -24.0, 24.0))
        for corner, (wx, wy, wz) in self.wheel_centers.items():
            yaw_w = road_steer_rad if corner.startswith("F") else 0.0
            M_wheel = self._local_to_world_matrix(
                yaw_rad=yaw_w,
                pivot=(wx, wy, wz),
            )
            self.wheel_actors[corner].user_matrix = M_wheel
            self.rim_actors[corner].user_matrix = M_wheel

        # 3. Update 6 White Force Arrows
        tire_loads = {
            "FL": float(sample["fz_fl_kn"]),
            "FR": float(sample["fz_fr_kn"]),
            "RL": float(sample["fz_rl_kn"]),
            "RR": float(sample["fz_rr_kn"]),
        }
        # Positioned at the visible contact/tire edge matching the reference broadcast graphic
        arrow_anchors = {
            "FL": (1.60,  0.76, 0.04),
            "FR": (1.60, -0.76, 0.04),
            "RL": (-0.98,  0.86, 0.16),
            "RR": (-0.62, -0.72, 0.26),
        }
        for corner, fz_kn in tire_loads.items():
            ax, ay, az = arrow_anchors[corner]
            length = float(np.clip(0.18 + 0.095 * fz_kn, 0.16, 0.78))
            M_arr = self._local_to_world_matrix(
                dx=ax,
                dy=ay,
                dz=az,
                scale_xyz=(0.55, 0.55, length),
            )
            self.arrow_tire_actors[corner].user_matrix = M_arr

        # Downforce arrow at center of gravity (pointing downward)
        df_kg = float(sample["downforce_kg"])
        df_len = float(np.clip(0.24 + 0.00032 * df_kg, 0.24, 0.70))
        M_df = self._local_to_world_matrix(
            dx=0.18,
            dy=-0.16,
            dz=0.60 + heave_m,
            scale_xyz=(0.58, 0.58, df_len),
        )
        self.actor_arrow_downforce.user_matrix = M_df

        # Drag arrow at rear wing (pointing rearward/downward)
        dr_kg = float(sample["drag_kg"])
        dr_len = float(np.clip(0.22 + 0.00095 * dr_kg, 0.22, 0.68))
        M_dr = self._local_to_world_matrix(
            dx=-1.45,
            dy=-0.04,
            dz=0.88 + heave_m,
            scale_xyz=(dr_len, 0.58, dr_len),
        )
        self.actor_arrow_drag.user_matrix = M_dr

        # Render against #08080A for true dark smoked-glass blending, then add alpha channel
        self.plotter.render()
        rgb_arr = self.plotter.screenshot(transparent_background=False, return_img=True)
        # Compute smooth alpha mask where pixels differ from #08080A (8, 8, 10)
        diff = np.max(np.abs(rgb_arr.astype(np.int16) - np.array([8, 8, 10], dtype=np.int16)), axis=-1)
        alpha = np.clip(diff * 18, 0, 255).astype(np.uint8)
        rgba_arr = np.dstack([rgb_arr, alpha])
        return Image.fromarray(rgba_arr, mode="RGBA")

    def close(self) -> None:
        self.plotter.close()


if __name__ == "__main__":
    from physics_2026 import simulate_brazil_2026_lap
    tel = simulate_brazil_2026_lap()
    sample = tel.sample_at_time(20.5)
    renderer = F1XRayCarRenderer()
    img = renderer.render_frame(sample)
    bg = Image.new("RGBA", img.size, (8, 8, 10, 255))
    bg.alpha_composite(img)
    bg.save("test_car_3d.png")
    renderer.close()
    print("Saved test_car_3d.png:", img.size, img.mode)
