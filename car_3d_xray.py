"""
car_3d_xray.py — Off-screen PyVista 3D translucent X-Ray F1 car renderer using
the segmented 235k-triangle Formula 1 CAD mesh assembly (assets/f1_*.vtp) with
articulated chassis pitch, roll, vertical heave (ride height), steering front wheels,
internal carbon/titanium suspension wishbones & rockers, and 6 dynamic red 3D force arrows.
"""

import os
from typing import Dict, Any, List, Tuple
import numpy as np
import pyvista as pv
from PIL import Image


def _make_tube(p1: np.ndarray, p2: np.ndarray, radius: float = 0.0055, n_sides: int = 14) -> pv.PolyData:
    """Creates a thin precision internal CAD linkage tube between two 3D points."""
    line = pv.Line(p1, p2, resolution=1)
    return line.tube(radius=radius, n_sides=n_sides)


def _merge_meshes(meshes: List[pv.DataSet]) -> pv.PolyData:
    poly_list = [m.extract_surface(algorithm="dataset_surface") if not isinstance(m, pv.PolyData) else m for m in meshes]
    merged = pv.merge(poly_list)
    return merged.compute_normals(auto_orient_normals=True, inplace=False)


class F1XRayCarRenderer:
    """
    Off-screen PyVista 3D renderer for the 2026 F1 translucent X-Ray CAD car rig.
    Uses in-place VTK actor matrix transformations per frame for high-speed video rendering.
    """

    def __init__(
        self,
        window_size: Tuple[int, int] = (710, 560),
        motion_scale: float = 2.5,
        assets_dir: str = "assets",
    ):
        self.window_size = list(window_size)
        self.motion_scale = motion_scale
        self.assets_dir = assets_dir

        self.base_yaw_rad = np.radians(-153.5)
        self.model_scale = 0.80
        c, s = np.cos(self.base_yaw_rad), np.sin(self.base_yaw_rad)
        sc = self.model_scale
        self.R_base = np.array([
            [c * sc, -s * sc, 0.0, 0.0],
            [s * sc,  c * sc, 0.0, 0.0],
            [0.0,     0.0,    sc,  0.23],
            [0.0,     0.0,    0.0, 1.0],
        ], dtype=np.float64)

        self.plotter = pv.Plotter(off_screen=True, window_size=self.window_size)
        self.plotter.set_background("#F3EFE6")
        try:
            self.plotter.enable_anti_aliasing("msaa")
        except Exception:
            pass

        # Exact camera specification from Section 4
        self.plotter.camera.position = (-3.8, -2.6, 2.3)
        self.plotter.camera.focal_point = (0.0, 0.0, 0.25)
        self.plotter.camera.up = (0.0, 0.0, 1.0)
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
        """Loads the high-detail F1 CAD meshes and sets up the X-Ray material shaders."""
        body_path = os.path.join(self.assets_dir, "f1_body.vtp")
        susp_path = os.path.join(self.assets_dir, "f1_susp.vtp")

        # 1. Smoked-Graphite Translucent F1 CAD Bodywork (#303642, opacity=0.35)
        body_mesh = pv.read(body_path)
        self.actor_body = self.plotter.add_mesh(
            body_mesh,
            color="#303642",
            opacity=0.35,
            smooth_shading=True,
            specular=0.75,
            specular_power=42,
            ambient=0.22,
        )

        # 2. Real CAD Aero-Section Suspension Wishbones & Pushrods
        susp_mesh = pv.read(susp_path)
        self.actor_susp = self.plotter.add_mesh(
            susp_mesh,
            color="#1A202C",
            opacity=0.95,
            smooth_shading=True,
            specular=0.65,
            specular_power=35,
            ambient=0.25,
        )

        # 3. Delicate Internal Inboard Damper / Bellcrank / Powertrain CAD Schematic Lines
        inboard_parts = []
        x_f, x_r = 1.135, -1.570
        # Front inboard torsion bar / heave damper assembly
        inboard_parts.append(_make_tube(np.array([x_f + 0.03, -0.11, 0.37]), np.array([x_f + 0.03, 0.11, 0.37]), 0.0065))
        inboard_parts.append(_make_tube(np.array([x_f - 0.10, -0.09, 0.34]), np.array([x_f + 0.10, -0.09, 0.38]), 0.0055))
        inboard_parts.append(_make_tube(np.array([x_f - 0.10,  0.09, 0.34]), np.array([x_f + 0.10,  0.09, 0.38]), 0.0055))
        inboard_parts.append(_make_tube(np.array([x_f + 0.03, 0.0, 0.18]), np.array([x_f + 0.03, 0.0, 0.37]), 0.0060))
        # Rear inboard damper assembly
        inboard_parts.append(_make_tube(np.array([x_r + 0.08, -0.10, 0.38]), np.array([x_r + 0.08, 0.10, 0.38]), 0.0065))
        inboard_parts.append(_make_tube(np.array([x_r - 0.10, -0.08, 0.34]), np.array([x_r + 0.12, -0.08, 0.38]), 0.0055))
        inboard_parts.append(_make_tube(np.array([x_r - 0.10,  0.08, 0.34]), np.array([x_r + 0.12,  0.08, 0.38]), 0.0055))
        # Thin internal driveline axis
        inboard_parts.append(_make_tube(np.array([x_r, 0.0, 0.24]), np.array([x_f, 0.0, 0.28]), 0.0060))

        inboard_mesh = _merge_meshes(inboard_parts)
        self.actor_inboard = self.plotter.add_mesh(
            inboard_mesh,
            color="#B85D19",
            opacity=0.88,
            smooth_shading=True,
            specular=0.60,
            ambient=0.35,
        )

        # Center of Gravity / Aero Center sphere marker inside chassis
        cg_sphere = pv.Sphere(radius=0.044, center=(-0.05, 0.0, 0.25))
        self.actor_cg = self.plotter.add_mesh(
            cg_sphere,
            color="#D92632",
            opacity=0.92,
            smooth_shading=True,
            ambient=0.50,
        )

        # 4. 4 Perfectly Round Revolved 18-Inch F1 Wheel & Tire Assemblies (FL, FR, RL, RR)
        self.wheel_centers = {
            "FL": (1.1353,  0.545, 0.266),
            "FR": (1.1353, -0.545, 0.266),
            "RL": (-1.5696,  0.540, 0.266),
            "RR": (-1.5696, -0.540, 0.266),
        }
        self.wheel_actors = {}
        for corner in ("FL", "FR", "RL", "RR"):
            w_path = os.path.join(self.assets_dir, f"f1_wheel_{corner}.vtp")
            w_mesh = pv.read(w_path)
            self.wheel_actors[corner] = self.plotter.add_mesh(
                w_mesh,
                color="#1E2128",
                opacity=0.92,
                smooth_shading=True,
                specular=0.35,
                specular_power=24,
                ambient=0.22,
            )

        # 5. 6 Dynamic 3D Force Vectors (#14161A high-contrast dark charcoal on beige)
        unit_arrow_up = pv.Arrow(
            start=(0.0, 0.0, 0.0),
            direction=(0.0, 0.0, 1.0),
            tip_length=0.28,
            tip_radius=0.088,
            shaft_radius=0.034,
            scale=1.0,
        )
        unit_arrow_down = pv.Arrow(
            start=(0.0, 0.0, 0.0),
            direction=(0.0, 0.0, -1.0),
            tip_length=0.28,
            tip_radius=0.088,
            shaft_radius=0.034,
            scale=1.0,
        )
        unit_arrow_drag = pv.Arrow(
            start=(0.0, 0.0, 0.0),
            direction=(-0.72, 0.0, -0.69),
            tip_length=0.28,
            tip_radius=0.088,
            shaft_radius=0.034,
            scale=1.0,
        )

        self.arrow_tire_actors = {}
        for corner in ("FL", "FR", "RL", "RR"):
            self.arrow_tire_actors[corner] = self.plotter.add_mesh(
                unit_arrow_up.copy(),
                color="#D92632",
                opacity=1.0,
                smooth_shading=True,
                ambient=0.65,
            )

        self.actor_arrow_downforce = self.plotter.add_mesh(
            unit_arrow_down.copy(),
            color="#D92632",
            opacity=1.0,
            smooth_shading=True,
            ambient=0.65,
        )
        self.actor_arrow_drag = self.plotter.add_mesh(
            unit_arrow_drag.copy(),
            color="#D92632",
            opacity=1.0,
            smooth_shading=True,
            ambient=0.65,
        )

    def render_frame(self, sample: Dict[str, Any]) -> Image.Image:
        """
        Updates the 3D car rig's pitch, roll, heave, front wheel steering, and 6 force
        vectors for the given telemetry sample, and returns a 710x560 RGBA PIL Image.
        """
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
            self.actor_susp,
            self.actor_inboard,
            self.actor_cg,
        ):
            actor.user_matrix = M_chassis

        # Front wheels yaw dynamically by steering_angle_deg
        road_steer_rad = np.radians(np.clip(float(sample["steering_deg"]) * 0.28, -22.0, 22.0))
        for corner, (wx, wy, wz) in self.wheel_centers.items():
            yaw_w = road_steer_rad if corner.startswith("F") else 0.0
            M_wheel = self._local_to_world_matrix(
                yaw_rad=yaw_w,
                pivot=(wx, wy, wz),
            )
            self.wheel_actors[corner].user_matrix = M_wheel

        # Update 6 Red Force Arrows
        tire_loads = {
            "FL": float(sample["fz_fl_kn"]),
            "FR": float(sample["fz_fr_kn"]),
            "RL": float(sample["fz_rl_kn"]),
            "RR": float(sample["fz_rr_kn"]),
        }
        arrow_anchors = {
            "FL": (1.42,  0.62, 0.04),
            "FR": (1.42, -0.62, 0.04),
            "RL": (-1.22,  0.72, 0.16),
            "RR": (-0.82, -0.62, 0.24),
        }
        for corner, fz_kn in tire_loads.items():
            ax, ay, az = arrow_anchors[corner]
            length = float(np.clip(0.18 + 0.095 * fz_kn, 0.16, 0.76))
            M_arr = self._local_to_world_matrix(
                dx=ax,
                dy=ay,
                dz=az,
                scale_xyz=(0.55, 0.55, length),
            )
            self.arrow_tire_actors[corner].user_matrix = M_arr

        df_kg = float(sample["downforce_kg"])
        df_len = float(np.clip(0.24 + 0.00032 * df_kg, 0.24, 0.68))
        M_df = self._local_to_world_matrix(
            dx=0.05,
            dy=-0.14,
            dz=0.58 + heave_m,
            scale_xyz=(0.56, 0.56, df_len),
        )
        self.actor_arrow_downforce.user_matrix = M_df

        dr_kg = float(sample["drag_kg"])
        dr_len = float(np.clip(0.22 + 0.00095 * dr_kg, 0.22, 0.66))
        M_dr = self._local_to_world_matrix(
            dx=-1.62,
            dy=-0.02,
            dz=0.86 + heave_m,
            scale_xyz=(dr_len, 0.56, dr_len),
        )
        self.actor_arrow_drag.user_matrix = M_dr

        self.plotter.render()
        rgb_arr = self.plotter.screenshot(transparent_background=False, return_img=True)
        diff = np.max(np.abs(rgb_arr.astype(np.int16) - np.array([243, 239, 230], dtype=np.int16)), axis=-1)
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
    bg = Image.new("RGBA", img.size, (243, 239, 230, 255))
    bg.alpha_composite(img)
    bg.save("test_car_3d.png")
    renderer.close()
    print("Saved test_car_3d.png:", img.size, img.mode)
