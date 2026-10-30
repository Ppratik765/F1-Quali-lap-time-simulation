"""
hud_renderer.py — Pixel-exact 1920x1080 2D broadcast compositor using Pillow and PyCairo
with condensed geometric tabular typography (Bahnschrift / DIN) replicating the
2026 F1 Qualifying Telemetry & 3D Vehicle Dynamics layout.
"""

import json
import math
import os
from typing import Dict, Any, Tuple
import cairo
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from track_loader import TrackData
from physics_2026 import TelemetryResult


def _hex_to_rgb_norm(hex_str: str) -> Tuple[float, float, float]:
    h = hex_str.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def _hex_to_rgba_int(hex_str: str, alpha: int = 255) -> Tuple[int, int, int, int]:
    h = hex_str.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), alpha)


def _load_font(size: int, weight: int = 400, width: int = 88) -> ImageFont.FreeTypeFont:
    """Loads Bahnschrift (condensed geometric tabular sans-serif) with fallback to Segoe UI / Arial."""
    candidates = [
        "C:/Windows/Fonts/bahnschrift.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if weight >= 600 else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if weight >= 600 else "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            font = ImageFont.truetype(path, size=size)
            if "bahnschrift" in path.lower():
                try:
                    font.set_variation_by_axes([weight, width])
                except Exception:
                    pass
            return font
    return ImageFont.load_default()


class BroadcastHUDRenderer:
    """
    Pixel-exact 1920x1080 broadcast compositor for the 2026 F1 São Paulo GP (Interlagos)
    Qualifying Telemetry & 3D Vehicle Dynamics simulation.
    """

    WIDTH = 1920
    HEIGHT = 1080

    def __init__(
        self,
        track: TrackData,
        telemetry: TelemetryResult,
        config_path: str = "config_brazil_2026.json",
    ):
        self.track = track
        self.telemetry = telemetry
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = json.load(f)

        # Pre-load typography hierarchy (condensed geometric tabular numerals)
        self.fonts = {
            "title_28b": _load_font(28, weight=680, width=88),
            "sub_14":    _load_font(14, weight=400, width=92),
            "wm_20sb":   _load_font(20, weight=600, width=92),
            "sec_14b":   _load_font(14, weight=650, width=88),
            "sec_13":    _load_font(13, weight=420, width=90),
            "axis_11":   _load_font(11, weight=400, width=90),
            "lap_52b":   _load_font(52, weight=680, width=86),
            "spd_44b":   _load_font(44, weight=680, width=86),
            "gear_56b":  _load_font(56, weight=700, width=86),
            "val_24b":   _load_font(24, weight=660, width=88),
            "val_22b":   _load_font(22, weight=660, width=88),
            "badge_19b": _load_font(19, weight=680, width=88),
            "tbl_16b":   _load_font(16, weight=650, width=88),
            "tbl_15b":   _load_font(15, weight=650, width=88),
            "tbl_14":    _load_font(14, weight=420, width=90),
            "tbl_13":    _load_font(13, weight=420, width=90),
            "tbl_12":    _load_font(12, weight=400, width=90),
        }

        # Map 2D Interlagos GPS coordinates into [715, 138, 1105, 425] preserving aspect ratio
        self._compute_track_map_projection()

        # Pre-render static background layer once for high-speed compositing
        self.static_base_img = self._build_static_background()

    def _compute_track_map_projection(self) -> None:
        x, y = self.track.x, self.track.y
        x_min, x_max = float(x.min()), float(x.max())
        y_min, y_max = float(y.min()), float(y.max())

        box_x1, box_y1, box_x2, box_y2 = 720.0, 140.0, 1100.0, 425.0
        bw = box_x2 - box_x1
        bh = box_y2 - box_y1

        scale = min(bw / max(1.0, x_max - x_min), bh / max(1.0, y_max - y_min))
        cx = 0.5 * (x_min + x_max)
        cy = 0.5 * (y_min + y_max)
        box_cx = 0.5 * (box_x1 + box_x2)
        box_cy = 0.5 * (box_y1 + box_y2)

        # Flip Y for screen coordinates (top-down)
        self.map_x = box_cx + (x - cx) * scale
        self.map_y = box_cy - (y - cy) * scale

        # Compute outer normal vectors for Active Aero X-Mode zone white indicator lines
        dx = np.gradient(self.map_x)
        dy = np.gradient(self.map_y)
        norm = np.maximum(1e-6, np.hypot(dx, dy))
        self.map_nx = -dy / norm
        self.map_ny = dx / norm

    def _cairo_surface_to_pil(self, surface: cairo.ImageSurface) -> Image.Image:
        buf = surface.get_data()
        arr = np.ndarray(shape=(self.HEIGHT, self.WIDTH, 4), dtype=np.uint8, buffer=buf)
        # Convert BGRA -> RGBA
        rgba = arr[:, :, [2, 1, 0, 3]].copy()
        return Image.fromarray(rgba, mode="RGBA")

    def _draw_brazil_flag_icon(self, ctx: cairo.Context, cx: float, cy: float, r: float) -> None:
        """Draws a crisp rounded circular Brazilian flag icon at (cx, cy)."""
        ctx.save()
        ctx.arc(cx, cy, r, 0, 2 * math.pi)
        ctx.clip()

        # Green background (#009739)
        ctx.set_source_rgb(*_hex_to_rgb_norm("#009739"))
        ctx.arc(cx, cy, r, 0, 2 * math.pi)
        ctx.fill()

        # Yellow rhombus (#FEDD00)
        ctx.set_source_rgb(*_hex_to_rgb_norm("#FEDD00"))
        ctx.move_to(cx, cy - r * 0.76)
        ctx.line_to(cx + r * 0.84, cy)
        ctx.line_to(cx, cy + r * 0.76)
        ctx.line_to(cx - r * 0.84, cy)
        ctx.close_path()
        ctx.fill()

        # Blue center globe (#012169)
        ctx.set_source_rgb(*_hex_to_rgb_norm("#012169"))
        ctx.arc(cx, cy, r * 0.44, 0, 2 * math.pi)
        ctx.fill()

        # White equatorial banner across globe
        ctx.set_source_rgb(1.0, 1.0, 1.0)
        ctx.set_line_width(1.7)
        ctx.arc(cx - r * 0.12, cy + r * 0.35, r * 0.55, -1.35, -0.25)
        ctx.stroke()

        # Subtle glossy edge shading
        grad = cairo.RadialGradient(cx - r * 0.3, cy - r * 0.3, r * 0.1, cx, cy, r)
        grad.add_color_stop_rgba(0.0, 1.0, 1.0, 1.0, 0.25)
        grad.add_color_stop_rgba(0.7, 0.0, 0.0, 0.0, 0.0)
        grad.add_color_stop_rgba(1.0, 0.0, 0.0, 0.0, 0.35)
        ctx.set_source(grad)
        ctx.arc(cx, cy, r, 0, 2 * math.pi)
        ctx.fill()
        ctx.restore()

    def _build_static_background(self) -> Image.Image:
        """Pre-renders all static grids, axes, headers, base track map, and watermarks."""
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.WIDTH, self.HEIGHT)
        ctx = cairo.Context(surface)

        # 1. Background #08080A
        ctx.set_source_rgb(*_hex_to_rgb_norm("#08080A"))
        ctx.paint()

        # 2. Left Column 3 Graph Grids
        # Graph 1: SPEED [45, 110, 635, 320] -> plot box [76, 132, 625, 298]
        # Graph 2: POWER [45, 365, 635, 585] -> plot box [76, 388, 625, 562]
        # Graph 3: SOC   [45, 630, 635, 840] -> plot box [76, 652, 625, 818]
        self.g1_box = (76.0, 132.0, 625.0, 298.0)
        self.g2_box = (76.0, 388.0, 625.0, 562.0)
        self.g3_box = (76.0, 652.0, 625.0, 818.0)
        s_max = float(self.track.length_m)

        for box, y_min, y_max, y_ticks in [
            (self.g1_box, 0.0, 350.0, [0, 100, 200, 300, 350]),
            (self.g2_box, -400.0, 450.0, [-400, -200, 0, 200, 400, 450]),
            (self.g3_box, 0.0, 4.0, [0, 1, 2, 3, 4]),
        ]:
            x1, y1, x2, y2 = box
            ctx.set_line_width(1.0)
            # Horizontal grid lines
            for yt in y_ticks:
                py = y2 - ((yt - y_min) / (y_max - y_min)) * (y2 - y1)
                if yt == 0 and y_min < 0:
                    ctx.set_source_rgb(*_hex_to_rgb_norm("#2C2D35"))
                else:
                    ctx.set_source_rgb(*_hex_to_rgb_norm("#1E1E24"))
                ctx.move_to(x1, py)
                ctx.line_to(x2, py)
                ctx.stroke()

            # Vertical grid lines at 0, 1000, 2000, 3000, 4000, 4309 m
            ctx.set_source_rgb(*_hex_to_rgb_norm("#1E1E24"))
            for xt in [0, 1000, 2000, 3000, 4000, s_max]:
                px = x1 + (xt / s_max) * (x2 - x1)
                ctx.move_to(px, y1)
                ctx.line_to(px, y2)
                ctx.stroke()

        # 3. 2D Interlagos Track Map Base Loop (#555860, 5.5px)
        ctx.save()
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        ctx.set_source_rgb(*_hex_to_rgb_norm("#555860"))
        ctx.set_line_width(5.5)
        ctx.move_to(self.map_x[0], self.map_y[0])
        for i in range(1, len(self.map_x)):
            ctx.line_to(self.map_x[i], self.map_y[i])
        ctx.close_path()
        ctx.stroke()

        # White outer Active Aero (X-Mode) zone indicator lines alongside straights
        ctx.set_source_rgba(0.92, 0.93, 0.95, 0.85)
        ctx.set_line_width(1.6)
        offset_px = 6.5
        for zone in self.cfg["aerodynamics_2026"]["x_mode"]["activation_zones_m"]:
            i_start = int(np.clip(zone["start_m"], 0, len(self.map_x) - 1))
            i_end = int(np.clip(zone["end_m"], 0, len(self.map_x) - 1))
            if i_end > i_start:
                ctx.move_to(
                    self.map_x[i_start] + offset_px * self.map_nx[i_start],
                    self.map_y[i_start] + offset_px * self.map_ny[i_start],
                )
                for k in range(i_start + 1, i_end + 1, 3):
                    ctx.line_to(
                        self.map_x[k] + offset_px * self.map_nx[k],
                        self.map_y[k] + offset_px * self.map_ny[k],
                    )
                ctx.stroke()
        ctx.restore()

        # 4. MGU-K Color Bar Legend [-350 to +350 kW] at [945, 456, 1105, 463]
        grad_mguk = cairo.LinearGradient(945.0, 456.0, 1105.0, 456.0)
        grad_mguk.add_color_stop_rgb(0.0, *_hex_to_rgb_norm("#A92323"))
        grad_mguk.add_color_stop_rgb(0.5, *_hex_to_rgb_norm("#EAECEF"))
        grad_mguk.add_color_stop_rgb(1.0, *_hex_to_rgb_norm("#2E75B6"))
        ctx.set_source(grad_mguk)
        ctx.rectangle(945.0, 457.0, 160.0, 6.0)
        ctx.fill()

        # Convert Cairo surface to PIL for crisp typography rendering
        img = self._cairo_surface_to_pil(surface)
        draw = ImageDraw.Draw(img)

        # 5. Top Header Title [600, 18, 1320, 68]
        ev = self.cfg["event"]
        title_text = ev["title"]      # "SÃO PAULO GRAND PRIX 2026"
        sub_text = ev["subtitle"]     # "Autódromo José Carlos Pace - 4309 m - qualifying"
        wm_text = ev["watermark"]     # "Priyanshu Pratik"

        tw_bbox = draw.textbbox((0, 0), title_text, font=self.fonts["title_28b"])
        tw = tw_bbox[2] - tw_bbox[0]
        flag_r = 12.5
        gap = 12.0
        total_w = (2 * flag_r) + gap + tw
        start_x = 960.0 - 0.5 * total_w

        # Draw Brazilian flag on Cairo surface at (start_x + flag_r, 31) and re-overlay
        flag_surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.WIDTH, 80)
        flag_ctx = cairo.Context(flag_surf)
        self._draw_brazil_flag_icon(flag_ctx, start_x + flag_r, 31.0, flag_r)
        flag_pil = Image.fromarray(
            np.ndarray(shape=(80, self.WIDTH, 4), dtype=np.uint8, buffer=flag_surf.get_data())[:, :, [2, 1, 0, 3]].copy(),
            mode="RGBA",
        )
        img.alpha_composite(flag_pil, (0, 0))
        draw = ImageDraw.Draw(img)

        draw.text((start_x + 2 * flag_r + gap, 15), title_text, font=self.fonts["title_28b"], fill="#FFFFFF")
        sub_bbox = draw.textbbox((0, 0), sub_text, font=self.fonts["sub_14"])
        sub_w = sub_bbox[2] - sub_bbox[0]
        draw.text((960.0 - 0.5 * sub_w, 50), sub_text, font=self.fonts["sub_14"], fill="#8A8D93")

        # 6. Top-Right Watermark [1620, 24, 1885, 55]
        wm_bbox = draw.textbbox((0, 0), wm_text, font=self.fonts["wm_20sb"])
        wm_w = wm_bbox[2] - wm_bbox[0]
        draw.text((1885.0 - wm_w, 22), wm_text, font=self.fonts["wm_20sb"], fill="#EAECEF")

        # 7. Left Graph Titles & Axis Tick Labels
        # Graph 1 Header
        draw.text((45, 105), "SPEED", font=self.fonts["sec_14b"], fill="#FFFFFF")
        draw.text((95, 106), "[km/h]", font=self.fonts["sec_13"], fill="#8A8D93")
        for yt in [0, 100, 200, 300]:
            py = self.g1_box[3] - (yt / 350.0) * (self.g1_box[3] - self.g1_box[1])
            lbl = str(yt)
            w = draw.textlength(lbl, font=self.fonts["axis_11"])
            draw.text((69 - w, py - 6), lbl, font=self.fonts["axis_11"], fill="#8A8D93")

        # Graph 2 Header + Inline Legend
        draw.text((45, 360), "POWER", font=self.fonts["sec_14b"], fill="#FFFFFF")
        draw.text((100, 361), "[kW]", font=self.fonts["sec_13"], fill="#8A8D93")
        draw.text((146, 360), "ICE demanded", font=self.fonts["tbl_14"], fill="#7A7D85")
        draw.text((252, 360), "MGU-K", font=self.fonts["tbl_14"], fill="#2E86DE")
        for yt in [-400, -200, 0, 200, 400]:
            py = self.g2_box[3] - ((yt - (-400.0)) / 850.0) * (self.g2_box[3] - self.g2_box[1])
            lbl = str(yt)
            w = draw.textlength(lbl, font=self.fonts["axis_11"])
            draw.text((69 - w, py - 6), lbl, font=self.fonts["axis_11"], fill="#8A8D93")

        # Graph 3 Header
        draw.text((45, 625), "SOC", font=self.fonts["sec_14b"], fill="#FFFFFF")
        draw.text((78, 626), "[MJ]", font=self.fonts["sec_13"], fill="#8A8D93")
        for yt in [0, 1, 2, 3, 4]:
            py = self.g3_box[3] - (yt / 4.0) * (self.g3_box[3] - self.g3_box[1])
            lbl = str(yt)
            w = draw.textlength(lbl, font=self.fonts["axis_11"])
            draw.text((69 - w, py - 6), lbl, font=self.fonts["axis_11"], fill="#8A8D93")

        # X-axis distance labels for all 3 graphs
        for box in (self.g1_box, self.g2_box, self.g3_box):
            x1, _, x2, y2 = box
            for xt, lbl in [(0, "0"), (1000, "1000"), (2000, "2000"), (3000, "3000"), (4000, "4000 m")]:
                px = x1 + (xt / s_max) * (x2 - x1)
                w = draw.textlength(lbl, font=self.fonts["axis_11"])
                draw.text((px - 0.5 * w, y2 + 5), lbl, font=self.fonts["axis_11"], fill="#8A8D93")

        # 8. Driver Inputs Cluster Static Labels
        draw.text((45, 918), "THROTTLE", font=self.fonts["sec_14b"], fill="#EAECEF")
        draw.text((45, 978), "BRAKE", font=self.fonts["sec_14b"], fill="#EAECEF")
        gw = draw.textlength("GEAR", font=self.fonts["sec_13"])
        draw.text((485 - 0.5 * gw, 898), "GEAR", font=self.fonts["sec_13"], fill="#8A8D93")

        # 9. Center Column Static Headers
        draw.text((705, 452), "MGU-K", font=self.fonts["sec_13"], fill="#8A8D93")
        draw.text((945, 440), "-350", font=self.fonts["axis_11"], fill="#8A8D93")
        w_p350 = draw.textlength("+350 kW", font=self.fonts["axis_11"])
        draw.text((1105 - w_p350, 440), "+350 kW", font=self.fonts["axis_11"], fill="#8A8D93")

        draw.text((705, 495), "LAP TIME", font=self.fonts["sec_13"], fill="#8A8D93")
        draw.text((705, 584), "SPEED", font=self.fonts["sec_13"], fill="#8A8D93")
        draw.text((930, 584), "MGU-K", font=self.fonts["sec_13"], fill="#8A8D93")
        draw.text((705, 672), "FUEL USED", font=self.fonts["sec_13"], fill="#8A8D93")
        draw.text((930, 672), "MGU-K POWER", font=self.fonts["sec_13"], fill="#8A8D93")

        draw.text((705, 760), "ENERGY RECOVERED", font=self.fonts["sec_13"], fill="#8A8D93")
        draw.text((705, 791), "Overrun", font=self.fonts["tbl_13"], fill="#D0D3D8")
        draw.text((705, 821), "Part throttle", font=self.fonts["tbl_13"], fill="#D0D3D8")
        draw.text((705, 851), "Superclip", font=self.fonts["tbl_13"], fill="#D0D3D8")

        draw.text((705, 905), "LAP TIME SENSITIVITY", font=self.fonts["sec_13"], fill="#8A8D93")
        draw.text((705, 936), "Power +10 kW", font=self.fonts["tbl_13"], fill="#D0D3D8")
        draw.text((705, 968), "Downforce +1%", font=self.fonts["tbl_13"], fill="#D0D3D8")
        draw.text((705, 1000), "Drag -1%", font=self.fonts["tbl_13"], fill="#D0D3D8")
        draw.text((915, 936), "Front grip +1%", font=self.fonts["tbl_13"], fill="#D0D3D8")
        draw.text((915, 968), "Rear grip +1%", font=self.fonts["tbl_13"], fill="#D0D3D8")

        # 10. Right Column Static Headers
        draw.text((1140, 705), "RIDE HEIGHT F / R", font=self.fonts["sec_13"], fill="#8A8D93")
        draw.text((1405, 705), "ROLL / PITCH", font=self.fonts["sec_13"], fill="#8A8D93")
        draw.text((1665, 705), "ACCELERATION", font=self.fonts["sec_13"], fill="#8A8D93")

        draw.text((1140, 785), "DOWNFORCE / DRAG", font=self.fonts["sec_13"], fill="#8A8D93")
        draw.text((1405, 785), "AERO BALANCE", font=self.fonts["sec_13"], fill="#8A8D93")
        draw.text((1665, 785), "AIR DENSITY", font=self.fonts["sec_13"], fill="#8A8D93")

        draw.text((1140, 905), "SECTORS", font=self.fonts["sec_13"], fill="#8A8D93")
        draw.text((1140, 936), "S1", font=self.fonts["tbl_13"], fill="#8A8D93")
        draw.text((1140, 968), "S2", font=self.fonts["tbl_13"], fill="#8A8D93")
        draw.text((1140, 1000), "S3", font=self.fonts["tbl_13"], fill="#8A8D93")

        draw.text((1390, 905), "DRIVER PHASES", font=self.fonts["sec_13"], fill="#8A8D93")
        draw.text((1645, 905), "LIMITED BY", font=self.fonts["sec_13"], fill="#8A8D93")

        return img

    def _draw_progressive_graphs_and_vectors(self, ctx: cairo.Context, sample: Dict[str, Any]) -> None:
        """Draws all anti-aliased Cairo vector traces, bars, steering wheel, and pie chart."""
        idx = int(sample["idx"])
        s_curr = float(sample["s"])
        s_max = float(self.track.length_m)
        s_arr = self.telemetry.s[:idx + 1]

        # -----------------------------------------------------------------
        # 1. Graph 1: SPEED [km/h]
        # -----------------------------------------------------------------
        x1, y1, x2, y2 = self.g1_box
        px_curr = x1 + (s_curr / s_max) * (x2 - x1)

        # Vertical grey cursor line
        ctx.set_line_width(1.0)
        ctx.set_source_rgba(0.60, 0.62, 0.66, 0.85)
        ctx.move_to(px_curr, y1)
        ctx.line_to(px_curr, y2)
        ctx.stroke()

        if idx >= 1:
            xs = x1 + (s_arr / s_max) * (x2 - x1)
            ys = y2 - (np.clip(self.telemetry.speed_kmh[:idx + 1], 0.0, 350.0) / 350.0) * (y2 - y1)
            ctx.set_source_rgb(*_hex_to_rgb_norm("#E12D39"))
            ctx.set_line_width(2.2)
            ctx.move_to(xs[0], ys[0])
            for k in range(1, len(xs)):
                ctx.line_to(xs[k], ys[k])
            ctx.stroke()

        py_spd = y2 - (np.clip(float(sample["speed_kmh"]), 0.0, 350.0) / 350.0) * (y2 - y1)
        ctx.set_source_rgba(*_hex_to_rgb_norm("#E12D39"), 0.32)
        ctx.arc(px_curr, py_spd, 6.5, 0, 2 * math.pi)
        ctx.fill()
        ctx.set_source_rgb(*_hex_to_rgb_norm("#FF525E"))
        ctx.arc(px_curr, py_spd, 3.2, 0, 2 * math.pi)
        ctx.fill()

        # -----------------------------------------------------------------
        # 2. Graph 2: POWER [kW] (ICE demanded & MGU-K)
        # -----------------------------------------------------------------
        x1, y1, x2, y2 = self.g2_box
        ctx.set_line_width(1.0)
        ctx.set_source_rgba(0.60, 0.62, 0.66, 0.85)
        ctx.move_to(px_curr, y1)
        ctx.line_to(px_curr, y2)
        ctx.stroke()

        if idx >= 1:
            xs = x1 + (s_arr / s_max) * (x2 - x1)
            ys_ice = y2 - ((np.clip(self.telemetry.ice_power_kw[:idx + 1], -400.0, 450.0) + 400.0) / 850.0) * (y2 - y1)
            ctx.set_source_rgb(*_hex_to_rgb_norm("#7A7D85"))
            ctx.set_line_width(1.9)
            ctx.move_to(xs[0], ys_ice[0])
            for k in range(1, len(xs)):
                ctx.line_to(xs[k], ys_ice[k])
            ctx.stroke()

            # MGU-K trace colored by deployment/harvesting/superclipping
            ys_mguk = y2 - ((np.clip(self.telemetry.mguk_power_kw[:idx + 1], -400.0, 450.0) + 400.0) / 850.0) * (y2 - y1)
            states = self.telemetry.mguk_state[:idx + 1]
            ctx.set_line_width(2.2)
            for k in range(1, len(xs)):
                st = states[k]
                if st == "SUPERCLIP":
                    ctx.set_source_rgb(*_hex_to_rgb_norm("#F39C12"))
                elif st == "HARVEST":
                    ctx.set_source_rgb(*_hex_to_rgb_norm("#E12D39"))
                elif st == "DEPLOY":
                    ctx.set_source_rgb(*_hex_to_rgb_norm("#2E86DE"))
                else:
                    ctx.set_source_rgb(*_hex_to_rgb_norm("#EAECEF"))
                ctx.move_to(xs[k - 1], ys_mguk[k - 1])
                ctx.line_to(xs[k], ys_mguk[k])
                ctx.stroke()

        py_ice = y2 - ((np.clip(float(sample["ice_power_kw"]), -400.0, 450.0) + 400.0) / 850.0) * (y2 - y1)
        py_mguk = y2 - ((np.clip(float(sample["mguk_power_kw"]), -400.0, 450.0) + 400.0) / 850.0) * (y2 - y1)
        ctx.set_source_rgb(*_hex_to_rgb_norm("#9FA3AC"))
        ctx.arc(px_curr, py_ice, 2.8, 0, 2 * math.pi)
        ctx.fill()

        st_curr = sample["mguk_state"]
        dot_hex = "#F39C12" if st_curr == "SUPERCLIP" else ("#E12D39" if st_curr == "HARVEST" else "#2E86DE")
        ctx.set_source_rgba(*_hex_to_rgb_norm(dot_hex), 0.35)
        ctx.arc(px_curr, py_mguk, 6.5, 0, 2 * math.pi)
        ctx.fill()
        ctx.set_source_rgb(*_hex_to_rgb_norm(dot_hex))
        ctx.arc(px_curr, py_mguk, 3.2, 0, 2 * math.pi)
        ctx.fill()

        # -----------------------------------------------------------------
        # 3. Graph 3: SOC [MJ] with subtle green fill under curve
        # -----------------------------------------------------------------
        x1, y1, x2, y2 = self.g3_box
        ctx.set_line_width(1.0)
        ctx.set_source_rgba(0.60, 0.62, 0.66, 0.85)
        ctx.move_to(px_curr, y1)
        ctx.line_to(px_curr, y2)
        ctx.stroke()

        if idx >= 1:
            xs = x1 + (s_arr / s_max) * (x2 - x1)
            ys_soc = y2 - (np.clip(self.telemetry.soc_mj[:idx + 1], 0.0, 4.0) / 4.0) * (y2 - y1)

            # Subtle green fill under curve (alpha=0.12)
            ctx.save()
            ctx.move_to(xs[0], y2)
            for k in range(len(xs)):
                ctx.line_to(xs[k], ys_soc[k])
            ctx.line_to(xs[-1], y2)
            ctx.close_path()
            grad_soc = cairo.LinearGradient(0.0, y1, 0.0, y2)
            grad_soc.add_color_stop_rgba(0.0, *_hex_to_rgb_norm("#2ECC71"), 0.16)
            grad_soc.add_color_stop_rgba(1.0, *_hex_to_rgb_norm("#2ECC71"), 0.02)
            ctx.set_source(grad_soc)
            ctx.fill()
            ctx.restore()

            # Green SOC line (#2ECC71, 2.5px)
            ctx.set_source_rgb(*_hex_to_rgb_norm("#2ECC71"))
            ctx.set_line_width(2.3)
            ctx.move_to(xs[0], ys_soc[0])
            for k in range(1, len(xs)):
                ctx.line_to(xs[k], ys_soc[k])
            ctx.stroke()

        py_soc = y2 - (np.clip(float(sample["soc_mj"]), 0.0, 4.0) / 4.0) * (y2 - y1)
        ctx.set_source_rgba(*_hex_to_rgb_norm("#2ECC71"), 0.32)
        ctx.arc(px_curr, py_soc, 6.5, 0, 2 * math.pi)
        ctx.fill()
        ctx.set_source_rgb(*_hex_to_rgb_norm("#58D68D"))
        ctx.arc(px_curr, py_soc, 3.2, 0, 2 * math.pi)
        ctx.fill()

        # -----------------------------------------------------------------
        # 4. 2D Interlagos Track Map Driven Trail + Leading Car Dot
        # -----------------------------------------------------------------
        if idx >= 1:
            ctx.save()
            ctx.set_line_cap(cairo.LINE_CAP_ROUND)
            ctx.set_line_join(cairo.LINE_JOIN_ROUND)
            ctx.set_line_width(5.5)
            states = self.telemetry.mguk_state[:idx + 1]
            step = 2
            for k in range(step, idx + 1, step):
                st = states[k]
                if st == "SUPERCLIP":
                    ctx.set_source_rgb(*_hex_to_rgb_norm("#F39C12"))
                elif st == "HARVEST":
                    ctx.set_source_rgb(*_hex_to_rgb_norm("#E12D39"))
                elif st == "DEPLOY":
                    ctx.set_source_rgb(*_hex_to_rgb_norm("#2E86DE"))
                else:
                    ctx.set_source_rgb(*_hex_to_rgb_norm("#FFFFFF"))
                ctx.move_to(self.map_x[k - step], self.map_y[k - step])
                ctx.line_to(self.map_x[k], self.map_y[k])
                ctx.stroke()
            ctx.restore()

        # White leading car dot (r=5)
        car_mx = float(self.map_x[idx])
        car_my = float(self.map_y[idx])
        ctx.set_source_rgba(1.0, 1.0, 1.0, 0.30)
        ctx.arc(car_mx, car_my, 8.5, 0, 2 * math.pi)
        ctx.fill()
        ctx.set_source_rgb(1.0, 1.0, 1.0)
        ctx.arc(car_mx, car_my, 5.2, 0, 2 * math.pi)
        ctx.fill()

        # -----------------------------------------------------------------
        # 5. Driver Inputs Cluster: Throttle/Brake Bars & Rotating Steering Wheel
        # -----------------------------------------------------------------
        # Throttle bar [150, 915, 355, 938]
        th_w = 205.0 * np.clip(float(sample["throttle_pct"]) / 100.0, 0.0, 1.0)
        ctx.set_source_rgb(*_hex_to_rgb_norm("#35383F"))
        ctx.rectangle(150.0, 915.0, 205.0, 23.0)
        ctx.fill()
        if th_w > 0.5:
            ctx.set_source_rgb(*_hex_to_rgb_norm("#4CD98C"))
            ctx.rectangle(150.0, 915.0, th_w, 23.0)
            ctx.fill()

        # Brake bar [150, 975, 355, 998]
        br_w = 205.0 * np.clip(float(sample["brake_pct"]) / 100.0, 0.0, 1.0)
        ctx.set_source_rgb(*_hex_to_rgb_norm("#35383F"))
        ctx.rectangle(150.0, 975.0, 205.0, 23.0)
        ctx.fill()
        if br_w > 0.5:
            ctx.set_source_rgb(*_hex_to_rgb_norm("#E12D39"))
            ctx.rectangle(150.0, 975.0, br_w, 23.0)
            ctx.fill()

        # Rotating Steering Wheel vector icon (center=[595, 950], r=34)
        ctx.save()
        ctx.translate(595.0, 950.0)
        steer_rad = math.radians(-float(sample["steering_deg"]))
        ctx.rotate(steer_rad)
        ctx.set_source_rgb(1.0, 1.0, 1.0)
        ctx.set_line_width(3.6)
        ctx.arc(0.0, 0.0, 33.0, 0, 2 * math.pi)
        ctx.stroke()
        # Horizontal crossbar and bottom spoke matching broadcast icon
        ctx.move_to(-32.0, -4.0)
        ctx.line_to(32.0, -4.0)
        ctx.stroke()
        ctx.move_to(0.0, -4.0)
        ctx.line_to(0.0, 32.0)
        ctx.stroke()
        ctx.restore()

        # -----------------------------------------------------------------
        # 6. MGU-K Dynamic Badge [930, 600, 1105, 645]
        # -----------------------------------------------------------------
        badge_colors = {
            "DEPLOY":    "#2E75B6",
            "HARVEST":   "#A92323",
            "SUPERCLIP": "#D68910",
            "OFF":       "#2C2F33",
        }
        b_col = badge_colors.get(st_curr, "#2C2F33")
        ctx.set_source_rgb(*_hex_to_rgb_norm(b_col))
        self._rounded_rect(ctx, 930.0, 600.0, 175.0, 45.0, 3.0)
        ctx.fill()

        # -----------------------------------------------------------------
        # 7. Energy Recovered Bars [840, y, 1055, y+9]
        # -----------------------------------------------------------------
        max_bar_mj = 2.0
        bars = [
            (795.0, float(sample["energy_overrun_mj"]), "#A92323"),
            (825.0, float(sample["energy_part_throttle_mj"]), "#A92323"),
            (855.0, float(sample["energy_superclip_mj"]), "#F39C12"),
        ]
        for by, val_mj, hex_c in bars:
            ctx.set_source_rgb(*_hex_to_rgb_norm("#2C2F33"))
            ctx.rectangle(840.0, by, 215.0, 9.0)
            ctx.fill()
            fill_w = 215.0 * np.clip(val_mj / max_bar_mj, 0.0, 1.0)
            if fill_w > 0.5:
                ctx.set_source_rgb(*_hex_to_rgb_norm(hex_c))
                ctx.rectangle(840.0, by, fill_w, 9.0)
                ctx.fill()

        # -----------------------------------------------------------------
        # 8. Aero Balance Slider Bar [1495, 816, 1615, 823]
        # -----------------------------------------------------------------
        ctx.set_source_rgb(*_hex_to_rgb_norm("#35383F"))
        ctx.rectangle(1495.0, 816.0, 120.0, 6.0)
        ctx.fill()
        bal_frac = np.clip((float(sample["aero_balance_pct"]) - 39.0) / 10.0, 0.0, 1.0)
        tick_x = 1495.0 + 120.0 * bal_frac
        ctx.set_source_rgb(*_hex_to_rgb_norm("#E12D39"))
        ctx.rectangle(tick_x - 1.5, 810.0, 3.0, 18.0)
        ctx.fill()

        # -----------------------------------------------------------------
        # 9. Driver Phases Pie Chart (center=[1430, 980], r=36) & Legend Squares
        # -----------------------------------------------------------------
        p_full = float(sample["phase_full_throttle_pct"])
        p_part = float(sample["phase_part_throttle_pct"])
        p_neut = float(sample["phase_neutral_pct"])
        p_brak = float(sample["phase_braking_pct"])
        total_p = max(1e-6, p_full + p_part + p_neut + p_brak)

        slices = [
            (p_full / total_p, "#2E86DE"),
            (p_part / total_p, "#6C7A89"),
            (p_neut / total_p, "#EAECEF"),
            (p_brak / total_p, "#A92323"),
        ]
        ang_curr = -0.5 * math.pi
        cx_p, cy_p, r_p = 1430.0, 980.0, 36.0
        for frac, hex_c in slices:
            if frac > 1e-4:
                ang_next = ang_curr + frac * 2.0 * math.pi
                ctx.set_source_rgb(*_hex_to_rgb_norm(hex_c))
                ctx.move_to(cx_p, cy_p)
                ctx.arc(cx_p, cy_p, r_p, ang_curr, ang_next)
                ctx.close_path()
                ctx.fill()
                ang_curr = ang_next

        # Legend color squares for Driver Phases
        for ly, hex_c in [(939.0, "#2E86DE"), (963.0, "#6C7A89"), (987.0, "#EAECEF"), (1011.0, "#A92323")]:
            ctx.set_source_rgb(*_hex_to_rgb_norm(hex_c))
            ctx.rectangle(1480.0, ly, 8.0, 8.0)
            ctx.fill()

        # -----------------------------------------------------------------
        # 10. Limited By Stacked Bar [1645, 930, 1885, 944]
        # -----------------------------------------------------------------
        ctx.set_source_rgb(*_hex_to_rgb_norm("#35383F"))
        ctx.rectangle(1645.0, 930.0, 240.0, 14.0)
        ctx.fill()

        total_lap_km = float(self.track.length_m) / 1000.0
        w_pow = 240.0 * np.clip(float(sample["limited_power_km"]) / total_lap_km, 0.0, 1.0)
        w_grp = 240.0 * np.clip(float(sample["limited_grip_km"]) / total_lap_km, 0.0, 1.0 - w_pow / 240.0)
        if w_pow > 0.5:
            ctx.set_source_rgb(*_hex_to_rgb_norm("#2E86DE"))
            ctx.rectangle(1645.0, 930.0, w_pow, 14.0)
            ctx.fill()
        if w_grp > 0.5:
            ctx.set_source_rgb(*_hex_to_rgb_norm("#EAECEF"))
            ctx.rectangle(1645.0 + w_pow, 930.0, w_grp, 14.0)
            ctx.fill()

        # Legend squares for Limited By
        ctx.set_source_rgb(*_hex_to_rgb_norm("#2E86DE"))
        ctx.rectangle(1645.0, 969.0, 8.0, 8.0)
        ctx.fill()
        ctx.set_source_rgb(*_hex_to_rgb_norm("#EAECEF"))
        ctx.rectangle(1645.0, 997.0, 8.0, 8.0)
        ctx.fill()

    def _draw_corner_telemetry_block(
        self,
        draw: ImageDraw.ImageDraw,
        corner_name: str,
        bbox: Tuple[int, int, int, int],
        fz_kn: float,
        camber_deg: float,
        slip_deg: float,
    ) -> None:
        """Draws a 4-corner tire telemetry block (RR, RL, FR, FL) with top force bar."""
        x1, y1, x2, _ = bbox
        bw = x2 - x1
        draw.text((x1, y1), corner_name, font=self.fonts["tbl_12"], fill="#8A8D93")

        # Horizontal force bar [x1, y1+17, x2, y1+23]
        bar_y1, bar_y2 = y1 + 17, y1 + 23
        draw.rectangle([x1, bar_y1, x2, bar_y2], fill="#2C2F33")
        fill_frac = float(np.clip(fz_kn / 6.5, 0.04, 1.0))
        fill_w = int(round(bw * fill_frac))
        bar_color = "#E12D39" if fz_kn >= 3.3 else "#4CD98C"
        if fill_w > 1:
            draw.rectangle([x1, bar_y1, x1 + fill_w, bar_y2], fill=bar_color)

        # 3 Telemetry rows: Fz [kN], gamma [deg], phi [deg]
        r1_y = y1 + 31
        r2_y = y1 + 55
        r3_y = y1 + 79

        draw.text((x1, r1_y), "Fz", font=self.fonts["tbl_13"], fill="#8A8D93")
        draw.text((x1 + 24, r1_y + 1), "kN", font=self.fonts["axis_11"], fill="#8A8D93")
        v1 = f"{fz_kn:.1f}"
        w1 = draw.textlength(v1, font=self.fonts["tbl_16b"])
        draw.text((x2 - w1, r1_y - 2), v1, font=self.fonts["tbl_16b"], fill="#FFFFFF")

        draw.text((x1, r2_y), "γ", font=self.fonts["tbl_13"], fill="#8A8D93")
        draw.text((x1 + 24, r2_y + 1), "°", font=self.fonts["axis_11"], fill="#8A8D93")
        v2 = f"{camber_deg:+.1f}"
        w2 = draw.textlength(v2, font=self.fonts["tbl_16b"])
        draw.text((x2 - w2, r2_y - 2), v2, font=self.fonts["tbl_16b"], fill="#FFFFFF")

        draw.text((x1, r3_y), "φ", font=self.fonts["tbl_13"], fill="#8A8D93")
        draw.text((x1 + 24, r3_y + 1), "°", font=self.fonts["axis_11"], fill="#8A8D93")
        v3 = f"{slip_deg:.1f}"
        w3 = draw.textlength(v3, font=self.fonts["tbl_16b"])
        draw.text((x2 - w3, r3_y - 2), v3, font=self.fonts["tbl_16b"], fill="#FFFFFF")

    @staticmethod
    def _rounded_rect(ctx: cairo.Context, x: float, y: float, w: float, h: float, r: float) -> None:
        ctx.new_sub_path()
        ctx.arc(x + w - r, y + r, r, -0.5 * math.pi, 0.0)
        ctx.arc(x + w - r, y + h - r, r, 0.0, 0.5 * math.pi)
        ctx.arc(x + r, y + h - r, r, 0.5 * math.pi, math.pi)
        ctx.arc(x + r, y + r, r, math.pi, 1.5 * math.pi)
        ctx.close_path()

    @staticmethod
    def _format_lap_time(t_sec: float) -> str:
        t_sec = max(0.0, float(t_sec))
        mins = int(t_sec // 60.0)
        rem = t_sec - mins * 60.0
        return f"{mins}:{rem:06.3f}"

    def render_hud_frame(self, sample: Dict[str, Any], car_3d_rgba: Image.Image) -> Image.Image:
        """
        Composites a complete 1920x1080 broadcast frame combining the static base layer,
        Cairo progressive traces/vectors, PyVista 3D X-Ray car render, and tabular readouts.
        """
        frame = self.static_base_img.copy()

        # 1. Draw dynamic Cairo vectors onto a transparent overlay
        vec_surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.WIDTH, self.HEIGHT)
        vec_ctx = cairo.Context(vec_surface)
        self._draw_progressive_graphs_and_vectors(vec_ctx, sample)
        vec_pil = self._cairo_surface_to_pil(vec_surface)
        frame.alpha_composite(vec_pil, (0, 0))

        # 2. Composite 710x560 off-screen PyVista 3D X-Ray car at [1180, 100, 1890, 660]
        frame.alpha_composite(car_3d_rgba, (1180, 100))

        # 3. Render all dynamic numerical readouts with tabular alignment
        draw = ImageDraw.Draw(frame)

        # Driver Inputs Cluster
        th_str = f"{int(round(sample['throttle_pct']))}%"
        w_th = draw.textlength(th_str, font=self.fonts["val_22b"])
        draw.text((425 - w_th, 914), th_str, font=self.fonts["val_22b"], fill="#FFFFFF")

        br_str = f"{int(round(sample['brake_pct']))}%"
        w_br = draw.textlength(br_str, font=self.fonts["val_22b"])
        draw.text((425 - w_br, 974), br_str, font=self.fonts["val_22b"], fill="#FFFFFF")

        gear_str = str(int(sample["gear"]))
        w_gr = draw.textlength(gear_str, font=self.fonts["gear_56b"])
        draw.text((485 - 0.5 * w_gr, 926), gear_str, font=self.fonts["gear_56b"], fill="#FFFFFF")

        st_deg_str = f"{int(round(abs(sample['steering_deg'])))}°"
        w_st = draw.textlength(st_deg_str, font=self.fonts["tbl_15b"])
        draw.text((595 - 0.5 * w_st, 994), st_deg_str, font=self.fonts["tbl_15b"], fill="#FFFFFF")

        # Core Digital Readouts [705, 495, 1120, 740]
        lap_str = self._format_lap_time(sample["t"])
        draw.text((705, 512), lap_str, font=self.fonts["lap_52b"], fill="#FFFFFF")

        spd_str = f"{int(round(sample['speed_kmh']))}"
        draw.text((705, 602), spd_str, font=self.fonts["spd_44b"], fill="#FFFFFF")
        w_spd = draw.textlength(spd_str, font=self.fonts["spd_44b"])
        draw.text((705 + w_spd + 10, 624), "km/h", font=self.fonts["tbl_14"], fill="#8A8D93")

        # MGU-K Badge Text
        st_lbl = str(sample["mguk_state"])
        w_bdg = draw.textlength(st_lbl, font=self.fonts["badge_19b"])
        draw.text((1017.5 - 0.5 * w_bdg, 612), st_lbl, font=self.fonts["badge_19b"], fill="#FFFFFF")

        # Fuel Used & MGU-K Power
        fuel_str = f"{sample['fuel_used_kg']:.3f} kg"
        draw.text((705, 692), fuel_str, font=self.fonts["val_24b"], fill="#FFFFFF")

        mguk_kw_int = int(round(sample["mguk_power_kw"]))
        mguk_str = f"{mguk_kw_int} kW"
        draw.text((930, 692), mguk_str, font=self.fonts["val_24b"], fill="#FFFFFF")

        # Energy Recovered Values
        tot_e_str = f"{sample['energy_total_recovered_mj']:.2f} MJ"
        w_te = draw.textlength(tot_e_str, font=self.fonts["tbl_16b"])
        draw.text((1110 - w_te, 756), tot_e_str, font=self.fonts["tbl_16b"], fill="#FFFFFF")

        for ry, val_mj in [
            (791, sample["energy_overrun_mj"]),
            (821, sample["energy_part_throttle_mj"]),
            (851, sample["energy_superclip_mj"]),
        ]:
            v_str = f"{val_mj:.2f}"
            w_v = draw.textlength(v_str, font=self.fonts["tbl_14"])
            draw.text((1110 - w_v, ry), v_str, font=self.fonts["tbl_14"], fill="#FFFFFF")

        # Lap Time Sensitivity Deltas
        sens_left = [
            (936, f"{sample['sens_power_s']:+.3f} s"),
            (968, f"{sample['sens_downforce_s']:+.3f} s"),
            (1000, f"{sample['sens_drag_s']:+.3f} s"),
        ]
        for sy, s_str in sens_left:
            w_s = draw.textlength(s_str, font=self.fonts["tbl_13"])
            draw.text((895 - w_s, sy), s_str, font=self.fonts["tbl_13"], fill="#FFFFFF")

        sens_right = [
            (936, f"{sample['sens_front_grip_s']:+.3f} s"),
            (968, f"{sample['sens_rear_grip_s']:+.3f} s"),
        ]
        for sy, s_str in sens_right:
            w_s = draw.textlength(s_str, font=self.fonts["tbl_13"])
            draw.text((1110 - w_s, sy), s_str, font=self.fonts["tbl_13"], fill="#FFFFFF")

        # 4-Corner Telemetry Blocks around 3D X-Ray Car
        self._draw_corner_telemetry_block(
            draw, "RR", (1400, 105, 1520, 210),
            sample["fz_rr_kn"], sample["camber_rr_deg"], sample["slip_rr_deg"],
        )
        self._draw_corner_telemetry_block(
            draw, "RL", (1695, 105, 1815, 210),
            sample["fz_rl_kn"], sample["camber_rl_deg"], sample["slip_rl_deg"],
        )
        self._draw_corner_telemetry_block(
            draw, "FR", (1285, 550, 1425, 665),
            sample["fz_fr_kn"], sample["camber_fr_deg"], sample["slip_fr_deg"],
        )
        self._draw_corner_telemetry_block(
            draw, "FL", (1630, 550, 1770, 665),
            sample["fz_fl_kn"], sample["camber_fl_deg"], sample["slip_fl_deg"],
        )

        # Aero & Chassis 3x2 Grid [1140, 700, 1890, 860]
        rh_f = int(round(sample["ride_height_front_mm"]))
        rh_r = int(round(sample["ride_height_rear_mm"]))
        draw.text((1140, 725), f"{rh_f} / {rh_r} mm", font=self.fonts["val_22b"], fill="#FFFFFF")

        roll_v = sample["roll_deg"]
        pitch_v = sample["pitch_deg"]
        draw.text((1405, 725), f"{roll_v:.1f}° / {pitch_v:.1f}°", font=self.fonts["val_22b"], fill="#FFFFFF")

        acc_v = sample["total_g"]
        draw.text((1665, 725), f"{acc_v:.2f} G", font=self.fonts["val_22b"], fill="#FFFFFF")

        df_i = int(round(sample["downforce_kg"]))
        dr_i = int(round(sample["drag_kg"]))
        draw.text((1140, 805), f"{df_i} / {dr_i} kg", font=self.fonts["val_22b"], fill="#FFFFFF")

        bal_v = sample["aero_balance_pct"]
        draw.text((1405, 805), f"{bal_v:.1f}%", font=self.fonts["val_22b"], fill="#FFFFFF")

        rho_v = sample["air_density"]
        draw.text((1665, 805), f"{rho_v:.3f} kg/m³", font=self.fonts["val_22b"], fill="#FFFFFF")

        # Sectors Table (S1, S2, S3) [1140, 905, 1370, 1040]
        for sy, s_val in [(935, sample["s1_str"]), (967, sample["s2_str"]), (999, sample["s3_str"])]:
            if s_val:
                w_sv = draw.textlength(s_val, font=self.fonts["tbl_15b"])
                draw.text((1355 - w_sv, sy), s_val, font=self.fonts["tbl_15b"], fill="#FFFFFF")

        # Driver Phases Legend & Percentages [1390, 905, 1890, 1045]
        phases = [
            (935, "Full throttle", f"{sample['phase_full_throttle_pct']:.1f}%"),
            (959, "Part throttle", f"{sample['phase_part_throttle_pct']:.1f}%"),
            (983, "Neutral",       f"{sample['phase_neutral_pct']:.1f}%"),
            (1007, "Braking",      f"{sample['phase_braking_pct']:.1f}%"),
        ]
        for py, p_lbl, p_val in phases:
            draw.text((1494, py), p_lbl, font=self.fonts["tbl_12"], fill="#8A8D93")
            w_pv = draw.textlength(p_val, font=self.fonts["tbl_13"])
            draw.text((1628 - w_pv, py), p_val, font=self.fonts["tbl_13"], fill="#FFFFFF")

        # Limited By Legend & Distances [1645, 905, 1890, 1045]
        lim_rows = [
            (964, "Power", f"{sample['limited_power_km']:.2f} km"),
            (992, "Grip",  f"{sample['limited_grip_km']:.2f} km"),
        ]
        for ly, l_lbl, l_val in lim_rows:
            draw.text((1660, ly), l_lbl, font=self.fonts["tbl_12"], fill="#8A8D93")
            w_lv = draw.textlength(l_val, font=self.fonts["tbl_13"])
            draw.text((1885 - w_lv, ly), l_val, font=self.fonts["tbl_13"], fill="#FFFFFF")

        return frame.convert("RGB")
