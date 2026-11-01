"""
generate_video.py — Master CLI script supporting:
  --preview : Renders a 10-second 30 FPS test clip (preview_brazil_2026.mp4)
              and exports 3 diagnostic PNG frames:
              - frame_01_superclip.png
              - frame_02_braking.png
              - frame_03_cornering.png
  --full    : Renders the full ~70-second qualifying lap (simulated lap time + 2.0s outro)
              at 60 FPS 1080p (brazil_2026_quali_1080p60.mp4) via an ffmpeg pipe.
"""

import argparse
import os
import shutil
import subprocess
import sys
import numpy as np
from tqdm import tqdm

from track_loader import load_interlagos_track
from physics_2026 import simulate_brazil_2026_lap, TelemetryResult
from car_3d_xray import F1XRayCarRenderer
from hud_renderer import BroadcastHUDRenderer


def _find_diagnostic_timestamps(tel: TelemetryResult) -> dict:
    """
    Finds representative timestamps for the 3 diagnostic PNG frames:
      1. frame_01_superclip.png : Active SUPERCLIP on Reta Oposta (100% throttle, -250 kW)
      2. frame_02_braking.png   : Heavy braking zone (HARVEST Overrun, -350 kW, nose dive)
      3. frame_03_cornering.png : High-G technical cornering / apex exit
    """
    # 1. Finish line crossing (full speed ~344 km/h, DEPLOY 100 kW, SOC hits 0.0 MJ)
    idx_sc = len(tel.s) - 2
    # 2. Heavy braking (into Descida do Lago T4 or Bico de Pato T10)
    br_indices = np.where((tel.brake_pct >= 80.0) & (tel.s >= 1310) & (tel.s <= 2800))[0]
    if len(br_indices) > 0:
        idx_br = int(br_indices[int(len(br_indices) * 0.25)])
    else:
        idx_br = int(np.argmax(tel.brake_pct))

    # 3. High-G cornering exit (DEPLOY state, high lateral G, inside front tire unloaded)
    cr_indices = np.where(
        (tel.mguk_state == "DEPLOY") & (np.abs(tel.lat_g) >= 1.8) & (tel.s >= 1400) & (tel.s <= 3260)
    )[0]
    if len(cr_indices) > 0:
        idx_cr = int(cr_indices[int(len(cr_indices) * 0.55)])
    else:
        idx_cr = int(np.argmax(np.abs(tel.lat_g)))

    return {
        "frame_01_superclip.png": float(tel.t[idx_sc]),
        "frame_02_braking.png":   float(tel.t[idx_br]),
        "frame_03_cornering.png": float(tel.t[idx_cr]),
    }


def export_diagnostic_frames(
    tel: TelemetryResult,
    car_renderer: F1XRayCarRenderer,
    hud_renderer: BroadcastHUDRenderer,
) -> None:
    """Renders and saves the 3 diagnostic 1920x1080 PNG frames."""
    diag_map = _find_diagnostic_timestamps(tel)
    for filename, t_sec in diag_map.items():
        sample = tel.sample_at_time(t_sec)
        car_img = car_renderer.render_frame(sample)
        hud_img = hud_renderer.render_hud_frame(sample, car_img)
        hud_img.save(filename, format="PNG")
        print(
            f"[Diagnostic] Saved {filename} at t={t_sec:.3f}s, s={sample['s']:.0f}m, "
            f"speed={sample['speed_kmh']:.1f}km/h, state={sample['mguk_state']}, "
            f"superclip={sample['energy_superclip_mj']:.2f}MJ"
        )
        try:
            art_dir = r"C:\Users\ppmak\.gemini\antigravity\brain\3bc2fc50-a50c-4fcb-9ab4-d340dbbac1ad"
            if os.path.exists(art_dir):
                shutil.copyfile(filename, os.path.join(art_dir, filename))
        except Exception:
            pass


def render_video_pipe(
    output_path: str,
    fps: int,
    t_start: float,
    t_end: float,
    outro_hold_s: float,
    tel: TelemetryResult,
    car_renderer: F1XRayCarRenderer,
    hud_renderer: BroadcastHUDRenderer,
) -> None:
    """Streams raw RGB24 frames directly into an FFmpeg H.264 MP4 encoding pipe."""
    play_duration = max(0.1, t_end - t_start)
    n_play_frames = int(round(play_duration * fps))
    n_outro_frames = int(round(outro_hold_s * fps))
    total_frames = n_play_frames + n_outro_frames

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", "1920x1080",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "-",
        "-an",
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-crf", "18",
        output_path,
    ]

    print(f"[Video] Starting FFmpeg pipe -> {output_path} ({total_frames} frames @ {fps} FPS)...")
    proc = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    last_frame_bytes = None
    try:
        for i in tqdm(range(total_frames), desc=f"Rendering {output_path}", unit="frame"):
            if i < n_play_frames:
                t_query = t_start + (i / float(fps))
                if i == n_play_frames - 1 and abs(t_end - tel.lap_time_s) < 1e-2:
                    t_query = tel.lap_time_s
                sample = tel.sample_at_time(t_query)
                car_img = car_renderer.render_frame(sample)
                hud_img = hud_renderer.render_hud_frame(sample, car_img)
                last_frame_bytes = hud_img.tobytes()
            else:
                # Hold final frame during outro
                if last_frame_bytes is None:
                    sample = tel.sample_at_time(t_end)
                    car_img = car_renderer.render_frame(sample)
                    hud_img = hud_renderer.render_hud_frame(sample, car_img)
                    last_frame_bytes = hud_img.tobytes()

            proc.stdin.write(last_frame_bytes)

        proc.stdin.close()
        stderr_bytes = proc.stderr.read()
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg exited with code {proc.returncode}:\n{stderr_bytes.decode(errors='ignore')}")
        print(f"[Video] Successfully generated {output_path}")
        try:
            art_dir = r"C:\Users\ppmak\.gemini\antigravity\brain\3bc2fc50-a50c-4fcb-9ab4-d340dbbac1ad"
            if os.path.exists(art_dir):
                import shutil
                shutil.copyfile(output_path, os.path.join(art_dir, os.path.basename(output_path)))
                print(f"[Artifact] Copied {output_path} to artifacts directory")
        except Exception as e:
            print(f"[Artifact] Note: {e}")
    finally:
        if proc.poll() is None:
            proc.kill()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="2026 F1 São Paulo GP (Interlagos) Qualifying Telemetry & 3D Vehicle Dynamics Video Generator"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Render a 10-second 30 FPS preview clip (preview_brazil_2026.mp4) and 3 diagnostic PNG frames.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Render the full ~70-second qualifying lap + 2s outro at 60 FPS 1080p (brazil_2026_quali_1080p60.mp4).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Target framerate for video rendering (default: 60).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config_brazil_2026.json",
        help="Path to JSON configuration file.",
    )
    args = parser.parse_args()

    if not args.preview and not args.full:
        args.preview = True

    print("[Init] Loading Interlagos 3D track geometry...")
    track = load_interlagos_track(config_path=args.config)

    print("[Init] Solving 2026 FIA QSS physics & energy telemetry...")
    tel = simulate_brazil_2026_lap(config_path=args.config, track=track)
    print(
        f"[Physics] Lap Time: {tel.lap_time_s:.3f}s "
        f"(S1: {tel.sector_times_s[0]:.3f}s, S2: {tel.sector_times_s[1]:.3f}s, S3: {tel.sector_times_s[2]:.3f}s)"
    )

    print("[Init] Initializing PyVista off-screen 3D X-Ray car renderer...")
    car_renderer = F1XRayCarRenderer()

    print("[Init] Initializing 1920x1080 Broadcast HUD compositor...")
    hud_renderer = BroadcastHUDRenderer(track=track, telemetry=tel, config_path=args.config)

    try:
        if args.preview:
            export_diagnostic_frames(tel, car_renderer, hud_renderer)
            # Render 10-second preview covering the finish straight (Juncao exit -> full power climb -> crossing finish line with SOC hitting 0)
            render_video_pipe(
                output_path="preview_brazil_2026.mp4",
                fps=30,
                t_start=max(0.0, tel.lap_time_s - 10.0),
                t_end=tel.lap_time_s,
                outro_hold_s=1.0,
                tel=tel,
                car_renderer=car_renderer,
                hud_renderer=hud_renderer,
            )

        if args.full:
            target_fps = getattr(args, 'fps', 60)
            out_filename = "brazil_2026_quali_1080p60.mp4" if target_fps == 60 else f"brazil_2026_quali_1080p{target_fps}.mp4"
            render_video_pipe(
                output_path=out_filename,
                fps=target_fps,
                t_start=0.0,
                t_end=tel.lap_time_s,
                outro_hold_s=2.0,
                tel=tel,
                car_renderer=car_renderer,
                hud_renderer=hud_renderer,
            )
    finally:
        car_renderer.close()


if __name__ == "__main__":
    main()
