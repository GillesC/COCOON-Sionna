"""Standalone visualization utilities for mobile AP schedules."""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Any

from matplotlib import animation
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

from .config import load_scenario_config
from .logging_utils import progress_bar
from .mobility import Trajectory, load_graph_json
from .pipeline import _draw_scene_background, _set_scene_axes, _trajectory_frame_rate
from .scene_builder import SceneArtifacts
from .sionna_rt_adapter import load_scene_metadata
from .sites import CandidateSite, load_candidate_sites

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ScheduleWindow:
    window_index: int
    start_time_s: float
    end_time_s: float
    sites: list[CandidateSite]


def _resolve_scene_artifacts(config) -> SceneArtifacts:
    if config.scene.kind == "builtin":
        return SceneArtifacts(scene_xml_path=Path("builtin"), metadata_path=None, walk_graph_path=config.mobility.graph_path)
    if config.scene.kind == "xml":
        metadata = config.scene.scene_output_dir / "scene_metadata.json" if config.scene.scene_output_dir else None
        return SceneArtifacts(
            scene_xml_path=config.scene.scene_xml_path or Path("builtin"),
            metadata_path=metadata,
            walk_graph_path=config.mobility.graph_path,
        )
    output_dir = config.scene.scene_output_dir
    if output_dir is None:
        raise ValueError("scene.scene_output_dir is required for OSM scenes")
    return SceneArtifacts(
        scene_xml_path=output_dir / "scene.xml",
        metadata_path=output_dir / "scene_metadata.json",
        walk_graph_path=output_dir / "walk_graph.json",
    )


def _load_trajectory_csv(path: Path) -> Trajectory:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Trajectory CSV is empty: {path}")

    times = sorted({float(row["time_s"]) for row in rows})
    ue_ids = sorted({str(row["ue_id"]) for row in rows})
    time_index = {value: idx for idx, value in enumerate(times)}
    ue_index = {value: idx for idx, value in enumerate(ue_ids)}

    positions = np.zeros((len(times), len(ue_ids), 3), dtype=float)
    velocities = np.zeros((len(times), len(ue_ids), 3), dtype=float)
    for row in rows:
        t_idx = time_index[float(row["time_s"])]
        u_idx = ue_index[str(row["ue_id"])]
        positions[t_idx, u_idx] = [float(row["x_m"]), float(row["y_m"]), float(row["z_m"])]
        velocities[t_idx, u_idx] = [float(row["vx_mps"]), float(row["vy_mps"]), float(row["vz_mps"])]
    return Trajectory(times_s=np.asarray(times, dtype=float), ue_ids=ue_ids, positions_m=positions, velocities_mps=velocities)


def load_mobile_ap_schedule(path: Path) -> list[ScheduleWindow]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(int(row["window_index"]), []).append(row)

    windows: list[ScheduleWindow] = []
    for window_index in sorted(grouped):
        window_rows = grouped[window_index]
        sites = [
            CandidateSite(
                site_id=str(row["ap_id"]),
                x_m=float(row["x_m"]),
                y_m=float(row["y_m"]),
                z_m=float(row["z_m"]),
                yaw_deg=0.0,
                pitch_deg=0.0,
                mount_type=str(row.get("source", "schedule")),
                source=str(row.get("source", "schedule")),
            )
            for row in window_rows
        ]
        windows.append(
            ScheduleWindow(
                window_index=window_index,
                start_time_s=float(window_rows[0]["start_time_s"]),
                end_time_s=float(window_rows[0]["end_time_s"]),
                sites=sites,
            )
        )
    return windows


def _window_for_time(windows: list[ScheduleWindow], time_s: float) -> ScheduleWindow:
    for window in windows:
        if window.start_time_s - 1e-9 <= time_s <= window.end_time_s + 1e-9:
            return window
    return windows[-1]


def _schedule_positions_array(windows: list[ScheduleWindow]) -> np.ndarray:
    points: list[list[float]] = []
    for window in windows:
        for site in window.sites:
            points.append([float(site.x_m), float(site.y_m)])
    if not points:
        return np.zeros((0, 2), dtype=float)
    return np.asarray(points, dtype=float)


def animate_mobile_ap_schedule(
    metadata: dict[str, Any] | None,
    graph,
    windows: list[ScheduleWindow],
    output_path: Path,
    trajectory: Trajectory | None = None,
    fixed_sites: list[CandidateSite] | None = None,
    speedup: float = 1.0,
) -> Path | None:
    if not windows:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    _draw_scene_background(ax, metadata, graph)

    if fixed_sites:
        ax.scatter(
            [site.x_m for site in fixed_sites],
            [site.y_m for site in fixed_sites],
            c="#2f5d8a",
            s=24,
            marker="s",
            alpha=0.45,
            zorder=3,
        )

    if trajectory is not None and trajectory.positions_m.size:
        positions = np.asarray(trajectory.positions_m[..., :2], dtype=float)
        ue_scatter = ax.scatter(
            positions[0, :, 0],
            positions[0, :, 1],
            c="#2ca25f",
            s=42,
            marker="o",
            edgecolors="black",
            linewidths=0.4,
            zorder=6,
        )
        ue_trails = [ax.plot([], [], color="#f28e2b", linewidth=1.4, alpha=0.55, zorder=4)[0] for _ in trajectory.ue_ids]
        frame_times = np.asarray(trajectory.times_s, dtype=float)
    else:
        positions = None
        ue_scatter = None
        ue_trails = []
        frame_times = np.asarray([0.5 * (window.start_time_s + window.end_time_s) for window in windows], dtype=float)

    ap_scatter = ax.scatter([], [], c="#cb3a2a", s=90, marker="^", edgecolors="black", linewidths=0.5, zorder=7)
    ap_trails: dict[str, Any] = {}
    for ap_id in sorted({site.site_id for window in windows for site in window.sites}):
        ap_trails[ap_id] = ax.plot([], [], color="#cb3a2a", linewidth=1.6, alpha=0.35, zorder=5)[0]

    timestamp = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox={"facecolor": "white", "edgecolor": "#444444", "alpha": 0.85, "boxstyle": "round,pad=0.25"},
    )

    legend_handles = []
    if fixed_sites:
        legend_handles.append(Line2D([0], [0], marker="s", linestyle="", color="#2f5d8a", markersize=7, label="Fixed APs"))
    legend_handles.append(
        Line2D([0], [0], marker="^", linestyle="", color="#cb3a2a", markeredgecolor="black", markersize=9, label="Mobile APs")
    )
    legend_handles.append(Line2D([0], [0], color="#cb3a2a", linewidth=1.6, alpha=0.35, label="AP path"))
    if trajectory is not None and trajectory.positions_m.size:
        legend_handles.extend(
            [
                Line2D([0], [0], marker="o", linestyle="", color="#2ca25f", markeredgecolor="black", markersize=7, label="UE"),
                Line2D([0], [0], color="#f28e2b", linewidth=1.6, alpha=0.55, label="UE trail"),
            ]
        )
    ax.legend(handles=legend_handles, loc="best")

    bounds_positions = positions if positions is not None else _schedule_positions_array(windows).reshape(-1, 1, 2)
    _set_scene_axes(ax, metadata, graph, positions=bounds_positions, sites=fixed_sites or [])
    ax.set_title("Mobile AP schedule animation")

    def _update(frame_index: int):
        time_s = float(frame_times[frame_index])
        window = _window_for_time(windows, time_s)
        current_sites = window.sites
        ap_offsets = np.asarray([[site.x_m, site.y_m] for site in current_sites], dtype=float)
        if ap_offsets.size == 0:
            ap_offsets = np.zeros((0, 2), dtype=float)
        ap_scatter.set_offsets(ap_offsets)

        for ap_id, line in ap_trails.items():
            path_points = []
            for candidate_window in windows:
                if candidate_window.start_time_s > window.end_time_s + 1e-9:
                    break
                for site in candidate_window.sites:
                    if site.site_id == ap_id:
                        path_points.append([site.x_m, site.y_m])
                        break
            if path_points:
                path = np.asarray(path_points, dtype=float)
                line.set_data(path[:, 0], path[:, 1])
            else:
                line.set_data([], [])

        artists: list[Any] = [ap_scatter, timestamp]
        artists.extend(ap_trails.values())

        if positions is not None and ue_scatter is not None:
            ue_scatter.set_offsets(positions[frame_index])
            for u_idx, line in enumerate(ue_trails):
                trail = positions[: frame_index + 1, u_idx, :]
                line.set_data(trail[:, 0], trail[:, 1])
            artists.append(ue_scatter)
            artists.extend(ue_trails)

        timestamp.set_text(
            f"t = {time_s:.1f} s\nwindow {window.window_index}: {window.start_time_s:.1f}-{window.end_time_s:.1f} s"
        )
        return artists

    if trajectory is not None:
        playback_fps = max(1.0, _trajectory_frame_rate(trajectory) * max(float(speedup), 1e-6))
    else:
        playback_fps = max(1.0, max(float(speedup), 1e-6))

    anim = animation.FuncAnimation(
        fig,
        _update,
        frames=len(frame_times),
        interval=max(1, int(round(1000.0 / playback_fps))),
        blit=False,
        repeat=True,
    )

    saved_path: Path | None = None
    try:
        ffmpeg_path = shutil.which("ffmpeg")
        target_path = output_path if ffmpeg_path else output_path.with_suffix(".gif")
        writer = (
            animation.FFMpegWriter(fps=playback_fps, bitrate=2400)
            if ffmpeg_path
            else animation.PillowWriter(fps=max(1, int(round(playback_fps))))
        )
        with progress_bar(total=len(frame_times), desc="Rendering AP schedule", unit="frame", leave=True) as render_progress:
            anim.save(
                str(target_path),
                writer=writer,
                dpi=180 if ffmpeg_path else 160,
                progress_callback=lambda frame_index, total_frames: render_progress.update(
                    max(0, int(frame_index) + 1 - getattr(render_progress, "n", 0))
                ),
            )
        saved_path = target_path
    except Exception:
        logger.exception("Failed to write mobile AP schedule animation")
    finally:
        plt.close(fig)

    return saved_path


def run_mobile_ap_schedule_visualization(
    scenario_path: str | Path,
    schedule_csv: Path | None = None,
    trajectory_csv: Path | None = None,
    output_path: Path | None = None,
    fixed_sites_csv: Path | None = None,
    speedup: float | None = None,
) -> Path | None:
    config = load_scenario_config(scenario_path)
    scene_artifacts = _resolve_scene_artifacts(config)
    metadata = load_scene_metadata(scene_artifacts.metadata_path)
    graph_path = config.mobility.graph_path or scene_artifacts.walk_graph_path
    if graph_path is None:
        raise ValueError("A mobility graph is required for schedule visualization")
    graph = load_graph_json(graph_path)

    output_dir = config.outputs.output_dir
    schedule_path = schedule_csv or output_dir / "random_baseline_schedule.csv"
    trajectory_path = trajectory_csv or output_dir / "trajectory.csv"
    fixed_path = fixed_sites_csv or output_dir / "fixed_aps.csv"
    animation_path = output_path or output_dir / "mobile_ap_schedule_animation.mp4"

    windows = load_mobile_ap_schedule(schedule_path)
    trajectory = _load_trajectory_csv(trajectory_path) if trajectory_path.exists() else None
    fixed_sites = load_candidate_sites(fixed_path) if fixed_path.exists() else None
    return animate_mobile_ap_schedule(
        metadata=metadata,
        graph=graph,
        windows=windows,
        output_path=animation_path,
        trajectory=trajectory,
        fixed_sites=fixed_sites,
        speedup=float(config.outputs.scene_animation_speedup if speedup is None else speedup),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize mobile AP schedules as an animation")
    parser.add_argument("scenario", help="Path to a scenario YAML file")
    parser.add_argument(
        "--schedule-csv",
        type=Path,
        default=None,
        help="Override path to a per-strategy schedule CSV such as random_baseline_schedule.csv",
    )
    parser.add_argument("--trajectory-csv", type=Path, default=None, help="Optional trajectory.csv for UE overlays")
    parser.add_argument("--fixed-sites-csv", type=Path, default=None, help="Optional fixed_aps.csv for reference markers")
    parser.add_argument("--output", type=Path, default=None, help="Output MP4 or GIF path")
    parser.add_argument("--speedup", type=float, default=None, help="Playback speed multiplier")
    args = parser.parse_args()

    result = run_mobile_ap_schedule_visualization(
        scenario_path=args.scenario,
        schedule_csv=args.schedule_csv,
        trajectory_csv=args.trajectory_csv,
        output_path=args.output,
        fixed_sites_csv=args.fixed_sites_csv,
        speedup=args.speedup,
    )
    if result is not None:
        print(result)


if __name__ == "__main__":
    main()
