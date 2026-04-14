"""Top-level orchestration for scene build, CSI extraction, and optimization."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, replace
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

from matplotlib import animation
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Point, Polygon

from .config import ScenarioConfig, load_scenario_config
from .logging_utils import progress_bar
from .mobility import Trajectory, generate_trajectory, load_graph_json
from .optimization import PlacementScore, greedy_one_swap, summarize_candidate_set
from .scene_builder import OSMSceneBuilder, SceneArtifacts
from .sionna_rt_adapter import SceneInputs, SionnaRtRunner, load_scene_metadata
from .sites import CandidateSite
from .sites import (
    augment_with_trajectory_sites,
    generate_wall_candidate_sites,
    load_candidate_sites,
    select_farthest_sites,
    write_candidate_sites,
)

logger = logging.getLogger(__name__)


def _resolve_scene_inputs(config: ScenarioConfig) -> SceneArtifacts:
    if config.scene.kind == "builtin":
        logger.info("Using builtin Sionna scene '%s'", config.scene.sionna_scene)
        return SceneArtifacts(scene_xml_path=Path("builtin"), metadata_path=None, walk_graph_path=config.mobility.graph_path)
    if config.scene.kind not in {"osm", "xml"}:
        raise ValueError(f"Unsupported scene kind: {config.scene.kind}")
    if config.scene.kind == "xml":
        if config.scene.scene_xml_path is None:
            raise ValueError("scene.scene_xml_path is required for xml scenes")
        logger.info("Using existing scene XML at %s", config.scene.scene_xml_path)
        metadata = config.scene.scene_output_dir / "scene_metadata.json" if config.scene.scene_output_dir else None
        walk_graph = config.mobility.graph_path
        return SceneArtifacts(scene_xml_path=config.scene.scene_xml_path, metadata_path=metadata, walk_graph_path=walk_graph)

    output_dir = config.scene.scene_output_dir
    assert output_dir is not None
    scene_xml = output_dir / "scene.xml"
    metadata = output_dir / "scene_metadata.json"
    walk_graph = output_dir / "walk_graph.json"
    if config.scene.rebuild or not scene_xml.exists():
        logger.info("Building OSM-derived scene assets in %s", output_dir)
        return OSMSceneBuilder(config.scene).build()
    logger.info("Reusing existing OSM scene assets from %s", output_dir)
    return SceneArtifacts(scene_xml_path=scene_xml, metadata_path=metadata, walk_graph_path=walk_graph)


def _mask_best_sinr(best_sinr_db: np.ndarray, cell_centers: np.ndarray, metadata: dict[str, Any] | None) -> np.ndarray:
    if metadata is None or "boundary_local" not in metadata:
        return best_sinr_db.reshape(-1)
    boundary = Polygon(metadata["boundary_local"])
    building_polygons = [Polygon(item["polygon_local"]) for item in metadata.get("buildings", [])]
    flat_centers = cell_centers.reshape(-1, 3)
    flat_values = best_sinr_db.reshape(-1)
    masked = []
    for center, value in zip(flat_centers, flat_values):
        point = Point(float(center[0]), float(center[1]))
        if not boundary.covers(point):
            continue
        if any(polygon.covers(point) for polygon in building_polygons):
            continue
        masked.append(float(value))
    return np.asarray(masked, dtype=float)


def _plot_polygon_outline(ax, coordinates: list[list[float]] | np.ndarray, **kwargs) -> None:
    xy = np.asarray(coordinates, dtype=float)
    if xy.ndim != 2 or xy.shape[0] < 2:
        return
    ax.plot(xy[:, 0], xy[:, 1], **kwargs)


def _plot_polygon_fill(ax, coordinates: list[list[float]] | np.ndarray, **kwargs) -> None:
    xy = np.asarray(coordinates, dtype=float)
    if xy.ndim != 2 or xy.shape[0] < 3:
        return
    ax.fill(xy[:, 0], xy[:, 1], **kwargs)


def _draw_scene_background(ax, metadata: dict[str, Any] | None, graph) -> None:
    if metadata is not None and "boundary_local" in metadata:
        _plot_polygon_fill(ax, metadata["boundary_local"], facecolor="#f3efe3", edgecolor="none", alpha=1.0)
        _plot_polygon_outline(ax, metadata["boundary_local"], color="#2d2d2d", linewidth=1.2)
        for building in metadata.get("buildings", []):
            _plot_polygon_fill(
                ax,
                building["polygon_local"],
                facecolor="#b9b2a3",
                edgecolor="#6e675d",
                linewidth=0.6,
                alpha=0.95,
            )

    if graph.number_of_edges():
        for u, v in graph.edges():
            start = graph.nodes[u]
            end = graph.nodes[v]
            ax.plot(
                [float(start["x"]), float(end["x"])],
                [float(start["y"]), float(end["y"])],
                color="#6ea6bf",
                linewidth=1.0,
                alpha=0.55,
                zorder=1,
            )


def _set_scene_axes(ax, metadata: dict[str, Any] | None, graph, positions: np.ndarray | None = None, sites: list[CandidateSite] | None = None) -> None:
    xs: list[float] = []
    ys: list[float] = []

    if metadata is not None and "boundary_local" in metadata:
        boundary = np.asarray(metadata["boundary_local"], dtype=float)
        xs.extend(boundary[:, 0].tolist())
        ys.extend(boundary[:, 1].tolist())
    else:
        for _, attrs in graph.nodes(data=True):
            xs.append(float(attrs["x"]))
            ys.append(float(attrs["y"]))

    if positions is not None and positions.size:
        flat = positions.reshape(-1, 2)
        xs.extend(flat[:, 0].tolist())
        ys.extend(flat[:, 1].tolist())
    if sites:
        xs.extend(float(site.x_m) for site in sites)
        ys.extend(float(site.y_m) for site in sites)

    if not xs or not ys:
        xs = [-1.0, 1.0]
        ys = [-1.0, 1.0]

    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)
    pad = 0.06 * max(x_max - x_min, y_max - y_min, 10.0)
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def _scene_xy_bounds(
    metadata: dict[str, Any] | None,
    graph,
    positions: np.ndarray | None = None,
    sites: list[CandidateSite] | None = None,
) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []

    if metadata is not None and "boundary_local" in metadata:
        boundary = np.asarray(metadata["boundary_local"], dtype=float)
        xs.extend(boundary[:, 0].tolist())
        ys.extend(boundary[:, 1].tolist())
    else:
        for _, attrs in graph.nodes(data=True):
            xs.append(float(attrs["x"]))
            ys.append(float(attrs["y"]))

    if positions is not None and positions.size:
        flat = positions.reshape(-1, 2)
        xs.extend(flat[:, 0].tolist())
        ys.extend(flat[:, 1].tolist())
    if sites:
        xs.extend(float(site.x_m) for site in sites)
        ys.extend(float(site.y_m) for site in sites)

    if not xs or not ys:
        return (-1.0, 1.0, -1.0, 1.0)
    return (min(xs), max(xs), min(ys), max(ys))


def _trajectory_frame_rate(trajectory) -> float:
    times = np.asarray(trajectory.times_s, dtype=float)
    if times.size <= 1:
        return 1.0
    deltas = np.diff(times)
    positive = deltas[deltas > 1e-9]
    if positive.size == 0:
        return 1.0
    return float(1.0 / np.median(positive))


def _plot_scene_layout(
    metadata: dict[str, Any] | None,
    graph,
    base_sites: list[CandidateSite],
    selected_sites: list[CandidateSite],
    trajectory,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    _draw_scene_background(ax, metadata, graph)

    if base_sites:
        ax.scatter(
            [site.x_m for site in base_sites],
            [site.y_m for site in base_sites],
            c="#2f5d8a",
            s=26,
            marker="s",
            alpha=0.7,
            label="Candidate APs",
            zorder=3,
        )
    if selected_sites:
        ax.scatter(
            [site.x_m for site in selected_sites],
            [site.y_m for site in selected_sites],
            c="#cb3a2a",
            s=80,
            marker="^",
            edgecolors="black",
            linewidths=0.5,
            label="Selected APs",
            zorder=5,
        )

    positions = np.asarray(trajectory.positions_m[..., :2], dtype=float)
    for u_idx in range(positions.shape[1]):
        xy = positions[:, u_idx, :]
        ax.plot(xy[:, 0], xy[:, 1], color="#f28e2b", linewidth=1.5, alpha=0.65, zorder=2)
    if positions.size:
        ax.scatter(
            positions[0, :, 0],
            positions[0, :, 1],
            c="#2ca25f",
            s=36,
            marker="o",
            edgecolors="black",
            linewidths=0.4,
            label="UE start",
            zorder=6,
        )
        ax.scatter(
            positions[-1, :, 0],
            positions[-1, :, 1],
            c="#f28e2b",
            s=50,
            marker="X",
            edgecolors="black",
            linewidths=0.4,
            label="UE end",
            zorder=6,
        )

    _set_scene_axes(ax, metadata, graph, positions=positions, sites=base_sites)
    ax.set_title("Scene layout with AP placement and UE motion")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _build_scene_camera(rt, metadata: dict[str, Any] | None, graph, positions: np.ndarray, sites: list[CandidateSite]):
    x_min, x_max, y_min, y_max = _scene_xy_bounds(metadata, graph, positions=positions[..., :2], sites=sites)
    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)
    span = max(x_max - x_min, y_max - y_min, 20.0)
    camera_height = max(0.9 * span, 35.0)
    camera_offset = 1.1 * span
    look_at = rt["mi"].Point3f(x_center, y_center, 1.5)
    return rt["rt"].Camera(
        position=rt["mi"].Point3f(x_center - camera_offset, y_center - 0.75 * camera_offset, camera_height),
        look_at=look_at,
    )


def _render_scene_view(
    runner: SionnaRtRunner,
    metadata: dict[str, Any] | None,
    graph,
    sites: list[CandidateSite],
    trajectory,
    output_path: Path,
    snapshot_index: int | None = None,
) -> Path | None:
    if len(trajectory.times_s) == 0:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_index = len(trajectory.times_s) // 2 if snapshot_index is None else int(snapshot_index)
    frame_index = max(0, min(frame_index, len(trajectory.times_s) - 1))

    scene, rt = runner._load_scene(tx_role="ap", rx_role="ue")
    runner._add_ap_transmitters(scene, rt, sites)
    rx_names = [f"ue_rx_{ue_id}" for ue_id in trajectory.ue_ids]
    runner._add_receivers(scene, rt, rx_names, trajectory.positions_m[frame_index], trajectory.velocities_mps[frame_index])

    camera = _build_scene_camera(rt, metadata, graph, np.asarray(trajectory.positions_m, dtype=float), sites)

    try:
        scene.render_to_file(
            camera=camera,
            filename=str(output_path),
            resolution=(1280, 720),
            num_samples=64,
            show_devices=True,
            show_orientations=False,
        )
    except Exception:
        logger.exception("Failed to render scene view to %s", output_path)
        return None
    return output_path


def _animate_scene(
    metadata: dict[str, Any] | None,
    graph,
    base_sites: list[CandidateSite],
    selected_sites: list[CandidateSite],
    trajectory,
    output_path: Path,
    speedup: float = 1.0,
) -> Path | None:
    positions = np.asarray(trajectory.positions_m[..., :2], dtype=float)
    if positions.ndim != 3 or positions.shape[0] == 0 or positions.shape[1] == 0:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    _draw_scene_background(ax, metadata, graph)

    if base_sites:
        ax.scatter(
            [site.x_m for site in base_sites],
            [site.y_m for site in base_sites],
            c="#2f5d8a",
            s=24,
            marker="s",
            alpha=0.5,
            zorder=3,
        )
    if selected_sites:
        ax.scatter(
            [site.x_m for site in selected_sites],
            [site.y_m for site in selected_sites],
            c="#cb3a2a",
            s=80,
            marker="^",
            edgecolors="black",
            linewidths=0.5,
            zorder=5,
        )

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
    trail_lines = [ax.plot([], [], color="#f28e2b", linewidth=1.6, alpha=0.7, zorder=4)[0] for _ in trajectory.ue_ids]
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

    legend_handles = [
        Line2D([0], [0], marker="s", linestyle="", color="#2f5d8a", markersize=7, label="Candidate APs"),
        Line2D(
            [0],
            [0],
            marker="^",
            linestyle="",
            color="#cb3a2a",
            markeredgecolor="black",
            markersize=9,
            label="Selected APs",
        ),
        Line2D([0], [0], marker="o", linestyle="", color="#2ca25f", markeredgecolor="black", markersize=7, label="UE"),
        Line2D([0], [0], color="#f28e2b", linewidth=1.8, label="UE trail"),
    ]
    ax.legend(handles=legend_handles, loc="best")
    _set_scene_axes(ax, metadata, graph, positions=positions, sites=base_sites)
    ax.set_title("Scene animation with AP placement and moving UEs")

    def _update(frame_index: int):
        ue_scatter.set_offsets(positions[frame_index])
        for u_idx, line in enumerate(trail_lines):
            trail = positions[: frame_index + 1, u_idx, :]
            line.set_data(trail[:, 0], trail[:, 1])
        timestamp.set_text(f"t = {float(trajectory.times_s[frame_index]):.1f} s")
        artists = [ue_scatter, timestamp]
        artists.extend(trail_lines)
        return artists

    playback_fps = max(1.0, _trajectory_frame_rate(trajectory) * max(float(speedup), 1e-6))
    anim = animation.FuncAnimation(
        fig,
        _update,
        frames=len(trajectory.times_s),
        interval=max(1, int(round(1000.0 / playback_fps))),
        blit=False,
        repeat=True,
    )

    saved_path: Path | None = None
    try:
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path:
            writer = animation.FFMpegWriter(fps=playback_fps, bitrate=2400)
            anim.save(str(output_path), writer=writer, dpi=180)
            saved_path = output_path
        else:
            gif_path = output_path.with_suffix(".gif")
            writer = animation.PillowWriter(fps=max(1, int(round(playback_fps))))
            anim.save(str(gif_path), writer=writer, dpi=160)
            saved_path = gif_path
    except Exception:
        logger.exception("Failed to write scene animation")
    finally:
        plt.close(fig)

    return saved_path


def _render_scene_video(
    runner: SionnaRtRunner,
    metadata: dict[str, Any] | None,
    graph,
    sites: list[CandidateSite],
    trajectory,
    output_path: Path,
) -> Path | None:
    if len(trajectory.times_s) == 0:
        return None

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        logger.warning("Skipping camera-rendered scene video because ffmpeg is unavailable")
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fps = _trajectory_frame_rate(trajectory)

    try:
        with tempfile.TemporaryDirectory(prefix="scene_camera_frames_") as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            for frame_index in range(len(trajectory.times_s)):
                scene, rt = runner._load_scene(tx_role="ap", rx_role="ue")
                runner._add_ap_transmitters(scene, rt, sites)
                rx_names = [f"ue_rx_{ue_id}" for ue_id in trajectory.ue_ids]
                runner._add_receivers(
                    scene,
                    rt,
                    rx_names,
                    trajectory.positions_m[frame_index],
                    trajectory.velocities_mps[frame_index],
                )
                camera = _build_scene_camera(rt, metadata, graph, np.asarray(trajectory.positions_m, dtype=float), sites)
                frame_path = tmp_dir / f"frame_{frame_index:04d}.png"
                scene.render_to_file(
                    camera=camera,
                    filename=str(frame_path),
                    resolution=(640, 360),
                    num_samples=8,
                    show_devices=True,
                    show_orientations=False,
                )

            command = [
                ffmpeg_path,
                "-y",
                "-framerate",
                f"{fps:.6f}",
                "-i",
                str(tmp_dir / "frame_%04d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output_path),
            ]
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                logger.warning("ffmpeg failed while writing scene camera video: %s", result.stderr.strip() or result.stdout.strip())
                return None
    except Exception:
        logger.exception("Failed to render camera scene video")
        return None

    return output_path


def _plot_coverage(best_sinr_db: np.ndarray, cell_centers: np.ndarray, sites, trajectory, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    if best_sinr_db.ndim == 2:
        centers = np.asarray(cell_centers, dtype=float)
        x_values = centers[..., 0]
        y_values = centers[..., 1]
        dx = float(np.median(np.diff(x_values, axis=1))) if centers.shape[1] > 1 else 1.0
        dy = float(np.median(np.diff(y_values, axis=0))) if centers.shape[0] > 1 else 1.0
        extent = (
            float(np.min(x_values) - 0.5 * dx),
            float(np.max(x_values) + 0.5 * dx),
            float(np.min(y_values) - 0.5 * dy),
            float(np.max(y_values) + 0.5 * dy),
        )
        plt.imshow(best_sinr_db, origin="lower", cmap="viridis", extent=extent, aspect="equal")
        plt.colorbar(label="Best-server SINR [dB]")
    x = [site.x_m for site in sites]
    y = [site.y_m for site in sites]
    plt.scatter(x, y, c="red", marker="^", label="Selected APs")
    positions = trajectory.positions_m[..., :2].reshape(-1, 2)
    plt.scatter(positions[:, 0], positions[:, 1], c="white", s=8, alpha=0.5, label="UE trajectory")
    plt.legend()
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _plot_colored_trajectories(trajectory, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    times = np.asarray(trajectory.times_s, dtype=float)
    cmap = plt.get_cmap("turbo")
    time_min = float(times.min()) if times.size else 0.0
    time_max = float(times.max()) if times.size else 1.0
    norm = Normalize(vmin=time_min, vmax=time_max if time_max > time_min else time_min + 1.0)

    for u_idx, _ue_id in enumerate(trajectory.ue_ids):
        xy = np.asarray(trajectory.positions_m[:, u_idx, :2], dtype=float)
        if len(xy) == 0:
            continue
        if len(xy) == 1:
            ax.scatter(
                xy[0, 0],
                xy[0, 1],
                c=[times[0] if times.size else 0.0],
                cmap=cmap,
                norm=norm,
                s=32,
                edgecolors="black",
                linewidths=0.4,
            )
            continue

        segments = np.stack([xy[:-1], xy[1:]], axis=1)
        collection = LineCollection(segments, cmap=cmap, norm=norm, linewidths=2.5, alpha=0.95)
        collection.set_array(times[:-1])
        ax.add_collection(collection)
        ax.scatter(
            xy[0, 0],
            xy[0, 1],
            c=[times[0]],
            cmap=cmap,
            norm=norm,
            s=32,
            marker="o",
            edgecolors="black",
            linewidths=0.4,
        )
        ax.scatter(
            xy[-1, 0],
            xy[-1, 1],
            c=[times[-1]],
            cmap=cmap,
            norm=norm,
            s=44,
            marker="X",
            edgecolors="black",
            linewidths=0.4,
        )

    ax.autoscale()
    ax.margins(0.05)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("UE trajectories colored by time")
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="Time [s]")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _select_static_baseline_sites(config: ScenarioConfig, base_sites: list[CandidateSite]) -> list[CandidateSite]:
    enabled_sites = [site for site in base_sites if site.enabled]
    if not enabled_sites:
        logger.warning("No enabled fixed baseline AP sites were found in %s", config.candidate_sites_path)
        return []

    if config.optimization.baseline_site_ids:
        enabled_ids = {site.site_id for site in enabled_sites}
        missing = [site_id for site_id in config.optimization.baseline_site_ids if site_id not in enabled_ids]
        if missing:
            logger.warning("Ignoring unknown baseline site ids: %s", ", ".join(missing))
        logger.info(
            "Using all %d enabled fixed APs as the movable baseline; baseline_site_ids no longer restrict selection",
            len(enabled_sites),
        )
    return enabled_sites


def _relocate_sites(
    baseline_sites: list[CandidateSite],
    selected_candidates: list[CandidateSite],
) -> list[CandidateSite]:
    if len(baseline_sites) != len(selected_candidates):
        raise ValueError("baseline site count must match selected candidate count for relocation")
    if not baseline_sites:
        return []

    baseline_xy = np.asarray([[site.x_m, site.y_m] for site in baseline_sites], dtype=float)
    candidate_xy = np.asarray([[site.x_m, site.y_m] for site in selected_candidates], dtype=float)
    row_ind, col_ind = linear_sum_assignment(np.sum((baseline_xy[:, None, :] - candidate_xy[None, :, :]) ** 2, axis=2))

    relocated: list[CandidateSite | None] = [None] * len(baseline_sites)
    for baseline_index, candidate_index in zip(row_ind.tolist(), col_ind.tolist(), strict=True):
        original = baseline_sites[baseline_index]
        candidate = selected_candidates[candidate_index]
        relocated[baseline_index] = CandidateSite(
            site_id=original.site_id,
            x_m=candidate.x_m,
            y_m=candidate.y_m,
            z_m=candidate.z_m,
            yaw_deg=candidate.yaw_deg,
            pitch_deg=candidate.pitch_deg,
            mount_type=original.mount_type,
            enabled=original.enabled,
            source=f"relocated:{candidate.site_id}",
        )
    return [site for site in relocated if site is not None]


def _write_ap_relocation_csv(
    output_path: Path,
    baseline_sites: list[CandidateSite],
    relocated_sites: list[CandidateSite],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_index = {site.site_id: site for site in baseline_sites}
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "site_id",
                "original_x_m",
                "original_y_m",
                "original_z_m",
                "relocated_x_m",
                "relocated_y_m",
                "relocated_z_m",
                "move_distance_m",
                "relocation_source",
            ]
        )
        for site in relocated_sites:
            original = baseline_index[site.site_id]
            writer.writerow(
                [
                    site.site_id,
                    original.x_m,
                    original.y_m,
                    original.z_m,
                    site.x_m,
                    site.y_m,
                    site.z_m,
                    float(np.hypot(site.x_m - original.x_m, site.y_m - original.y_m)),
                    site.source,
                ]
            )


def _slice_trajectory(trajectory: Trajectory, indices: np.ndarray) -> Trajectory:
    return Trajectory(
        times_s=np.asarray(trajectory.times_s[indices], dtype=float),
        ue_ids=list(trajectory.ue_ids),
        positions_m=np.asarray(trajectory.positions_m[indices], dtype=float),
        velocities_mps=np.asarray(trajectory.velocities_mps[indices], dtype=float),
    )


def _window_slices(times_s: np.ndarray, relocation_interval_s: float) -> list[np.ndarray]:
    times = np.asarray(times_s, dtype=float)
    if times.size == 0:
        return []
    interval_s = max(float(relocation_interval_s), 1e-6)
    window_ids = np.floor((times - times[0]) / interval_s).astype(int)
    return [np.flatnonzero(window_ids == window_id) for window_id in np.unique(window_ids)]


def _concat_ap_ue_segments(segments: list[dict[str, Any]]) -> dict[str, Any]:
    if not segments:
        raise ValueError("segments must not be empty")
    result = {
        "tx_site_ids": list(segments[0]["tx_site_ids"]),
        "rx_ue_ids": list(segments[0]["rx_ue_ids"]),
        "times_s": np.concatenate([np.asarray(segment["times_s"], dtype=float) for segment in segments], axis=0),
        "best_sinr_db": np.concatenate([np.asarray(segment["best_sinr_db"], dtype=float) for segment in segments], axis=0),
        "link_power_w": np.concatenate([np.asarray(segment["link_power_w"]) for segment in segments], axis=0),
    }
    if all("cfr" in segment for segment in segments):
        result["cfr"] = np.concatenate([np.asarray(segment["cfr"]) for segment in segments], axis=0)
    if all("cir" in segment for segment in segments):
        result["cir"] = np.concatenate([np.asarray(segment["cir"]) for segment in segments], axis=0)
    if all("tau" in segment for segment in segments):
        result["tau"] = np.concatenate([np.asarray(segment["tau"]) for segment in segments], axis=0)
    return result


def _update_prefixed_export(
    target: dict[str, Any],
    prefix: str,
    source: dict[str, Any],
    keys: tuple[str, ...],
) -> None:
    for key in keys:
        if key in source:
            target[f"{prefix}_{key}"] = source[key]


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _hash_bytes(*parts: bytes) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part)
    return digest.hexdigest()


def _hash_array(array: np.ndarray) -> str:
    arr = np.asarray(array)
    return _hash_bytes(
        str(arr.dtype).encode("utf-8"),
        json.dumps(arr.shape).encode("utf-8"),
        arr.tobytes(),
    )


def _file_digest(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _site_to_dict(site: CandidateSite) -> dict[str, Any]:
    return {
        "site_id": site.site_id,
        "x_m": float(site.x_m),
        "y_m": float(site.y_m),
        "z_m": float(site.z_m),
        "yaw_deg": float(site.yaw_deg),
        "pitch_deg": float(site.pitch_deg),
        "mount_type": site.mount_type,
        "enabled": bool(site.enabled),
        "source": site.source,
    }


def _site_from_dict(data: dict[str, Any]) -> CandidateSite:
    return CandidateSite(
        site_id=str(data["site_id"]),
        x_m=float(data["x_m"]),
        y_m=float(data["y_m"]),
        z_m=float(data["z_m"]),
        yaw_deg=float(data["yaw_deg"]),
        pitch_deg=float(data["pitch_deg"]),
        mount_type=str(data["mount_type"]),
        enabled=bool(data.get("enabled", True)),
        source=str(data.get("source", "cache")),
    )


def _placement_score_to_dict(score: PlacementScore) -> dict[str, Any]:
    return {
        "score": float(score.score),
        "outage": float(score.outage),
        "percentile_10_db": float(score.percentile_10_db),
        "peer_tiebreak": float(score.peer_tiebreak),
        "grid_outage": float(score.grid_outage),
        "trajectory_outage": float(score.trajectory_outage),
    }


def _placement_score_from_dict(data: dict[str, Any]) -> PlacementScore:
    return PlacementScore(
        score=float(data["score"]),
        outage=float(data["outage"]),
        percentile_10_db=float(data["percentile_10_db"]),
        peer_tiebreak=float(data["peer_tiebreak"]),
        grid_outage=float(data["grid_outage"]),
        trajectory_outage=float(data["trajectory_outage"]),
    )


def _build_csi_cache_key(
    config: ScenarioConfig,
    scene_artifacts: SceneArtifacts,
    graph_path: Path,
    trajectory: Trajectory,
    base_sites: list[CandidateSite],
    candidate_sites: list[CandidateSite],
    baseline_sites: list[CandidateSite],
) -> str:
    config_payload = asdict(config)
    config_payload.pop("outputs", None)
    config_payload.pop("scenario_path", None)
    if "scene" in config_payload:
        config_payload["scene"].pop("rebuild", None)
        config_payload["scene"].pop("scene_output_dir", None)

    payload = {
        "config": _json_ready(config_payload),
        "scene_xml_digest": _file_digest(scene_artifacts.scene_xml_path if scene_artifacts.scene_xml_path != Path("builtin") else None),
        "scene_metadata_digest": _file_digest(scene_artifacts.metadata_path),
        "graph_digest": _file_digest(graph_path),
        "trajectory_times_digest": _hash_array(trajectory.times_s),
        "trajectory_positions_digest": _hash_array(trajectory.positions_m),
        "trajectory_velocities_digest": _hash_array(trajectory.velocities_mps),
        "base_sites": [_site_to_dict(site) for site in base_sites],
        "candidate_sites": [_site_to_dict(site) for site in candidate_sites],
        "baseline_sites": [_site_to_dict(site) for site in baseline_sites],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _cache_dir(output_dir: Path, cache_key: str) -> Path:
    return output_dir / ".csi_cache" / cache_key


def _should_render_sionna_scene_artifacts(runtime_info: dict[str, str] | None) -> bool:
    if runtime_info is None:
        return False
    device = str(runtime_info.get("device", "")).upper()
    variant = str(runtime_info.get("variant", ""))
    return not (device == "CPU" and variant.startswith("llvm_ad"))


def _split_prefixed_export(payload: dict[str, Any], prefix: str, base_keys: tuple[str, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in base_keys:
        prefixed = f"{prefix}_{key}"
        if prefixed in payload:
            value = payload[prefixed]
            if isinstance(value, np.ndarray) and value.dtype == object:
                result[key] = value.tolist()
            else:
                result[key] = value
    for optional in ("cfr", "cir", "tau"):
        prefixed = f"{prefix}_{optional}"
        if prefixed in payload:
            result[optional] = payload[prefixed]
    return result


def _load_npz_payload(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _try_load_csi_cache(output_dir: Path, cache_key: str) -> dict[str, Any] | None:
    cache_dir = _cache_dir(output_dir, cache_key)
    manifest_path = cache_dir / "manifest.json"
    peer_path = cache_dir / "peer_csi_snapshots.npz"
    infra_path = cache_dir / "infra_csi_snapshots.npz"
    coverage_path = cache_dir / "coverage_map.npz"
    fixed_coverage_path = cache_dir / "fixed_coverage_map.npz"
    required = [manifest_path, peer_path, infra_path]
    if not all(path.exists() for path in required):
        return None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    peer_csi = _load_npz_payload(peer_path)
    infra = _load_npz_payload(infra_path)
    final_radio_map = _load_npz_payload(coverage_path) if coverage_path.exists() else None
    fixed_radio_map = _load_npz_payload(fixed_coverage_path) if fixed_coverage_path.exists() else None
    return {
        "runtime_info": manifest["runtime_info"],
        "peer_csi": peer_csi,
        "fixed_ap_ue": _split_prefixed_export(
            infra,
            "fixed_ap_ue",
            ("tx_site_ids", "rx_ue_ids", "times_s", "best_sinr_db", "link_power_w"),
        ),
        "final_ap_ue": _split_prefixed_export(
            infra,
            "mobile_ap_ue",
            ("tx_site_ids", "rx_ue_ids", "times_s", "best_sinr_db", "link_power_w"),
        ),
        "fixed_ap_ap": _split_prefixed_export(
            infra,
            "fixed_ap_ap",
            ("tx_site_ids", "rx_site_ids", "link_power_w"),
        ),
        "final_ap_ap": _split_prefixed_export(
            infra,
            "mobile_ap_ap",
            ("tx_site_ids", "rx_site_ids", "link_power_w"),
        ),
        "fixed_radio_map": fixed_radio_map,
        "final_radio_map": final_radio_map,
        "selected_sites": [_site_from_dict(item) for item in manifest["selected_sites"]],
        "final_selected_candidate_ids": [str(item) for item in manifest["final_selected_candidate_ids"]],
        "selected_candidate_union": set(str(item) for item in manifest["selected_candidate_union"]),
        "schedule_rows": list(manifest["schedule_rows"]),
        "fixed_score": _placement_score_from_dict(manifest["fixed_score"]),
        "best_score": _placement_score_from_dict(manifest["best_score"]),
        "num_relocation_windows": int(manifest["num_relocation_windows"]),
        "cache_key": cache_key,
    }


def _cache_optional_artifact(output_dir: Path, cache_key: str, source_path: Path, cache_name: str) -> None:
    if not source_path.exists():
        return
    cache_dir = _cache_dir(output_dir, cache_key)
    cache_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, cache_dir / cache_name)


def _restore_cached_artifact(output_dir: Path, cache_key: str, cache_name: str, destination_path: Path) -> Path | None:
    cached_path = _cache_dir(output_dir, cache_key) / cache_name
    if not cached_path.exists():
        return None
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached_path, destination_path)
    return destination_path


def _remove_artifacts(paths: list[Path]) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def _write_csi_cache(
    output_dir: Path,
    cache_key: str,
    runtime_info: dict[str, str],
    peer_csi: dict[str, Any],
    fixed_ap_ue: dict[str, Any],
    final_ap_ue: dict[str, Any],
    fixed_ap_ap: dict[str, Any],
    final_ap_ap: dict[str, Any],
    fixed_radio_map: dict[str, Any] | None,
    final_radio_map: dict[str, Any] | None,
    selected_sites: list[CandidateSite],
    final_selected_candidate_ids: list[str],
    selected_candidate_union: set[str],
    schedule_rows: list[dict[str, Any]],
    fixed_score: PlacementScore,
    best_score: PlacementScore,
    num_relocation_windows: int,
) -> None:
    cache_dir = _cache_dir(output_dir, cache_key)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_dir / "peer_csi_snapshots.npz", **peer_csi)

    infra_export: dict[str, Any] = {
        "fixed_ap_ue_tx_site_ids": np.asarray(fixed_ap_ue["tx_site_ids"], dtype=object),
        "fixed_ap_ue_rx_ue_ids": np.asarray(fixed_ap_ue["rx_ue_ids"], dtype=object),
        "fixed_ap_ue_times_s": fixed_ap_ue["times_s"],
        "fixed_ap_ue_best_sinr_db": fixed_ap_ue["best_sinr_db"],
        "fixed_ap_ue_link_power_w": fixed_ap_ue["link_power_w"],
        "mobile_ap_ue_tx_site_ids": np.asarray(final_ap_ue["tx_site_ids"], dtype=object),
        "mobile_ap_ue_rx_ue_ids": np.asarray(final_ap_ue["rx_ue_ids"], dtype=object),
        "mobile_ap_ue_times_s": final_ap_ue["times_s"],
        "mobile_ap_ue_best_sinr_db": final_ap_ue["best_sinr_db"],
        "mobile_ap_ue_link_power_w": final_ap_ue["link_power_w"],
        "fixed_ap_ap_tx_site_ids": np.asarray(fixed_ap_ap["tx_site_ids"], dtype=object),
        "fixed_ap_ap_rx_site_ids": np.asarray(fixed_ap_ap["rx_site_ids"], dtype=object),
        "fixed_ap_ap_link_power_w": fixed_ap_ap["link_power_w"],
        "mobile_ap_ap_tx_site_ids": np.asarray(final_ap_ap["tx_site_ids"], dtype=object),
        "mobile_ap_ap_rx_site_ids": np.asarray(final_ap_ap["rx_site_ids"], dtype=object),
        "mobile_ap_ap_link_power_w": final_ap_ap["link_power_w"],
    }
    _update_prefixed_export(infra_export, "fixed_ap_ue", fixed_ap_ue, ("cfr", "cir", "tau"))
    _update_prefixed_export(infra_export, "mobile_ap_ue", final_ap_ue, ("cfr", "cir", "tau"))
    _update_prefixed_export(infra_export, "fixed_ap_ap", fixed_ap_ap, ("cfr", "cir", "tau"))
    _update_prefixed_export(infra_export, "mobile_ap_ap", final_ap_ap, ("cfr", "cir", "tau"))
    np.savez_compressed(cache_dir / "infra_csi_snapshots.npz", **infra_export)
    if final_radio_map is not None:
        np.savez_compressed(cache_dir / "coverage_map.npz", **final_radio_map)
    if fixed_radio_map is not None:
        np.savez_compressed(cache_dir / "fixed_coverage_map.npz", **fixed_radio_map)
    manifest = {
        "runtime_info": runtime_info,
        "selected_sites": [_site_to_dict(site) for site in selected_sites],
        "final_selected_candidate_ids": list(final_selected_candidate_ids),
        "selected_candidate_union": sorted(selected_candidate_union),
        "schedule_rows": schedule_rows,
        "fixed_score": _placement_score_to_dict(fixed_score),
        "best_score": _placement_score_to_dict(best_score),
        "num_relocation_windows": num_relocation_windows,
    }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _write_user_sinr_timeseries_csv(
    output_path: Path,
    trajectory: Trajectory,
    fixed_sinr_db: np.ndarray,
    mobile_sinr_db: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_s", "ue_id", "fixed_sinr_db", "mobile_sinr_db"])
        for t_idx, time_s in enumerate(trajectory.times_s):
            for u_idx, ue_id in enumerate(trajectory.ue_ids):
                writer.writerow(
                    [
                        float(time_s),
                        ue_id,
                        float(fixed_sinr_db[t_idx, u_idx]),
                        float(mobile_sinr_db[t_idx, u_idx]),
                    ]
                )


def _write_mobile_ap_schedule_csv(
    output_path: Path,
    schedule_rows: list[dict[str, Any]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["window_index", "start_time_s", "end_time_s", "ap_id", "x_m", "y_m", "z_m", "source"])
        for row in schedule_rows:
            writer.writerow(
                [
                    row["window_index"],
                    row["start_time_s"],
                    row["end_time_s"],
                    row["ap_id"],
                    row["x_m"],
                    row["y_m"],
                    row["z_m"],
                    row["source"],
                ]
            )


def _load_ap_site_pool(config: ScenarioConfig, metadata: dict[str, Any] | None) -> tuple[list[CandidateSite], list[CandidateSite]]:
    if metadata is not None and metadata.get("buildings"):
        candidate_sites = generate_wall_candidate_sites(
            metadata=metadata,
            spacing_m=config.optimization.wall_candidate_spacing_m,
            mount_height_m=config.optimization.wall_mount_height_m,
            corner_clearance_m=config.optimization.wall_corner_clearance_m,
            mount_offset_m=config.optimization.wall_mount_offset_m,
            min_spacing_m=config.optimization.candidate_min_spacing_m,
        )
        if candidate_sites:
            fixed_sites = select_farthest_sites(candidate_sites, config.optimization.num_selected_aps)
            logger.info(
                "Generated %d wall-mounted AP anchors and selected %d far-apart fixed anchors",
                len(candidate_sites),
                len(fixed_sites),
            )
            return fixed_sites, candidate_sites

    base_sites = load_candidate_sites(config.candidate_sites_path)
    return base_sites, list(base_sites)


def _build_static_mobile_schedule(
    trajectory: Trajectory,
    sites: list[CandidateSite],
    relocation_interval_s: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for window_index, indices in enumerate(_window_slices(trajectory.times_s, relocation_interval_s)):
        for site in sites:
            rows.append(
                {
                    "window_index": window_index,
                    "start_time_s": float(trajectory.times_s[indices[0]]),
                    "end_time_s": float(trajectory.times_s[indices[-1]]),
                    "ap_id": site.site_id,
                    "x_m": float(site.x_m),
                    "y_m": float(site.y_m),
                    "z_m": float(site.z_m),
                    "source": site.source,
                }
            )
    return rows

def _per_user_mean_best_sinr(best_sinr_db: np.ndarray) -> np.ndarray:
    values = np.asarray(best_sinr_db, dtype=float)
    if values.ndim == 1:
        return values
    return np.nanmean(values, axis=0)


def _instantaneous_user_sinr_samples(best_sinr_db: np.ndarray) -> np.ndarray:
    return np.asarray(best_sinr_db, dtype=float).reshape(-1)


def _cdf_points(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.asarray(values, dtype=float).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    sorted_values = np.sort(finite)
    probabilities = np.arange(1, sorted_values.size + 1, dtype=float) / sorted_values.size
    return sorted_values, probabilities


def _write_user_sinr_csv(
    output_path: Path,
    ue_ids: list[str],
    static_sinr_db: np.ndarray,
    optimized_sinr_db: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ue_id", "baseline_mean_zf_sinr_db", "relocated_mean_zf_sinr_db"])
        for ue_id, static_value, optimized_value in zip(ue_ids, static_sinr_db, optimized_sinr_db, strict=True):
            writer.writerow([ue_id, float(static_value), float(optimized_value)])


def _plot_user_sinr_cdf(
    static_sinr_db: np.ndarray,
    optimized_sinr_db: np.ndarray,
    output_path: Path,
    static_label: str = "Original APs",
    optimized_label: str = "Relocated APs",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    static_x, static_y = _cdf_points(static_sinr_db)
    optimized_x, optimized_y = _cdf_points(optimized_sinr_db)

    plt.figure(figsize=(8, 6))
    if static_x.size:
        plt.step(static_x, static_y, where="post", linewidth=2.0, linestyle="-", color="#1f77b4", label=static_label)
    if optimized_x.size:
        plt.step(
            optimized_x,
            optimized_y,
            where="post",
            linewidth=2.0,
            linestyle="--",
            color="#ff7f0e",
            label=optimized_label,
        )
    plt.xlabel("Instantaneous distributed-MIMO ZF SINR per user snapshot [dB]")
    plt.ylabel("CDF")
    plt.xlim(left=min(np.min(static_x) if static_x.size else 0.0, np.min(optimized_x) if optimized_x.size else 0.0) - 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _write_user_sinr_artifacts(
    output_dir: Path,
    trajectory: Trajectory,
    fixed_ap_ue: dict[str, Any],
    comparison_ap_ue: dict[str, Any],
    comparison_label: str,
) -> None:
    static_per_user_sinr = _per_user_mean_best_sinr(fixed_ap_ue["best_sinr_db"])
    comparison_per_user_sinr = _per_user_mean_best_sinr(comparison_ap_ue["best_sinr_db"])
    _write_user_sinr_csv(
        output_dir / "user_sinr_summary.csv",
        trajectory.ue_ids,
        static_per_user_sinr,
        comparison_per_user_sinr,
    )
    _write_user_sinr_timeseries_csv(
        output_dir / "user_sinr_timeseries.csv",
        trajectory,
        np.asarray(fixed_ap_ue["best_sinr_db"], dtype=float),
        np.asarray(comparison_ap_ue["best_sinr_db"], dtype=float),
    )
    _plot_user_sinr_cdf(
        _instantaneous_user_sinr_samples(fixed_ap_ue["best_sinr_db"]),
        _instantaneous_user_sinr_samples(comparison_ap_ue["best_sinr_db"]),
        output_dir / "user_sinr_cdf.png",
        static_label="Fixed APs",
        optimized_label=comparison_label,
    )


def build_scene_only(config_or_path: ScenarioConfig | str | Path) -> SceneArtifacts:
    config = load_scenario_config(config_or_path) if not isinstance(config_or_path, ScenarioConfig) else config_or_path
    logger.info("Starting scene build for scenario '%s'", config.name)
    if config.scene.kind == "osm":
        config = replace(config, scene=replace(config.scene, rebuild=True))
        artifacts = OSMSceneBuilder(config.scene).build()
    else:
        artifacts = _resolve_scene_inputs(config)
    logger.info("Scene build complete: %s", artifacts.scene_xml_path)
    return artifacts


def run_scenario(config_or_path: ScenarioConfig | str | Path) -> dict[str, Any]:
    config = load_scenario_config(config_or_path) if not isinstance(config_or_path, ScenarioConfig) else config_or_path
    output_dir = config.outputs.output_dir
    radio_map_enabled = bool(config.coverage.enabled)
    csi_exports_enabled = bool(config.outputs.write_csi_exports)
    csi_cache_enabled = bool(config.outputs.enable_csi_cache)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Starting scenario '%s'", config.name)
    logger.info("Writing outputs to %s", output_dir)

    with progress_bar(total=10, desc=f"Scenario {config.name}", unit="stage", leave=True) as scenario_progress:
        scene_artifacts = _resolve_scene_inputs(config)
        metadata = load_scene_metadata(scene_artifacts.metadata_path)
        if config.solver.enable_ray_tracing and config.solver.require_gpu:
            logger.info("Validating required GPU backend before trajectory generation")
            SionnaRtRunner(
                scene_cfg=config.scene,
                radio=config.radio,
                solver_cfg=config.solver,
                scene_inputs=SceneInputs(
                    scene_path=None if config.scene.kind == "builtin" else scene_artifacts.scene_xml_path,
                    metadata=metadata,
                ),
            ).runtime_info()
        scenario_progress.update(1)

        graph_path = config.mobility.graph_path or scene_artifacts.walk_graph_path
        if graph_path is None:
            raise ValueError("A mobility graph is required. Provide mobility.graph_path or build an OSM scene.")

        logger.info("Loading mobility graph from %s", graph_path)
        graph = load_graph_json(graph_path)
        trajectory = generate_trajectory(graph, config.mobility, config.radio.ue_height_m, metadata=metadata)
        trajectory.write_csv(output_dir / "trajectory.csv")
        logger.info(
            "Generated trajectory with %d users across %d snapshots",
            len(trajectory.ue_ids),
            len(trajectory.times_s),
        )
        scenario_progress.update(1)

        if metadata is not None and metadata.get("buildings"):
            base_sites, candidate_sites = _load_ap_site_pool(config, metadata)
            baseline_sites = list(base_sites)
            logger.info(
                "Using %d fixed wall-mounted APs and %d wall anchor candidates",
                len(baseline_sites),
                len(candidate_sites),
            )
        else:
            base_sites = load_candidate_sites(config.candidate_sites_path)
            candidate_sites = augment_with_trajectory_sites(
                base_sites=base_sites,
                trajectory=trajectory,
                min_spacing_m=config.optimization.candidate_min_spacing_m,
                max_new_sites=config.optimization.max_candidate_ue_positions,
            )
            logger.info(
                "Loaded %d base candidate sites and expanded to %d candidates with UE-derived positions",
                len(base_sites),
                len(candidate_sites),
            )
            baseline_sites = _select_static_baseline_sites(config, base_sites)
        if not baseline_sites:
            raise ValueError("At least one enabled fixed AP site is required for fixed/mobile comparison")
        logger.info("Using fixed AP sites: %s", ", ".join(site.site_id for site in baseline_sites))
        scenario_progress.update(1)

        if not config.solver.enable_ray_tracing:
            logger.info("Ray tracing disabled by configuration; generating trajectory/layout outputs only")
            _remove_artifacts(
                [
                    output_dir / "coverage_map.npz",
                    output_dir / "coverage_map.png",
                    output_dir / "fixed_coverage_map.npz",
                    output_dir / "fixed_coverage_map.png",
                    output_dir / "peer_csi_snapshots.npz",
                    output_dir / "infra_csi_snapshots.npz",
                ]
            )
            selected_sites = list(baseline_sites)
            write_candidate_sites(
                output_dir / "recommended_aps.csv",
                candidate_sites,
                selected_ids={site.site_id for site in selected_sites},
            )
            write_candidate_sites(
                output_dir / "fixed_aps.csv",
                baseline_sites,
                selected_ids={site.site_id for site in baseline_sites},
            )
            _write_ap_relocation_csv(output_dir / "ap_relocation_summary.csv", baseline_sites, selected_sites)
            _write_mobile_ap_schedule_csv(
                output_dir / "mobile_ap_schedule.csv",
                _build_static_mobile_schedule(
                    trajectory,
                    selected_sites,
                    config.optimization.relocation_interval_s,
                ),
            )
            _plot_scene_layout(metadata, graph, base_sites, selected_sites, trajectory, output_dir / "scene_layout.png")
            animation_path = _animate_scene(
                metadata,
                graph,
                base_sites,
                selected_sites,
                trajectory,
                output_dir / "scene_animation.mp4",
                speedup=config.outputs.scene_animation_speedup,
            )
            _plot_colored_trajectories(trajectory, output_dir / "trajectory_colormap.png")
            if animation_path is not None:
                logger.info("Wrote scene animation to %s", animation_path)
            logger.info("Wrote non-ray-traced trajectory and layout artifacts to %s", output_dir)
            scenario_progress.update(7)

            summary = {
                "scenario": config.name,
                "ray_tracing_enabled": False,
                "radio_map_enabled": radio_map_enabled,
                "csi_exports_enabled": csi_exports_enabled,
                "csi_cache_enabled": csi_cache_enabled,
                "optimization_enabled": config.optimization.enable_optimization,
                "compute_device": "SKIPPED",
                "mitsuba_variant": "",
                "relocation_interval_s": config.optimization.relocation_interval_s,
                "fixed_site_ids": [site.site_id for site in baseline_sites],
                "mobile_site_ids": [site.site_id for site in selected_sites],
                "final_mobile_candidate_ids": [site.site_id for site in selected_sites],
                "status": "trajectory_only",
            }
            (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            return summary

        cache_key = _build_csi_cache_key(
            config,
            scene_artifacts,
            Path(graph_path),
            trajectory,
            base_sites,
            candidate_sites,
            baseline_sites,
        )
        cache_hit = _try_load_csi_cache(output_dir, cache_key) if csi_cache_enabled else None
        sinr_outputs_written = False

        early_scene_render_path = None
        early_scene_camera_video_path = None

        if cache_hit is not None:
            runtime_info = cache_hit["runtime_info"]
            logger.info(
                "Reusing cached compute payload from backend %s via Mitsuba variant %s",
                runtime_info["device"],
                runtime_info["variant"],
            )
            logger.info("Loaded cached CSI payload for hash %s", cache_key[:12])
            peer_csi = cache_hit["peer_csi"]
            fixed_ap_ue = cache_hit["fixed_ap_ue"]
            fixed_ap_ap = cache_hit["fixed_ap_ap"]
            final_ap_ue = cache_hit["final_ap_ue"]
            final_ap_ap = cache_hit["final_ap_ap"]
            fixed_radio_map = cache_hit["fixed_radio_map"] if radio_map_enabled else None
            final_radio_map = cache_hit["final_radio_map"] if radio_map_enabled else None
            selected_sites = cache_hit["selected_sites"]
            final_selected_candidate_ids = cache_hit["final_selected_candidate_ids"]
            selected_candidate_union = cache_hit["selected_candidate_union"]
            schedule_rows = cache_hit["schedule_rows"]
            fixed_score = cache_hit["fixed_score"]
            best_score = cache_hit["best_score"]
            relocation_windows = [None] * cache_hit["num_relocation_windows"]
            scenario_progress.update(3)
        else:
            runner = SionnaRtRunner(
                scene_cfg=config.scene,
                radio=config.radio,
                solver_cfg=config.solver,
                scene_inputs=SceneInputs(
                    scene_path=None if config.scene.kind == "builtin" else scene_artifacts.scene_xml_path,
                    metadata=metadata,
                ),
            )
            runtime_info = runner.runtime_info()
            logger.info(
                "Compute backend: %s via Mitsuba variant %s",
                runtime_info["device"],
                runtime_info["variant"],
            )
            scenario_progress.update(1)
            if _should_render_sionna_scene_artifacts(runtime_info):
                # Render a baseline scene view/video before the heavy CSI/radio-map phases so
                # the Sionna camera artifacts still exist if the backend dies later.
                early_scene_render_path = _render_scene_view(
                    runner,
                    metadata,
                    graph,
                    baseline_sites,
                    trajectory,
                    output_dir / "scene_render.png",
                )
                early_scene_camera_video_path = _render_scene_video(
                    runner,
                    metadata,
                    graph,
                    baseline_sites,
                    trajectory,
                    output_dir / "scene_camera.mp4",
                )
                if early_scene_render_path is not None:
                    logger.info("Wrote early scene render to %s", early_scene_render_path)
                if early_scene_camera_video_path is not None:
                    logger.info("Wrote early scene camera video to %s", early_scene_camera_video_path)
            else:
                logger.info("Skipping early Sionna scene render/video on CPU LLVM backend to avoid Dr.Jit instability")
            logger.info("Computing UE-UE CSI snapshots")
            peer_csi = runner.compute_ue_ue_csi(trajectory, export_full=csi_exports_enabled)
            if csi_exports_enabled:
                np.savez_compressed(output_dir / "peer_csi_snapshots.npz", **peer_csi)
                logger.info("Wrote peer CSI export to %s", output_dir / "peer_csi_snapshots.npz")
            scenario_progress.update(1)

            logger.info("Computing fixed-constellation AP-UE and AP-AP CSI")
            fixed_ap_ue = runner.compute_ap_ue_csi(baseline_sites, trajectory, export_full=csi_exports_enabled)
            fixed_ap_ap = runner.compute_ap_ap_csi(baseline_sites, export_full=csi_exports_enabled)
            if not config.optimization.enable_optimization:
                _write_user_sinr_artifacts(
                    output_dir,
                    trajectory,
                    fixed_ap_ue,
                    fixed_ap_ue,
                    comparison_label="Fixed APs (same deployment)",
                )
                logger.info("Wrote fixed-AP user SINR artifacts before optimization/radio-map stages")
                sinr_outputs_written = True
            fixed_score = summarize_candidate_set(
                grid_best_sinr_db=np.asarray([], dtype=float),
                trajectory_best_sinr_db=fixed_ap_ue["best_sinr_db"],
                peer_need_weights=peer_csi["need_weights"],
                cfg=config.optimization,
            )
            scenario_progress.update(1)

            if not config.optimization.enable_optimization:
                logger.info("AP optimization disabled by configuration; reusing fixed AP constellation")
                relocation_windows = _window_slices(trajectory.times_s, config.optimization.relocation_interval_s)
                selected_sites = list(baseline_sites)
                final_selected_candidate_ids = [site.site_id for site in baseline_sites]
                selected_candidate_union = set(final_selected_candidate_ids)
                schedule_rows = _build_static_mobile_schedule(
                    trajectory,
                    selected_sites,
                    config.optimization.relocation_interval_s,
                )
                final_ap_ue = fixed_ap_ue
                final_ap_ap = fixed_ap_ap
                best_score = fixed_score
            else:
                candidate_index = {site.site_id: site for site in candidate_sites if site.enabled}
                mobile_window_segments: list[dict[str, Any]] = []
                selected_candidate_union = set()
                schedule_rows = []
                relocation_windows = _window_slices(trajectory.times_s, config.optimization.relocation_interval_s)
                mobile_reference_sites = [
                    CandidateSite(
                        site_id=f"mobile_ap_{index + 1:02d}",
                        x_m=site.x_m,
                        y_m=site.y_m,
                        z_m=site.z_m,
                        yaw_deg=site.yaw_deg,
                        pitch_deg=site.pitch_deg,
                        mount_type=site.mount_type,
                        enabled=True,
                        source=f"seed:{site.site_id}",
                    )
                    for index, site in enumerate(baseline_sites)
                ]
                mobile_sites = list(mobile_reference_sites)
                final_selected_candidate_ids = []
                final_ap_ap = None

                logger.info(
                    "Optimizing mobile AP constellation over %d relocation windows with %.2f second spacing",
                    len(relocation_windows),
                    config.optimization.relocation_interval_s,
                )
                for window_index, indices in enumerate(relocation_windows):
                    window_trajectory = _slice_trajectory(trajectory, indices)
                    window_need_weights = np.asarray(peer_csi["need_weights"][indices], dtype=float)
                    evaluation_cache: dict[tuple[str, ...], dict[str, Any]] = {}

                    def evaluate_window(subset: tuple[str, ...]) -> PlacementScore:
                        if subset not in evaluation_cache:
                            selected = [candidate_index[site_id] for site_id in subset]
                            ap_ue = runner.compute_ap_ue_csi(selected, window_trajectory, export_full=False)
                            score = summarize_candidate_set(
                                grid_best_sinr_db=np.asarray([], dtype=float),
                                trajectory_best_sinr_db=ap_ue["best_sinr_db"],
                                peer_need_weights=window_need_weights,
                                cfg=config.optimization,
                            )
                            evaluation_cache[subset] = {"score": score, "ap_ue": ap_ue}
                        return evaluation_cache[subset]["score"]

                    selected_candidate_ids, _window_score = greedy_one_swap(
                        candidate_ids=sorted(candidate_index),
                        select_count=len(mobile_reference_sites),
                        evaluator=evaluate_window,
                    )
                    selected_candidates = [candidate_index[site_id] for site_id in selected_candidate_ids]
                    mobile_sites = _relocate_sites(mobile_sites, selected_candidates)
                    mobile_segment = runner.compute_ap_ue_csi(
                        mobile_sites,
                        window_trajectory,
                        export_full=csi_exports_enabled,
                    )
                    final_ap_ap = runner.compute_ap_ap_csi(mobile_sites, export_full=csi_exports_enabled)
                    final_selected_candidate_ids = selected_candidate_ids
                    mobile_window_segments.append(mobile_segment)
                    selected_candidate_union.update(selected_candidate_ids)
                    for site in mobile_sites:
                        schedule_rows.append(
                            {
                                "window_index": window_index,
                                "start_time_s": float(window_trajectory.times_s[0]),
                                "end_time_s": float(window_trajectory.times_s[-1]),
                                "ap_id": site.site_id,
                                "x_m": float(site.x_m),
                                "y_m": float(site.y_m),
                                "z_m": float(site.z_m),
                                "source": site.source,
                            }
                        )

                if not mobile_window_segments or final_ap_ap is None:
                    raise RuntimeError("Mobile AP optimization produced no relocation windows")

                selected_sites = list(mobile_sites)
                final_ap_ue = _concat_ap_ue_segments(mobile_window_segments)
                best_score = summarize_candidate_set(
                    grid_best_sinr_db=np.asarray([], dtype=float),
                    trajectory_best_sinr_db=final_ap_ue["best_sinr_db"],
                    peer_need_weights=peer_csi["need_weights"],
                    cfg=config.optimization,
                )
            fixed_radio_map = None
            final_radio_map = None
            if radio_map_enabled:
                logger.info("Computing fixed coverage map")
                fixed_radio_map = runner.compute_radio_map(baseline_sites, config.coverage)
                if config.optimization.enable_optimization:
                    logger.info("Computing final coverage map")
                    final_radio_map = runner.compute_radio_map(selected_sites, config.coverage)
                else:
                    final_radio_map = fixed_radio_map
            if csi_cache_enabled:
                _write_csi_cache(
                    output_dir=output_dir,
                    cache_key=cache_key,
                    runtime_info=runtime_info,
                    peer_csi=peer_csi,
                    fixed_ap_ue=fixed_ap_ue,
                    final_ap_ue=final_ap_ue,
                    fixed_ap_ap=fixed_ap_ap,
                    final_ap_ap=final_ap_ap,
                    fixed_radio_map=fixed_radio_map,
                    final_radio_map=final_radio_map,
                    selected_sites=selected_sites,
                    final_selected_candidate_ids=final_selected_candidate_ids,
                    selected_candidate_union=selected_candidate_union,
                    schedule_rows=schedule_rows,
                    fixed_score=fixed_score,
                    best_score=best_score,
                    num_relocation_windows=len(relocation_windows),
                )
                logger.info("Stored CSI cache under %s", _cache_dir(output_dir, cache_key))
        scenario_progress.update(1)

        if csi_exports_enabled:
            np.savez_compressed(output_dir / "peer_csi_snapshots.npz", **peer_csi)
            logger.info("Wrote peer CSI export to %s", output_dir / "peer_csi_snapshots.npz")
        else:
            _remove_artifacts([output_dir / "peer_csi_snapshots.npz"])

        infra_export: dict[str, Any] = {
            "fixed_ap_ue_tx_site_ids": np.asarray(fixed_ap_ue["tx_site_ids"], dtype=object),
            "fixed_ap_ue_rx_ue_ids": np.asarray(fixed_ap_ue["rx_ue_ids"], dtype=object),
            "fixed_ap_ue_times_s": fixed_ap_ue["times_s"],
            "fixed_ap_ue_best_sinr_db": fixed_ap_ue["best_sinr_db"],
            "fixed_ap_ue_link_power_w": fixed_ap_ue["link_power_w"],
            "mobile_ap_ue_tx_site_ids": np.asarray(final_ap_ue["tx_site_ids"], dtype=object),
            "mobile_ap_ue_rx_ue_ids": np.asarray(final_ap_ue["rx_ue_ids"], dtype=object),
            "mobile_ap_ue_times_s": final_ap_ue["times_s"],
            "mobile_ap_ue_best_sinr_db": final_ap_ue["best_sinr_db"],
            "mobile_ap_ue_link_power_w": final_ap_ue["link_power_w"],
            "fixed_ap_ap_tx_site_ids": np.asarray(fixed_ap_ap["tx_site_ids"], dtype=object),
            "fixed_ap_ap_rx_site_ids": np.asarray(fixed_ap_ap["rx_site_ids"], dtype=object),
            "fixed_ap_ap_link_power_w": fixed_ap_ap["link_power_w"],
            "mobile_ap_ap_tx_site_ids": np.asarray(final_ap_ap["tx_site_ids"], dtype=object),
            "mobile_ap_ap_rx_site_ids": np.asarray(final_ap_ap["rx_site_ids"], dtype=object),
            "mobile_ap_ap_link_power_w": final_ap_ap["link_power_w"],
        }
        _update_prefixed_export(infra_export, "fixed_ap_ue", fixed_ap_ue, ("cfr", "cir", "tau"))
        _update_prefixed_export(infra_export, "mobile_ap_ue", final_ap_ue, ("cfr", "cir", "tau"))
        _update_prefixed_export(infra_export, "fixed_ap_ap", fixed_ap_ap, ("cfr", "cir", "tau"))
        _update_prefixed_export(infra_export, "mobile_ap_ap", final_ap_ap, ("cfr", "cir", "tau"))
        if csi_exports_enabled:
            np.savez_compressed(output_dir / "infra_csi_snapshots.npz", **infra_export)
            logger.info("Wrote infrastructure CSI export to %s", output_dir / "infra_csi_snapshots.npz")
        else:
            _remove_artifacts([output_dir / "infra_csi_snapshots.npz"])

        if fixed_radio_map is not None and final_radio_map is not None:
            np.savez_compressed(
                output_dir / "coverage_map.npz",
                path_gain=final_radio_map["path_gain"],
                rss=final_radio_map["rss"],
                sinr=final_radio_map["sinr"],
                best_sinr_db=final_radio_map["best_sinr_db"],
                cell_centers=final_radio_map["cell_centers"],
            )
            np.savez_compressed(
                output_dir / "fixed_coverage_map.npz",
                path_gain=fixed_radio_map["path_gain"],
                rss=fixed_radio_map["rss"],
                sinr=fixed_radio_map["sinr"],
                best_sinr_db=fixed_radio_map["best_sinr_db"],
                cell_centers=fixed_radio_map["cell_centers"],
            )
            _plot_coverage(
                fixed_radio_map["best_sinr_db"],
                fixed_radio_map["cell_centers"],
                baseline_sites,
                trajectory,
                output_dir / "fixed_coverage_map.png",
            )
            _plot_coverage(
                final_radio_map["best_sinr_db"],
                final_radio_map["cell_centers"],
                selected_sites,
                trajectory,
                output_dir / "coverage_map.png",
            )
            logger.info("Wrote coverage-map exports to %s", output_dir)
        else:
            _remove_artifacts(
                [
                    output_dir / "coverage_map.npz",
                    output_dir / "coverage_map.png",
                    output_dir / "fixed_coverage_map.npz",
                    output_dir / "fixed_coverage_map.png",
                ]
            )
            logger.info("Coverage-map computation disabled; skipping coverage-map exports")
        scene_render_path = early_scene_render_path
        scene_camera_video_path = early_scene_camera_video_path
        if cache_hit is None:
            if _should_render_sionna_scene_artifacts(runtime_info):
                final_scene_render_path = _render_scene_view(
                    runner,
                    metadata,
                    graph,
                    selected_sites,
                    trajectory,
                    output_dir / "scene_render.png",
                )
                final_scene_camera_video_path = _render_scene_video(
                    runner,
                    metadata,
                    graph,
                    selected_sites,
                    trajectory,
                    output_dir / "scene_camera.mp4",
                )
                if final_scene_render_path is not None:
                    scene_render_path = final_scene_render_path
                if final_scene_camera_video_path is not None:
                    scene_camera_video_path = final_scene_camera_video_path
                if scene_render_path is not None:
                    _cache_optional_artifact(output_dir, cache_key, scene_render_path, "scene_render.png")
                if scene_camera_video_path is not None:
                    _cache_optional_artifact(output_dir, cache_key, scene_camera_video_path, "scene_camera.mp4")
            else:
                logger.info("Skipping final Sionna scene render/video on CPU LLVM backend to avoid Dr.Jit instability")
        else:
            scene_render_path = _restore_cached_artifact(
                output_dir,
                cache_key,
                "scene_render.png",
                output_dir / "scene_render.png",
            )
            scene_camera_video_path = _restore_cached_artifact(
                output_dir,
                cache_key,
                "scene_camera.mp4",
                output_dir / "scene_camera.mp4",
            )
            logger.info("Skipping fresh scene render/video generation on CSI cache hit to avoid Dr.Jit re-entry")
        _plot_scene_layout(metadata, graph, base_sites, selected_sites, trajectory, output_dir / "scene_layout.png")
        animation_path = _animate_scene(
            metadata,
            graph,
            base_sites,
            selected_sites,
            trajectory,
            output_dir / "scene_animation.mp4",
            speedup=config.outputs.scene_animation_speedup,
        )
        _plot_colored_trajectories(trajectory, output_dir / "trajectory_colormap.png")
        write_candidate_sites(output_dir / "recommended_aps.csv", candidate_sites, selected_ids=selected_candidate_union)
        write_candidate_sites(output_dir / "fixed_aps.csv", baseline_sites, selected_ids={site.site_id for site in baseline_sites})
        _write_ap_relocation_csv(output_dir / "ap_relocation_summary.csv", baseline_sites, selected_sites)
        _write_mobile_ap_schedule_csv(output_dir / "mobile_ap_schedule.csv", schedule_rows)
        if scene_render_path is not None:
            logger.info("Wrote scene render to %s", scene_render_path)
        if scene_camera_video_path is not None:
            logger.info("Wrote scene camera video to %s", scene_camera_video_path)
        if animation_path is not None:
            logger.info("Wrote scene animation to %s", animation_path)
        logger.info("Wrote scene and recommendation artifacts to %s", output_dir)
        scenario_progress.update(1)

        if fixed_ap_ue is not None and not sinr_outputs_written:
            comparison_label = "Relocated APs" if config.optimization.enable_optimization else "Fixed APs (same deployment)"
            _write_user_sinr_artifacts(
                output_dir,
                trajectory,
                fixed_ap_ue,
                final_ap_ue,
                comparison_label=comparison_label,
            )
            logger.info("Wrote user SINR comparison artifacts to %s", output_dir)
        scenario_progress.update(1)

        summary = {
            "scenario": config.name,
            "compute_device": runtime_info["device"],
            "mitsuba_variant": runtime_info["variant"],
            "csi_cache_key": cache_key,
            "csi_cache_hit": cache_hit is not None,
            "radio_map_enabled": radio_map_enabled,
            "csi_exports_enabled": csi_exports_enabled,
            "csi_cache_enabled": csi_cache_enabled,
            "optimization_enabled": config.optimization.enable_optimization,
            "relocation_interval_s": config.optimization.relocation_interval_s,
            "fixed_site_ids": [site.site_id for site in baseline_sites],
            "mobile_site_ids": [site.site_id for site in selected_sites],
            "final_mobile_candidate_ids": final_selected_candidate_ids,
            "fixed_score": fixed_score.score,
            "fixed_outage": fixed_score.outage,
            "fixed_percentile_10_db": fixed_score.percentile_10_db,
            "score": best_score.score,
            "score_gain": best_score.score - fixed_score.score,
            "outage": best_score.outage,
            "outage_delta": best_score.outage - fixed_score.outage,
            "percentile_10_db": best_score.percentile_10_db,
            "percentile_10_db_gain": best_score.percentile_10_db - fixed_score.percentile_10_db,
            "peer_tiebreak": best_score.peer_tiebreak,
            "num_relocation_windows": len(relocation_windows),
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info("Scenario '%s' complete with score %.3f", config.name, best_score.score)
        return summary
