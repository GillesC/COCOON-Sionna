"""Top-level orchestration for scene build, CSI extraction, and optimization."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass, field, replace
import hashlib
import itertools
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
from scipy.constants import Boltzmann
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Point, Polygon

from .config import ScenarioConfig, load_scenario_config
from .logging_utils import progress_bar
from .mobility import Trajectory, generate_trajectory, load_graph_json
from .optimization import (
    PlacementScore,
    sample_random_candidates,
    select_local_csi_candidates,
    summarize_candidate_set,
)
from .scene_builder import OSMSceneBuilder, SceneArtifacts
from .sionna_rt_adapter import (
    SceneInputs,
    SionnaRtRunner,
    _zf_sinr_terms_from_mimo_channel,
    load_scene_metadata,
)
from .sites import CandidateSite
from .sites import (
    generate_wall_candidate_sites,
    load_candidate_sites,
    select_farthest_sites,
    write_candidate_sites,
)

logger = logging.getLogger(__name__)

ALL_STRATEGY_NAMES = (
    "central_massive_mimo",
    "distributed_fixed",
    "distributed_movable",
    "distributed_movable_optimization_2",
    "distributed_movable_optimization_3",
)
LEGACY_STRATEGY_NAMES = ("random_baseline", "local_csi_p10", "capped_exact_search")
CENTRAL_AP_SITE_ID = "central_ap_01"
CENTRAL_AP_ROOF_OFFSET_M = 1.5
CENTRAL_AP_DOWNTILT_DEG = -10.0
VISUALIZATION_ARTIFACT_NAMES = (
    "coverage_map.png",
    "fixed_coverage_map.png",
    "scene_render.png",
    "scene_camera.mp4",
    "scene_layout.png",
    "scene_animation.mp4",
    "scene_animation.gif",
    "scene_animation_with_central_massive_mimo.mp4",
    "scene_animation_with_central_massive_mimo.gif",
    "trajectory_colormap.png",
    "user_sinr_cdf.png",
)


@dataclass(slots=True)
class StrategyArtifacts:
    name: str
    selected_sites: list[CandidateSite]
    movable_sites: list[CandidateSite]
    ap_ue: dict[str, Any]
    ap_ap: dict[str, Any]
    score: PlacementScore
    schedule_rows: list[dict[str, Any]]
    final_candidate_ids: list[str]
    selected_candidate_union: set[str]
    capped: bool = False
    evaluated_combinations: int = 0
    details: dict[str, Any] = field(default_factory=dict)


def _active_strategy_names(config: ScenarioConfig) -> tuple[str, ...]:
    names = ["central_massive_mimo", "distributed_fixed"]
    if config.placement.enable_optimization_1:
        names.append("distributed_movable")
    if config.placement.enable_optimization_2:
        names.append("distributed_movable_optimization_2")
    if config.placement.enable_optimization_3:
        names.append("distributed_movable_optimization_3")
    return tuple(names)


def _validate_three_mode_config(config: ScenarioConfig) -> None:
    if int(config.placement.num_fixed_aps) != 0:
        raise ValueError(
            "Comparison requires placement.num_fixed_aps == 0 because "
            "the distributed baseline is the initial full AP constellation."
        )


def _strategy_linestyle(strategy_name: str) -> str:
    styles = {
        "central_massive_mimo": "-.",
        "distributed_fixed": "-",
        "distributed_movable": "--",
        "distributed_movable_optimization_2": "-",
        "distributed_movable_optimization_3": ":",
    }
    return styles.get(strategy_name, "-")


def _strategy_site_csv_name(strategy_name: str) -> str:
    if strategy_name == "central_massive_mimo":
        return "central_massive_mimo_ap.csv"
    return f"{strategy_name}_aps.csv"


def _legacy_output_paths(output_dir: Path) -> list[Path]:
    paths = [output_dir / "fixed_aps.csv"]
    for name in LEGACY_STRATEGY_NAMES:
        paths.append(output_dir / f"{name}_movable_aps.csv")
        paths.append(output_dir / f"{name}_schedule.csv")
    return paths


def _strategy_output_paths(output_dir: Path) -> list[Path]:
    paths: list[Path] = [output_dir / "strategy_comparison.csv"]
    for name in ALL_STRATEGY_NAMES:
        paths.append(output_dir / _strategy_site_csv_name(name))
        paths.append(output_dir / f"{name}_schedule.csv")
    return paths


def _factor_central_ap_array(total_elements: int) -> tuple[int, int]:
    if total_elements <= 0:
        raise ValueError("Central AP antenna budget must be positive")
    rows = int(np.floor(np.sqrt(float(total_elements))))
    while rows > 1 and total_elements % rows != 0:
        rows -= 1
    cols = total_elements // rows
    return rows, cols


def _central_ap_radio(base_radio, distributed_ap_count: int):
    if distributed_ap_count <= 0:
        raise ValueError("distributed_ap_count must be positive for the central AP radio budget")
    # This normalizes the rooftop proxy against the distributed AP budget; it does
    # not model a full placement optimization over a true co-located massive-MIMO array.
    total_elements = distributed_ap_count * int(base_radio.ap_num_rows) * int(base_radio.ap_num_cols)
    rows, cols = _factor_central_ap_array(total_elements)
    total_power_dbm = float(base_radio.tx_power_dbm_ap + 10.0 * np.log10(float(distributed_ap_count)))
    return replace(base_radio, ap_num_rows=rows, ap_num_cols=cols, tx_power_dbm_ap=total_power_dbm)


def _make_central_ap_site(candidate: CandidateSite) -> CandidateSite:
    return CandidateSite(
        site_id=CENTRAL_AP_SITE_ID,
        x_m=candidate.x_m,
        y_m=candidate.y_m,
        z_m=candidate.z_m,
        yaw_deg=candidate.yaw_deg,
        pitch_deg=candidate.pitch_deg,
        mount_type=candidate.mount_type,
        enabled=True,
        source=f"selected:{candidate.site_id}",
    )


def _area_center_xy(metadata: dict[str, Any] | None) -> np.ndarray | None:
    if metadata is None:
        return None
    if "boundary_local" in metadata:
        boundary = np.asarray(metadata["boundary_local"], dtype=float)
        if boundary.ndim == 2 and boundary.shape[1] >= 2 and boundary.shape[0] > 0:
            polygon = Polygon(boundary[:, :2])
            if not polygon.is_empty:
                centroid = polygon.centroid
                return np.asarray([float(centroid.x), float(centroid.y)], dtype=float)
    buildings = metadata.get("buildings", [])
    if not isinstance(buildings, list) or not buildings:
        return None
    points: list[list[float]] = []
    for building in buildings:
        polygon_coords = np.asarray(building.get("polygon_local", []), dtype=float)
        if polygon_coords.ndim == 2 and polygon_coords.shape[1] == 2 and polygon_coords.shape[0] > 0:
            points.extend(polygon_coords[:, :2].tolist())
    if not points:
        return None
    return np.mean(np.asarray(points, dtype=float), axis=0)


def _generate_rooftop_candidates(metadata: dict[str, Any] | None) -> list[CandidateSite]:
    if metadata is None:
        return []
    buildings = metadata.get("buildings", [])
    if not isinstance(buildings, list):
        return []

    centroid_xy = _area_center_xy(metadata)
    indexed_candidates: list[tuple[float, CandidateSite]] = []
    for index, building in enumerate(buildings):
        polygon_coords = np.asarray(building.get("polygon_local", []), dtype=float)
        if polygon_coords.ndim != 2 or polygon_coords.shape[0] < 4 or polygon_coords.shape[1] != 2:
            continue
        polygon = Polygon(polygon_coords)
        if polygon.is_empty:
            continue
        representative = polygon.representative_point()
        building_name = str(building.get("name", f"building_{index:03d}"))
        target_xy = centroid_xy if centroid_xy is not None else np.asarray([representative.x, representative.y], dtype=float)
        delta = target_xy - np.asarray([representative.x, representative.y], dtype=float)
        yaw_deg = float(np.degrees(np.arctan2(delta[1], delta[0]))) if np.linalg.norm(delta) > 1e-9 else 0.0
        candidate = CandidateSite(
            site_id=f"roof_{building_name}",
            x_m=float(representative.x),
            y_m=float(representative.y),
            z_m=float(building.get("height_m", 0.0)) + CENTRAL_AP_ROOF_OFFSET_M,
            yaw_deg=yaw_deg,
            pitch_deg=CENTRAL_AP_DOWNTILT_DEG,
            mount_type="rooftop",
            enabled=True,
            source="roof_metadata:center_building",
        )
        distance = float(np.hypot(candidate.x_m - target_xy[0], candidate.y_m - target_xy[1]))
        indexed_candidates.append((distance, candidate))

    if not indexed_candidates:
        return []
    indexed_candidates.sort(key=lambda item: item[0])
    return [indexed_candidates[0][1]]


def _select_centroid_rooftop_candidate(
    metadata: dict[str, Any] | None,
    rooftop_candidates: list[CandidateSite],
) -> CandidateSite:
    if not rooftop_candidates:
        raise ValueError("At least one rooftop candidate is required for the central AP comparison")
    centroid_xy = _area_center_xy(metadata)
    if centroid_xy is None:
        return rooftop_candidates[0]
    return min(rooftop_candidates, key=lambda site: float(np.hypot(site.x_m - centroid_xy[0], site.y_m - centroid_xy[1])))


def _resolve_scene_inputs(config: ScenarioConfig) -> SceneArtifacts:
    if config.scene.kind == "builtin":
        logger.info("Using builtin Sionna scene '%s'", config.scene.sionna_scene)
        return SceneArtifacts(scene_xml_path=Path("builtin"), metadata_path=None, walk_graph_path=config.mobility.graph_path)
    if config.scene.kind not in {"osm", "xml"}:
        raise ValueError(f"Unsupported scene kind: {config.scene.kind}")
    if config.scene.kind == "xml":
        if config.scene.scene_xml_path is None:
            raise ValueError("scene.scene_xml_path is required for xml scenes")
        if not config.scene.scene_xml_path.exists():
            raise FileNotFoundError(f"Scene XML is missing: {config.scene.scene_xml_path}")
        logger.info("Using existing scene XML at %s", config.scene.scene_xml_path)
        metadata = config.scene.scene_output_dir / "scene_metadata.json" if config.scene.scene_output_dir else None
        walk_graph = config.mobility.graph_path
        return SceneArtifacts(scene_xml_path=config.scene.scene_xml_path, metadata_path=metadata, walk_graph_path=walk_graph)

    output_dir = config.scene.scene_output_dir
    if output_dir is None:
        raise ValueError("scene.scene_output_dir is required for OSM scenes")
    scene_xml = output_dir / "scene.xml"
    metadata = output_dir / "scene_metadata.json"
    walk_graph = output_dir / "walk_graph.json"
    if not scene_xml.exists():
        raise FileNotFoundError(
            f"Scene assets are missing in {output_dir}. Run `cocoon-sionna build-scene {config.scenario_path}` first."
        )
    logger.info("Reusing existing OSM scene assets from %s", output_dir)
    return SceneArtifacts(scene_xml_path=scene_xml, metadata_path=metadata, walk_graph_path=walk_graph)


def _visualization_artifact_paths(output_dir: Path) -> list[Path]:
    return [output_dir / name for name in VISUALIZATION_ARTIFACT_NAMES]


def _copy_optional_artifact(source: Path | None, destination: Path) -> Path | None:
    if source is None or not source.exists():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return destination


def _store_scene_context(output_dir: Path, scene_artifacts: SceneArtifacts) -> dict[str, str]:
    metadata_copy = _copy_optional_artifact(scene_artifacts.metadata_path, output_dir / "scene_metadata.json")
    graph_copy = _copy_optional_artifact(scene_artifacts.walk_graph_path, output_dir / "walk_graph.json")
    return {
        "scene_xml_path": str(scene_artifacts.scene_xml_path) if scene_artifacts.scene_xml_path != Path("builtin") else "",
        "scene_metadata_path": str(metadata_copy) if metadata_copy is not None else "",
        "walk_graph_path": str(graph_copy) if graph_copy is not None else "",
    }


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
    reference_sites: list[CandidateSite] | None = None,
    reference_label: str = "Central massive-MIMO BS",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    _draw_scene_background(ax, metadata, graph)
    display_sites = [*base_sites, *(reference_sites or [])]

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
    if reference_sites:
        ax.scatter(
            [site.x_m for site in reference_sites],
            [site.y_m for site in reference_sites],
            c="#d4a017",
            s=140,
            marker="*",
            edgecolors="black",
            linewidths=0.5,
            label=reference_label,
            zorder=6,
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

    _set_scene_axes(ax, metadata, graph, positions=positions, sites=display_sites)
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


def _group_schedule_rows(schedule_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in schedule_rows:
        grouped.setdefault(int(row["window_index"]), []).append(row)
    windows: list[dict[str, Any]] = []
    for window_index in sorted(grouped):
        rows = grouped[window_index]
        windows.append(
            {
                "window_index": window_index,
                "start_time_s": float(rows[0]["start_time_s"]),
                "end_time_s": float(rows[0]["end_time_s"]),
                "sites": [
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
                    for row in rows
                ],
            }
        )
    return windows


def _schedule_window_for_time(schedule_windows: list[dict[str, Any]], time_s: float) -> dict[str, Any]:
    if not schedule_windows:
        raise ValueError("schedule_windows must not be empty")
    for window in schedule_windows:
        if window["start_time_s"] - 1e-9 <= time_s <= window["end_time_s"] + 1e-9:
            return window
    return schedule_windows[-1]


def _schedule_positions_array(schedule_windows: list[dict[str, Any]]) -> np.ndarray:
    points: list[list[float]] = []
    for window in schedule_windows:
        for site in window["sites"]:
            points.append([float(site.x_m), float(site.y_m)])
    if not points:
        return np.zeros((0, 2), dtype=float)
    return np.asarray(points, dtype=float)


def _schedule_has_movement(schedule_rows: list[dict[str, Any]]) -> bool:
    by_ap: dict[str, set[tuple[float, float, float]]] = {}
    for row in schedule_rows:
        by_ap.setdefault(str(row["ap_id"]), set()).add(
            (float(row["x_m"]), float(row["y_m"]), float(row["z_m"]))
        )
    return any(len(positions) > 1 for positions in by_ap.values())


def _scene_animation_strategy_name(
    strategy_results: dict[str, StrategyArtifacts],
    preferred_name: str,
) -> str:
    preferred = strategy_results[preferred_name]
    if _schedule_has_movement(preferred.schedule_rows):
        return preferred_name

    moving_names = [
        name
        for name in ALL_STRATEGY_NAMES
        if name in strategy_results
        and name != preferred_name
        and _schedule_has_movement(strategy_results[name].schedule_rows)
    ]
    if not moving_names:
        return preferred_name
    return max(moving_names, key=lambda name: strategy_results[name].score.score)


def _animate_scene(
    metadata: dict[str, Any] | None,
    graph,
    base_sites: list[CandidateSite],
    selected_sites: list[CandidateSite],
    trajectory,
    output_path: Path,
    speedup: float = 1.0,
    schedule_rows: list[dict[str, Any]] | None = None,
    fixed_sites: list[CandidateSite] | None = None,
    reference_sites: list[CandidateSite] | None = None,
    reference_label: str = "Reference APs",
) -> Path | None:
    positions = np.asarray(trajectory.positions_m[..., :2], dtype=float)
    if positions.ndim != 3 or positions.shape[0] == 0 or positions.shape[1] == 0:
        return None
    schedule_windows = _group_schedule_rows(schedule_rows or [])
    schedule_has_movement = _schedule_has_movement(schedule_rows or [])
    display_sites = [*base_sites, *(fixed_sites or []), *(reference_sites or [])]

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
    if fixed_sites:
        ax.scatter(
            [site.x_m for site in fixed_sites],
            [site.y_m for site in fixed_sites],
            c="#4c566a",
            s=55,
            marker="P",
            edgecolors="black",
            linewidths=0.4,
            zorder=5,
        )
    if reference_sites:
        ax.scatter(
            [site.x_m for site in reference_sites],
            [site.y_m for site in reference_sites],
            c="#d4a017",
            s=140,
            marker="*",
            edgecolors="black",
            linewidths=0.5,
            zorder=6,
        )
    if schedule_windows:
        initial_window = schedule_windows[0]
        movable_scatter = ax.scatter(
            [site.x_m for site in initial_window["sites"]],
            [site.y_m for site in initial_window["sites"]],
            c="#cb3a2a",
            s=80,
            marker="^",
            edgecolors="black",
            linewidths=0.5,
            zorder=6,
        )
        movable_trails = (
            {
                site.site_id: ax.plot([], [], color="#cb3a2a", linewidth=1.5, alpha=0.35, zorder=4)[0]
                for site in initial_window["sites"]
            }
            if schedule_has_movement
            else {}
        )
    elif selected_sites:
        movable_scatter = ax.scatter(
            [site.x_m for site in selected_sites],
            [site.y_m for site in selected_sites],
            c="#cb3a2a",
            s=80,
            marker="^",
            edgecolors="black",
            linewidths=0.5,
            zorder=6,
        )
        movable_trails = {}
    else:
        movable_scatter = ax.scatter([], [], c="#cb3a2a", s=80, marker="^", edgecolors="black", linewidths=0.5, zorder=6)
        movable_trails = {}

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
    ]
    if fixed_sites:
        legend_handles.append(
            Line2D([0], [0], marker="P", linestyle="", color="#4c566a", markeredgecolor="black", markersize=8, label="Fixed APs")
        )
    if reference_sites:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="*",
                linestyle="",
                color="#d4a017",
                markeredgecolor="black",
                markersize=12,
                label=reference_label,
            )
        )
    legend_handles.extend(
        [
            Line2D(
                [0],
                [0],
                marker="^",
                linestyle="",
                color="#cb3a2a",
                markeredgecolor="black",
                markersize=9,
                label="Movable APs",
            ),
            Line2D([0], [0], marker="o", linestyle="", color="#2ca25f", markeredgecolor="black", markersize=7, label="UE"),
            Line2D([0], [0], color="#f28e2b", linewidth=1.8, label="UE trail"),
        ]
    )
    if schedule_has_movement:
        legend_handles.append(Line2D([0], [0], color="#cb3a2a", linewidth=1.5, alpha=0.35, label="AP path"))
    ax.legend(handles=legend_handles, loc="best")
    if schedule_windows:
        schedule_positions = _schedule_positions_array(schedule_windows)
        bounds_positions = (
            np.concatenate([positions.reshape(-1, 2), schedule_positions], axis=0)
            if schedule_positions.size
            else positions.reshape(-1, 2)
        )
        _set_scene_axes(
            ax,
            metadata,
            graph,
            positions=bounds_positions,
            sites=display_sites,
        )
    else:
        _set_scene_axes(ax, metadata, graph, positions=positions, sites=display_sites)
    ax.set_title(
        "Scene animation with AP movement and moving UEs"
        if schedule_has_movement
        else "Scene animation with AP placement and moving UEs"
    )

    def _update(frame_index: int):
        ue_scatter.set_offsets(positions[frame_index])
        for u_idx, line in enumerate(trail_lines):
            trail = positions[: frame_index + 1, u_idx, :]
            line.set_data(trail[:, 0], trail[:, 1])
        artists = [ue_scatter, timestamp, movable_scatter]
        if schedule_windows:
            time_s = float(trajectory.times_s[frame_index])
            current_window = _schedule_window_for_time(schedule_windows, time_s)
            current_sites = current_window["sites"]
            offsets = np.asarray([[site.x_m, site.y_m] for site in current_sites], dtype=float)
            if offsets.size == 0:
                offsets = np.zeros((0, 2), dtype=float)
            movable_scatter.set_offsets(offsets)
            if schedule_has_movement:
                for ap_id, line in movable_trails.items():
                    path_points = []
                    for window in schedule_windows:
                        if window["start_time_s"] > current_window["end_time_s"] + 1e-9:
                            break
                        for site in window["sites"]:
                            if site.site_id == ap_id:
                                path_points.append([site.x_m, site.y_m])
                                break
                    if path_points:
                        path = np.asarray(path_points, dtype=float)
                        line.set_data(path[:, 0], path[:, 1])
                    else:
                        line.set_data([], [])
            artists.extend(movable_trails.values())
            timestamp.set_text(
                f"t = {time_s:.1f} s\nwindow {current_window['window_index']}: "
                f"{current_window['start_time_s']:.1f}-{current_window['end_time_s']:.1f} s"
            )
        else:
            timestamp.set_text(f"t = {float(trajectory.times_s[frame_index]):.1f} s")
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


def _movable_ap_count(config: ScenarioConfig) -> int:
    return max(int(config.placement.num_movable_aps), 0)


def _select_fixed_sites(config: ScenarioConfig, candidate_sites: list[CandidateSite]) -> list[CandidateSite]:
    fixed_count = max(int(config.placement.num_fixed_aps), 0)
    enabled_sites = [site for site in candidate_sites if site.enabled]
    if fixed_count > len(enabled_sites):
        raise ValueError(
            "Requested %d fixed APs, but only %d enabled candidate AP positions are available"
            % (fixed_count, len(enabled_sites))
        )
    return select_farthest_sites(enabled_sites, fixed_count)


def _make_reference_movable_sites(seed_sites: list[CandidateSite]) -> list[CandidateSite]:
    return [
        CandidateSite(
            site_id=f"movable_ap_{index + 1:02d}",
            x_m=site.x_m,
            y_m=site.y_m,
            z_m=site.z_m,
            yaw_deg=site.yaw_deg,
            pitch_deg=site.pitch_deg,
            mount_type=site.mount_type,
            enabled=True,
            source=f"seed:{site.site_id}",
        )
        for index, site in enumerate(seed_sites)
    ]


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
    use_positional_fallback = any(site.site_id not in baseline_index for site in relocated_sites)
    if use_positional_fallback:
        if len(baseline_sites) != len(relocated_sites):
            raise ValueError(
                "Cannot summarize AP relocations when relocated site ids do not match the baseline ids "
                f"and the site counts differ ({len(baseline_sites)} baseline vs {len(relocated_sites)} relocated)"
            )
        logger.info(
            "Writing AP relocation summary by baseline/mobile AP order because relocated ids differ from baseline ids"
        )
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
        for index, site in enumerate(relocated_sites):
            original = baseline_index.get(site.site_id)
            if original is None:
                original = baseline_sites[index]
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
        "sinr_linear": np.concatenate([np.asarray(segment["sinr_linear"], dtype=float) for segment in segments], axis=0),
        "best_sinr_db": np.concatenate([np.asarray(segment["best_sinr_db"], dtype=float) for segment in segments], axis=0),
        "desired_power_w": np.concatenate([np.asarray(segment["desired_power_w"], dtype=float) for segment in segments], axis=0),
        "interference_power_w": np.concatenate([np.asarray(segment["interference_power_w"], dtype=float) for segment in segments], axis=0),
        "noise_power_w": np.concatenate([np.asarray(segment["noise_power_w"], dtype=float) for segment in segments], axis=0),
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
            (
                "tx_site_ids",
                "rx_ue_ids",
                "times_s",
                "sinr_linear",
                "best_sinr_db",
                "desired_power_w",
                "interference_power_w",
                "noise_power_w",
                "link_power_w",
            ),
        ),
        "final_ap_ue": _split_prefixed_export(
            infra,
            "mobile_ap_ue",
            (
                "tx_site_ids",
                "rx_ue_ids",
                "times_s",
                "sinr_linear",
                "best_sinr_db",
                "desired_power_w",
                "interference_power_w",
                "noise_power_w",
                "link_power_w",
            ),
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
        "fixed_ap_ue_sinr_linear": fixed_ap_ue["sinr_linear"],
        "fixed_ap_ue_best_sinr_db": fixed_ap_ue["best_sinr_db"],
        "fixed_ap_ue_desired_power_w": fixed_ap_ue["desired_power_w"],
        "fixed_ap_ue_interference_power_w": fixed_ap_ue["interference_power_w"],
        "fixed_ap_ue_noise_power_w": fixed_ap_ue["noise_power_w"],
        "fixed_ap_ue_link_power_w": fixed_ap_ue["link_power_w"],
        "mobile_ap_ue_tx_site_ids": np.asarray(final_ap_ue["tx_site_ids"], dtype=object),
        "mobile_ap_ue_rx_ue_ids": np.asarray(final_ap_ue["rx_ue_ids"], dtype=object),
        "mobile_ap_ue_times_s": final_ap_ue["times_s"],
        "mobile_ap_ue_sinr_linear": final_ap_ue["sinr_linear"],
        "mobile_ap_ue_best_sinr_db": final_ap_ue["best_sinr_db"],
        "mobile_ap_ue_desired_power_w": final_ap_ue["desired_power_w"],
        "mobile_ap_ue_interference_power_w": final_ap_ue["interference_power_w"],
        "mobile_ap_ue_noise_power_w": final_ap_ue["noise_power_w"],
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


def _write_strategy_site_csv(output_dir: Path, artifact: StrategyArtifacts) -> None:
    write_candidate_sites(
        output_dir / _strategy_site_csv_name(artifact.name),
        artifact.selected_sites,
        selected_ids={site.site_id for site in artifact.selected_sites},
    )


def _load_ap_site_pool(config: ScenarioConfig, metadata: dict[str, Any] | None) -> tuple[list[CandidateSite], list[CandidateSite]]:
    if metadata is not None and metadata.get("buildings"):
        candidate_sites = generate_wall_candidate_sites(
            metadata=metadata,
            spacing_m=config.placement.candidate_wall_spacing_m,
            mount_height_m=config.placement.candidate_wall_height_m,
            corner_clearance_m=config.placement.candidate_corner_clearance_m,
            mount_offset_m=config.placement.candidate_wall_offset_m,
            min_spacing_m=config.placement.candidate_min_spacing_m,
        )
        if candidate_sites:
            logger.info(
                "Generated %d wall-mounted candidate AP positions",
                len(candidate_sites),
            )
            return candidate_sites, candidate_sites

    base_sites = load_candidate_sites(config.candidate_sites_path)
    return base_sites, list(base_sites)


def _build_static_movable_schedule(
    trajectory: Trajectory,
    sites: list[CandidateSite],
    window_interval_s: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for window_index, indices in enumerate(_window_slices(trajectory.times_s, window_interval_s)):
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


def _nearest_snapshot_mask(trajectory: Trajectory, site: CandidateSite, k_nearest: int) -> np.ndarray:
    positions = np.asarray(trajectory.positions_m[..., :2], dtype=float)
    if positions.size == 0:
        return np.zeros((0,), dtype=bool)
    flat_positions = positions.reshape(-1, 2)
    distances = np.linalg.norm(flat_positions - np.asarray([[site.x_m, site.y_m]], dtype=float), axis=1)
    k = max(1, min(int(k_nearest), len(distances)))
    nearest_indices = np.argpartition(distances, k - 1)[:k]
    mask = np.zeros(len(distances), dtype=bool)
    mask[nearest_indices] = True
    return mask


def _local_percentile_10(best_sinr_db: np.ndarray, snapshot_mask: np.ndarray) -> float:
    values = np.asarray(best_sinr_db, dtype=float).reshape(-1)
    if values.size == 0:
        return -120.0
    if snapshot_mask.size != values.size:
        raise ValueError("snapshot mask size must match flattened SINR samples")
    local_values = values[snapshot_mask]
    if local_values.size == 0:
        return -120.0
    return float(np.percentile(local_values, 10))


def _weighted_percentile(values: np.ndarray, weights: np.ndarray, percentile: float) -> float:
    finite_values = np.asarray(values, dtype=float).reshape(-1)
    finite_weights = np.asarray(weights, dtype=float).reshape(-1)
    if finite_values.size == 0 or finite_weights.size != finite_values.size:
        return -120.0
    mask = np.isfinite(finite_values) & np.isfinite(finite_weights) & (finite_weights > 0.0)
    if not np.any(mask):
        return -120.0
    finite_values = finite_values[mask]
    finite_weights = finite_weights[mask]
    order = np.argsort(finite_values, kind="mergesort")
    sorted_values = finite_values[order]
    sorted_weights = finite_weights[order]
    cumulative = np.cumsum(sorted_weights)
    target = 0.01 * float(percentile) * cumulative[-1]
    return float(sorted_values[np.searchsorted(cumulative, target, side="left")])


def _historical_local_percentile_10(
    history_segments: list[dict[str, Any]],
    selected_candidate_ids: tuple[str, ...],
    candidate_index: dict[str, CandidateSite],
    k_nearest: int,
    decay_rate_per_s: float,
) -> float:
    if not history_segments:
        return -120.0

    now_s = max(float(segment["trajectory"].times_s[-1]) for segment in history_segments)
    weighted_values: list[np.ndarray] = []
    weighted_samples: list[np.ndarray] = []
    for segment in history_segments:
        window_trajectory = segment["trajectory"]
        snapshot_mask = np.zeros(len(window_trajectory.times_s) * len(window_trajectory.ue_ids), dtype=bool)
        for site_id in selected_candidate_ids:
            snapshot_mask |= _nearest_snapshot_mask(
                window_trajectory,
                candidate_index[site_id],
                k_nearest,
            )
        if not np.any(snapshot_mask):
            continue

        values = np.asarray(segment["ap_ue"]["best_sinr_db"], dtype=float).reshape(-1)
        sample_weights = np.repeat(
            np.exp(-max(float(decay_rate_per_s), 0.0) * (now_s - np.asarray(window_trajectory.times_s, dtype=float))),
            len(window_trajectory.ue_ids),
        )
        weighted_values.append(values[snapshot_mask])
        weighted_samples.append(sample_weights[snapshot_mask])

    if not weighted_values:
        return -120.0
    return _weighted_percentile(
        np.concatenate(weighted_values, axis=0),
        np.concatenate(weighted_samples, axis=0),
        10.0,
    )


def _distance_threshold_snapshot_mask(
    trajectory: Trajectory,
    sites: list[CandidateSite],
    distance_threshold_m: float,
) -> np.ndarray:
    positions = np.asarray(trajectory.positions_m[..., :2], dtype=float)
    if positions.ndim != 3 or positions.size == 0:
        return np.zeros((0, 0), dtype=bool)
    if not sites:
        return np.zeros(positions.shape[:2], dtype=bool)
    threshold_m = max(float(distance_threshold_m), 0.0)
    if threshold_m <= 0.0:
        return np.zeros(positions.shape[:2], dtype=bool)

    flat_positions = positions.reshape(-1, 2)
    site_xy = np.asarray([[float(site.x_m), float(site.y_m)] for site in sites], dtype=float)
    distances = np.linalg.norm(flat_positions[:, None, :] - site_xy[None, :, :], axis=2)
    return np.any(distances <= threshold_m, axis=1).reshape(positions.shape[0], positions.shape[1])


def _local_window_sum_rate(
    ap_ue_segment: dict[str, Any],
    trajectory: Trajectory,
    selected_candidate_ids: tuple[str, ...],
    candidate_index: dict[str, CandidateSite],
    distance_threshold_m: float,
) -> float:
    if not selected_candidate_ids:
        return -1e12
    local_sites = [candidate_index[site_id] for site_id in selected_candidate_ids]
    local_mask = _distance_threshold_snapshot_mask(trajectory, local_sites, distance_threshold_m)
    if local_mask.size == 0 or not np.any(local_mask):
        return -1e12

    sinr_linear = np.asarray(ap_ue_segment["sinr_linear"], dtype=float)
    if sinr_linear.shape != local_mask.shape:
        raise ValueError("distance-threshold mask shape must match AP-UE SINR samples")

    local_rates = np.where(local_mask, np.log2(1.0 + np.clip(sinr_linear, 0.0, None)), 0.0)
    return float(np.mean(np.sum(local_rates, axis=1)))


def _local_window_average_power(
    ap_ue_segment: dict[str, Any],
    trajectory: Trajectory,
    selected_candidate_ids: tuple[str, ...],
    candidate_index: dict[str, CandidateSite],
    distance_threshold_m: float,
) -> float:
    if not selected_candidate_ids:
        return -1e12
    local_sites = [candidate_index[site_id] for site_id in selected_candidate_ids]
    local_mask = _distance_threshold_snapshot_mask(trajectory, local_sites, distance_threshold_m)
    if local_mask.size == 0 or not np.any(local_mask):
        return -1e12

    link_power_w = np.asarray(ap_ue_segment["link_power_w"], dtype=float)
    target_shape = np.asarray(ap_ue_segment.get("sinr_linear", local_mask), dtype=float).shape
    ap_count = len(ap_ue_segment.get("tx_site_ids", selected_candidate_ids))
    if link_power_w.shape == target_shape:
        total_power_w = link_power_w
    else:
        total_power_w = None
        axes = tuple(range(1, link_power_w.ndim))
        candidate_reductions: list[tuple[int, tuple[int, ...], np.ndarray]] = []
        for num_axes in range(1, len(axes) + 1):
            for reduce_axes in itertools.combinations(axes, num_axes):
                reduced = np.sum(link_power_w, axis=reduce_axes)
                if reduced.shape == target_shape:
                    reduced_extent = int(np.prod([link_power_w.shape[axis] for axis in reduce_axes], dtype=int))
                    priority = 0 if reduced_extent == ap_count else 1
                    candidate_reductions.append((priority, reduce_axes, reduced))
        if candidate_reductions:
            candidate_reductions.sort(key=lambda item: (item[0], len(item[1]), item[1]))
            total_power_w = candidate_reductions[0][2]
        if total_power_w is None:
            raise ValueError(
                "distance-threshold mask shape must match AP-UE power samples after AP-axis reduction"
            )

    return float(np.mean(total_power_w[local_mask]))


def _window_candidate_anchor_users(
    trajectory: Trajectory,
    sites: list[CandidateSite],
    distance_threshold_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(trajectory.positions_m[..., :2], dtype=float)
    if positions.ndim != 3 or positions.size == 0 or not sites or positions.shape[1] == 0:
        return np.zeros((positions.shape[0] if positions.ndim == 3 else 0, len(sites)), dtype=int), np.zeros(
            (positions.shape[0] if positions.ndim == 3 else 0, len(sites)),
            dtype=bool,
        )
    threshold_m = max(float(distance_threshold_m), 0.0)
    if threshold_m <= 0.0:
        return np.zeros((positions.shape[0], len(sites)), dtype=int), np.zeros((positions.shape[0], len(sites)), dtype=bool)

    site_xy = np.asarray([[float(site.x_m), float(site.y_m)] for site in sites], dtype=float)
    distances = np.linalg.norm(positions[:, :, None, :] - site_xy[None, None, :, :], axis=3)
    anchor_indices = np.argmin(distances, axis=1)
    anchor_distances = np.take_along_axis(distances, anchor_indices[:, None, :], axis=1).reshape(
        positions.shape[0],
        len(sites),
    )
    return anchor_indices.astype(int), anchor_distances <= threshold_m


def _proxy_ap_candidate_power_from_peer_csi(
    peer_link_power_w: np.ndarray,
    trajectory: Trajectory,
    selected_candidate_ids: tuple[str, ...],
    candidate_index: dict[str, CandidateSite],
    distance_threshold_m: float,
    tx_power_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not selected_candidate_ids:
        return (
            np.zeros((0, 0, 0), dtype=float),
            np.zeros((0, 0), dtype=int),
            np.zeros((0, 0), dtype=bool),
        )

    local_sites = [candidate_index[site_id] for site_id in selected_candidate_ids]
    anchor_indices, valid_mask = _window_candidate_anchor_users(trajectory, local_sites, distance_threshold_m)
    expected_shape = (len(trajectory.times_s), len(trajectory.ue_ids), len(trajectory.ue_ids))
    link_power = np.asarray(peer_link_power_w, dtype=float)
    if link_power.shape != expected_shape:
        raise ValueError("UE-UE proxy power samples must match [time, user, user]")

    proxy_power_w = np.zeros((len(trajectory.times_s), len(local_sites), len(trajectory.ue_ids)), dtype=float)
    if proxy_power_w.size == 0:
        return proxy_power_w, anchor_indices, valid_mask

    snapshot_indices = np.arange(len(trajectory.times_s), dtype=int)
    for site_index in range(len(local_sites)):
        anchor_rows = np.asarray(link_power[snapshot_indices, anchor_indices[:, site_index], :], dtype=float)
        anchor_nonself = np.array(anchor_rows, copy=True)
        anchor_nonself[snapshot_indices, anchor_indices[:, site_index]] = -np.inf
        anchor_strength = np.max(anchor_nonself, axis=1, initial=0.0)
        anchor_strength[~np.isfinite(anchor_strength)] = 0.0
        anchor_rows[snapshot_indices, anchor_indices[:, site_index]] = anchor_strength
        anchor_rows[~valid_mask[:, site_index], :] = 0.0
        proxy_power_w[:, site_index, :] = max(float(tx_power_scale), 0.0) * np.clip(anchor_rows, 0.0, None)

    return proxy_power_w, anchor_indices, valid_mask


def _peer_cfr_to_narrowband(peer_cfr: np.ndarray, peer_link_power_w: np.ndarray) -> np.ndarray:
    cfr = np.asarray(peer_cfr)
    link_power = np.asarray(peer_link_power_w, dtype=float)
    if cfr.ndim < 4:
        raise ValueError("UE-UE proxy CFR must retain time, user, and feature axes")
    if cfr.shape[0] != link_power.shape[0]:
        raise ValueError("UE-UE proxy CFR time axis must match UE-UE power samples")

    num_users = int(link_power.shape[1])
    candidate_axes = [axis for axis in range(1, cfr.ndim) if cfr.shape[axis] == num_users]
    best_channel = None
    best_error = float("inf")
    for rx_axis, tx_axis in itertools.permutations(candidate_axes, 2):
        reoriented = np.moveaxis(cfr, (0, rx_axis, tx_axis), (0, 1, 2))
        reduce_axes = tuple(range(3, reoriented.ndim))
        power = np.mean(np.abs(reoriented) ** 2, axis=reduce_axes) if reduce_axes else np.abs(reoriented) ** 2
        if power.shape != link_power.shape:
            continue
        error = float(np.nanmean(np.abs(power - link_power)))
        if error < best_error:
            best_error = error
            best_channel = np.mean(reoriented, axis=reduce_axes) if reduce_axes else reoriented
    if best_channel is None:
        raise ValueError("Unable to infer UE-UE CFR user axes from the stored proxy payload")
    return np.asarray(best_channel, dtype=np.complex128)


def _proxy_ap_candidate_channel_from_peer_csi(
    peer_csi_segment: dict[str, Any],
    trajectory: Trajectory,
    selected_candidate_ids: tuple[str, ...],
    candidate_index: dict[str, CandidateSite],
    distance_threshold_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if "cfr" not in peer_csi_segment:
        raise ValueError("UE-UE proxy CSI requires a CFR export")

    local_sites = [candidate_index[site_id] for site_id in selected_candidate_ids]
    anchor_indices, valid_mask = _window_candidate_anchor_users(trajectory, local_sites, distance_threshold_m)
    narrowband = _peer_cfr_to_narrowband(peer_csi_segment["cfr"], peer_csi_segment["link_power_w"])
    if narrowband.shape != (len(trajectory.times_s), len(trajectory.ue_ids), len(trajectory.ue_ids)):
        raise ValueError("UE-UE proxy narrowband channel must match [time, user, user]")

    proxy_channel = np.zeros((len(trajectory.times_s), len(trajectory.ue_ids), len(local_sites)), dtype=np.complex128)
    snapshot_indices = np.arange(len(trajectory.times_s), dtype=int)
    for site_index in range(len(local_sites)):
        anchor_columns = np.asarray(narrowband[snapshot_indices, :, anchor_indices[:, site_index]], dtype=np.complex128)
        nonself_magnitudes = np.abs(anchor_columns)
        nonself_magnitudes[snapshot_indices, anchor_indices[:, site_index]] = -1.0
        strongest_peer_indices = np.argmax(nonself_magnitudes, axis=1)
        strongest_peer_values = anchor_columns[snapshot_indices, strongest_peer_indices]
        anchor_columns[snapshot_indices, anchor_indices[:, site_index]] = 0.0 + 0.0j
        valid_self_fill = np.max(nonself_magnitudes, axis=1) >= 0.0
        anchor_columns[
            snapshot_indices[valid_self_fill],
            anchor_indices[valid_self_fill, site_index],
        ] = strongest_peer_values[valid_self_fill]
        anchor_columns[~valid_mask[:, site_index], :] = 0.0 + 0.0j
        proxy_channel[:, :, site_index] = anchor_columns
    return proxy_channel, anchor_indices, valid_mask


def _proxy_window_sum_rate_from_peer_csi(
    peer_csi_segment: dict[str, Any],
    trajectory: Trajectory,
    selected_candidate_ids: tuple[str, ...],
    candidate_index: dict[str, CandidateSite],
    distance_threshold_m: float,
    noise_power_w: float,
    total_tx_power_w: float,
    tx_power_scale: float,
) -> float:
    if "cfr" in peer_csi_segment:
        try:
            proxy_channel, _anchor_indices, valid_mask = _proxy_ap_candidate_channel_from_peer_csi(
                peer_csi_segment,
                trajectory,
                selected_candidate_ids,
                candidate_index,
                distance_threshold_m,
            )
        except ValueError as exc:
            logger.warning(
                "Falling back to UE-UE power-proxy ESR because proxy CSI decoding failed: %s "
                "(cfr_shape=%s, link_power_shape=%s)",
                exc,
                np.asarray(peer_csi_segment["cfr"]).shape,
                np.asarray(peer_csi_segment["link_power_w"]).shape,
            )
        else:
            if proxy_channel.size == 0 or not np.any(valid_mask):
                return -1e12

            esr_samples: list[float] = []
            for snapshot_index in range(proxy_channel.shape[0]):
                channel = proxy_channel[snapshot_index][:, None, :, None]
                sinr_terms = _zf_sinr_terms_from_mimo_channel(
                    channel,
                    total_tx_power_w=max(float(total_tx_power_w), 0.0),
                    noise_power_w=max(float(noise_power_w), 1e-12),
                )
                esr_samples.append(float(np.sum(np.log2(1.0 + np.clip(sinr_terms["sinr"], 0.0, None)))))
            return float(np.mean(esr_samples)) if esr_samples else -1e12

    proxy_power_w, _anchor_indices, valid_mask = _proxy_ap_candidate_power_from_peer_csi(
        peer_csi_segment["link_power_w"],
        trajectory,
        selected_candidate_ids,
        candidate_index,
        distance_threshold_m,
        tx_power_scale,
    )
    if proxy_power_w.size == 0 or not np.any(valid_mask):
        return -1e12

    desired_power_w = np.max(proxy_power_w, axis=1)
    total_power_w = np.sum(proxy_power_w, axis=1)
    interference_power_w = np.clip(total_power_w - desired_power_w, 0.0, None)
    sinr_linear = desired_power_w / (interference_power_w + max(float(noise_power_w), 1e-12))
    return float(np.mean(np.sum(np.log2(1.0 + np.clip(sinr_linear, 0.0, None)), axis=1)))


def _proxy_window_average_power_from_peer_csi(
    peer_link_power_w: np.ndarray,
    trajectory: Trajectory,
    selected_candidate_ids: tuple[str, ...],
    candidate_index: dict[str, CandidateSite],
    distance_threshold_m: float,
    tx_power_scale: float,
) -> float:
    proxy_power_w, _anchor_indices, valid_mask = _proxy_ap_candidate_power_from_peer_csi(
        peer_link_power_w,
        trajectory,
        selected_candidate_ids,
        candidate_index,
        distance_threshold_m,
        tx_power_scale,
    )
    if proxy_power_w.size == 0 or not np.any(valid_mask):
        return -1e12
    return float(np.mean(np.sum(proxy_power_w, axis=1)))


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
    strategy_ap_ue: dict[str, dict[str, Any]],
) -> None:
    strategy_names = [name for name in ALL_STRATEGY_NAMES if name in strategy_ap_ue]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ue_id", *[f"{name}_mean_zf_sinr_db" for name in strategy_names]])
        strategy_values = {
            name: _per_user_mean_best_sinr(payload["best_sinr_db"])
            for name, payload in strategy_ap_ue.items()
        }
        for index, ue_id in enumerate(ue_ids):
            row = [ue_id]
            for name in strategy_names:
                if name in strategy_values:
                    row.append(float(strategy_values[name][index]))
            writer.writerow(row)


def _plot_user_sinr_cdf(strategy_ap_ue: dict[str, dict[str, Any]], output_path: Path) -> None:
    strategy_names = [name for name in ALL_STRATEGY_NAMES if name in strategy_ap_ue]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    styles = {
        "central_massive_mimo": ("#2ca02c", "-."),
        "distributed_fixed": ("#1f77b4", "-"),
        "distributed_movable": ("#ff7f0e", "--"),
        "distributed_movable_optimization_2": ("#ff7f0e", "-"),
        "distributed_movable_optimization_3": ("#ff7f0e", ":"),
    }
    minimum = None
    for name in strategy_names:
        payload = strategy_ap_ue.get(name)
        if payload is None:
            continue
        x_values, y_values = _cdf_points(_instantaneous_user_sinr_samples(payload["best_sinr_db"]))
        if x_values.size == 0:
            continue
        color, linestyle = styles.get(name, ("#444444", _strategy_linestyle(name)))
        minimum = float(np.min(x_values)) if minimum is None else min(minimum, float(np.min(x_values)))
        plt.step(x_values, y_values, where="post", linewidth=2.0, linestyle=linestyle, color=color, label=name)
    plt.xlabel("Instantaneous distributed-MIMO ZF SINR per user snapshot [dB]")
    plt.ylabel("CDF")
    plt.xlim(left=(minimum if minimum is not None else 0.0) - 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _write_user_sinr_artifacts(
    output_dir: Path,
    trajectory: Trajectory,
    strategy_ap_ue: dict[str, dict[str, Any]],
    *,
    include_plots: bool = True,
) -> None:
    strategy_names = [name for name in ALL_STRATEGY_NAMES if name in strategy_ap_ue]
    _write_user_sinr_csv(
        output_dir / "user_sinr_summary.csv",
        trajectory.ue_ids,
        strategy_ap_ue,
    )
    output_path = output_dir / "user_sinr_timeseries.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["snapshot_index", "time_s", "ue_id", *[f"{name}_sinr_db" for name in strategy_names]])
        for t_idx, time_s in enumerate(trajectory.times_s):
            for u_idx, ue_id in enumerate(trajectory.ue_ids):
                row = [int(t_idx), float(time_s), ue_id]
                for name in strategy_names:
                    payload = strategy_ap_ue.get(name)
                    if payload is not None:
                        row.append(float(np.asarray(payload["best_sinr_db"], dtype=float)[t_idx, u_idx]))
                writer.writerow(row)
    npz_payload: dict[str, Any] = {
        "snapshot_index": np.arange(len(trajectory.times_s), dtype=int),
        "times_s": np.asarray(trajectory.times_s, dtype=float),
        "ue_ids": np.asarray(trajectory.ue_ids, dtype=object),
        "strategy_names": np.asarray(strategy_names, dtype=object),
    }
    for name in strategy_names:
        payload = strategy_ap_ue.get(name)
        if payload is not None:
            if "sinr_linear" in payload:
                npz_payload[f"{name}_sinr_linear"] = np.asarray(payload["sinr_linear"], dtype=float)
            npz_payload[f"{name}_sinr_db"] = np.asarray(payload["best_sinr_db"], dtype=float)
            if "desired_power_w" in payload:
                npz_payload[f"{name}_desired_power_w"] = np.asarray(payload["desired_power_w"], dtype=float)
            if "interference_power_w" in payload:
                npz_payload[f"{name}_interference_power_w"] = np.asarray(payload["interference_power_w"], dtype=float)
            if "noise_power_w" in payload:
                npz_payload[f"{name}_noise_power_w"] = np.asarray(payload["noise_power_w"], dtype=float)
    np.savez_compressed(output_dir / "user_sinr_snapshots.npz", **npz_payload)
    if include_plots:
        _plot_user_sinr_cdf(strategy_ap_ue, output_dir / "user_sinr_cdf.png")


def _write_strategy_comparison_csv(output_path: Path, strategies: dict[str, StrategyArtifacts]) -> None:
    strategy_names = [name for name in ALL_STRATEGY_NAMES if name in strategies]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "strategy",
                "score",
                "outage",
                "percentile_10_db",
                "peer_tiebreak",
                "capped",
                "evaluated_combinations",
                "final_candidate_ids",
            ]
        )
        for name in strategy_names:
            if name not in strategies:
                continue
            artifact = strategies[name]
            writer.writerow(
                [
                    name,
                    float(artifact.score.score),
                    float(artifact.score.outage),
                    float(artifact.score.percentile_10_db),
                    float(artifact.score.peer_tiebreak),
                    bool(artifact.capped),
                    int(artifact.evaluated_combinations),
                    " ".join(artifact.final_candidate_ids),
                ]
            )


def _candidate_selection_to_sites(
    reference_sites: list[CandidateSite],
    selected_candidate_ids: list[str],
    candidate_index: dict[str, CandidateSite],
) -> list[CandidateSite]:
    selected_candidates = [candidate_index[site_id] for site_id in selected_candidate_ids]
    return _relocate_sites(reference_sites, selected_candidates)


def _append_schedule_rows(
    schedule_rows: list[dict[str, Any]],
    window_index: int,
    window_trajectory: Trajectory,
    movable_sites: list[CandidateSite],
) -> None:
    for site in movable_sites:
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


def _evaluate_strategy_windows(
    strategy_name: str,
    runner: SionnaRtRunner,
    config: ScenarioConfig,
    fixed_sites: list[CandidateSite],
    initial_movable_sites: list[CandidateSite],
    candidate_index: dict[str, CandidateSite],
    trajectory: Trajectory,
    peer_need_weights: np.ndarray,
    candidate_selector,
    export_full: bool,
) -> StrategyArtifacts:
    movable_sites = list(initial_movable_sites)
    ap_ue_segments: list[dict[str, Any]] = []
    schedule_rows: list[dict[str, Any]] = []
    selected_candidate_union: set[str] = set()
    final_candidate_ids: list[str] = []
    any_capped = False
    total_evaluations = 0

    for window_index, indices in enumerate(_window_slices(trajectory.times_s, config.placement.window_interval_s)):
        window_trajectory = _slice_trajectory(trajectory, indices)
        window_need_weights = np.asarray(peer_need_weights[indices], dtype=float)
        selected_candidate_ids, capped, evaluations = candidate_selector(
            window_index,
            window_trajectory,
            window_need_weights,
            movable_sites,
        )
        movable_sites = _candidate_selection_to_sites(movable_sites, selected_candidate_ids, candidate_index)
        active_sites = [*fixed_sites, *movable_sites]
        ap_ue_segments.append(runner.compute_ap_ue_csi(active_sites, window_trajectory, export_full=export_full))
        _append_schedule_rows(schedule_rows, window_index, window_trajectory, movable_sites)
        final_candidate_ids = list(selected_candidate_ids)
        selected_candidate_union.update(selected_candidate_ids)
        any_capped = any_capped or capped
        total_evaluations += evaluations

    if not ap_ue_segments:
        raise RuntimeError(f"{strategy_name} produced no placement windows")

    selected_sites = [*fixed_sites, *movable_sites]
    ap_ue = _concat_ap_ue_segments(ap_ue_segments)
    ap_ap = runner.compute_ap_ap_csi(selected_sites, export_full=export_full)
    score = summarize_candidate_set(
        grid_best_sinr_db=np.asarray([], dtype=float),
        trajectory_best_sinr_db=ap_ue["best_sinr_db"],
        peer_need_weights=peer_need_weights,
        cfg=config.placement,
    )
    return StrategyArtifacts(
        name=strategy_name,
        selected_sites=selected_sites,
        movable_sites=list(movable_sites),
        ap_ue=ap_ue,
        ap_ap=ap_ap,
        score=score,
        schedule_rows=schedule_rows,
        final_candidate_ids=final_candidate_ids,
        selected_candidate_union=selected_candidate_union,
        capped=any_capped,
        evaluated_combinations=total_evaluations,
    )


def _evaluate_static_strategy(
    strategy_name: str,
    runner: SionnaRtRunner,
    config: ScenarioConfig,
    selected_sites: list[CandidateSite],
    trajectory: Trajectory,
    peer_need_weights: np.ndarray,
    export_full: bool,
    final_candidate_ids: list[str],
    details: dict[str, Any] | None = None,
) -> StrategyArtifacts:
    ap_ue = runner.compute_ap_ue_csi(selected_sites, trajectory, export_full=export_full)
    ap_ap = runner.compute_ap_ap_csi(selected_sites, export_full=export_full)
    score = summarize_candidate_set(
        grid_best_sinr_db=np.asarray([], dtype=float),
        trajectory_best_sinr_db=ap_ue["best_sinr_db"],
        peer_need_weights=peer_need_weights,
        cfg=config.placement,
    )
    return StrategyArtifacts(
        name=strategy_name,
        selected_sites=list(selected_sites),
        movable_sites=list(selected_sites),
        ap_ue=ap_ue,
        ap_ap=ap_ap,
        score=score,
        schedule_rows=_build_static_movable_schedule(
            trajectory,
            selected_sites,
            config.placement.window_interval_s,
        ),
        final_candidate_ids=list(final_candidate_ids),
        selected_candidate_union=set(final_candidate_ids),
        details=dict(details or {}),
    )


def _evaluate_central_massive_mimo(
    runner: SionnaRtRunner,
    config: ScenarioConfig,
    rooftop_candidates: list[CandidateSite],
    trajectory: Trajectory,
    peer_need_weights: np.ndarray,
    export_full: bool,
) -> StrategyArtifacts:
    if not rooftop_candidates:
        raise ValueError("At least one rooftop candidate is required for central_massive_mimo")
    best_candidate = rooftop_candidates[0]
    best_site = _make_central_ap_site(best_candidate)
    best_ap_ue = runner.compute_ap_ue_csi([best_site], trajectory, export_full=export_full)
    best_score = summarize_candidate_set(
        grid_best_sinr_db=np.asarray([], dtype=float),
        trajectory_best_sinr_db=best_ap_ue["best_sinr_db"],
        peer_need_weights=peer_need_weights,
        cfg=config.placement,
    )
    ap_ap = runner.compute_ap_ap_csi([best_site], export_full=export_full)
    return StrategyArtifacts(
        name="central_massive_mimo",
        selected_sites=[best_site],
        movable_sites=[best_site],
        ap_ue=best_ap_ue,
        ap_ap=ap_ap,
        score=best_score,
        schedule_rows=_build_static_movable_schedule(
            trajectory,
            [best_site],
            config.placement.window_interval_s,
        ),
        final_candidate_ids=[best_candidate.site_id],
        selected_candidate_union={best_candidate.site_id},
        details={
            "selection_mode": "fixed_center_rooftop",
            "rooftop_candidate_id": best_candidate.site_id,
        },
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
    _validate_three_mode_config(config)
    output_dir = config.outputs.output_dir
    radio_map_enabled = bool(config.coverage.enabled)
    csi_exports_enabled = bool(config.outputs.write_csi_exports)
    csi_cache_enabled = bool(config.outputs.enable_csi_cache)
    strategy_names = _active_strategy_names(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Starting scenario '%s'", config.name)
    logger.info("Writing outputs to %s", output_dir)
    _remove_artifacts(_legacy_output_paths(output_dir))
    _remove_artifacts(_strategy_output_paths(output_dir))
    _remove_artifacts(_visualization_artifact_paths(output_dir))
    if csi_cache_enabled:
        logger.warning("CSI cache reuse is not yet implemented for the multi-strategy placement pipeline; skipping cache")
        csi_cache_enabled = False

    with progress_bar(total=10, desc=f"Scenario {config.name}", unit="stage", leave=True) as scenario_progress:
        scene_artifacts = _resolve_scene_inputs(config)
        metadata = load_scene_metadata(scene_artifacts.metadata_path)
        scene_context = _store_scene_context(output_dir, scene_artifacts)
        rooftop_candidates = _generate_rooftop_candidates(metadata)
        if not rooftop_candidates:
            raise ValueError(
                "Three-mode comparison requires rooftop candidates derived from scene metadata buildings"
            )
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

        _base_sites, candidate_sites = _load_ap_site_pool(config, metadata)
        fixed_sites: list[CandidateSite] = []
        movable_candidate_sites = [site for site in candidate_sites if site.enabled]
        movable_candidate_ids = sorted(site.site_id for site in movable_candidate_sites)
        movable_count = _movable_ap_count(config)
        if movable_count > len(movable_candidate_sites):
            raise ValueError(
                "Requested %d distributed APs, but only %d candidate AP positions are available"
                % (movable_count, len(movable_candidate_sites))
            )
        random_candidate_ids = sample_random_candidates(
            movable_candidate_ids,
            movable_count,
            config.placement.random_seed,
        )
        candidate_index = {site.site_id: site for site in movable_candidate_sites}
        baseline_seed_sites = [candidate_index[site_id] for site_id in random_candidate_ids]
        baseline_sites = _make_reference_movable_sites(baseline_seed_sites)
        if not baseline_sites:
            raise ValueError("At least one AP is required for placement comparison")
        central_radio = _central_ap_radio(config.radio, movable_count)
        logger.info(
            "Comparing 1 central AP against %d distributed APs using %d wall candidates and %d rooftop candidates",
            len(baseline_sites),
            len(movable_candidate_sites),
            len(rooftop_candidates),
        )
        scenario_progress.update(1)

        if not config.solver.enable_ray_tracing:
            logger.info("Ray tracing disabled by configuration; exporting geometric placement artifacts only")
            _remove_artifacts(
                [
                    output_dir / "coverage_map.npz",
                    output_dir / "fixed_coverage_map.npz",
                    output_dir / "peer_csi_snapshots.npz",
                    output_dir / "infra_csi_snapshots.npz",
                    output_dir / "user_sinr_summary.csv",
                    output_dir / "user_sinr_timeseries.csv",
                    output_dir / "user_sinr_snapshots.npz",
                ]
            )
            static_schedule = _build_static_movable_schedule(
                trajectory,
                baseline_sites,
                config.placement.window_interval_s,
            )
            central_placeholder_candidate = _select_centroid_rooftop_candidate(metadata, rooftop_candidates)
            central_placeholder_site = _make_central_ap_site(central_placeholder_candidate)
            strategy_results = {
                "distributed_fixed": StrategyArtifacts(
                    name="distributed_fixed",
                    selected_sites=list(baseline_sites),
                    movable_sites=list(baseline_sites),
                    ap_ue={},
                    ap_ap={},
                    score=PlacementScore(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    schedule_rows=list(static_schedule),
                    final_candidate_ids=list(random_candidate_ids),
                    selected_candidate_union=set(random_candidate_ids),
                    details={"selection_mode": "geometric_placeholder"},
                ),
                "central_massive_mimo": StrategyArtifacts(
                    name="central_massive_mimo",
                    selected_sites=[central_placeholder_site],
                    movable_sites=[central_placeholder_site],
                    ap_ue={},
                    ap_ap={},
                    score=PlacementScore(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    schedule_rows=_build_static_movable_schedule(
                        trajectory,
                        [central_placeholder_site],
                        config.placement.window_interval_s,
                    ),
                    final_candidate_ids=[central_placeholder_candidate.site_id],
                    selected_candidate_union={central_placeholder_candidate.site_id},
                    details={
                        "selection_mode": "geometric_placeholder",
                        "is_placeholder": True,
                        "rooftop_candidate_id": central_placeholder_candidate.site_id,
                        "central_ap_num_rows": int(central_radio.ap_num_rows),
                        "central_ap_num_cols": int(central_radio.ap_num_cols),
                        "central_ap_tx_power_dbm": float(central_radio.tx_power_dbm_ap),
                    },
                ),
            }
            if config.placement.enable_optimization_1:
                strategy_results["distributed_movable"] = StrategyArtifacts(
                    name="distributed_movable",
                    selected_sites=list(baseline_sites),
                    movable_sites=list(baseline_sites),
                    ap_ue={},
                    ap_ap={},
                    score=PlacementScore(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    schedule_rows=list(static_schedule),
                    final_candidate_ids=list(random_candidate_ids),
                    selected_candidate_union=set(random_candidate_ids),
                    details={"selection_mode": "geometric_placeholder"},
                )
            if config.placement.enable_optimization_2:
                strategy_results["distributed_movable_optimization_2"] = StrategyArtifacts(
                    name="distributed_movable_optimization_2",
                    selected_sites=list(baseline_sites),
                    movable_sites=list(baseline_sites),
                    ap_ue={},
                    ap_ap={},
                    score=PlacementScore(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    schedule_rows=list(static_schedule),
                    final_candidate_ids=list(random_candidate_ids),
                    selected_candidate_union=set(random_candidate_ids),
                    details={"selection_mode": "geometric_placeholder"},
                )
            if config.placement.enable_optimization_3:
                strategy_results["distributed_movable_optimization_3"] = StrategyArtifacts(
                    name="distributed_movable_optimization_3",
                    selected_sites=list(baseline_sites),
                    movable_sites=list(baseline_sites),
                    ap_ue={},
                    ap_ap={},
                    score=PlacementScore(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    schedule_rows=list(static_schedule),
                    final_candidate_ids=list(random_candidate_ids),
                    selected_candidate_union=set(random_candidate_ids),
                    details={"selection_mode": "geometric_placeholder"},
                )
            best_strategy_name = "distributed_fixed"
            animation_strategy_name = _scene_animation_strategy_name(strategy_results, best_strategy_name)
            animation_strategy = strategy_results[animation_strategy_name]
            visualized_sites = list(strategy_results[best_strategy_name].selected_sites)
            distributed_selected_union = set().union(
                *(
                    artifact.selected_candidate_union
                    for name, artifact in strategy_results.items()
                    if name != "central_massive_mimo"
                )
            )
            write_candidate_sites(
                output_dir / "candidate_ap_positions.csv",
                movable_candidate_sites,
                selected_ids=distributed_selected_union,
            )
            write_candidate_sites(
                output_dir / "central_ap_rooftop_candidates.csv",
                rooftop_candidates,
                selected_ids=strategy_results["central_massive_mimo"].selected_candidate_union,
            )
            for artifact in strategy_results.values():
                _write_strategy_site_csv(output_dir, artifact)
                _write_mobile_ap_schedule_csv(output_dir / f"{artifact.name}_schedule.csv", artifact.schedule_rows)
            _write_strategy_comparison_csv(output_dir / "strategy_comparison.csv", strategy_results)
            logger.info("Wrote non-ray-traced trajectory and placement data to %s", output_dir)
            scenario_progress.update(7)

            summary = {
                "scenario": config.name,
                "ray_tracing_enabled": False,
                "radio_map_enabled": radio_map_enabled,
                "csi_exports_enabled": csi_exports_enabled,
                "csi_cache_enabled": csi_cache_enabled,
                "baseline_strategy": "distributed_fixed",
                "best_strategy": best_strategy_name,
                "scene_animation_strategy": animation_strategy_name,
                "scene_animation_with_central_massive_mimo": "",
                "scene_animation_speedup": float(config.outputs.scene_animation_speedup),
                "scene_context": scene_context,
                "compute_device": "SKIPPED",
                "mitsuba_variant": "",
                "window_interval_s": config.placement.window_interval_s,
                "candidate_site_ids": movable_candidate_ids,
                "central_rooftop_candidate_ids": [site.site_id for site in rooftop_candidates],
                "selected_site_ids": [site.site_id for site in visualized_sites],
                "strategies": {
                    name: {
                        "selected_site_ids": [site.site_id for site in artifact.selected_sites],
                        "movable_site_ids": [site.site_id for site in artifact.movable_sites],
                        "final_candidate_ids": list(artifact.final_candidate_ids),
                        "capped": bool(artifact.capped),
                        **artifact.details,
                    }
                    for name, artifact in strategy_results.items()
                },
                "status": "trajectory_only",
            }
            (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            return summary
        distributed_runner = SionnaRtRunner(
            scene_cfg=config.scene,
            radio=config.radio,
            solver_cfg=config.solver,
            scene_inputs=SceneInputs(
                scene_path=None if config.scene.kind == "builtin" else scene_artifacts.scene_xml_path,
                metadata=metadata,
            ),
        )
        central_runner = SionnaRtRunner(
            scene_cfg=config.scene,
            radio=central_radio,
            solver_cfg=config.solver,
            scene_inputs=SceneInputs(
                scene_path=None if config.scene.kind == "builtin" else scene_artifacts.scene_xml_path,
                metadata=metadata,
            ),
        )
        runtime_info = distributed_runner.runtime_info()
        logger.info(
            "Compute backend: %s via Mitsuba variant %s",
            runtime_info["device"],
            runtime_info["variant"],
        )
        scenario_progress.update(1)

        logger.info("Computing UE-UE CSI snapshots")
        peer_csi = distributed_runner.compute_ue_ue_csi(trajectory, export_full=True)
        scenario_progress.update(1)

        initial_movable_sites = list(baseline_sites)
        historical_decay_rate = float(config.placement.historical_csi_decay_rate_per_s)
        optimization_2_distance_threshold_m = float(config.placement.optimization_2_distance_threshold_m)
        subset_history: dict[tuple[str, ...], list[dict[str, Any]]] = {}
        window_indices = _window_slices(trajectory.times_s, config.placement.window_interval_s)
        peer_csi_windows: list[dict[str, Any]] = []
        for indices in window_indices:
            segment = {
                "link_power_w": np.asarray(peer_csi["link_power_w"], dtype=float)[indices],
            }
            if "cfr" in peer_csi:
                segment["cfr"] = np.asarray(peer_csi["cfr"])[indices]
            peer_csi_windows.append(segment)
        ap_tx_power_w = float(10 ** (config.radio.tx_power_dbm_ap / 10.0) / 1000.0)
        ue_tx_power_w = float(10 ** (config.radio.tx_power_dbm_ue / 10.0) / 1000.0)
        proxy_tx_power_scale = ap_tx_power_w / max(ue_tx_power_w, 1e-12)
        proxy_noise_power_w = float(Boltzmann * config.radio.temperature_k * config.radio.bandwidth_hz)

        def local_selector(window_index: int, window_trajectory: Trajectory, window_need_weights: np.ndarray, _current_movable_sites):
            del window_need_weights
            evaluated_subsets: set[tuple[str, ...]] = set()

            def history_for_subset(subset: tuple[str, ...]) -> list[dict[str, Any]]:
                history = subset_history.setdefault(subset, [])
                if len(history) <= window_index:
                    active_sites = [candidate_index[site_id] for site_id in subset]
                    ap_ue = distributed_runner.compute_ap_ue_csi(active_sites, window_trajectory, export_full=False)
                    history.append({"trajectory": window_trajectory, "ap_ue": ap_ue})
                    evaluated_subsets.add(subset)
                return history

            def local_evaluator(subset: tuple[str, ...]) -> float:
                return _historical_local_percentile_10(
                    history_for_subset(subset),
                    subset,
                    candidate_index,
                    config.placement.heuristic_k_nearest,
                    historical_decay_rate,
                )

            selected_ids = select_local_csi_candidates(
                movable_candidate_ids,
                len(initial_movable_sites),
                local_evaluator,
            )
            return selected_ids, False, len(evaluated_subsets)

        def optimization_2_selector(window_index: int, window_trajectory: Trajectory, window_need_weights: np.ndarray, _current_movable_sites):
            del window_need_weights
            subset_cache: dict[tuple[str, ...], float] = {}
            peer_csi_window = peer_csi_windows[window_index]

            def local_evaluator(subset: tuple[str, ...]) -> float:
                if subset not in subset_cache:
                    subset_cache[subset] = _proxy_window_sum_rate_from_peer_csi(
                        peer_csi_window,
                        window_trajectory,
                        subset,
                        candidate_index,
                        optimization_2_distance_threshold_m,
                        proxy_noise_power_w,
                        len(subset) * ap_tx_power_w,
                        proxy_tx_power_scale,
                    )
                return subset_cache[subset]

            selected_ids = select_local_csi_candidates(
                movable_candidate_ids,
                len(initial_movable_sites),
                local_evaluator,
            )
            return selected_ids, False, len(subset_cache)

        def optimization_3_selector(window_index: int, window_trajectory: Trajectory, window_need_weights: np.ndarray, _current_movable_sites):
            del window_need_weights
            subset_cache: dict[tuple[str, ...], float] = {}
            peer_csi_window = peer_csi_windows[window_index]

            def local_evaluator(subset: tuple[str, ...]) -> float:
                if subset not in subset_cache:
                    subset_cache[subset] = _proxy_window_average_power_from_peer_csi(
                        peer_csi_window["link_power_w"],
                        window_trajectory,
                        subset,
                        candidate_index,
                        optimization_2_distance_threshold_m,
                        proxy_tx_power_scale,
                    )
                return subset_cache[subset]

            selected_ids = select_local_csi_candidates(
                movable_candidate_ids,
                len(initial_movable_sites),
                local_evaluator,
            )
            return selected_ids, False, len(subset_cache)

        peer_need_weights = np.asarray(peer_csi["need_weights"], dtype=float)
        strategy_results: dict[str, StrategyArtifacts] = {
            "distributed_fixed": _evaluate_static_strategy(
                "distributed_fixed",
                distributed_runner,
                config,
                baseline_sites,
                trajectory,
                peer_need_weights,
                csi_exports_enabled,
                list(random_candidate_ids),
                details={"selection_mode": "seed_constellation"},
            ),
            "central_massive_mimo": _evaluate_central_massive_mimo(
                central_runner,
                config,
                rooftop_candidates,
                trajectory,
                peer_need_weights,
                csi_exports_enabled,
            ),
        }
        if config.placement.enable_optimization_1:
            strategy_results["distributed_movable"] = _evaluate_strategy_windows(
                "distributed_movable",
                distributed_runner,
                config,
                [],
                initial_movable_sites,
                candidate_index,
                trajectory,
                peer_need_weights,
                local_selector,
                csi_exports_enabled,
            )
            strategy_results["distributed_movable"].details = {
                "selection_mode": "windowed_local_csi_history",
                "historical_csi_decay_rate_per_s": historical_decay_rate,
                "heuristic_k_nearest": int(config.placement.heuristic_k_nearest),
            }
        if config.placement.enable_optimization_2:
            strategy_results["distributed_movable_optimization_2"] = _evaluate_strategy_windows(
                "distributed_movable_optimization_2",
                distributed_runner,
                config,
                [],
                initial_movable_sites,
                candidate_index,
                trajectory,
                peer_need_weights,
                optimization_2_selector,
                csi_exports_enabled,
            )
            strategy_results["distributed_movable_optimization_2"].details = {
                "selection_mode": "windowed_ue_proxy_esr",
                "selection_proxy": "ue_ue_csi",
                "optimization_2_distance_threshold_m": optimization_2_distance_threshold_m,
            }
        if config.placement.enable_optimization_3:
            strategy_results["distributed_movable_optimization_3"] = _evaluate_strategy_windows(
                "distributed_movable_optimization_3",
                distributed_runner,
                config,
                [],
                initial_movable_sites,
                candidate_index,
                trajectory,
                peer_need_weights,
                optimization_3_selector,
                csi_exports_enabled,
            )
            strategy_results["distributed_movable_optimization_3"].details = {
                "selection_mode": "windowed_ue_proxy_average_power",
                "selection_proxy": "ue_ue_power",
                "optimization_2_distance_threshold_m": optimization_2_distance_threshold_m,
            }
        strategy_results["central_massive_mimo"].details.update(
            {
                "central_ap_num_rows": int(central_radio.ap_num_rows),
                "central_ap_num_cols": int(central_radio.ap_num_cols),
                "central_ap_tx_power_dbm": float(central_radio.tx_power_dbm_ap),
            }
        )
        scenario_progress.update(2)

        best_strategy_name = max(
            strategy_names,
            key=lambda name: strategy_results[name].score.score,
        )
        baseline_strategy = strategy_results["distributed_fixed"]
        best_strategy = strategy_results[best_strategy_name]
        animation_strategy_name = _scene_animation_strategy_name(strategy_results, best_strategy_name)
        animation_strategy = strategy_results[animation_strategy_name]
        runner_by_strategy = {
            name: (central_runner if name == "central_massive_mimo" else distributed_runner)
            for name in strategy_names
        }
        if animation_strategy_name != best_strategy_name:
            logger.info(
                "Using %s schedule as postprocess animation reference because best strategy %s keeps movable APs static",
                animation_strategy_name,
                best_strategy_name,
            )

        if csi_exports_enabled:
            np.savez_compressed(output_dir / "peer_csi_snapshots.npz", **peer_csi)
            logger.info("Wrote peer CSI export to %s", output_dir / "peer_csi_snapshots.npz")
        else:
            _remove_artifacts([output_dir / "peer_csi_snapshots.npz"])

        infra_export: dict[str, Any] = {}
        for name in strategy_names:
            artifact = strategy_results[name]
            ap_ue = artifact.ap_ue
            ap_ap = artifact.ap_ap
            prefix_ap_ue = f"{name}_ap_ue"
            prefix_ap_ap = f"{name}_ap_ap"
            infra_export[f"{prefix_ap_ue}_tx_site_ids"] = np.asarray(ap_ue["tx_site_ids"], dtype=object)
            infra_export[f"{prefix_ap_ue}_rx_ue_ids"] = np.asarray(ap_ue["rx_ue_ids"], dtype=object)
            infra_export[f"{prefix_ap_ue}_times_s"] = ap_ue["times_s"]
            infra_export[f"{prefix_ap_ue}_sinr_linear"] = ap_ue["sinr_linear"]
            infra_export[f"{prefix_ap_ue}_best_sinr_db"] = ap_ue["best_sinr_db"]
            infra_export[f"{prefix_ap_ue}_desired_power_w"] = ap_ue["desired_power_w"]
            infra_export[f"{prefix_ap_ue}_interference_power_w"] = ap_ue["interference_power_w"]
            infra_export[f"{prefix_ap_ue}_noise_power_w"] = ap_ue["noise_power_w"]
            infra_export[f"{prefix_ap_ue}_link_power_w"] = ap_ue["link_power_w"]
            infra_export[f"{prefix_ap_ap}_tx_site_ids"] = np.asarray(ap_ap["tx_site_ids"], dtype=object)
            infra_export[f"{prefix_ap_ap}_rx_site_ids"] = np.asarray(ap_ap["rx_site_ids"], dtype=object)
            infra_export[f"{prefix_ap_ap}_link_power_w"] = ap_ap["link_power_w"]
            _update_prefixed_export(infra_export, prefix_ap_ue, ap_ue, ("cfr", "cir", "tau"))
            _update_prefixed_export(infra_export, prefix_ap_ap, ap_ap, ("cfr", "cir", "tau"))
        if csi_exports_enabled:
            np.savez_compressed(output_dir / "infra_csi_snapshots.npz", **infra_export)
            logger.info("Wrote infrastructure CSI export to %s", output_dir / "infra_csi_snapshots.npz")
        else:
            _remove_artifacts([output_dir / "infra_csi_snapshots.npz"])

        baseline_radio_map = None
        best_radio_map = None
        if radio_map_enabled:
            logger.info("Computing baseline and best-strategy coverage maps")
            baseline_radio_map = runner_by_strategy["distributed_fixed"].compute_radio_map(
                baseline_strategy.selected_sites,
                config.coverage,
            )
            best_radio_map = runner_by_strategy[best_strategy_name].compute_radio_map(
                best_strategy.selected_sites,
                config.coverage,
            )
            np.savez_compressed(
                output_dir / "fixed_coverage_map.npz",
                path_gain=baseline_radio_map["path_gain"],
                rss=baseline_radio_map["rss"],
                sinr=baseline_radio_map["sinr"],
                best_sinr_db=baseline_radio_map["best_sinr_db"],
                cell_centers=baseline_radio_map["cell_centers"],
            )
            np.savez_compressed(
                output_dir / "coverage_map.npz",
                path_gain=best_radio_map["path_gain"],
                rss=best_radio_map["rss"],
                sinr=best_radio_map["sinr"],
                best_sinr_db=best_radio_map["best_sinr_db"],
                cell_centers=best_radio_map["cell_centers"],
            )
        else:
            _remove_artifacts(
                [
                    output_dir / "coverage_map.npz",
                    output_dir / "fixed_coverage_map.npz",
                ]
            )
            logger.info("Coverage-map computation disabled; skipping coverage-map exports")
        scenario_progress.update(1)

        distributed_selected_union = set().union(
            *(
                artifact.selected_candidate_union
                for name, artifact in strategy_results.items()
                if name != "central_massive_mimo"
            )
        )
        write_candidate_sites(
            output_dir / "candidate_ap_positions.csv",
            movable_candidate_sites,
            selected_ids=distributed_selected_union,
        )
        write_candidate_sites(
            output_dir / "central_ap_rooftop_candidates.csv",
            rooftop_candidates,
            selected_ids=strategy_results["central_massive_mimo"].selected_candidate_union,
        )
        for artifact in strategy_results.values():
            _write_strategy_site_csv(output_dir, artifact)
            _write_mobile_ap_schedule_csv(output_dir / f"{artifact.name}_schedule.csv", artifact.schedule_rows)
        _remove_artifacts(_legacy_output_paths(output_dir))
        _write_strategy_comparison_csv(output_dir / "strategy_comparison.csv", strategy_results)
        logger.info("Wrote placement and comparison data to %s", output_dir)
        scenario_progress.update(1)

        _write_user_sinr_artifacts(
            output_dir,
            trajectory,
            {name: strategy_results[name].ap_ue for name in strategy_names},
            include_plots=False,
        )
        logger.info("Wrote user SINR data artifacts to %s", output_dir)
        scenario_progress.update(1)

        summary = {
            "scenario": config.name,
            "compute_device": runtime_info["device"],
            "mitsuba_variant": runtime_info["variant"],
            "radio_map_enabled": radio_map_enabled,
            "csi_exports_enabled": csi_exports_enabled,
            "csi_cache_enabled": csi_cache_enabled,
            "baseline_strategy": "distributed_fixed",
            "best_strategy": best_strategy_name,
            "scene_animation_strategy": animation_strategy_name,
            "scene_animation_with_central_massive_mimo": "",
            "scene_animation_speedup": float(config.outputs.scene_animation_speedup),
            "scene_context": scene_context,
            "window_interval_s": config.placement.window_interval_s,
            "candidate_site_ids": movable_candidate_ids,
            "central_rooftop_candidate_ids": [site.site_id for site in rooftop_candidates],
            "selected_site_ids": [site.site_id for site in best_strategy.selected_sites],
            "strategies": {
                name: {
                    "selected_site_ids": [site.site_id for site in artifact.selected_sites],
                    "movable_site_ids": [site.site_id for site in artifact.movable_sites],
                    "final_candidate_ids": list(artifact.final_candidate_ids),
                    "score": float(artifact.score.score),
                    "outage": float(artifact.score.outage),
                    "percentile_10_db": float(artifact.score.percentile_10_db),
                    "peer_tiebreak": float(artifact.score.peer_tiebreak),
                    "capped": bool(artifact.capped),
                    "evaluated_combinations": int(artifact.evaluated_combinations),
                    **artifact.details,
                }
                for name, artifact in strategy_results.items()
            },
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info(
            "Scenario '%s' complete; best strategy %s scored %.3f",
            config.name,
            best_strategy_name,
            best_strategy.score.score,
        )
        return summary
