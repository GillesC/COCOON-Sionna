"""Candidate AP/CSP site handling."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geo import inward_yaw_deg
from .mobility import Trajectory


@dataclass(slots=True)
class CandidateSite:
    site_id: str
    x_m: float
    y_m: float
    z_m: float
    yaw_deg: float
    pitch_deg: float
    mount_type: str
    enabled: bool = True
    source: str = "file"

    @property
    def position(self) -> tuple[float, float, float]:
        return (self.x_m, self.y_m, self.z_m)


def load_candidate_sites(path: str | Path) -> list[CandidateSite]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            CandidateSite(
                site_id=row["site_id"],
                x_m=float(row["x_m"]),
                y_m=float(row["y_m"]),
                z_m=float(row["z_m"]),
                yaw_deg=float(row.get("yaw_deg", 0.0)),
                pitch_deg=float(row.get("pitch_deg", 0.0)),
                mount_type=row.get("mount_type", "unknown"),
                enabled=str(row.get("enabled", "true")).lower() in {"1", "true", "yes"},
                source=row.get("source", "file"),
            )
            for row in reader
        ]


def write_candidate_sites(path: str | Path, sites: list[CandidateSite], selected_ids: set[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    selected_ids = selected_ids or set()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "site_id",
                "x_m",
                "y_m",
                "z_m",
                "yaw_deg",
                "pitch_deg",
                "mount_type",
                "enabled",
                "source",
                "selected",
            ]
        )
        for site in sites:
            writer.writerow(
                [
                    site.site_id,
                    site.x_m,
                    site.y_m,
                    site.z_m,
                    site.yaw_deg,
                    site.pitch_deg,
                    site.mount_type,
                    site.enabled,
                    site.source,
                    site.site_id in selected_ids,
                ]
            )


def _farthest_point_indices(xy: np.ndarray, count: int) -> list[int]:
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("Expected Nx2 candidate positions")
    if count <= 0 or len(xy) == 0:
        return []
    count = min(count, len(xy))
    centroid = np.mean(xy, axis=0)
    first = int(np.argmax(np.linalg.norm(xy - centroid[None, :], axis=1)))
    selected = [first]
    min_distances = np.linalg.norm(xy - xy[first][None, :], axis=1)
    while len(selected) < count:
        next_index = int(np.argmax(min_distances))
        if next_index in selected:
            break
        selected.append(next_index)
        min_distances = np.minimum(min_distances, np.linalg.norm(xy - xy[next_index][None, :], axis=1))
    return selected


def select_farthest_sites(sites: list[CandidateSite], count: int) -> list[CandidateSite]:
    enabled_sites = [site for site in sites if site.enabled]
    if count >= len(enabled_sites):
        return enabled_sites
    xy = np.asarray([[site.x_m, site.y_m] for site in enabled_sites], dtype=float)
    return [enabled_sites[index] for index in _farthest_point_indices(xy, count)]


def _wall_normal(start_xy: np.ndarray, end_xy: np.ndarray, is_ccw: bool) -> np.ndarray:
    direction = end_xy - start_xy
    length = max(float(np.linalg.norm(direction)), 1e-9)
    dx, dy = direction / length
    if is_ccw:
        normal = np.array([dy, -dx], dtype=float)
    else:
        normal = np.array([-dy, dx], dtype=float)
    return normal / max(float(np.linalg.norm(normal)), 1e-9)


def generate_wall_candidate_sites(
    metadata: dict[str, object] | None,
    spacing_m: float,
    mount_height_m: float,
    corner_clearance_m: float,
    mount_offset_m: float,
    min_spacing_m: float,
) -> list[CandidateSite]:
    if metadata is None:
        return []

    sites: list[CandidateSite] = []
    accepted_xy: list[np.ndarray] = []
    buildings = metadata.get("buildings", [])
    building_items = buildings if isinstance(buildings, list) else []
    for building in building_items:
        polygon_coords = np.asarray(building.get("polygon_local", []), dtype=float)
        if polygon_coords.ndim != 2 or polygon_coords.shape[0] < 4 or polygon_coords.shape[1] != 2:
            continue
        ring = polygon_coords[:-1]
        signed_area = 0.5 * np.sum(ring[:, 0] * np.roll(ring[:, 1], -1) - np.roll(ring[:, 0], -1) * ring[:, 1])
        is_ccw = signed_area >= 0.0
        building_name = str(building.get("name", f"building_{len(sites):03d}"))

        for edge_index in range(len(ring)):
            start_xy = ring[edge_index]
            end_xy = ring[(edge_index + 1) % len(ring)]
            edge_length = float(np.linalg.norm(end_xy - start_xy))
            if edge_length < 1e-6:
                continue
            margin = min(float(corner_clearance_m), max(0.0, 0.5 * edge_length - 1e-6))
            usable_length = edge_length - 2.0 * margin
            if usable_length <= 0.0:
                sample_distances = np.array([0.5 * edge_length], dtype=float)
            else:
                steps = max(1, int(math.floor(usable_length / max(spacing_m, 1e-6))))
                sample_distances = np.linspace(margin, edge_length - margin, num=steps + 1, dtype=float)

            normal = _wall_normal(start_xy, end_xy, is_ccw)
            for sample_index, distance in enumerate(sample_distances):
                alpha = float(distance / edge_length)
                point_xy = start_xy + alpha * (end_xy - start_xy)
                mount_xy = point_xy + mount_offset_m * normal
                if accepted_xy:
                    distances = np.linalg.norm(np.asarray(accepted_xy, dtype=float) - mount_xy[None, :], axis=1)
                    if np.min(distances) < min_spacing_m:
                        continue
                yaw_deg = math.degrees(math.atan2(normal[1], normal[0]))
                site = CandidateSite(
                    site_id=f"wall_{building_name}_{edge_index:02d}_{sample_index:02d}",
                    x_m=float(mount_xy[0]),
                    y_m=float(mount_xy[1]),
                    z_m=float(mount_height_m),
                    yaw_deg=float(yaw_deg),
                    pitch_deg=-10.0,
                    mount_type="facade",
                    enabled=True,
                    source="wall_metadata",
                )
                sites.append(site)
                accepted_xy.append(mount_xy)
    return sites


def augment_with_trajectory_sites(
    base_sites: list[CandidateSite],
    trajectory: Trajectory,
    min_spacing_m: float,
    max_new_sites: int,
    pitch_deg: float = -10.0,
) -> list[CandidateSite]:
    sites = list(base_sites)
    accepted_xy = np.asarray([[site.x_m, site.y_m] for site in sites], dtype=float) if sites else np.empty((0, 2))
    seen = 0
    for point in trajectory.positions_m[..., :2].reshape(-1, 2):
        if accepted_xy.size and np.min(np.linalg.norm(accepted_xy - point, axis=1)) < min_spacing_m:
            continue
        site = CandidateSite(
            site_id=f"ue_candidate_{seen:03d}",
            x_m=float(point[0]),
            y_m=float(point[1]),
            z_m=float(trajectory.positions_m[0, 0, 2]),
            yaw_deg=inward_yaw_deg(float(point[0]), float(point[1])),
            pitch_deg=pitch_deg,
            mount_type="ue_candidate",
            enabled=True,
            source="trajectory",
        )
        sites.append(site)
        accepted_xy = np.vstack([accepted_xy, point]) if accepted_xy.size else point[None, :]
        seen += 1
        if seen >= max_new_sites:
            break
    return sites
