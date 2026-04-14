"""Simple ASCII PLY mesh generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import triangulate


@dataclass(slots=True)
class MeshData:
    vertices: np.ndarray
    faces: np.ndarray


class MeshBuilder:
    """Accumulates vertices/faces while deduplicating repeated vertices."""

    def __init__(self) -> None:
        self._vertices: list[tuple[float, float, float]] = []
        self._faces: list[tuple[int, int, int]] = []
        self._index: dict[tuple[float, float, float], int] = {}

    def add_vertex(self, vertex: tuple[float, float, float]) -> int:
        key = tuple(round(v, 6) for v in vertex)
        if key in self._index:
            return self._index[key]
        idx = len(self._vertices)
        self._vertices.append(tuple(float(v) for v in vertex))
        self._index[key] = idx
        return idx

    def add_face(self, a: int, b: int, c: int) -> None:
        if len({a, b, c}) == 3:
            self._faces.append((a, b, c))

    def to_mesh(self) -> MeshData:
        return MeshData(
            vertices=np.asarray(self._vertices, dtype=float),
            faces=np.asarray(self._faces, dtype=int),
        )


def _iter_rings(polygon: Polygon) -> list[list[tuple[float, float]]]:
    rings = [list(polygon.exterior.coords)]
    rings.extend(list(ring.coords) for ring in polygon.interiors)
    return rings


def build_ground_mesh(boundary: Polygon) -> MeshData:
    builder = MeshBuilder()
    for triangle in triangulate(boundary):
        if not triangle.representative_point().covered_by(boundary):
            continue
        coords = list(triangle.exterior.coords)[:3]
        face = [builder.add_vertex((x, y, 0.0)) for x, y in coords]
        builder.add_face(*face)
    return builder.to_mesh()


def build_roof_mesh(footprint: Polygon, height_m: float) -> MeshData:
    builder = MeshBuilder()
    for triangle in triangulate(footprint):
        if not triangle.representative_point().covered_by(footprint):
            continue
        coords = list(triangle.exterior.coords)[:3]
        face = [builder.add_vertex((x, y, height_m)) for x, y in coords]
        builder.add_face(*face)
    return builder.to_mesh()


def build_wall_mesh(footprint: Polygon, height_m: float) -> MeshData:
    builder = MeshBuilder()
    for ring in _iter_rings(footprint):
        for start, end in zip(ring[:-1], ring[1:]):
            sx, sy = start
            ex, ey = end
            v0 = builder.add_vertex((sx, sy, 0.0))
            v1 = builder.add_vertex((ex, ey, 0.0))
            v2 = builder.add_vertex((ex, ey, height_m))
            v3 = builder.add_vertex((sx, sy, height_m))
            builder.add_face(v0, v1, v2)
            builder.add_face(v0, v2, v3)
    return builder.to_mesh()


def write_ascii_ply(path: str | Path, mesh: MeshData) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(mesh.vertices)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write(f"element face {len(mesh.faces)}\n")
        handle.write("property list uchar int vertex_indices\n")
        handle.write("end_header\n")
        for x, y, z in mesh.vertices:
            handle.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
        for a, b, c in mesh.faces:
            handle.write(f"3 {a} {b} {c}\n")
