"""Geometry and coordinate helpers."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from pyproj import CRS, Transformer
from shapely.geometry import LineString, Point, Polygon, shape


def sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("_") or "element"


def parse_osm_height(
    tags: dict[str, str],
    default_height_m: float,
    meters_per_level: float,
) -> float:
    raw_height = tags.get("height")
    if raw_height:
        match = re.search(r"[-+]?[0-9]*\.?[0-9]+", raw_height)
        if match:
            return max(float(match.group(0)), 1.0)
    raw_levels = tags.get("building:levels")
    if raw_levels:
        match = re.search(r"[-+]?[0-9]*\.?[0-9]+", raw_levels)
        if match:
            return max(float(match.group(0)) * meters_per_level, 1.0)
    return default_height_m


def utm_crs_for_lonlat(lon: float, lat: float) -> CRS:
    zone = int((lon + 180.0) / 6.0) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)


@dataclass(slots=True)
class LocalFrame:
    origin_lon: float
    origin_lat: float
    epsg: int
    to_local: Transformer
    to_wgs84: Transformer

    @classmethod
    def from_lonlat(cls, lon: float, lat: float) -> "LocalFrame":
        utm = utm_crs_for_lonlat(lon, lat)
        to_utm = Transformer.from_crs("EPSG:4326", utm, always_xy=True)
        to_wgs84 = Transformer.from_crs(utm, "EPSG:4326", always_xy=True)
        return cls(
            origin_lon=lon,
            origin_lat=lat,
            epsg=int(utm.to_epsg()),
            to_local=to_utm,
            to_wgs84=to_wgs84,
        )

    def lonlat_to_local_xy(self, lon: float, lat: float) -> tuple[float, float]:
        x0, y0 = self.to_local.transform(self.origin_lon, self.origin_lat)
        x, y = self.to_local.transform(lon, lat)
        return x - x0, y - y0

    def local_xy_to_lonlat(self, x: float, y: float) -> tuple[float, float]:
        x0, y0 = self.to_local.transform(self.origin_lon, self.origin_lat)
        return self.to_wgs84.transform(x + x0, y + y0)


def load_geojson_polygon(path: str | Path) -> Polygon:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if raw.get("type") == "FeatureCollection":
        geometry = raw["features"][0]["geometry"]
    elif raw.get("type") == "Feature":
        geometry = raw["geometry"]
    else:
        geometry = raw
    polygon = shape(geometry)
    if not isinstance(polygon, Polygon):
        raise TypeError(f"Expected Polygon geometry in {path}")
    return polygon


def osm_export_bbox_polygon(
    west: float,
    south: float,
    east: float,
    north: float,
) -> Polygon:
    if west >= east:
        raise ValueError(f"Expected west < east, got {west} >= {east}")
    if south >= north:
        raise ValueError(f"Expected south < north, got {south} >= {north}")
    return Polygon(
        [
            (west, south),
            (east, south),
            (east, north),
            (west, north),
            (west, south),
        ]
    )


def polygon_to_local(polygon: Polygon, frame: LocalFrame) -> Polygon:
    exterior = [frame.lonlat_to_local_xy(x, y) for x, y in polygon.exterior.coords]
    interiors = [
        [frame.lonlat_to_local_xy(x, y) for x, y in ring.coords]
        for ring in polygon.interiors
    ]
    return Polygon(exterior, interiors)


def line_to_local(line: LineString, frame: LocalFrame) -> LineString:
    return LineString([frame.lonlat_to_local_xy(x, y) for x, y in line.coords])


def to_point_array(coords: Iterable[tuple[float, float]], z: float) -> np.ndarray:
    xy = np.asarray(list(coords), dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("Expected Nx2 coordinates")
    zz = np.full((xy.shape[0], 1), z, dtype=float)
    return np.hstack([xy, zz])


def db10(value: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    return 10.0 * np.log10(np.clip(value, floor, None))


def inward_yaw_deg(x: float, y: float) -> float:
    return math.degrees(math.atan2(-y, -x))


def boundary_entry_distance(point: Point, polygon: Polygon) -> float:
    return point.distance(polygon.boundary)
