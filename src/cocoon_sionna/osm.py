"""Minimal OSM download and parsing utilities."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

import networkx as nx
import requests
from shapely.geometry import LineString, Point, Polygon

from .geo import LocalFrame, boundary_entry_distance, line_to_local, parse_osm_height, polygon_to_local, sanitize_name

logger = logging.getLogger(__name__)

WALKABLE_HIGHWAYS = {
    "cycleway",
    "footway",
    "living_street",
    "path",
    "pedestrian",
    "residential",
    "service",
    "steps",
    "track",
    "unclassified",
}


@dataclass(slots=True)
class OSMBuilding:
    name: str
    height_m: float
    polygon_local: Polygon
    tags: dict[str, str]


@dataclass(slots=True)
class ParsedOSM:
    nodes: dict[int, tuple[float, float]]
    ways: dict[int, dict[str, Any]]
    relations: list[dict[str, Any]]


class OverpassClient:
    def __init__(self, url: str) -> None:
        self.url = url

    def fetch(self, boundary_lonlat: Polygon) -> ParsedOSM:
        min_lon, min_lat, max_lon, max_lat = boundary_lonlat.bounds
        logger.info(
            "Fetching OSM data for bbox lon/lat=(%.6f, %.6f, %.6f, %.6f)",
            min_lon,
            min_lat,
            max_lon,
            max_lat,
        )
        query = f"""
[out:json][timeout:120];
(
  way["building"]({min_lat},{min_lon},{max_lat},{max_lon});
  relation["building"]({min_lat},{min_lon},{max_lat},{max_lon});
  way["highway"]({min_lat},{min_lon},{max_lat},{max_lon});
);
(._;>;);
out body;
"""
        urls = [
            self.url,
            "https://overpass.kumi.systems/api/interpreter",
            "https://lz4.overpass-api.de/api/interpreter",
        ]
        errors: list[str] = []
        payload = None
        for url in urls:
            try:
                logger.info("Querying Overpass endpoint %s", url)
                response = requests.post(url, data=query.encode("utf-8"), timeout=180)
                response.raise_for_status()
                payload = response.json()
                logger.info("Overpass request succeeded via %s", url)
                break
            except requests.RequestException as exc:
                logger.warning("Overpass request failed via %s: %s", url, exc)
                errors.append(f"{url}: {exc}")
        if payload is None:
            raise RuntimeError("All Overpass endpoints failed: " + " | ".join(errors))
        nodes: dict[int, tuple[float, float]] = {}
        ways: dict[int, dict[str, Any]] = {}
        relations: list[dict[str, Any]] = []
        for element in payload["elements"]:
            kind = element["type"]
            if kind == "node":
                nodes[int(element["id"])] = (float(element["lon"]), float(element["lat"]))
            elif kind == "way":
                ways[int(element["id"])] = element
            elif kind == "relation":
                relations.append(element)
        logger.info("Parsed OSM payload: %d nodes, %d ways, %d relations", len(nodes), len(ways), len(relations))
        return ParsedOSM(nodes=nodes, ways=ways, relations=relations)


def _way_polygon_lonlat(way: dict[str, Any], nodes: dict[int, tuple[float, float]]) -> Polygon | None:
    refs = way.get("nodes", [])
    if len(refs) < 4 or refs[0] != refs[-1]:
        return None
    coords = [nodes.get(int(ref)) for ref in refs]
    if any(coord is None for coord in coords):
        return None
    polygon = Polygon(coords)
    if polygon.is_empty or polygon.area <= 0:
        return None
    return polygon


def _assemble_relation_outer_ring(
    relation: dict[str, Any],
    ways: dict[int, dict[str, Any]],
    nodes: dict[int, tuple[float, float]],
) -> Polygon | None:
    segments: list[list[tuple[float, float]]] = []
    for member in relation.get("members", []):
        if member.get("type") != "way" or member.get("role") not in ("", "outer"):
            continue
        way = ways.get(int(member["ref"]))
        if not way:
            continue
        coords = [nodes.get(int(ref)) for ref in way.get("nodes", [])]
        if any(coord is None for coord in coords) or len(coords) < 2:
            continue
        segments.append(coords)
    if not segments:
        return None

    ring = list(segments.pop(0))
    while segments:
        last = ring[-1]
        progress = False
        for idx, segment in enumerate(segments):
            if last == segment[0]:
                ring.extend(segment[1:])
                segments.pop(idx)
                progress = True
                break
            if last == segment[-1]:
                ring.extend(reversed(segment[:-1]))
                segments.pop(idx)
                progress = True
                break
        if not progress:
            return None
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    polygon = Polygon(ring)
    if polygon.is_empty or polygon.area <= 0:
        return None
    return polygon


def extract_buildings(
    parsed: ParsedOSM,
    boundary_lonlat: Polygon,
    frame: LocalFrame,
    default_height_m: float,
    meters_per_level: float,
) -> list[OSMBuilding]:
    buildings: list[OSMBuilding] = []
    seen_names: set[str] = set()

    def append_polygon(polygon_lonlat: Polygon, tags: dict[str, str], fallback_name: str) -> None:
        clipped = polygon_lonlat.intersection(boundary_lonlat)
        if clipped.is_empty:
            return
        if clipped.geom_type == "MultiPolygon":
            polygons = list(clipped.geoms)
        else:
            polygons = [clipped]
        for index, polygon in enumerate(polygons):
            if polygon.is_empty or polygon.area <= 0:
                continue
            polygon_local = polygon_to_local(polygon, frame).buffer(0)
            if polygon_local.is_empty or polygon_local.area <= 0:
                continue
            name = sanitize_name(tags.get("name", fallback_name))
            if index:
                name = f"{name}_{index}"
            if name in seen_names:
                suffix = len(seen_names)
                name = f"{name}_{suffix}"
            seen_names.add(name)
            buildings.append(
                OSMBuilding(
                    name=name,
                    height_m=parse_osm_height(tags, default_height_m, meters_per_level),
                    polygon_local=polygon_local,
                    tags=tags,
                )
            )

    for way_id, way in parsed.ways.items():
        tags = way.get("tags", {})
        if "building" not in tags:
            continue
        polygon = _way_polygon_lonlat(way, parsed.nodes)
        if polygon is not None:
            append_polygon(polygon, tags, f"building_{way_id}")

    for relation in parsed.relations:
        tags = relation.get("tags", {})
        if "building" not in tags:
            continue
        polygon = _assemble_relation_outer_ring(relation, parsed.ways, parsed.nodes)
        if polygon is not None:
            append_polygon(polygon, tags, f"building_rel_{relation['id']}")

    logger.info("Extracted %d clipped building footprints", len(buildings))
    return buildings


def extract_walk_graph(
    parsed: ParsedOSM,
    boundary_local: Polygon,
    frame: LocalFrame,
    building_polygons_local: list[Polygon],
    entry_buffer_m: float,
) -> nx.Graph:
    graph = nx.Graph()
    for way in parsed.ways.values():
        tags = way.get("tags", {})
        if tags.get("highway") not in WALKABLE_HIGHWAYS:
            continue
        refs = [int(ref) for ref in way.get("nodes", [])]
        for start_ref, end_ref in zip(refs[:-1], refs[1:]):
            start_lonlat = parsed.nodes.get(start_ref)
            end_lonlat = parsed.nodes.get(end_ref)
            if start_lonlat is None or end_lonlat is None:
                continue
            line_local = line_to_local(LineString([start_lonlat, end_lonlat]), frame)
            start_local = Point(line_local.coords[0])
            end_local = Point(line_local.coords[1])
            midpoint = line_local.interpolate(0.5, normalized=True)
            if not boundary_local.covers(start_local) or not boundary_local.covers(end_local):
                continue
            if any(polygon.covers(midpoint) for polygon in building_polygons_local):
                continue

            for node_id, point in ((start_ref, start_local), (end_ref, end_local)):
                graph.add_node(
                    node_id,
                    x=float(point.x),
                    y=float(point.y),
                    entry_candidate=boundary_entry_distance(point, boundary_local) <= entry_buffer_m,
                )
            graph.add_edge(start_ref, end_ref, length=float(line_local.length))

    isolates = list(nx.isolates(graph))
    if isolates:
        graph.remove_nodes_from(isolates)
    logger.info("Extracted walk graph with %d nodes and %d edges", graph.number_of_nodes(), graph.number_of_edges())
    return graph
