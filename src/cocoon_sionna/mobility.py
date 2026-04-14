"""Pedestrian mobility generation over a walk graph."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from .config import MobilityConfig

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Trajectory:
    times_s: np.ndarray
    ue_ids: list[str]
    positions_m: np.ndarray
    velocities_mps: np.ndarray

    def write_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time_s", "ue_id", "x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps"])
            for t_idx, time_s in enumerate(self.times_s):
                for u_idx, ue_id in enumerate(self.ue_ids):
                    position = self.positions_m[t_idx, u_idx]
                    velocity = self.velocities_mps[t_idx, u_idx]
                    writer.writerow([time_s, ue_id, *position.tolist(), *velocity.tolist()])


def load_graph_json(path: str | Path) -> nx.Graph:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    graph = nx.Graph()
    for node in raw["nodes"]:
        graph.add_node(
            int(node["id"]),
            x=float(node["x"]),
            y=float(node["y"]),
            entry_candidate=bool(node.get("entry_candidate", False)),
        )
    for edge in raw["edges"]:
        graph.add_edge(int(edge["u"]), int(edge["v"]), length=float(edge["length"]))
    logger.info("Loaded walk graph with %d nodes and %d edges", graph.number_of_nodes(), graph.number_of_edges())
    return graph


def _free_space_geometry(
    metadata: dict[str, Any],
    clearance_m: float,
):
    boundary = Polygon(metadata["boundary_local"])
    building_geometries = []
    for item in metadata.get("buildings", []):
        polygon = Polygon(item["polygon_local"])
        if polygon.is_empty:
            continue
        building_geometries.append(polygon.buffer(clearance_m))
    if building_geometries:
        return boundary.difference(unary_union(building_geometries)).buffer(0)
    return boundary


def _connect_visible_neighbors(
    graph: nx.Graph,
    node_ids: list[int],
    radius_m: float,
    free_space,
) -> None:
    if radius_m <= 0.0:
        return
    coordinates = {
        node_id: np.array([float(graph.nodes[node_id]["x"]), float(graph.nodes[node_id]["y"])], dtype=float)
        for node_id in node_ids
    }
    for index, node_id in enumerate(node_ids):
        start_xy = coordinates[node_id]
        for other_id in node_ids[index + 1 :]:
            end_xy = coordinates[other_id]
            distance = float(np.linalg.norm(start_xy - end_xy))
            if distance <= 1e-6 or distance > radius_m:
                continue
            segment = LineString([tuple(start_xy), tuple(end_xy)])
            if free_space.covers(segment):
                graph.add_edge(node_id, other_id, length=distance)


def augment_graph_with_open_area(
    graph: nx.Graph,
    mobility: MobilityConfig,
    metadata: dict[str, Any] | None,
) -> nx.Graph:
    if not mobility.allow_open_area or metadata is None or "boundary_local" not in metadata:
        return graph

    spacing = max(float(mobility.open_area_grid_spacing_m), 1.0)
    radius = max(float(mobility.open_area_connection_radius_m), spacing * 1.01)
    clearance = max(float(mobility.open_area_clearance_m), 0.0)
    boundary = Polygon(metadata["boundary_local"])
    free_space = _free_space_geometry(metadata, clearance)
    if free_space.is_empty:
        return graph

    augmented = graph.copy()
    next_node_id = (max((int(node_id) for node_id in augmented.nodes), default=0) + 1) if augmented.number_of_nodes() else 1

    existing_node_ids = list(augmented.nodes)
    existing_node_points = {
        node_id: Point(float(augmented.nodes[node_id]["x"]), float(augmented.nodes[node_id]["y"]))
        for node_id in existing_node_ids
    }

    free_node_ids: list[int] = []
    min_x, min_y, max_x, max_y = free_space.bounds
    x_values = np.arange(min_x, max_x + 0.5 * spacing, spacing, dtype=float)
    y_values = np.arange(min_y, max_y + 0.5 * spacing, spacing, dtype=float)
    for x in x_values:
        for y in y_values:
            point = Point(float(x), float(y))
            if not free_space.covers(point):
                continue
            if any(point.distance(existing_point) < 0.35 * spacing for existing_point in existing_node_points.values()):
                continue
            augmented.add_node(
                next_node_id,
                x=float(x),
                y=float(y),
                entry_candidate=boundary.boundary.distance(point) <= max(spacing, radius),
            )
            free_node_ids.append(next_node_id)
            next_node_id += 1

    if free_node_ids:
        _connect_visible_neighbors(augmented, free_node_ids, radius, free_space)
        _connect_visible_neighbors(augmented, existing_node_ids + free_node_ids, radius, free_space)

    isolates = list(nx.isolates(augmented))
    if isolates:
        augmented.remove_nodes_from(isolates)
    logger.info(
        "Augmented mobility graph with %d off-path nodes; graph now has %d nodes and %d edges",
        len(free_node_ids),
        augmented.number_of_nodes(),
        augmented.number_of_edges(),
    )
    return augmented


def _path_segments(graph: nx.Graph, route: list[int]) -> list[tuple[np.ndarray, np.ndarray, float]]:
    segments: list[tuple[np.ndarray, np.ndarray, float]] = []
    for u, v in zip(route[:-1], route[1:]):
        start = np.array([graph.nodes[u]["x"], graph.nodes[u]["y"]], dtype=float)
        end = np.array([graph.nodes[v]["x"], graph.nodes[v]["y"]], dtype=float)
        length = float(graph.edges[u, v]["length"])
        segments.append((start, end, max(length, 1e-6)))
    return segments


def _edge_key(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u <= v else (v, u)


def _node_xy(graph: nx.Graph, node: int) -> np.ndarray:
    return np.array([graph.nodes[node]["x"], graph.nodes[node]["y"]], dtype=float)


def _spread_nodes(graph: nx.Graph, rng: np.random.Generator, nodes: list[int]) -> list[int]:
    unique_nodes = [int(node) for node in dict.fromkeys(nodes)]
    if len(unique_nodes) <= 1:
        return unique_nodes

    selected = [int(rng.choice(unique_nodes))]
    remaining = {node for node in unique_nodes if node != selected[0]}
    coordinates = {node: _node_xy(graph, node) for node in unique_nodes}

    while remaining:
        best_nodes: list[int] = []
        best_score = -np.inf
        for node in remaining:
            xy = coordinates[node]
            min_distance = min(float(np.linalg.norm(xy - coordinates[chosen])) for chosen in selected)
            if min_distance > best_score + 1e-9:
                best_score = min_distance
                best_nodes = [node]
            elif abs(min_distance - best_score) <= 1e-9:
                best_nodes.append(node)
        chosen = int(rng.choice(best_nodes))
        selected.append(chosen)
        remaining.remove(chosen)
    return selected


def _choose_start_nodes(
    graph: nx.Graph,
    rng: np.random.Generator,
    entry_nodes: list[int],
    num_users: int,
) -> list[int]:
    all_nodes = [int(node) for node in graph.nodes]
    interior_nodes = [node for node in all_nodes if node not in set(entry_nodes)]

    primary_nodes = interior_nodes or entry_nodes or all_nodes
    spread_nodes = _spread_nodes(graph, rng, primary_nodes)
    if not spread_nodes:
        return []

    if len(spread_nodes) < num_users:
        supplemental_nodes = [node for node in _spread_nodes(graph, rng, all_nodes) if node not in spread_nodes]
        spread_nodes.extend(supplemental_nodes)

    if num_users <= len(spread_nodes):
        return spread_nodes[:num_users]
    return [spread_nodes[index % len(spread_nodes)] for index in range(num_users)]


def _sample_user_profile(mobility: MobilityConfig, rng: np.random.Generator) -> tuple[str, float]:
    pedestrian_range = mobility.pedestrian_speed_mps_range or mobility.speed_mps_range
    bike_enabled = mobility.bike_speed_mps_range is not None and mobility.bike_fraction > 0.0
    if bike_enabled and rng.random() < mobility.bike_fraction:
        low, high = mobility.bike_speed_mps_range
        return "bike", float(rng.uniform(low, high))
    low, high = pedestrian_range
    return "pedestrian", float(rng.uniform(low, high))


def _sample_step_speed(preferred_speed: float, mobility: MobilityConfig, rng: np.random.Generator) -> float:
    variation = max(0.0, float(mobility.speed_variation_fraction))
    if variation <= 0.0:
        return preferred_speed
    scale = rng.uniform(max(0.1, 1.0 - variation), 1.0 + variation)
    return max(0.0, preferred_speed * float(scale))


def _choose_next_node(
    graph: nx.Graph,
    rng: np.random.Generator,
    current: int,
    previous: int | None,
    user_edge_counts: dict[tuple[int, int], int],
    shared_edge_counts: dict[tuple[int, int], int],
) -> int | None:
    neighbors = [int(node) for node in graph.neighbors(current)]
    if not neighbors:
        return None

    weights = []
    for neighbor in neighbors:
        edge_key = _edge_key(current, neighbor)
        weight = 1.0 / (1.0 + 0.8 * shared_edge_counts.get(edge_key, 0) + 1.6 * user_edge_counts.get(edge_key, 0))
        if previous is not None and neighbor == previous and len(neighbors) > 1:
            weight *= 0.15
        weights.append(weight)

    probabilities = np.asarray(weights, dtype=float)
    probabilities /= np.sum(probabilities)
    return int(rng.choice(neighbors, p=probabilities))


def generate_trajectory(
    graph: nx.Graph,
    mobility: MobilityConfig,
    ue_height_m: float,
    metadata: dict[str, Any] | None = None,
) -> Trajectory:
    graph = augment_graph_with_open_area(graph, mobility, metadata)
    if graph.number_of_nodes() == 0:
        raise ValueError("Mobility graph is empty")
    logger.info(
        "Generating trajectory for %d UEs over %.1fs with %.1fs steps",
        mobility.num_users,
        mobility.duration_s,
        mobility.step_s,
    )
    rng = np.random.default_rng(mobility.seed)
    times = np.arange(0.0, mobility.duration_s + 1e-9, mobility.step_s, dtype=float)
    ue_ids = [f"ue_{index:03d}" for index in range(mobility.num_users)]
    positions = np.zeros((len(times), mobility.num_users, 3), dtype=float)

    entry_nodes = [node for node, attrs in graph.nodes(data=True) if attrs.get("entry_candidate")]
    if not entry_nodes:
        entry_nodes = list(graph.nodes)

    start_nodes = _choose_start_nodes(graph, rng, entry_nodes, mobility.num_users)
    shared_edge_counts: dict[tuple[int, int], int] = {}
    users: list[dict[str, object]] = []
    for start_node in start_nodes:
        mode, preferred_speed = _sample_user_profile(mobility, rng)
        user_edge_counts: dict[tuple[int, int], int] = {}
        next_node = _choose_next_node(graph, rng, start_node, None, user_edge_counts, shared_edge_counts)
        if next_node is not None:
            edge_key = _edge_key(start_node, next_node)
            user_edge_counts[edge_key] = user_edge_counts.get(edge_key, 0) + 1
            shared_edge_counts[edge_key] = shared_edge_counts.get(edge_key, 0) + 1
        users.append(
            {
                "current_node": start_node,
                "previous_node": None,
                "next_node": next_node,
                "segment_progress": 0.0,
                "preferred_speed": preferred_speed,
                "mode": mode,
                "user_edge_counts": user_edge_counts,
                "dwell_left_s": 0.0,
            }
        )

    for t_idx, _time_s in enumerate(times):
        for u_idx, state in enumerate(users):
            current_node = int(state["current_node"])
            if state["dwell_left_s"] > 0:
                state["dwell_left_s"] = max(0.0, float(state["dwell_left_s"]) - mobility.step_s)
            else:
                distance_left = _sample_step_speed(float(state["preferred_speed"]), mobility, rng) * mobility.step_s
                while distance_left > 0:
                    next_node = state["next_node"]
                    if next_node is None:
                        next_node = _choose_next_node(
                            graph,
                            rng,
                            current_node,
                            int(state["previous_node"]) if state["previous_node"] is not None else None,
                            state["user_edge_counts"],
                            shared_edge_counts,
                        )
                        state["next_node"] = next_node
                        if next_node is None:
                            break
                        edge_key = _edge_key(current_node, int(next_node))
                        state["user_edge_counts"][edge_key] = state["user_edge_counts"].get(edge_key, 0) + 1
                        shared_edge_counts[edge_key] = shared_edge_counts.get(edge_key, 0) + 1
                    start = _node_xy(graph, current_node)
                    end = _node_xy(graph, int(next_node))
                    seg_len = max(float(graph.edges[current_node, int(next_node)]["length"]), 1e-6)
                    remaining = seg_len - float(state["segment_progress"])
                    step = min(distance_left, remaining)
                    state["segment_progress"] = float(state["segment_progress"]) + step
                    distance_left -= step
                    if state["segment_progress"] >= seg_len - 1e-9:
                        previous_node = current_node
                        current_node = int(next_node)
                        state["current_node"] = current_node
                        state["previous_node"] = previous_node
                        state["next_node"] = None
                        state["segment_progress"] = 0.0
                        if mobility.dwell_s_range[1] > 0.0 and (rng.random() < 0.2 or graph.degree[current_node] <= 1):
                            state["dwell_left_s"] = float(rng.uniform(*mobility.dwell_s_range))
                            break
                    else:
                        state["current_node"] = current_node
                        break

            current_node = int(state["current_node"])
            next_node = state["next_node"]
            if next_node is None:
                xy = _node_xy(graph, current_node)
            else:
                start = _node_xy(graph, current_node)
                end = _node_xy(graph, int(next_node))
                seg_len = max(float(graph.edges[current_node, int(next_node)]["length"]), 1e-6)
                alpha = float(state["segment_progress"]) / seg_len
                xy = start + alpha * (end - start)
            positions[t_idx, u_idx, :2] = xy
            positions[t_idx, u_idx, 2] = ue_height_m

    velocities = np.zeros_like(positions)
    if len(times) > 1:
        velocities[1:] = (positions[1:] - positions[:-1]) / mobility.step_s
        velocities[0] = velocities[1]
    logger.info("Trajectory generation complete")
    return Trajectory(times_s=times, ue_ids=ue_ids, positions_m=positions, velocities_mps=velocities)
