import networkx as nx
import numpy as np
from shapely.geometry import Point, Polygon

from cocoon_sionna.config import MobilityConfig
from cocoon_sionna.mobility import augment_graph_with_open_area, generate_trajectory


def test_generate_trajectory_shapes():
    graph = nx.Graph()
    graph.add_node(1, x=0.0, y=0.0, entry_candidate=True)
    graph.add_node(2, x=10.0, y=0.0, entry_candidate=True)
    graph.add_edge(1, 2, length=10.0)

    trajectory = generate_trajectory(
        graph,
        MobilityConfig(
            source="file",
            graph_path=None,
            num_users=2,
            duration_s=4.0,
            step_s=1.0,
            speed_mps_range=(1.0, 1.0),
            dwell_s_range=(0.0, 0.0),
            seed=1,
        ),
        ue_height_m=1.5,
    )

    assert trajectory.positions_m.shape == (5, 2, 3)
    assert trajectory.velocities_mps.shape == (5, 2, 3)
    assert (trajectory.positions_m[..., 2] == 1.5).all()


def test_generate_trajectory_spreads_initial_users_across_entry_nodes():
    graph = nx.Graph()
    graph.add_node(1, x=0.0, y=0.0, entry_candidate=True)
    graph.add_node(2, x=10.0, y=0.0, entry_candidate=True)
    graph.add_node(3, x=10.0, y=10.0, entry_candidate=True)
    graph.add_node(4, x=0.0, y=10.0, entry_candidate=True)
    graph.add_edges_from(
        [
            (1, 2, {"length": 10.0}),
            (2, 3, {"length": 10.0}),
            (3, 4, {"length": 10.0}),
            (4, 1, {"length": 10.0}),
        ]
    )

    trajectory = generate_trajectory(
        graph,
        MobilityConfig(
            source="file",
            graph_path=None,
            num_users=4,
            duration_s=0.0,
            step_s=1.0,
            speed_mps_range=(0.0, 0.0),
            dwell_s_range=(0.0, 0.0),
            seed=3,
        ),
        ue_height_m=1.5,
    )

    positions = {tuple(point) for point in trajectory.positions_m[0, :, :2]}

    assert positions == {(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)}


def test_generate_trajectory_prefers_interior_start_nodes_when_available():
    graph = nx.Graph()
    graph.add_node(1, x=0.0, y=0.0, entry_candidate=True)
    graph.add_node(2, x=20.0, y=0.0, entry_candidate=True)
    graph.add_node(3, x=20.0, y=20.0, entry_candidate=True)
    graph.add_node(4, x=0.0, y=20.0, entry_candidate=True)
    graph.add_node(5, x=8.0, y=8.0, entry_candidate=False)
    graph.add_node(6, x=12.0, y=8.0, entry_candidate=False)
    graph.add_node(7, x=12.0, y=12.0, entry_candidate=False)
    graph.add_node(8, x=8.0, y=12.0, entry_candidate=False)
    graph.add_edges_from(
        [
            (1, 5, {"length": 11.3137}),
            (2, 6, {"length": 11.3137}),
            (3, 7, {"length": 11.3137}),
            (4, 8, {"length": 11.3137}),
            (5, 6, {"length": 4.0}),
            (6, 7, {"length": 4.0}),
            (7, 8, {"length": 4.0}),
            (8, 5, {"length": 4.0}),
        ]
    )

    trajectory = generate_trajectory(
        graph,
        MobilityConfig(
            source="file",
            graph_path=None,
            num_users=4,
            duration_s=0.0,
            step_s=1.0,
            speed_mps_range=(0.0, 0.0),
            dwell_s_range=(0.0, 0.0),
            seed=7,
        ),
        ue_height_m=1.5,
    )

    positions = {tuple(point) for point in trajectory.positions_m[0, :, :2]}

    assert positions == {(8.0, 8.0), (12.0, 8.0), (12.0, 12.0), (8.0, 12.0)}


def test_generate_trajectory_varies_speed_over_time():
    graph = nx.Graph()
    graph.add_node(1, x=0.0, y=0.0, entry_candidate=True)
    graph.add_node(2, x=50.0, y=0.0, entry_candidate=True)
    graph.add_edge(1, 2, length=50.0)

    trajectory = generate_trajectory(
        graph,
        MobilityConfig(
            source="file",
            graph_path=None,
            num_users=1,
            duration_s=5.0,
            step_s=1.0,
            speed_mps_range=(1.0, 1.0),
            speed_variation_fraction=0.25,
            dwell_s_range=(0.0, 0.0),
            seed=5,
        ),
        ue_height_m=1.5,
    )

    speeds = trajectory.velocities_mps[:, 0, 0]

    assert np.ptp(speeds) > 0.05


def test_generate_trajectory_supports_bike_profiles():
    graph = nx.Graph()
    graph.add_node(1, x=0.0, y=0.0, entry_candidate=True)
    graph.add_node(2, x=80.0, y=0.0, entry_candidate=True)
    graph.add_edge(1, 2, length=80.0)

    trajectory = generate_trajectory(
        graph,
        MobilityConfig(
            source="file",
            graph_path=None,
            num_users=1,
            duration_s=3.0,
            step_s=1.0,
            speed_mps_range=(1.0, 1.0),
            bike_speed_mps_range=(4.0, 4.0),
            bike_fraction=1.0,
            speed_variation_fraction=0.0,
            dwell_s_range=(0.0, 0.0),
            seed=9,
        ),
        ue_height_m=1.5,
    )

    assert (trajectory.velocities_mps[:, 0, 0] >= 4.0 - 1e-6).all()


def test_augment_graph_with_open_area_adds_free_space_nodes_outside_buildings():
    graph = nx.Graph()
    graph.add_node(1, x=0.0, y=10.0, entry_candidate=True)
    graph.add_node(2, x=20.0, y=10.0, entry_candidate=True)
    graph.add_edge(1, 2, length=20.0)
    metadata = {
        "boundary_local": [[0.0, 0.0], [20.0, 0.0], [20.0, 20.0], [0.0, 20.0], [0.0, 0.0]],
        "buildings": [
            {"polygon_local": [[8.0, 8.0], [12.0, 8.0], [12.0, 12.0], [8.0, 12.0], [8.0, 8.0]]},
        ],
    }

    augmented = augment_graph_with_open_area(
        graph,
        MobilityConfig(
            source="scene_metadata",
            allow_open_area=True,
            open_area_grid_spacing_m=5.0,
            open_area_connection_radius_m=8.0,
            open_area_clearance_m=0.5,
        ),
        metadata,
    )

    building = Polygon(metadata["buildings"][0]["polygon_local"])
    free_nodes = [
        node_id
        for node_id in augmented.nodes
        if node_id not in {1, 2}
    ]

    assert free_nodes
    assert any(abs(float(augmented.nodes[node_id]["y"]) - 10.0) > 1e-6 for node_id in free_nodes)
    assert all(
        not building.covers(Point(float(augmented.nodes[node_id]["x"]), float(augmented.nodes[node_id]["y"])))
        for node_id in free_nodes
    )


def test_generate_trajectory_can_use_open_area_without_walk_edges():
    metadata = {
        "boundary_local": [[0.0, 0.0], [18.0, 0.0], [18.0, 18.0], [0.0, 18.0], [0.0, 0.0]],
        "buildings": [
            {"polygon_local": [[7.0, 7.0], [11.0, 7.0], [11.0, 11.0], [7.0, 11.0], [7.0, 7.0]]},
        ],
    }
    building = Polygon(metadata["buildings"][0]["polygon_local"])

    trajectory = generate_trajectory(
        nx.Graph(),
        MobilityConfig(
            source="scene_metadata",
            num_users=1,
            duration_s=2.0,
            step_s=1.0,
            allow_open_area=True,
            open_area_grid_spacing_m=6.0,
            open_area_connection_radius_m=9.0,
            speed_mps_range=(1.0, 1.0),
            speed_variation_fraction=0.0,
            dwell_s_range=(0.0, 0.0),
            seed=4,
        ),
        ue_height_m=1.5,
        metadata=metadata,
    )

    assert trajectory.positions_m.shape == (3, 1, 3)
    assert all(
        not building.covers(Point(float(position[0]), float(position[1])))
        for position in trajectory.positions_m[:, 0, :2]
    )
