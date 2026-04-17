import matplotlib
import networkx as nx
import numpy as np
from matplotlib import animation as mpl_animation

from cocoon_sionna.config import PlacementConfig, RadioConfig, load_scenario_config
from cocoon_sionna.mobility import Trajectory
from cocoon_sionna.optimization import PlacementScore
from cocoon_sionna.pipeline import (
    StrategyArtifacts,
    _central_ap_radio,
    _cache_dir,
    _animate_scene,
    _build_csi_cache_key,
    _cache_optional_artifact,
    _distance_threshold_snapshot_mask,
    _historical_local_percentile_10,
    _factor_central_ap_array,
    _generate_rooftop_candidates,
    _instantaneous_user_sinr_samples,
    _local_percentile_10,
    _local_window_average_power,
    _local_window_sum_rate,
    _make_reference_movable_sites,
    _movable_ap_count,
    _nearest_snapshot_mask,
    _per_user_mean_best_sinr,
    _plot_scene_layout,
    _proxy_ap_candidate_power_from_peer_csi,
    _proxy_window_sum_rate_from_peer_csi,
    _window_candidate_anchor_users,
    _relocate_sites,
    _restore_cached_artifact,
    _scene_animation_strategy_name,
    _select_fixed_sites,
    _should_render_sionna_scene_artifacts,
    _try_load_csi_cache,
    _update_prefixed_export,
    _write_ap_relocation_csv,
    _write_csi_cache,
    _write_user_sinr_artifacts,
    _window_slices,
)
from cocoon_sionna.scene_builder import SceneArtifacts
from cocoon_sionna.sites import CandidateSite


class _DummyConfig:
    def __init__(self, num_fixed_aps=0, num_movable_aps=2):
        self.candidate_sites_path = "dummy.csv"
        self.placement = PlacementConfig(
            num_fixed_aps=num_fixed_aps,
            num_movable_aps=num_movable_aps,
        )


def test_movable_ap_count_uses_placement_config():
    config = _DummyConfig(num_movable_aps=3)
    assert _movable_ap_count(config) == 3


def test_select_fixed_sites_uses_farthest_spacing():
    config = _DummyConfig(num_fixed_aps=2)
    sites = [
        CandidateSite("site_a", 0.0, 0.0, 5.0, 0.0, -10.0, "pole"),
        CandidateSite("site_b", 1.0, 0.0, 5.0, 0.0, -10.0, "pole"),
        CandidateSite("site_c", 50.0, 0.0, 5.0, 0.0, -10.0, "pole"),
    ]

    selected = _select_fixed_sites(config, sites)

    assert len(selected) == 2
    assert {site.site_id for site in selected} == {"site_a", "site_c"}


def test_make_reference_movable_sites_assigns_stable_ids():
    sites = [
        CandidateSite("cand_a", 0.0, 0.0, 1.5, 0.0, -10.0, "wall"),
        CandidateSite("cand_b", 2.0, 0.0, 1.5, 0.0, -10.0, "wall"),
    ]

    movable_sites = _make_reference_movable_sites(sites)

    assert [site.site_id for site in movable_sites] == ["movable_ap_01", "movable_ap_02"]
    assert movable_sites[0].source == "seed:cand_a"
    assert movable_sites[1].source == "seed:cand_b"


def test_factor_central_ap_array_prefers_near_square_layout():
    assert _factor_central_ap_array(24) == (4, 6)
    assert _factor_central_ap_array(36) == (6, 6)


def test_central_ap_radio_matches_total_antenna_and_power_budget():
    base = RadioConfig(ap_num_rows=2, ap_num_cols=3, tx_power_dbm_ap=28.0)

    central = _central_ap_radio(base, distributed_ap_count=6)

    assert central.ap_num_rows == 6
    assert central.ap_num_cols == 6
    assert central.ap_num_rows * central.ap_num_cols == 36
    assert np.isclose(central.tx_power_dbm_ap, 28.0 + 10.0 * np.log10(6.0))


def test_generate_rooftop_candidates_uses_representative_point_and_roof_offset():
    metadata = {
        "boundary_local": [[0.0, 0.0], [20.0, 0.0], [20.0, 10.0], [0.0, 10.0], [0.0, 0.0]],
        "buildings": [
            {
                "name": "blok_a",
                "height_m": 12.0,
                "polygon_local": [[2.0, 1.0], [4.0, 1.0], [4.0, 3.0], [2.0, 3.0], [2.0, 1.0]],
            },
            {
                "name": "blok_b",
                "height_m": 9.0,
                "polygon_local": [[6.0, 6.0], [8.0, 6.0], [8.0, 8.0], [6.0, 8.0], [6.0, 6.0]],
            },
        ]
    }

    candidates = _generate_rooftop_candidates(metadata)

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.site_id == "roof_blok_b"
    assert np.isclose(candidate.x_m, 7.0)
    assert np.isclose(candidate.y_m, 7.0)
    assert np.isclose(candidate.z_m, 10.5)
    assert np.isclose(candidate.yaw_deg, np.degrees(np.arctan2(-2.0, 3.0)))
    assert np.isclose(candidate.pitch_deg, -10.0)
    assert candidate.mount_type == "rooftop"
    assert candidate.source == "roof_metadata:center_building"


def test_relocate_sites_preserves_ap_ids_and_matches_nearest_candidates():
    baseline_sites = [
        CandidateSite("site_a", 0.0, 0.0, 5.0, 0.0, -10.0, "pole"),
        CandidateSite("site_b", 10.0, 0.0, 5.0, 0.0, -10.0, "pole"),
    ]
    selected_candidates = [
        CandidateSite("cand_far", 9.0, 1.0, 7.0, 45.0, -8.0, "candidate"),
        CandidateSite("cand_near", 1.0, 1.0, 7.0, 90.0, -6.0, "candidate"),
    ]

    relocated = _relocate_sites(baseline_sites, selected_candidates)

    assert [site.site_id for site in relocated] == ["site_a", "site_b"]
    assert relocated[0].position == (1.0, 1.0, 7.0)
    assert relocated[1].position == (9.0, 1.0, 7.0)
    assert relocated[0].source == "relocated:cand_near"
    assert relocated[1].source == "relocated:cand_far"


def test_window_slices_groups_snapshots_by_relocation_interval():
    windows = _window_slices(np.array([0.0, 2.0, 4.0, 7.0, 8.0, 14.0]), relocation_interval_s=6.0)

    assert [window.tolist() for window in windows] == [[0, 1, 2], [3, 4], [5]]


def test_nearest_snapshot_mask_and_local_percentile():
    trajectory = Trajectory(
        times_s=np.array([0.0, 1.0]),
        ue_ids=["ue_0", "ue_1"],
        positions_m=np.array(
            [
                [[0.0, 0.0, 1.5], [10.0, 0.0, 1.5]],
                [[1.0, 0.0, 1.5], [20.0, 0.0, 1.5]],
            ]
        ),
        velocities_mps=np.zeros((2, 2, 3), dtype=float),
    )
    site = CandidateSite("cand", 0.0, 0.0, 1.5, 0.0, -10.0, "wall")

    mask = _nearest_snapshot_mask(trajectory, site, k_nearest=2)

    assert mask.tolist() == [True, False, True, False]
    assert np.isclose(_local_percentile_10(np.array([[1.0, 100.0], [3.0, 200.0]]), mask), 1.2)


def test_historical_local_percentile_10_applies_exponential_decay():
    site = CandidateSite("cand", 0.0, 0.0, 1.5, 0.0, -10.0, "wall")
    history_segments = [
        {
            "trajectory": Trajectory(
                times_s=np.array([0.0]),
                ue_ids=["ue_0"],
                positions_m=np.array([[[0.0, 0.0, 1.5]]]),
                velocities_mps=np.zeros((1, 1, 3), dtype=float),
            ),
            "ap_ue": {"best_sinr_db": np.array([[-20.0]], dtype=float)},
        },
        {
            "trajectory": Trajectory(
                times_s=np.array([10.0]),
                ue_ids=["ue_0"],
                positions_m=np.array([[[0.0, 0.0, 1.5]]]),
                velocities_mps=np.zeros((1, 1, 3), dtype=float),
            ),
            "ap_ue": {"best_sinr_db": np.array([[10.0]], dtype=float)},
        },
    ]

    reduced = _historical_local_percentile_10(
        history_segments,
        ("cand",),
        {"cand": site},
        k_nearest=1,
        decay_rate_per_s=0.5,
    )

    assert reduced == 10.0


def test_distance_threshold_mask_and_local_window_sum_rate_use_local_users_only():
    trajectory = Trajectory(
        times_s=np.array([0.0, 5.0]),
        ue_ids=["ue_0", "ue_1"],
        positions_m=np.array(
            [
                [[0.0, 0.0, 1.5], [30.0, 0.0, 1.5]],
                [[1.0, 0.0, 1.5], [40.0, 0.0, 1.5]],
            ],
            dtype=float,
        ),
        velocities_mps=np.zeros((2, 2, 3), dtype=float),
    )
    site = CandidateSite("cand", 0.0, 0.0, 1.5, 0.0, -10.0, "wall")

    mask = _distance_threshold_snapshot_mask(trajectory, [site], distance_threshold_m=5.0)
    assert mask.tolist() == [[True, False], [True, False]]

    score = _local_window_sum_rate(
        {"sinr_linear": np.array([[3.0, 100.0], [15.0, 100.0]], dtype=float)},
        trajectory,
        ("cand",),
        {"cand": site},
        distance_threshold_m=5.0,
    )

    expected = 0.5 * (np.log2(1.0 + 3.0) + np.log2(1.0 + 15.0))
    assert np.isclose(score, expected)


def test_local_window_average_power_uses_same_radius_mask_as_optimization_2():
    trajectory = Trajectory(
        times_s=np.array([0.0, 5.0]),
        ue_ids=["ue_0", "ue_1"],
        positions_m=np.array(
            [
                [[0.0, 0.0, 1.5], [30.0, 0.0, 1.5]],
                [[1.0, 0.0, 1.5], [40.0, 0.0, 1.5]],
            ],
            dtype=float,
        ),
        velocities_mps=np.zeros((2, 2, 3), dtype=float),
    )
    site = CandidateSite("cand", 0.0, 0.0, 1.5, 0.0, -10.0, "wall")

    score = _local_window_average_power(
        {
            "tx_site_ids": ["cand"],
            "sinr_linear": np.ones((2, 2), dtype=float),
            "link_power_w": np.array(
                [
                    [[5.0, 300.0]],
                    [[12.0, 300.0]],
                ],
                dtype=float,
            )
        },
        trajectory,
        ("cand",),
        {"cand": site},
        distance_threshold_m=5.0,
    )

    expected = np.mean([5.0, 12.0])
    assert np.isclose(score, expected)


def test_local_window_average_power_accepts_user_ap_power_axis_order():
    trajectory = Trajectory(
        times_s=np.array([0.0, 5.0]),
        ue_ids=["ue_0", "ue_1"],
        positions_m=np.array(
            [
                [[0.0, 0.0, 1.5], [30.0, 0.0, 1.5]],
                [[1.0, 0.0, 1.5], [40.0, 0.0, 1.5]],
            ],
            dtype=float,
        ),
        velocities_mps=np.zeros((2, 2, 3), dtype=float),
    )
    site = CandidateSite("cand", 0.0, 0.0, 1.5, 0.0, -10.0, "wall")

    score = _local_window_average_power(
        {
            "tx_site_ids": ["cand"],
            "sinr_linear": np.ones((2, 2), dtype=float),
            "link_power_w": np.array(
                [
                    [[5.0], [300.0]],
                    [[12.0], [300.0]],
                ],
                dtype=float,
            )
        },
        trajectory,
        ("cand",),
        {"cand": site},
        distance_threshold_m=5.0,
    )

    expected = np.mean([5.0, 12.0])
    assert np.isclose(score, expected)


def test_window_candidate_anchor_users_selects_nearest_user_per_snapshot():
    trajectory = Trajectory(
        times_s=np.array([0.0, 5.0]),
        ue_ids=["ue_0", "ue_1"],
        positions_m=np.array(
            [
                [[0.0, 0.0, 1.5], [10.0, 0.0, 1.5]],
                [[10.0, 0.0, 1.5], [0.0, 0.0, 1.5]],
            ],
            dtype=float,
        ),
        velocities_mps=np.zeros((2, 2, 3), dtype=float),
    )
    site = CandidateSite("cand", 1.0, 0.0, 1.5, 0.0, -10.0, "wall")

    anchor_indices, valid_mask = _window_candidate_anchor_users(trajectory, [site], distance_threshold_m=5.0)

    assert anchor_indices.tolist() == [[0], [1]]
    assert valid_mask.tolist() == [[True], [True]]


def test_proxy_ap_candidate_power_from_peer_csi_scales_abs_csi_squared_proxy():
    trajectory = Trajectory(
        times_s=np.array([0.0]),
        ue_ids=["ue_0", "ue_1"],
        positions_m=np.array([[[0.0, 0.0, 1.5], [10.0, 0.0, 1.5]]], dtype=float),
        velocities_mps=np.zeros((1, 2, 3), dtype=float),
    )
    site = CandidateSite("cand", 0.0, 0.0, 1.5, 0.0, -10.0, "wall")

    proxy_power_w, anchor_indices, valid_mask = _proxy_ap_candidate_power_from_peer_csi(
        np.array([[[0.0, 5.0], [2.0, 0.0]]], dtype=float),
        trajectory,
        ("cand",),
        {"cand": site},
        distance_threshold_m=5.0,
        tx_power_scale=2.0,
    )

    assert anchor_indices.tolist() == [[0]]
    assert valid_mask.tolist() == [[True]]
    np.testing.assert_allclose(proxy_power_w[0, 0], np.array([10.0, 10.0]))


def test_proxy_window_sum_rate_uses_csi_when_cfr_is_available(monkeypatch):
    trajectory = Trajectory(
        times_s=np.array([0.0]),
        ue_ids=["ue_0", "ue_1"],
        positions_m=np.array([[[0.0, 0.0, 1.5], [10.0, 0.0, 1.5]]], dtype=float),
        velocities_mps=np.zeros((1, 2, 3), dtype=float),
    )
    site = CandidateSite("cand", 0.0, 0.0, 1.5, 0.0, -10.0, "wall")
    calls: list[np.ndarray] = []

    def fake_zf(channel, total_tx_power_w, noise_power_w):
        calls.append(np.asarray(channel))
        assert np.isclose(total_tx_power_w, 1.0)
        assert np.isclose(noise_power_w, 0.5)
        return {
            "desired_power_w": np.array([0.0, 0.0], dtype=float),
            "interference_power_w": np.array([0.0, 0.0], dtype=float),
            "noise_power_w": np.array([0.5, 0.5], dtype=float),
            "sinr": np.array([3.0, 1.0], dtype=float),
        }

    monkeypatch.setattr("cocoon_sionna.pipeline._zf_sinr_terms_from_mimo_channel", fake_zf)
    score = _proxy_window_sum_rate_from_peer_csi(
        {
            "link_power_w": np.array([[[0.0, 9.0], [4.0, 0.0]]], dtype=float),
            "cfr": np.array(
                [
                    [
                        [[0.0 + 0.0j], [3.0 + 0.0j]],
                        [[2.0 + 0.0j], [0.0 + 0.0j]],
                    ]
                ],
                dtype=np.complex128,
            ),
        },
        trajectory,
        ("cand",),
        {"cand": site},
        distance_threshold_m=5.0,
        noise_power_w=0.5,
        total_tx_power_w=1.0,
        tx_power_scale=1.0,
    )

    assert len(calls) == 1
    np.testing.assert_allclose(np.abs(calls[0][:, 0, 0, 0]), np.array([2.0, 2.0]))
    assert np.isclose(score, np.log2(1.0 + 3.0) + np.log2(1.0 + 1.0))


def test_should_render_sionna_scene_artifacts_skips_cpu_llvm_backend():
    assert not _should_render_sionna_scene_artifacts({"device": "CPU", "variant": "llvm_ad_mono_polarized"})
    assert _should_render_sionna_scene_artifacts({"device": "GPU", "variant": "cuda_ad_mono_polarized"})


def test_per_user_mean_best_sinr_reduces_over_time():
    best_sinr_db = np.array(
        [
            [0.0, 10.0, 20.0],
            [2.0, 14.0, 18.0],
            [4.0, 16.0, 22.0],
        ]
    )

    reduced = _per_user_mean_best_sinr(best_sinr_db)

    np.testing.assert_allclose(reduced, np.array([2.0, 13.3333333333, 20.0]))


def test_instantaneous_user_sinr_samples_preserves_all_snapshots():
    best_sinr_db = np.array(
        [
            [0.0, 10.0],
            [2.0, 12.0],
            [4.0, 14.0],
        ]
    )

    reduced = _instantaneous_user_sinr_samples(best_sinr_db)

    np.testing.assert_allclose(reduced, np.array([0.0, 10.0, 2.0, 12.0, 4.0, 14.0]))


def test_scene_layout_and_animation_are_written(tmp_path, monkeypatch):
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
    metadata = {
        "boundary_local": [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [0.0, 0.0]],
        "buildings": [
            {"polygon_local": [[3.0, 3.0], [5.0, 3.0], [5.0, 5.0], [3.0, 5.0], [3.0, 3.0]]},
        ],
    }
    trajectory = Trajectory(
        times_s=np.array([0.0, 1.0]),
        ue_ids=["ue_000", "ue_001"],
        positions_m=np.array(
            [
                [[1.0, 1.0, 1.5], [9.0, 1.0, 1.5]],
                [[1.0, 9.0, 1.5], [9.0, 9.0, 1.5]],
            ]
        ),
        velocities_mps=np.zeros((2, 2, 3), dtype=float),
    )
    base_sites = [
        CandidateSite("site_a", 2.0, 2.0, 5.0, 0.0, -10.0, "pole"),
        CandidateSite("site_b", 8.0, 8.0, 5.0, 0.0, -10.0, "pole"),
    ]
    selected_sites = [base_sites[1]]
    reference_sites = [CandidateSite("central_ap_01", 5.0, 5.0, 13.5, 0.0, -90.0, "rooftop")]

    layout_path = tmp_path / "scene_layout.png"
    _plot_scene_layout(metadata, graph, base_sites, selected_sites, trajectory, layout_path, reference_sites=reference_sites)

    monkeypatch.setattr("cocoon_sionna.pipeline.shutil.which", lambda _name: None)
    animation_path = _animate_scene(metadata, graph, base_sites, selected_sites, trajectory, tmp_path / "scene_animation.mp4")

    assert layout_path.exists()
    assert animation_path == tmp_path / "scene_animation.gif"
    assert animation_path.exists()


def test_animate_scene_applies_speedup_to_gif_fps(tmp_path, monkeypatch):
    matplotlib.use("Agg", force=True)
    graph = nx.Graph()
    graph.add_node(1, x=0.0, y=0.0, entry_candidate=True)
    graph.add_node(2, x=1.0, y=0.0, entry_candidate=True)
    graph.add_edge(1, 2, length=1.0)
    trajectory = Trajectory(
        times_s=np.array([0.0, 1.0, 2.0]),
        ue_ids=["ue_000"],
        positions_m=np.array(
            [
                [[0.0, 0.0, 1.5]],
                [[0.5, 0.0, 1.5]],
                [[1.0, 0.0, 1.5]],
            ]
        ),
        velocities_mps=np.zeros((3, 1, 3), dtype=float),
    )

    observed = {}

    class _DummyWriter:
        def __init__(self, fps):
            observed["fps"] = fps

    def _fake_save(self, path, writer=None, dpi=None):
        target = tmp_path / "scene_animation.gif"
        target.write_bytes(b"GIF89a")

    monkeypatch.setattr("cocoon_sionna.pipeline.shutil.which", lambda _name: None)
    monkeypatch.setattr(mpl_animation, "PillowWriter", _DummyWriter)
    monkeypatch.setattr(mpl_animation.FuncAnimation, "save", _fake_save, raising=False)

    animation_path = _animate_scene(
        None,
        graph,
        [],
        [],
        trajectory,
        tmp_path / "scene_animation.mp4",
        speedup=10.0,
    )

    assert animation_path == tmp_path / "scene_animation.gif"
    assert observed["fps"] == 10


def test_animate_scene_writes_schedule_overlay_with_fixed_sites(tmp_path, monkeypatch):
    matplotlib.use("Agg", force=True)
    graph = nx.Graph()
    graph.add_node(1, x=0.0, y=0.0, entry_candidate=True)
    graph.add_node(2, x=10.0, y=0.0, entry_candidate=True)
    graph.add_edge(1, 2, length=10.0)
    trajectory = Trajectory(
        times_s=np.array([0.0, 5.0, 10.0]),
        ue_ids=["ue_000"],
        positions_m=np.array(
            [
                [[0.0, 1.0, 1.5]],
                [[5.0, 1.0, 1.5]],
                [[10.0, 1.0, 1.5]],
            ]
        ),
        velocities_mps=np.zeros((3, 1, 3), dtype=float),
    )
    candidate_sites = [
        CandidateSite("cand_a", 1.0, 0.0, 1.5, 0.0, -10.0, "wall"),
        CandidateSite("cand_b", 9.0, 0.0, 1.5, 0.0, -10.0, "wall"),
    ]
    fixed_sites = [CandidateSite("fixed_ap_01", 5.0, 5.0, 1.5, 0.0, -10.0, "wall")]
    reference_sites = [CandidateSite("central_ap_01", 5.0, -3.0, 13.5, 0.0, -90.0, "rooftop")]
    schedule_rows = [
        {
            "window_index": 0,
            "start_time_s": 0.0,
            "end_time_s": 5.0,
            "ap_id": "movable_ap_01",
            "x_m": 1.0,
            "y_m": 0.0,
            "z_m": 1.5,
            "source": "relocated:cand_a",
        },
        {
            "window_index": 1,
            "start_time_s": 10.0,
            "end_time_s": 10.0,
            "ap_id": "movable_ap_01",
            "x_m": 9.0,
            "y_m": 0.0,
            "z_m": 1.5,
            "source": "relocated:cand_b",
        },
    ]

    def _fake_save(self, path, writer=None, dpi=None):
        target = tmp_path / "scene_animation.gif"
        target.write_bytes(b"GIF89a")

    monkeypatch.setattr("cocoon_sionna.pipeline.shutil.which", lambda _name: None)
    monkeypatch.setattr(mpl_animation.FuncAnimation, "save", _fake_save, raising=False)

    animation_path = _animate_scene(
        None,
        graph,
        candidate_sites,
        [],
        trajectory,
        tmp_path / "scene_animation.mp4",
        schedule_rows=schedule_rows,
        fixed_sites=fixed_sites,
        reference_sites=reference_sites,
        reference_label="Central massive-MIMO BS",
    )

    assert animation_path == tmp_path / "scene_animation.gif"
    assert animation_path.exists()


def test_scene_animation_prefers_moving_optimized_strategy():
    static_schedule = [
        {
            "window_index": 0,
            "start_time_s": 0.0,
            "end_time_s": 10.0,
            "ap_id": "movable_ap_01",
            "x_m": 1.0,
            "y_m": 0.0,
            "z_m": 1.5,
            "source": "seed:cand_a",
        },
        {
            "window_index": 1,
            "start_time_s": 10.0,
            "end_time_s": 20.0,
            "ap_id": "movable_ap_01",
            "x_m": 1.0,
            "y_m": 0.0,
            "z_m": 1.5,
            "source": "seed:cand_a",
        },
    ]
    moving_schedule = [
        dict(static_schedule[0]),
        {
            "window_index": 1,
            "start_time_s": 10.0,
            "end_time_s": 20.0,
            "ap_id": "movable_ap_01",
            "x_m": 8.0,
            "y_m": 0.0,
            "z_m": 1.5,
            "source": "relocated:cand_b",
        },
    ]

    def _artifact(name, score, schedule_rows):
        return StrategyArtifacts(
            name=name,
            selected_sites=[],
            movable_sites=[],
            ap_ue={},
            ap_ap={},
            score=PlacementScore(score, 0.0, 0.0, 0.0, 0.0, 0.0),
            schedule_rows=schedule_rows,
            final_candidate_ids=[],
            selected_candidate_union=set(),
        )

    strategy_results = {
        "central_massive_mimo": _artifact("central_massive_mimo", 10.0, static_schedule),
        "distributed_fixed": _artifact("distributed_fixed", 8.0, static_schedule),
        "distributed_movable": _artifact("distributed_movable", 7.0, moving_schedule),
    }

    assert _scene_animation_strategy_name(strategy_results, "central_massive_mimo") == "distributed_movable"


def test_update_prefixed_export_only_writes_available_keys():
    target = {"base": 1}
    _update_prefixed_export(target, "peer", {"cfr": np.array([1.0]), "tau": np.array([2.0])}, ("cfr", "cir", "tau"))

    assert "peer_cfr" in target
    assert "peer_tau" in target
    assert "peer_cir" not in target


def test_write_user_sinr_artifacts_exports_snapshot_csv_and_npz(tmp_path):
    trajectory = Trajectory(
        times_s=np.array([0.0, 5.0]),
        ue_ids=["ue_000", "ue_001"],
        positions_m=np.zeros((2, 2, 3), dtype=float),
        velocities_mps=np.zeros((2, 2, 3), dtype=float),
    )
    strategy_ap_ue = {
        "central_massive_mimo": {
            "sinr_linear": np.array([[10.0, 100.0], [1000.0, 10000.0]], dtype=float),
            "best_sinr_db": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
            "desired_power_w": np.array([[1.0, 1.5], [2.0, 2.5]], dtype=float),
            "interference_power_w": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=float),
            "noise_power_w": np.array([[0.01, 0.01], [0.01, 0.01]], dtype=float),
        },
        "distributed_fixed": {
            "sinr_linear": np.array([[100000.0, 1000000.0], [10000000.0, 100000000.0]], dtype=float),
            "best_sinr_db": np.array([[5.0, 6.0], [7.0, 8.0]], dtype=float),
            "desired_power_w": np.array([[3.0, 3.5], [4.0, 4.5]], dtype=float),
            "interference_power_w": np.array([[0.5, 0.6], [0.7, 0.8]], dtype=float),
            "noise_power_w": np.array([[0.02, 0.02], [0.02, 0.02]], dtype=float),
        },
        "distributed_movable": {
            "sinr_linear": np.array([[1000000000.0, 10000000000.0], [100000000000.0, 1000000000000.0]], dtype=float),
            "best_sinr_db": np.array([[9.0, 10.0], [11.0, 12.0]], dtype=float),
            "desired_power_w": np.array([[5.0, 5.5], [6.0, 6.5]], dtype=float),
            "interference_power_w": np.array([[0.9, 1.0], [1.1, 1.2]], dtype=float),
            "noise_power_w": np.array([[0.03, 0.03], [0.03, 0.03]], dtype=float),
        },
    }

    _write_user_sinr_artifacts(tmp_path, trajectory, strategy_ap_ue)

    csv_lines = (tmp_path / "user_sinr_timeseries.csv").read_text(encoding="utf-8").splitlines()
    assert (
        csv_lines[0]
        == "snapshot_index,time_s,ue_id,central_massive_mimo_sinr_db,distributed_fixed_sinr_db,distributed_movable_sinr_db"
    )
    assert csv_lines[1] == "0,0.0,ue_000,1.0,5.0,9.0"
    assert csv_lines[4] == "1,5.0,ue_001,4.0,8.0,12.0"

    payload = np.load(tmp_path / "user_sinr_snapshots.npz", allow_pickle=True)
    np.testing.assert_array_equal(payload["snapshot_index"], np.array([0, 1], dtype=int))
    np.testing.assert_array_equal(payload["times_s"], np.array([0.0, 5.0], dtype=float))
    np.testing.assert_array_equal(payload["ue_ids"], np.array(["ue_000", "ue_001"], dtype=object))
    np.testing.assert_array_equal(
        payload["strategy_names"],
        np.array(["central_massive_mimo", "distributed_fixed", "distributed_movable"], dtype=object),
    )
    np.testing.assert_allclose(payload["central_massive_mimo_sinr_db"], np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float))
    np.testing.assert_allclose(payload["central_massive_mimo_sinr_linear"], np.array([[10.0, 100.0], [1000.0, 10000.0]], dtype=float))
    np.testing.assert_allclose(payload["central_massive_mimo_interference_power_w"], np.array([[0.1, 0.2], [0.3, 0.4]], dtype=float))
    np.testing.assert_allclose(payload["distributed_fixed_sinr_db"], np.array([[5.0, 6.0], [7.0, 8.0]], dtype=float))
    np.testing.assert_allclose(payload["distributed_movable_sinr_db"], np.array([[9.0, 10.0], [11.0, 12.0]], dtype=float))
    assert (tmp_path / "user_sinr_summary.csv").exists()
    assert (tmp_path / "user_sinr_cdf.png").exists()


def test_write_ap_relocation_csv_supports_mobile_ap_ids(tmp_path):
    output_path = tmp_path / "ap_relocation_summary.csv"
    baseline_sites = [
        CandidateSite("fixed_ap_01", 0.0, 0.0, 5.0, 0.0, -10.0, "pole"),
        CandidateSite("fixed_ap_02", 10.0, 0.0, 5.0, 0.0, -10.0, "pole"),
    ]
    relocated_sites = [
        CandidateSite("mobile_ap_01", 1.0, 1.0, 5.0, 0.0, -10.0, "pole", source="relocated:cand_a"),
        CandidateSite("mobile_ap_02", 12.0, 1.0, 5.0, 0.0, -10.0, "pole", source="relocated:cand_b"),
    ]

    _write_ap_relocation_csv(output_path, baseline_sites, relocated_sites)

    rows = output_path.read_text(encoding="utf-8").splitlines()
    assert rows[0] == (
        "site_id,original_x_m,original_y_m,original_z_m,relocated_x_m,"
        "relocated_y_m,relocated_z_m,move_distance_m,relocation_source"
    )
    assert rows[1].startswith("mobile_ap_01,0.0,0.0,5.0,1.0,1.0,5.0,")
    assert rows[2].startswith("mobile_ap_02,10.0,0.0,5.0,12.0,1.0,5.0,")


def test_csi_cache_roundtrip(tmp_path):
    config = load_scenario_config("scenarios/rabot.yaml")
    config.outputs.output_dir = tmp_path / "out"
    trajectory = Trajectory(
        times_s=np.array([0.0, 1.0]),
        ue_ids=["ue_000"],
        positions_m=np.array([[[0.0, 0.0, 1.5]], [[1.0, 0.0, 1.5]]]),
        velocities_mps=np.zeros((2, 1, 3), dtype=float),
    )
    base_sites = [CandidateSite("site_a", 0.0, 0.0, 5.0, 0.0, 0.0, "wall")]
    scene_dir = tmp_path / "scene"
    scene_dir.mkdir()
    graph_path = scene_dir / "walk_graph.json"
    graph_path.write_text("{}", encoding="utf-8")
    scene_xml = scene_dir / "scene.xml"
    scene_xml.write_text("<scene/>", encoding="utf-8")
    metadata_path = scene_dir / "scene_metadata.json"
    metadata_path.write_text("{}", encoding="utf-8")
    artifacts = SceneArtifacts(scene_xml_path=scene_xml, metadata_path=metadata_path, walk_graph_path=graph_path)
    runtime_info = {"device": "CPU", "variant": "llvm_ad_mono_polarized", "note": "test"}
    cache_key = _build_csi_cache_key(config, artifacts, graph_path, trajectory, base_sites, base_sites, base_sites)

    peer_csi = {
        "ue_ids": ["ue_000"],
        "times_s": trajectory.times_s,
        "link_power_w": np.ones((2, 1, 1), dtype=float),
        "need_weights": np.ones((2, 1), dtype=float),
    }
    fixed_ap_ue = {
        "tx_site_ids": ["site_a"],
        "rx_ue_ids": ["ue_000"],
        "times_s": trajectory.times_s,
        "sinr_linear": np.ones((2, 1), dtype=float),
        "best_sinr_db": np.ones((2, 1), dtype=float),
        "desired_power_w": np.ones((2, 1), dtype=float),
        "interference_power_w": np.zeros((2, 1), dtype=float),
        "noise_power_w": 0.1 * np.ones((2, 1), dtype=float),
        "link_power_w": np.ones((2, 1, 1), dtype=float),
    }
    fixed_ap_ap = {
        "tx_site_ids": ["site_a"],
        "rx_site_ids": ["site_a"],
        "link_power_w": np.zeros((1, 1), dtype=float),
    }
    fixed_radio_map = {
        "path_gain": np.ones((1, 1), dtype=float),
        "rss": np.ones((1, 1), dtype=float),
        "sinr": np.ones((1, 1), dtype=float),
        "best_sinr_db": np.ones((1, 1), dtype=float),
        "cell_centers": np.zeros((1, 1, 3), dtype=float),
    }

    _write_csi_cache(
        output_dir=config.outputs.output_dir,
        cache_key=cache_key,
        runtime_info=runtime_info,
        peer_csi=peer_csi,
        fixed_ap_ue=fixed_ap_ue,
        final_ap_ue=fixed_ap_ue,
        fixed_ap_ap=fixed_ap_ap,
        final_ap_ap=fixed_ap_ap,
        fixed_radio_map=fixed_radio_map,
        final_radio_map=fixed_radio_map,
        selected_sites=base_sites,
        final_selected_candidate_ids=["site_a"],
        selected_candidate_union={"site_a"},
        schedule_rows=[],
        fixed_score=PlacementScore(1.0, 0.0, 5.0, 1.0, 0.0, 0.0),
        best_score=PlacementScore(1.0, 0.0, 5.0, 1.0, 0.0, 0.0),
        num_relocation_windows=1,
    )

    loaded = _try_load_csi_cache(config.outputs.output_dir, cache_key)

    assert loaded is not None
    assert (_cache_dir(config.outputs.output_dir, cache_key) / "manifest.json").exists()
    assert loaded["fixed_ap_ue"]["tx_site_ids"] == ["site_a"]
    assert loaded["selected_sites"][0].site_id == "site_a"
    assert loaded["fixed_radio_map"] is not None


def test_csi_cache_roundtrip_without_radiomap(tmp_path):
    config = load_scenario_config("scenarios/rabot.yaml")
    config.outputs.output_dir = tmp_path / "out"
    config.coverage.enabled = False
    trajectory = Trajectory(
        times_s=np.array([0.0, 1.0]),
        ue_ids=["ue_000"],
        positions_m=np.array([[[0.0, 0.0, 1.5]], [[1.0, 0.0, 1.5]]]),
        velocities_mps=np.zeros((2, 1, 3), dtype=float),
    )
    base_sites = [CandidateSite("site_a", 0.0, 0.0, 5.0, 0.0, 0.0, "wall")]
    scene_dir = tmp_path / "scene"
    scene_dir.mkdir()
    graph_path = scene_dir / "walk_graph.json"
    graph_path.write_text("{}", encoding="utf-8")
    scene_xml = scene_dir / "scene.xml"
    scene_xml.write_text("<scene/>", encoding="utf-8")
    metadata_path = scene_dir / "scene_metadata.json"
    metadata_path.write_text("{}", encoding="utf-8")
    artifacts = SceneArtifacts(scene_xml_path=scene_xml, metadata_path=metadata_path, walk_graph_path=graph_path)
    runtime_info = {"device": "CPU", "variant": "llvm_ad_mono_polarized", "note": "test"}
    cache_key = _build_csi_cache_key(config, artifacts, graph_path, trajectory, base_sites, base_sites, base_sites)

    peer_csi = {
        "ue_ids": ["ue_000"],
        "times_s": trajectory.times_s,
        "link_power_w": np.ones((2, 1, 1), dtype=float),
        "need_weights": np.ones((2, 1), dtype=float),
    }
    fixed_ap_ue = {
        "tx_site_ids": ["site_a"],
        "rx_ue_ids": ["ue_000"],
        "times_s": trajectory.times_s,
        "sinr_linear": np.ones((2, 1), dtype=float),
        "best_sinr_db": np.ones((2, 1), dtype=float),
        "desired_power_w": np.ones((2, 1), dtype=float),
        "interference_power_w": np.zeros((2, 1), dtype=float),
        "noise_power_w": 0.1 * np.ones((2, 1), dtype=float),
        "link_power_w": np.ones((2, 1, 1), dtype=float),
    }
    fixed_ap_ap = {
        "tx_site_ids": ["site_a"],
        "rx_site_ids": ["site_a"],
        "link_power_w": np.zeros((1, 1), dtype=float),
    }

    _write_csi_cache(
        output_dir=config.outputs.output_dir,
        cache_key=cache_key,
        runtime_info=runtime_info,
        peer_csi=peer_csi,
        fixed_ap_ue=fixed_ap_ue,
        final_ap_ue=fixed_ap_ue,
        fixed_ap_ap=fixed_ap_ap,
        final_ap_ap=fixed_ap_ap,
        fixed_radio_map=None,
        final_radio_map=None,
        selected_sites=base_sites,
        final_selected_candidate_ids=["site_a"],
        selected_candidate_union={"site_a"},
        schedule_rows=[],
        fixed_score=PlacementScore(1.0, 0.0, 5.0, 1.0, 0.0, 0.0),
        best_score=PlacementScore(1.0, 0.0, 5.0, 1.0, 0.0, 0.0),
        num_relocation_windows=1,
    )

    loaded = _try_load_csi_cache(config.outputs.output_dir, cache_key)

    assert loaded is not None
    assert loaded["fixed_radio_map"] is None
    assert loaded["final_radio_map"] is None
    assert not (_cache_dir(config.outputs.output_dir, cache_key) / "coverage_map.npz").exists()
    assert not (_cache_dir(config.outputs.output_dir, cache_key) / "fixed_coverage_map.npz").exists()


def test_cache_optional_artifacts_roundtrip(tmp_path):
    output_dir = tmp_path / "out"
    cache_key = "abc123"
    source_render = tmp_path / "scene_render.png"
    source_video = tmp_path / "scene_camera.mp4"
    source_render.write_bytes(b"render")
    source_video.write_bytes(b"video")

    _cache_optional_artifact(output_dir, cache_key, source_render, "scene_render.png")
    _cache_optional_artifact(output_dir, cache_key, source_video, "scene_camera.mp4")

    restored_render = output_dir / "scene_render.png"
    restored_video = output_dir / "scene_camera.mp4"

    render_path = _restore_cached_artifact(output_dir, cache_key, "scene_render.png", restored_render)
    video_path = _restore_cached_artifact(output_dir, cache_key, "scene_camera.mp4", restored_video)

    assert render_path == restored_render
    assert video_path == restored_video
    assert restored_render.read_bytes() == b"render"
    assert restored_video.read_bytes() == b"video"
