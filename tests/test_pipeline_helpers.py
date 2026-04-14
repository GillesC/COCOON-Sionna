import matplotlib
import networkx as nx
import numpy as np
from matplotlib import animation as mpl_animation

from cocoon_sionna.config import OptimizationConfig, load_scenario_config
from cocoon_sionna.mobility import Trajectory
from cocoon_sionna.optimization import PlacementScore
from cocoon_sionna.pipeline import (
    _cache_dir,
    _animate_scene,
    _build_csi_cache_key,
    _cache_optional_artifact,
    _instantaneous_user_sinr_samples,
    _ordered_enabled_sites,
    _per_user_mean_best_sinr,
    _plot_scene_layout,
    _relocate_sites,
    _restore_cached_artifact,
    _select_fixed_and_mobile_seed_sites,
    _should_render_sionna_scene_artifacts,
    _try_load_csi_cache,
    _update_prefixed_export,
    _write_ap_relocation_csv,
    _write_csi_cache,
    _window_slices,
)
from cocoon_sionna.scene_builder import SceneArtifacts
from cocoon_sionna.sites import CandidateSite


class _DummyConfig:
    def __init__(self, baseline_site_ids, num_fixed_aps=0, num_mobile_aps=None, num_selected_aps=2):
        self.candidate_sites_path = "dummy.csv"
        self.optimization = OptimizationConfig(
            num_selected_aps=num_selected_aps,
            num_fixed_aps=num_fixed_aps,
            num_mobile_aps=num_mobile_aps,
            baseline_site_ids=list(baseline_site_ids),
        )


def test_ordered_enabled_sites_prioritizes_baseline_site_ids():
    config = _DummyConfig(["site_b", "site_a"])
    sites = [
        CandidateSite("site_a", 0.0, 0.0, 5.0, 0.0, -10.0, "pole"),
        CandidateSite("site_b", 1.0, 0.0, 5.0, 0.0, -10.0, "pole"),
        CandidateSite("site_c", 2.0, 0.0, 5.0, 0.0, -10.0, "pole"),
    ]

    selected = _ordered_enabled_sites(config, sites)

    assert [site.site_id for site in selected] == ["site_b", "site_a", "site_c"]


def test_select_fixed_and_mobile_seed_sites_splits_ordered_pool():
    config = _DummyConfig([])
    sites = [
        CandidateSite("site_a", 0.0, 0.0, 5.0, 0.0, -10.0, "pole"),
        CandidateSite("site_b", 1.0, 0.0, 5.0, 0.0, -10.0, "pole", enabled=False),
        CandidateSite("site_c", 2.0, 0.0, 5.0, 0.0, -10.0, "pole"),
        CandidateSite("site_d", 3.0, 0.0, 5.0, 0.0, -10.0, "pole"),
    ]

    config.optimization.num_fixed_aps = 1
    config.optimization.num_mobile_aps = 2
    fixed_sites, mobile_sites = _select_fixed_and_mobile_seed_sites(config, sites, strategy="ordered")

    assert [site.site_id for site in fixed_sites] == ["site_a"]
    assert [site.site_id for site in mobile_sites] == ["site_c", "site_d"]


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

    layout_path = tmp_path / "scene_layout.png"
    _plot_scene_layout(metadata, graph, base_sites, selected_sites, trajectory, layout_path)

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


def test_update_prefixed_export_only_writes_available_keys():
    target = {"base": 1}
    _update_prefixed_export(target, "peer", {"cfr": np.array([1.0]), "tau": np.array([2.0])}, ("cfr", "cir", "tau"))

    assert "peer_cfr" in target
    assert "peer_tau" in target
    assert "peer_cir" not in target


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
    config = load_scenario_config("scenarios/etoile_demo.yaml")
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
        "best_sinr_db": np.ones((2, 1), dtype=float),
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
    config = load_scenario_config("scenarios/etoile_demo.yaml")
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
        "best_sinr_db": np.ones((2, 1), dtype=float),
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
