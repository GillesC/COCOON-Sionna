from pathlib import Path

import matplotlib
import networkx as nx
import numpy as np

from cocoon_sionna.config import load_scenario_config
from cocoon_sionna.mobility import Trajectory
from cocoon_sionna.pipeline import run_scenario
from cocoon_sionna.scene_builder import SceneArtifacts
from cocoon_sionna.sites import CandidateSite


matplotlib.use("Agg", force=True)


def test_optimization_scoring_does_not_call_radiomap_per_candidate(monkeypatch, tmp_path: Path):
    config = load_scenario_config("scenarios/rabot.yaml")
    config.outputs.output_dir = tmp_path / "rabot"
    config.coverage.enabled = True
    config.outputs.write_csi_exports = True
    config.placement.num_fixed_aps = 0
    config.placement.num_movable_aps = 1
    config.mobility.graph_path = tmp_path / "walk_graph.json"
    config.mobility.graph_path.write_text("{}", encoding="utf-8")
    metadata_path = tmp_path / "scene_metadata.json"
    metadata_path.write_text(
        '{"buildings":[{"name":"blok_a","height_m":12.0,"polygon_local":[[0.0,0.0],[4.0,0.0],[4.0,4.0],[0.0,4.0],[0.0,0.0]]}]}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "cocoon_sionna.pipeline._resolve_scene_inputs",
        lambda _config: SceneArtifacts(
            scene_xml_path=tmp_path / "scene.xml",
            metadata_path=metadata_path,
            walk_graph_path=config.mobility.graph_path,
        ),
    )

    trajectory = Trajectory(
        times_s=np.array([0.0, 5.0]),
        ue_ids=["ue_000"],
        positions_m=np.array([[[0.0, 0.0, 1.5]], [[1.0, 0.0, 1.5]]]),
        velocities_mps=np.zeros((2, 1, 3), dtype=float),
    )
    graph = nx.Graph()
    candidate_sites = [
        CandidateSite("cand_a", 0.0, 0.0, 5.0, 0.0, -10.0, "pole"),
        CandidateSite("cand_b", 10.0, 0.0, 5.0, 0.0, -10.0, "pole"),
        CandidateSite("cand_c", 50.0, 0.0, 5.0, 0.0, -10.0, "pole"),
    ]
    calls = {"radio_map": 0, "ap_ue_site_counts": []}

    class FakeRunner:
        def __init__(self, **_kwargs):
            pass

        def runtime_info(self):
            return {"device": "CPU", "variant": "llvm_ad_mono_polarized"}

        def compute_ue_ue_csi(self, trajectory, export_full):
            assert export_full is True
            return {
                "ue_ids": list(trajectory.ue_ids),
                "times_s": trajectory.times_s,
                "link_power_w": np.ones((len(trajectory.times_s), 1, 1), dtype=float),
                "need_weights": np.ones((len(trajectory.times_s), len(trajectory.ue_ids)), dtype=float),
            }

        def compute_ap_ue_csi(self, sites, trajectory, export_full):
            calls["ap_ue_site_counts"].append(len(sites))
            signal = sum(float(site.x_m) for site in sites)
            return {
                "tx_site_ids": [site.site_id for site in sites],
                "rx_ue_ids": list(trajectory.ue_ids),
                "times_s": trajectory.times_s,
                "sinr_linear": np.full((len(trajectory.times_s), len(trajectory.ue_ids)), max(signal, 1.0), dtype=float),
                "best_sinr_db": np.full((len(trajectory.times_s), len(trajectory.ue_ids)), signal, dtype=float),
                "desired_power_w": np.full((len(trajectory.times_s), len(trajectory.ue_ids)), max(signal, 1.0), dtype=float),
                "interference_power_w": np.zeros((len(trajectory.times_s), len(trajectory.ue_ids)), dtype=float),
                "noise_power_w": 0.1 * np.ones((len(trajectory.times_s), len(trajectory.ue_ids)), dtype=float),
                "link_power_w": np.full((len(trajectory.times_s), len(sites), len(trajectory.ue_ids)), max(signal, 1.0), dtype=float),
            }

        def compute_ap_ap_csi(self, sites, export_full):
            return {
                "tx_site_ids": [site.site_id for site in sites],
                "rx_site_ids": [site.site_id for site in sites],
                "link_power_w": np.ones((len(sites), len(sites)), dtype=float),
            }

        def compute_radio_map(self, sites, coverage):
            assert coverage.enabled is True
            calls["radio_map"] += 1
            signal = sum(float(site.x_m) for site in sites)
            return {
                "path_gain": np.full((1, 1), signal, dtype=float),
                "rss": np.full((1, 1), signal, dtype=float),
                "sinr": np.full((1, 1), signal, dtype=float),
                "best_sinr_db": np.full((1, 1), signal, dtype=float),
                "cell_centers": np.zeros((1, 1, 3), dtype=float),
            }

    monkeypatch.setattr("cocoon_sionna.pipeline.SionnaRtRunner", FakeRunner)
    monkeypatch.setattr("cocoon_sionna.pipeline.load_graph_json", lambda _path: graph)
    monkeypatch.setattr("cocoon_sionna.pipeline.generate_trajectory", lambda *args, **kwargs: trajectory)
    monkeypatch.setattr("cocoon_sionna.pipeline.load_candidate_sites", lambda _path: list(candidate_sites))
    monkeypatch.setattr("cocoon_sionna.pipeline._plot_scene_layout", lambda *args, **kwargs: None)
    monkeypatch.setattr("cocoon_sionna.pipeline._animate_scene", lambda *args, **kwargs: None)
    monkeypatch.setattr("cocoon_sionna.pipeline._plot_colored_trajectories", lambda *args, **kwargs: None)
    monkeypatch.setattr("cocoon_sionna.pipeline._write_user_sinr_artifacts", lambda *args, **kwargs: None)

    summary = run_scenario(config)

    assert summary["radio_map_enabled"] is True
    assert summary["baseline_strategy"] == "distributed_fixed"
    assert summary["best_strategy"] in {
        "central_massive_mimo",
        "distributed_fixed",
        "distributed_movable",
        "distributed_movable_optimization_2",
        "distributed_movable_optimization_3",
    }
    assert calls["radio_map"] == 2
    assert all(site_count == 1 for site_count in calls["ap_ue_site_counts"])
    assert (config.outputs.output_dir / "coverage_map.npz").exists()
    assert (config.outputs.output_dir / "fixed_coverage_map.npz").exists()


def test_run_scenario_can_skip_csi_storage_and_full_exports(monkeypatch, tmp_path: Path):
    config = load_scenario_config("scenarios/rabot.yaml")
    config.outputs.output_dir = tmp_path / "rabot_no_store"
    config.outputs.write_csi_exports = False
    config.outputs.enable_csi_cache = False
    config.coverage.enabled = False
    config.placement.num_fixed_aps = 0
    config.placement.num_movable_aps = 1
    config.mobility.graph_path = tmp_path / "walk_graph.json"
    config.mobility.graph_path.write_text("{}", encoding="utf-8")
    metadata_path = tmp_path / "scene_metadata.json"
    metadata_path.write_text(
        '{"buildings":[{"name":"blok_a","height_m":12.0,"polygon_local":[[0.0,0.0],[4.0,0.0],[4.0,4.0],[0.0,4.0],[0.0,0.0]]}]}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "cocoon_sionna.pipeline._resolve_scene_inputs",
        lambda _config: SceneArtifacts(
            scene_xml_path=tmp_path / "scene.xml",
            metadata_path=metadata_path,
            walk_graph_path=config.mobility.graph_path,
        ),
    )

    trajectory = Trajectory(
        times_s=np.array([0.0, 5.0]),
        ue_ids=["ue_000"],
        positions_m=np.array([[[0.0, 0.0, 1.5]], [[1.0, 0.0, 1.5]]]),
        velocities_mps=np.zeros((2, 1, 3), dtype=float),
    )
    graph = nx.Graph()
    candidate_sites = [
        CandidateSite("base_01", 0.0, 0.0, 5.0, 0.0, -10.0, "pole"),
        CandidateSite("base_02", 10.0, 0.0, 5.0, 0.0, -10.0, "pole"),
    ]
    calls = {"ue_ue_export_full": [], "ap_ue_export_full": [], "ap_ap_export_full": []}

    class FakeRunner:
        def __init__(self, **_kwargs):
            pass

        def runtime_info(self):
            return {"device": "CPU", "variant": "llvm_ad_mono_polarized"}

        def compute_ue_ue_csi(self, trajectory, export_full):
            calls["ue_ue_export_full"].append(export_full)
            return {
                "ue_ids": list(trajectory.ue_ids),
                "times_s": trajectory.times_s,
                "link_power_w": np.ones((len(trajectory.times_s), 1, 1), dtype=float),
                "need_weights": np.ones((len(trajectory.times_s), len(trajectory.ue_ids)), dtype=float),
            }

        def compute_ap_ue_csi(self, sites, trajectory, export_full):
            calls["ap_ue_export_full"].append(export_full)
            return {
                "tx_site_ids": [site.site_id for site in sites],
                "rx_ue_ids": list(trajectory.ue_ids),
                "times_s": trajectory.times_s,
                "sinr_linear": np.ones((len(trajectory.times_s), len(trajectory.ue_ids)), dtype=float),
                "best_sinr_db": np.full((len(trajectory.times_s), len(trajectory.ue_ids)), 1.0, dtype=float),
                "desired_power_w": np.ones((len(trajectory.times_s), len(trajectory.ue_ids)), dtype=float),
                "interference_power_w": np.zeros((len(trajectory.times_s), len(trajectory.ue_ids)), dtype=float),
                "noise_power_w": 0.1 * np.ones((len(trajectory.times_s), len(trajectory.ue_ids)), dtype=float),
                "link_power_w": np.ones((len(trajectory.times_s), len(sites), len(trajectory.ue_ids)), dtype=float),
            }

        def compute_ap_ap_csi(self, sites, export_full):
            calls["ap_ap_export_full"].append(export_full)
            return {
                "tx_site_ids": [site.site_id for site in sites],
                "rx_site_ids": [site.site_id for site in sites],
                "link_power_w": np.ones((len(sites), len(sites)), dtype=float),
            }

    monkeypatch.setattr("cocoon_sionna.pipeline.SionnaRtRunner", FakeRunner)
    monkeypatch.setattr("cocoon_sionna.pipeline.load_graph_json", lambda _path: graph)
    monkeypatch.setattr("cocoon_sionna.pipeline.generate_trajectory", lambda *args, **kwargs: trajectory)
    monkeypatch.setattr("cocoon_sionna.pipeline.load_candidate_sites", lambda _path: list(candidate_sites))
    monkeypatch.setattr("cocoon_sionna.pipeline._plot_scene_layout", lambda *args, **kwargs: None)
    monkeypatch.setattr("cocoon_sionna.pipeline._animate_scene", lambda *args, **kwargs: None)
    monkeypatch.setattr("cocoon_sionna.pipeline._plot_colored_trajectories", lambda *args, **kwargs: None)
    monkeypatch.setattr("cocoon_sionna.pipeline._write_user_sinr_artifacts", lambda *args, **kwargs: None)

    summary = run_scenario(config)

    assert summary["csi_exports_enabled"] is False
    assert summary["csi_cache_enabled"] is False
    assert calls["ue_ue_export_full"] == [False]
    assert calls["ap_ue_export_full"]
    assert calls["ap_ap_export_full"]
    assert all(value is False for value in calls["ap_ue_export_full"])
    assert all(value is False for value in calls["ap_ap_export_full"])
    assert not (config.outputs.output_dir / "peer_csi_snapshots.npz").exists()
    assert not (config.outputs.output_dir / "infra_csi_snapshots.npz").exists()
    assert not (config.outputs.output_dir / ".csi_cache").exists()
