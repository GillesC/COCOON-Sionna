from pathlib import Path

import matplotlib
import networkx as nx
import numpy as np

from cocoon_sionna.config import load_scenario_config
from cocoon_sionna.mobility import Trajectory
from cocoon_sionna.pipeline import run_scenario
from cocoon_sionna.sites import CandidateSite


matplotlib.use("Agg", force=True)


def test_optimization_scoring_does_not_call_radiomap_per_candidate(monkeypatch, tmp_path: Path):
    config = load_scenario_config("scenarios/etoile_demo.yaml")
    config.outputs.output_dir = tmp_path / "etoile"
    config.coverage.enabled = True
    config.optimization.baseline_site_ids = []
    config.mobility.graph_path = tmp_path / "walk_graph.json"
    config.mobility.graph_path.write_text("{}", encoding="utf-8")

    trajectory = Trajectory(
        times_s=np.array([0.0, 5.0]),
        ue_ids=["ue_000"],
        positions_m=np.array([[[0.0, 0.0, 1.5]], [[1.0, 0.0, 1.5]]]),
        velocities_mps=np.zeros((2, 1, 3), dtype=float),
    )
    graph = nx.Graph()
    base_sites = [
        CandidateSite("base_01", 0.0, 0.0, 5.0, 0.0, -10.0, "pole"),
        CandidateSite("base_02", 1.0, 0.0, 5.0, 0.0, -10.0, "pole"),
    ]
    candidate_sites = [
        CandidateSite("cand_a", 10.0, 0.0, 5.0, 0.0, -10.0, "pole"),
        CandidateSite("cand_b", 20.0, 0.0, 5.0, 0.0, -10.0, "pole"),
        CandidateSite("cand_c", 50.0, 0.0, 5.0, 0.0, -10.0, "pole"),
    ]
    calls = {"radio_map": 0}

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
            signal = sum(float(site.x_m) for site in sites)
            return {
                "tx_site_ids": [site.site_id for site in sites],
                "rx_ue_ids": list(trajectory.ue_ids),
                "times_s": trajectory.times_s,
                "best_sinr_db": np.full((len(trajectory.times_s), len(trajectory.ue_ids)), signal, dtype=float),
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

    def fake_greedy_one_swap(candidate_ids, select_count, evaluator):
        assert select_count == 2
        subset_ab = tuple(sorted(candidate_ids[:2]))
        subset_ac = tuple(sorted((candidate_ids[0], candidate_ids[2])))
        score_ab = evaluator(subset_ab)
        score_ac = evaluator(subset_ac)
        if score_ac.score > score_ab.score:
            return list(subset_ac), score_ac
        return list(subset_ab), score_ab

    monkeypatch.setattr("cocoon_sionna.pipeline.SionnaRtRunner", FakeRunner)
    monkeypatch.setattr("cocoon_sionna.pipeline.load_graph_json", lambda _path: graph)
    monkeypatch.setattr("cocoon_sionna.pipeline.generate_trajectory", lambda *args, **kwargs: trajectory)
    monkeypatch.setattr("cocoon_sionna.pipeline.load_candidate_sites", lambda _path: list(base_sites))
    monkeypatch.setattr("cocoon_sionna.pipeline.augment_with_trajectory_sites", lambda **_kwargs: list(candidate_sites))
    monkeypatch.setattr("cocoon_sionna.pipeline.greedy_one_swap", fake_greedy_one_swap)
    monkeypatch.setattr("cocoon_sionna.pipeline._plot_scene_layout", lambda *args, **kwargs: None)
    monkeypatch.setattr("cocoon_sionna.pipeline._animate_scene", lambda *args, **kwargs: None)
    monkeypatch.setattr("cocoon_sionna.pipeline._plot_colored_trajectories", lambda *args, **kwargs: None)
    monkeypatch.setattr("cocoon_sionna.pipeline._write_ap_relocation_csv", lambda *args, **kwargs: None)
    monkeypatch.setattr("cocoon_sionna.pipeline._write_user_sinr_artifacts", lambda *args, **kwargs: None)

    summary = run_scenario(config)

    assert summary["radio_map_enabled"] is True
    assert calls["radio_map"] == 2
    assert (config.outputs.output_dir / "coverage_map.npz").exists()
    assert (config.outputs.output_dir / "fixed_coverage_map.npz").exists()


def test_run_scenario_can_skip_csi_storage_and_full_exports(monkeypatch, tmp_path: Path):
    config = load_scenario_config("scenarios/etoile_demo.yaml")
    config.outputs.output_dir = tmp_path / "etoile_no_store"
    config.outputs.write_csi_exports = False
    config.outputs.enable_csi_cache = False
    config.coverage.enabled = False
    config.optimization.enable_optimization = False
    config.optimization.baseline_site_ids = []
    config.mobility.graph_path = tmp_path / "walk_graph.json"
    config.mobility.graph_path.write_text("{}", encoding="utf-8")

    trajectory = Trajectory(
        times_s=np.array([0.0, 5.0]),
        ue_ids=["ue_000"],
        positions_m=np.array([[[0.0, 0.0, 1.5]], [[1.0, 0.0, 1.5]]]),
        velocities_mps=np.zeros((2, 1, 3), dtype=float),
    )
    graph = nx.Graph()
    base_sites = [
        CandidateSite("base_01", 0.0, 0.0, 5.0, 0.0, -10.0, "pole"),
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
                "best_sinr_db": np.full((len(trajectory.times_s), len(trajectory.ue_ids)), 1.0, dtype=float),
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
    monkeypatch.setattr("cocoon_sionna.pipeline.load_candidate_sites", lambda _path: list(base_sites))
    monkeypatch.setattr("cocoon_sionna.pipeline.augment_with_trajectory_sites", lambda **_kwargs: list(base_sites))
    monkeypatch.setattr("cocoon_sionna.pipeline._plot_scene_layout", lambda *args, **kwargs: None)
    monkeypatch.setattr("cocoon_sionna.pipeline._animate_scene", lambda *args, **kwargs: None)
    monkeypatch.setattr("cocoon_sionna.pipeline._plot_colored_trajectories", lambda *args, **kwargs: None)
    monkeypatch.setattr("cocoon_sionna.pipeline._write_ap_relocation_csv", lambda *args, **kwargs: None)
    monkeypatch.setattr("cocoon_sionna.pipeline._write_user_sinr_artifacts", lambda *args, **kwargs: None)

    summary = run_scenario(config)

    assert summary["csi_exports_enabled"] is False
    assert summary["csi_cache_enabled"] is False
    assert calls["ue_ue_export_full"] == [False]
    assert calls["ap_ue_export_full"] == [False]
    assert calls["ap_ap_export_full"] == [False]
    assert not (config.outputs.output_dir / "peer_csi_snapshots.npz").exists()
    assert not (config.outputs.output_dir / "infra_csi_snapshots.npz").exists()
    assert not (config.outputs.output_dir / ".csi_cache").exists()
