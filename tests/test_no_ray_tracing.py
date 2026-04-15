from pathlib import Path

import matplotlib

from cocoon_sionna.config import load_scenario_config
from cocoon_sionna.pipeline import run_scenario


matplotlib.use("Agg", force=True)


def test_run_scenario_without_ray_tracing_writes_trajectory_outputs(tmp_path: Path):
    config = load_scenario_config("scenarios/etoile_demo.yaml")
    config.outputs.output_dir = tmp_path / "etoile_no_rt"
    config.solver.enable_ray_tracing = False
    config.outputs.output_dir.mkdir(parents=True, exist_ok=True)
    (config.outputs.output_dir / "coverage_map.npz").write_bytes(b"stale")
    (config.outputs.output_dir / "fixed_coverage_map.npz").write_bytes(b"stale")

    summary = run_scenario(config)

    assert summary["ray_tracing_enabled"] is False
    assert summary["status"] == "trajectory_only"
    assert (config.outputs.output_dir / "trajectory.csv").exists()
    assert (config.outputs.output_dir / "scene_layout.png").exists()
    assert (config.outputs.output_dir / "trajectory_colormap.png").exists()
    assert (config.outputs.output_dir / "candidate_ap_positions.csv").exists()
    assert (config.outputs.output_dir / "fixed_aps.csv").exists()
    assert (config.outputs.output_dir / "random_baseline_schedule.csv").exists()
    assert (config.outputs.output_dir / "local_csi_p90_schedule.csv").exists()
    assert (config.outputs.output_dir / "capped_exact_search_schedule.csv").exists()
    assert (config.outputs.output_dir / "strategy_comparison.csv").exists()
    assert not (config.outputs.output_dir / "coverage_map.npz").exists()
    assert not (config.outputs.output_dir / "infra_csi_snapshots.npz").exists()
    assert not (config.outputs.output_dir / "peer_csi_snapshots.npz").exists()


def test_run_scenario_without_ray_tracing_uses_random_baseline_as_reference(tmp_path: Path):
    config = load_scenario_config("scenarios/etoile_demo.yaml")
    config.outputs.output_dir = tmp_path / "etoile_no_opt"
    config.solver.enable_ray_tracing = False

    summary = run_scenario(config)

    assert summary["ray_tracing_enabled"] is False
    assert summary["baseline_strategy"] == "random_baseline"
    assert summary["best_strategy"] == "random_baseline"


def test_run_scenario_can_disable_capped_exact_search(tmp_path: Path):
    config = load_scenario_config("scenarios/etoile_demo.yaml")
    config.outputs.output_dir = tmp_path / "etoile_no_exact"
    config.solver.enable_ray_tracing = False
    config.placement.enable_capped_exact_search = False
    config.outputs.output_dir.mkdir(parents=True, exist_ok=True)
    (config.outputs.output_dir / "capped_exact_search_movable_aps.csv").write_text("stale", encoding="utf-8")
    (config.outputs.output_dir / "capped_exact_search_schedule.csv").write_text("stale", encoding="utf-8")

    summary = run_scenario(config)

    assert summary["ray_tracing_enabled"] is False
    assert set(summary["strategies"]) == {"random_baseline", "local_csi_p90"}
    assert not (config.outputs.output_dir / "capped_exact_search_movable_aps.csv").exists()
    assert not (config.outputs.output_dir / "capped_exact_search_schedule.csv").exists()
