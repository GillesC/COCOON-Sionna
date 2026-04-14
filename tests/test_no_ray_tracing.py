from pathlib import Path

import matplotlib

from cocoon_sionna.config import load_scenario_config
from cocoon_sionna.pipeline import run_scenario


matplotlib.use("Agg", force=True)


def test_run_scenario_without_ray_tracing_writes_trajectory_outputs(tmp_path: Path):
    config = load_scenario_config("scenarios/etoile_demo.yaml")
    config.outputs.output_dir = tmp_path / "etoile_no_rt"
    config.solver.enable_ray_tracing = False

    summary = run_scenario(config)

    assert summary["ray_tracing_enabled"] is False
    assert summary["status"] == "trajectory_only"
    assert (config.outputs.output_dir / "trajectory.csv").exists()
    assert (config.outputs.output_dir / "scene_layout.png").exists()
    assert (config.outputs.output_dir / "trajectory_colormap.png").exists()
    assert (config.outputs.output_dir / "recommended_aps.csv").exists()
    assert (config.outputs.output_dir / "fixed_aps.csv").exists()
    assert (config.outputs.output_dir / "mobile_ap_schedule.csv").exists()
    assert not (config.outputs.output_dir / "coverage_map.npz").exists()
    assert not (config.outputs.output_dir / "infra_csi_snapshots.npz").exists()
    assert not (config.outputs.output_dir / "peer_csi_snapshots.npz").exists()


def test_run_scenario_without_optimization_keeps_fixed_layout(tmp_path: Path):
    config = load_scenario_config("scenarios/etoile_demo.yaml")
    config.outputs.output_dir = tmp_path / "etoile_no_opt"
    config.solver.enable_ray_tracing = False
    config.optimization.enable_optimization = False

    summary = run_scenario(config)

    assert summary["ray_tracing_enabled"] is False
    assert summary["optimization_enabled"] is False
    assert summary["fixed_site_ids"] == summary["mobile_site_ids"]
