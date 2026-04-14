import shutil
from pathlib import Path

import pytest

from cocoon_sionna.config import load_scenario_config
from cocoon_sionna.pipeline import run_scenario


@pytest.mark.integration
def test_etoile_smoke_pipeline(tmp_path: Path):
    pytest.importorskip("mitsuba")

    config = load_scenario_config("scenarios/etoile_demo.yaml")
    config.outputs.output_dir = tmp_path / "etoile"
    config.radio.cfr_bins = 16
    config.solver.samples_per_src = 500
    config.solver.samples_per_tx = 500
    config.placement.exact_max_iterations = 200

    summary = run_scenario(config)

    assert summary["selected_site_ids"]
    assert (config.outputs.output_dir / "candidate_ap_positions.csv").exists()
    assert (config.outputs.output_dir / "fixed_aps.csv").exists()
    assert (config.outputs.output_dir / "random_baseline_movable_aps.csv").exists()
    assert (config.outputs.output_dir / "local_csi_p90_movable_aps.csv").exists()
    assert (config.outputs.output_dir / "capped_exact_search_movable_aps.csv").exists()
    assert (config.outputs.output_dir / "strategy_comparison.csv").exists()
    assert (config.outputs.output_dir / "coverage_map.npz").exists()
    assert (config.outputs.output_dir / "fixed_coverage_map.npz").exists()
    assert (config.outputs.output_dir / "infra_csi_snapshots.npz").exists()
    assert (config.outputs.output_dir / "peer_csi_snapshots.npz").exists()
    assert (config.outputs.output_dir / "scene_render.png").exists()
    assert (config.outputs.output_dir / "scene_layout.png").exists()
    assert (config.outputs.output_dir / "trajectory_colormap.png").exists()
    if shutil.which("ffmpeg"):
        assert (config.outputs.output_dir / "scene_camera.mp4").exists()
    assert (
        (config.outputs.output_dir / "scene_animation.mp4").exists()
        or (config.outputs.output_dir / "scene_animation.gif").exists()
    )
    assert (config.outputs.output_dir / "user_sinr_cdf.png").exists()
    assert (config.outputs.output_dir / "user_sinr_summary.csv").exists()
    assert (config.outputs.output_dir / "user_sinr_timeseries.csv").exists()
    assert (config.outputs.output_dir / "random_baseline_schedule.csv").exists()
