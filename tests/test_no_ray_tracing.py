import json
from pathlib import Path

import matplotlib

from cocoon_sionna.config import load_scenario_config
from cocoon_sionna.pipeline import run_scenario
from cocoon_sionna.scene_builder import SceneArtifacts


matplotlib.use("Agg", force=True)


def _stub_scene(monkeypatch, tmp_path: Path) -> Path:
    graph_path = tmp_path / "walk_graph.json"
    graph_path.write_text(
        json.dumps(
            {
                "nodes": [
                    {"id": 1, "x": 0.0, "y": 0.0, "entry_candidate": True},
                    {"id": 2, "x": 10.0, "y": 0.0, "entry_candidate": True},
                    {"id": 3, "x": 10.0, "y": 10.0, "entry_candidate": True},
                    {"id": 4, "x": 0.0, "y": 10.0, "entry_candidate": True},
                ],
                "edges": [
                    {"u": 1, "v": 2, "length": 10.0},
                    {"u": 2, "v": 3, "length": 10.0},
                    {"u": 3, "v": 4, "length": 10.0},
                    {"u": 4, "v": 1, "length": 10.0},
                ],
            }
        ),
        encoding="utf-8",
    )
    scene_xml = tmp_path / "scene.xml"
    scene_xml.write_text("<scene version='3.0.0'/>", encoding="utf-8")
    metadata_path = tmp_path / "scene_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "boundary_local": [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [0.0, 0.0]],
                "buildings": [
                    {
                        "name": "blok_a",
                        "height_m": 12.0,
                        "polygon_local": [[2.0, 2.0], [4.0, 2.0], [4.0, 4.0], [2.0, 4.0], [2.0, 2.0]],
                    },
                    {
                        "name": "blok_b",
                        "height_m": 9.0,
                        "polygon_local": [[6.0, 6.0], [8.0, 6.0], [8.0, 8.0], [6.0, 8.0], [6.0, 6.0]],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "cocoon_sionna.pipeline._resolve_scene_inputs",
        lambda _config: SceneArtifacts(scene_xml_path=scene_xml, metadata_path=metadata_path, walk_graph_path=graph_path),
    )
    return graph_path


def test_run_scenario_without_ray_tracing_writes_trajectory_outputs(monkeypatch, tmp_path: Path):
    config = load_scenario_config("scenarios/rabot.yaml")
    config.outputs.output_dir = tmp_path / "rabot_no_rt"
    config.solver.enable_ray_tracing = False
    config.mobility.graph_path = _stub_scene(monkeypatch, tmp_path)
    config.outputs.output_dir.mkdir(parents=True, exist_ok=True)
    (config.outputs.output_dir / "coverage_map.npz").write_bytes(b"stale")
    (config.outputs.output_dir / "fixed_coverage_map.npz").write_bytes(b"stale")

    summary = run_scenario(config)

    assert summary["ray_tracing_enabled"] is False
    assert summary["status"] == "trajectory_only"
    assert (config.outputs.output_dir / "trajectory.csv").exists()
    assert (config.outputs.output_dir / "scene_metadata.json").exists()
    assert (config.outputs.output_dir / "walk_graph.json").exists()
    assert (config.outputs.output_dir / "candidate_ap_positions.csv").exists()
    assert (config.outputs.output_dir / "central_ap_rooftop_candidates.csv").exists()
    assert (config.outputs.output_dir / "central_massive_mimo_ap.csv").exists()
    assert (config.outputs.output_dir / "central_massive_mimo_schedule.csv").exists()
    assert (config.outputs.output_dir / "distributed_fixed_aps.csv").exists()
    assert (config.outputs.output_dir / "distributed_fixed_schedule.csv").exists()
    assert (config.outputs.output_dir / "distributed_movable_aps.csv").exists()
    assert (config.outputs.output_dir / "distributed_movable_schedule.csv").exists()
    assert (config.outputs.output_dir / "strategy_comparison.csv").exists()
    assert not (config.outputs.output_dir / "coverage_map.npz").exists()
    assert not (config.outputs.output_dir / "scene_layout.png").exists()
    assert not (config.outputs.output_dir / "trajectory_colormap.png").exists()
    assert not (config.outputs.output_dir / "infra_csi_snapshots.npz").exists()
    assert not (config.outputs.output_dir / "peer_csi_snapshots.npz").exists()


def test_run_scenario_without_ray_tracing_uses_distributed_fixed_as_reference(monkeypatch, tmp_path: Path):
    config = load_scenario_config("scenarios/rabot.yaml")
    config.outputs.output_dir = tmp_path / "rabot_no_opt"
    config.solver.enable_ray_tracing = False
    config.mobility.graph_path = _stub_scene(monkeypatch, tmp_path)

    summary = run_scenario(config)

    assert summary["ray_tracing_enabled"] is False
    assert summary["baseline_strategy"] == "distributed_fixed"
    assert summary["best_strategy"] == "distributed_fixed"
    assert summary["strategies"]["central_massive_mimo"]["is_placeholder"] is True


def test_run_scenario_rejects_nonzero_fixed_ap_count(monkeypatch, tmp_path: Path):
    config = load_scenario_config("scenarios/rabot.yaml")
    config.outputs.output_dir = tmp_path / "rabot_invalid"
    config.solver.enable_ray_tracing = False
    config.placement.num_fixed_aps = 1
    config.mobility.graph_path = _stub_scene(monkeypatch, tmp_path)

    try:
        run_scenario(config)
    except ValueError as exc:
        assert "num_fixed_aps == 0" in str(exc)
    else:
        raise AssertionError("Expected run_scenario() to reject nonzero num_fixed_aps")
