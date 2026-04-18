import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

from cocoon_sionna.postprocess import (
    resolve_output_dir_argument,
    run_manuscript_report,
    run_scene_visualization_postprocess,
    run_schedule_analysis,
    run_sinr_snapshot_analysis,
    run_strategy_summary_analysis,
    run_visualization_postprocess,
)


matplotlib.use("Agg", force=True)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_fake_output_dir(tmp_path: Path) -> Path:
    output_dir = tmp_path / "rabot"
    output_dir.mkdir()
    summary = {
        "scenario": "rabot_outdoor",
        "baseline_strategy": "distributed_fixed",
        "best_strategy": "distributed_movable",
        "scene_animation_strategy": "distributed_movable",
        "scene_animation_speedup": 10.0,
        "scene_context": {
            "scene_xml_path": "",
            "scene_metadata_path": str(output_dir / "scene_metadata.json"),
            "walk_graph_path": str(output_dir / "walk_graph.json"),
        },
        "strategies": {
            "central_massive_mimo": {
                "selected_site_ids": ["central_ap_01"],
                "movable_site_ids": ["central_ap_01"],
                "final_candidate_ids": ["roof_a"],
                "score": 0.5,
                "outage": 0.35,
                "percentile_10_db": -2.0,
                "peer_tiebreak": 1.5,
                "capped": False,
                "evaluated_combinations": 2,
            },
            "distributed_fixed": {
                "selected_site_ids": ["movable_ap_01", "movable_ap_02"],
                "movable_site_ids": ["movable_ap_01", "movable_ap_02"],
                "final_candidate_ids": ["cand_a"],
                "score": 1.0,
                "outage": 0.30,
                "percentile_10_db": 0.5,
                "peer_tiebreak": 2.0,
                "capped": False,
                "evaluated_combinations": 1,
            },
            "distributed_movable": {
                "selected_site_ids": ["movable_ap_01", "movable_ap_02"],
                "movable_site_ids": ["movable_ap_01", "movable_ap_02"],
                "final_candidate_ids": ["cand_b"],
                "score": 2.0,
                "outage": 0.10,
                "percentile_10_db": 4.0,
                "peer_tiebreak": 3.0,
                "capped": False,
                "evaluated_combinations": 8,
            },
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _write_csv(
        output_dir / "strategy_comparison.csv",
        [
            "strategy",
            "score",
            "outage",
            "percentile_10_db",
            "peer_tiebreak",
            "capped",
            "evaluated_combinations",
            "final_candidate_ids",
        ],
        [
            {
                "strategy": "central_massive_mimo",
                "score": 0.5,
                "outage": 0.35,
                "percentile_10_db": -2.0,
                "peer_tiebreak": 1.5,
                "capped": False,
                "evaluated_combinations": 2,
                "final_candidate_ids": "roof_a",
            },
            {
                "strategy": "distributed_fixed",
                "score": 1.0,
                "outage": 0.30,
                "percentile_10_db": 0.5,
                "peer_tiebreak": 2.0,
                "capped": False,
                "evaluated_combinations": 1,
                "final_candidate_ids": "cand_a",
            },
            {
                "strategy": "distributed_movable",
                "score": 2.0,
                "outage": 0.10,
                "percentile_10_db": 4.0,
                "peer_tiebreak": 3.0,
                "capped": False,
                "evaluated_combinations": 8,
                "final_candidate_ids": "cand_b",
            },
        ],
    )

    np.savez_compressed(
        output_dir / "user_sinr_snapshots.npz",
        snapshot_index=np.array([0, 1, 2], dtype=int),
        times_s=np.array([0.0, 10.0, 20.0], dtype=float),
        ue_ids=np.array(["ue_000", "ue_001"], dtype=object),
        strategy_names=np.array(["central_massive_mimo", "distributed_fixed", "distributed_movable"], dtype=object),
        central_massive_mimo_sinr_linear=np.power(10.0, np.array([[-2.0, 0.0], [-1.0, 1.0], [0.0, 2.0]], dtype=float) / 10.0),
        central_massive_mimo_sinr_db=np.array([[-2.0, 0.0], [-1.0, 1.0], [0.0, 2.0]], dtype=float),
        distributed_fixed_sinr_linear=np.power(10.0, np.array([[0.0, 2.0], [1.0, 3.0], [2.0, 4.0]], dtype=float) / 10.0),
        distributed_fixed_sinr_db=np.array([[0.0, 2.0], [1.0, 3.0], [2.0, 4.0]], dtype=float),
        distributed_fixed_spectral_efficiency_bps_hz=2.0 * np.ones((3, 2), dtype=float),
        distributed_movable_sinr_linear=np.power(10.0, np.array([[3.0, 5.0], [4.0, 6.0], [5.0, 7.0]], dtype=float) / 10.0),
        distributed_movable_sinr_db=np.array([[3.0, 5.0], [4.0, 6.0], [5.0, 7.0]], dtype=float),
    )

    (output_dir / "scene_metadata.json").write_text(
        json.dumps(
            {
                "boundary_local": [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [0.0, 0.0]],
                "buildings": [
                    {
                        "name": "blok_a",
                        "height_m": 12.0,
                        "polygon_local": [[2.0, 2.0], [4.0, 2.0], [4.0, 4.0], [2.0, 4.0], [2.0, 2.0]],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "walk_graph.json").write_text(
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
    _write_csv(
        output_dir / "trajectory.csv",
        ["time_s", "ue_id", "x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps"],
        [
            {"time_s": 0.0, "ue_id": "ue_000", "x_m": 1.0, "y_m": 1.0, "z_m": 1.5, "vx_mps": 0.0, "vy_mps": 0.0, "vz_mps": 0.0},
            {"time_s": 0.0, "ue_id": "ue_001", "x_m": 9.0, "y_m": 1.0, "z_m": 1.5, "vx_mps": 0.0, "vy_mps": 0.0, "vz_mps": 0.0},
            {"time_s": 10.0, "ue_id": "ue_000", "x_m": 1.0, "y_m": 9.0, "z_m": 1.5, "vx_mps": 0.0, "vy_mps": 0.0, "vz_mps": 0.0},
            {"time_s": 10.0, "ue_id": "ue_001", "x_m": 9.0, "y_m": 9.0, "z_m": 1.5, "vx_mps": 0.0, "vy_mps": 0.0, "vz_mps": 0.0},
            {"time_s": 20.0, "ue_id": "ue_000", "x_m": 1.0, "y_m": 5.0, "z_m": 1.5, "vx_mps": 0.0, "vy_mps": 0.0, "vz_mps": 0.0},
            {"time_s": 20.0, "ue_id": "ue_001", "x_m": 9.0, "y_m": 5.0, "z_m": 1.5, "vx_mps": 0.0, "vy_mps": 0.0, "vz_mps": 0.0},
        ],
    )
    site_fields = ["site_id", "x_m", "y_m", "z_m", "yaw_deg", "pitch_deg", "mount_type", "enabled", "source", "selected"]
    _write_csv(
        output_dir / "candidate_ap_positions.csv",
        site_fields,
        [
            {"site_id": "cand_a", "x_m": 0.0, "y_m": 0.0, "z_m": 1.5, "yaw_deg": 0.0, "pitch_deg": -10.0, "mount_type": "facade", "enabled": True, "source": "wall", "selected": True},
            {"site_id": "cand_b", "x_m": 10.0, "y_m": 0.0, "z_m": 1.5, "yaw_deg": 180.0, "pitch_deg": -10.0, "mount_type": "facade", "enabled": True, "source": "wall", "selected": True},
        ],
    )
    _write_csv(
        output_dir / "central_ap_rooftop_candidates.csv",
        site_fields,
        [
            {"site_id": "roof_a", "x_m": 5.0, "y_m": 5.0, "z_m": 13.5, "yaw_deg": 0.0, "pitch_deg": -10.0, "mount_type": "rooftop", "enabled": True, "source": "roof", "selected": True},
        ],
    )
    _write_csv(
        output_dir / "central_massive_mimo_ap.csv",
        site_fields,
        [
            {"site_id": "central_ap_01", "x_m": 5.0, "y_m": 5.0, "z_m": 13.5, "yaw_deg": 0.0, "pitch_deg": -10.0, "mount_type": "rooftop", "enabled": True, "source": "selected:roof_a", "selected": True},
        ],
    )
    _write_csv(
        output_dir / "distributed_fixed_aps.csv",
        site_fields,
        [
            {"site_id": "movable_ap_01", "x_m": 0.0, "y_m": 0.0, "z_m": 1.5, "yaw_deg": 0.0, "pitch_deg": -10.0, "mount_type": "facade", "enabled": True, "source": "seed:cand_a", "selected": True},
        ],
    )
    _write_csv(
        output_dir / "distributed_movable_aps.csv",
        site_fields,
        [
            {"site_id": "movable_ap_01", "x_m": 10.0, "y_m": 0.0, "z_m": 1.5, "yaw_deg": 180.0, "pitch_deg": -10.0, "mount_type": "facade", "enabled": True, "source": "relocated:cand_b", "selected": True},
        ],
    )
    np.savez_compressed(
        output_dir / "fixed_coverage_map.npz",
        path_gain=np.ones((1, 1), dtype=float),
        rss=np.ones((1, 1), dtype=float),
        sinr=np.ones((1, 1), dtype=float),
        best_sinr_db=np.ones((1, 1), dtype=float),
        cell_centers=np.zeros((1, 1, 3), dtype=float),
    )
    np.savez_compressed(
        output_dir / "coverage_map.npz",
        path_gain=2.0 * np.ones((1, 1), dtype=float),
        rss=2.0 * np.ones((1, 1), dtype=float),
        sinr=2.0 * np.ones((1, 1), dtype=float),
        best_sinr_db=2.0 * np.ones((1, 1), dtype=float),
        cell_centers=np.zeros((1, 1, 3), dtype=float),
    )

    _write_csv(
        output_dir / "central_massive_mimo_schedule.csv",
        ["window_index", "start_time_s", "end_time_s", "ap_id", "x_m", "y_m", "z_m", "source"],
        [
            {"window_index": 0, "start_time_s": 0.0, "end_time_s": 10.0, "ap_id": "central_ap_01", "x_m": 5.0, "y_m": 5.0, "z_m": 13.5, "source": "selected:roof_a"},
            {"window_index": 1, "start_time_s": 20.0, "end_time_s": 20.0, "ap_id": "central_ap_01", "x_m": 5.0, "y_m": 5.0, "z_m": 13.5, "source": "selected:roof_a"},
        ],
    )
    _write_csv(
        output_dir / "distributed_fixed_schedule.csv",
        ["window_index", "start_time_s", "end_time_s", "ap_id", "x_m", "y_m", "z_m", "source"],
        [
            {"window_index": 0, "start_time_s": 0.0, "end_time_s": 10.0, "ap_id": "movable_ap_01", "x_m": 0.0, "y_m": 0.0, "z_m": 1.5, "source": "seed:cand_a"},
            {"window_index": 1, "start_time_s": 20.0, "end_time_s": 20.0, "ap_id": "movable_ap_01", "x_m": 0.0, "y_m": 0.0, "z_m": 1.5, "source": "seed:cand_a"},
        ],
    )
    _write_csv(
        output_dir / "distributed_movable_schedule.csv",
        ["window_index", "start_time_s", "end_time_s", "ap_id", "x_m", "y_m", "z_m", "source"],
        [
            {"window_index": 0, "start_time_s": 0.0, "end_time_s": 10.0, "ap_id": "movable_ap_01", "x_m": 0.0, "y_m": 0.0, "z_m": 1.5, "source": "relocated:cand_a"},
            {"window_index": 1, "start_time_s": 20.0, "end_time_s": 20.0, "ap_id": "movable_ap_01", "x_m": 10.0, "y_m": 0.0, "z_m": 1.5, "source": "relocated:cand_b"},
        ],
    )
    return output_dir


def test_run_strategy_summary_analysis_writes_tables(tmp_path: Path):
    output_dir = _build_fake_output_dir(tmp_path)

    artifacts = run_strategy_summary_analysis([output_dir])

    assert artifacts["csv"].exists()
    assert artifacts["markdown"].exists()
    rows = list(csv.DictReader(artifacts["csv"].open("r", encoding="utf-8", newline="")))
    assert [row["strategy"] for row in rows] == ["central_massive_mimo", "distributed_fixed", "distributed_movable"]
    assert rows[2]["is_best_strategy"] == "True"


def test_resolve_output_dir_argument_accepts_yaml_and_output_dir():
    assert resolve_output_dir_argument("scenarios/rabot.yaml").name == "rabot"
    assert resolve_output_dir_argument("outputs/rabot") == Path("outputs/rabot")


def test_run_sinr_snapshot_analysis_writes_csvs_and_plots(tmp_path: Path):
    output_dir = _build_fake_output_dir(tmp_path)

    artifacts = run_sinr_snapshot_analysis(output_dir, outage_threshold_db=3.0)

    assert artifacts["summary_csv"].exists()
    assert artifacts["per_user_csv"].exists()
    assert artifacts["threshold_csv"].exists()
    assert artifacts["cdf_plot"].exists()
    assert artifacts["cdf_plot_tikz"].exists()
    assert artifacts["threshold_plot"].exists()
    assert artifacts["esr_summary_csv"].exists()
    assert artifacts["esr_timeseries_plot"].exists()
    assert artifacts["esr_timeseries_plot_tikz"].exists()
    assert artifacts["esr_cdf_plot"].exists()
    assert artifacts["esr_window_cdf_plot"].exists()
    tikz_text = artifacts["cdf_plot_tikz"].read_text(encoding="utf-8")
    assert "\\addplot" in tikz_text
    assert "table[" in tikz_text
    assert "\\includegraphics" not in tikz_text
    assert "title={" not in tikz_text
    assert "\\definecolor{DistributedFixedColor}{HTML}{2F5D8A}" in tikz_text
    assert "\\definecolor{DistributedMovableColor}{HTML}{CB3A2A}" in tikz_text
    assert "each nth point=5" in tikz_text
    assert "xmin=-20" in tikz_text
    esr_tikz_text = artifacts["esr_timeseries_plot_tikz"].read_text(encoding="utf-8")
    assert "title={" not in esr_tikz_text
    assert "forget plot" in esr_tikz_text
    assert "window_mean_esr_bps_hz" not in esr_tikz_text
    boxplot_tikz_text = artifacts["boxplot_tikz"].read_text(encoding="utf-8")
    assert "title={" not in boxplot_tikz_text
    assert "xlabel={Per-user mean SINR [dB]}" in boxplot_tikz_text
    rows = list(csv.DictReader(artifacts["summary_csv"].open("r", encoding="utf-8", newline="")))
    assert rows[0]["strategy"] == "central_massive_mimo"
    assert rows[1]["strategy"] == "distributed_fixed"
    assert rows[2]["strategy"] == "distributed_movable"
    assert "outage_at_3db" in rows[0]
    esr_rows = {row["strategy"]: row for row in csv.DictReader(artifacts["esr_summary_csv"].open("r", encoding="utf-8", newline=""))}
    assert float(esr_rows["distributed_fixed"]["mean_esr_bps_hz"]) == 4.0


def test_run_scene_visualization_postprocess_rebuilds_visuals(tmp_path: Path, monkeypatch):
    output_dir = _build_fake_output_dir(tmp_path)

    def _fake_animate(*args, **kwargs):
        target = Path(args[5]).with_suffix(".gif")
        target.write_bytes(b"GIF89a")
        return target

    monkeypatch.setattr("cocoon_sionna.postprocess._animate_scene", _fake_animate)

    artifacts = run_scene_visualization_postprocess(output_dir)

    assert artifacts["scene_layout"].exists()
    assert artifacts["scene_layout_tikz"].exists()
    scene_layout_tikz = artifacts["scene_layout_tikz"].read_text(encoding="utf-8")
    assert "\\addplot" in scene_layout_tikz
    assert "scene_layout_central_massive_mimo.csv" in scene_layout_tikz
    assert "Central massive-MIMO BS" in scene_layout_tikz
    assert artifacts["trajectory_colormap"].exists()
    assert artifacts["trajectory_colormap_tikz"].exists()
    assert artifacts["scene_animation"].exists()
    assert artifacts["scene_animation_with_central"].exists()
    assert artifacts["fixed_coverage_plot"].exists()
    assert artifacts["fixed_coverage_plot_tikz"].exists()
    assert "fixed_coverage_grid.csv" in artifacts["fixed_coverage_plot_tikz"].read_text(encoding="utf-8")
    assert artifacts["coverage_plot"].exists()
    assert artifacts["coverage_plot_tikz"].exists()


def test_run_schedule_analysis_and_manuscript_report(tmp_path: Path):
    output_dir = _build_fake_output_dir(tmp_path)

    schedule_artifacts = run_schedule_analysis(output_dir)
    report_artifacts = run_manuscript_report(output_dir)

    assert schedule_artifacts["summary_csv"].exists()
    assert schedule_artifacts["transition_csv"].exists()
    assert schedule_artifacts["overview_tikz"].exists()
    assert "\\addplot" in schedule_artifacts["overview_tikz"].read_text(encoding="utf-8")
    schedule_rows = list(csv.DictReader(schedule_artifacts["summary_csv"].open("r", encoding="utf-8", newline="")))
    local_row = next(row for row in schedule_rows if row["strategy"] == "distributed_movable")
    assert float(local_row["total_distance_m"]) == 10.0
    assert report_artifacts["summary_markdown"].exists()
    assert report_artifacts["manifest"].exists()
    summary_text = report_artifacts["summary_markdown"].read_text(encoding="utf-8")
    assert "Best strategy: `distributed_movable`" in summary_text


def test_run_visualization_postprocess_runs_bundle(tmp_path: Path, monkeypatch):
    output_dir = _build_fake_output_dir(tmp_path)

    def _fake_animate(*args, **kwargs):
        target = Path(args[5]).with_suffix(".gif")
        target.write_bytes(b"GIF89a")
        return target

    monkeypatch.setattr("cocoon_sionna.postprocess._animate_scene", _fake_animate)

    artifacts = run_visualization_postprocess(output_dir)

    assert artifacts["scene_scene_layout"].exists()
    assert artifacts["analysis_summary_markdown"].exists()
    assert artifacts["manifest"].exists()
