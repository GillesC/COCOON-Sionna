import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

from cocoon_sionna.postprocess import (
    resolve_output_dir_argument,
    run_manuscript_report,
    run_schedule_analysis,
    run_sinr_snapshot_analysis,
    run_strategy_summary_analysis,
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
        "baseline_strategy": "random_baseline",
        "best_strategy": "local_csi_p10",
        "scene_animation_strategy": "local_csi_p10",
        "strategies": {
            "random_baseline": {
                "selected_site_ids": ["fixed_ap_01", "movable_ap_01"],
                "movable_site_ids": ["movable_ap_01"],
                "final_candidate_ids": ["cand_a"],
                "score": 1.0,
                "outage": 0.30,
                "percentile_10_db": 0.5,
                "peer_tiebreak": 2.0,
                "capped": False,
                "evaluated_combinations": 1,
            },
            "local_csi_p10": {
                "selected_site_ids": ["fixed_ap_01", "movable_ap_01"],
                "movable_site_ids": ["movable_ap_01"],
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
                "strategy": "random_baseline",
                "score": 1.0,
                "outage": 0.30,
                "percentile_10_db": 0.5,
                "peer_tiebreak": 2.0,
                "capped": False,
                "evaluated_combinations": 1,
                "final_candidate_ids": "cand_a",
            },
            {
                "strategy": "local_csi_p10",
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
        strategy_names=np.array(["random_baseline", "local_csi_p10"], dtype=object),
        random_baseline_sinr_db=np.array([[0.0, 2.0], [1.0, 3.0], [2.0, 4.0]], dtype=float),
        local_csi_p10_sinr_db=np.array([[3.0, 5.0], [4.0, 6.0], [5.0, 7.0]], dtype=float),
    )

    _write_csv(
        output_dir / "random_baseline_schedule.csv",
        ["window_index", "start_time_s", "end_time_s", "ap_id", "x_m", "y_m", "z_m", "source"],
        [
            {"window_index": 0, "start_time_s": 0.0, "end_time_s": 10.0, "ap_id": "movable_ap_01", "x_m": 0.0, "y_m": 0.0, "z_m": 1.5, "source": "seed:cand_a"},
            {"window_index": 1, "start_time_s": 20.0, "end_time_s": 20.0, "ap_id": "movable_ap_01", "x_m": 0.0, "y_m": 0.0, "z_m": 1.5, "source": "seed:cand_a"},
        ],
    )
    _write_csv(
        output_dir / "local_csi_p10_schedule.csv",
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
    assert [row["strategy"] for row in rows] == ["random_baseline", "local_csi_p10"]
    assert rows[1]["is_best_strategy"] == "True"


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
    assert artifacts["threshold_plot"].exists()
    rows = list(csv.DictReader(artifacts["summary_csv"].open("r", encoding="utf-8", newline="")))
    assert rows[0]["strategy"] == "random_baseline"
    assert rows[1]["strategy"] == "local_csi_p10"
    assert "outage_at_3db" in rows[0]


def test_run_schedule_analysis_and_manuscript_report(tmp_path: Path):
    output_dir = _build_fake_output_dir(tmp_path)

    schedule_artifacts = run_schedule_analysis(output_dir)
    report_artifacts = run_manuscript_report(output_dir)

    assert schedule_artifacts["summary_csv"].exists()
    assert schedule_artifacts["transition_csv"].exists()
    schedule_rows = list(csv.DictReader(schedule_artifacts["summary_csv"].open("r", encoding="utf-8", newline="")))
    local_row = next(row for row in schedule_rows if row["strategy"] == "local_csi_p10")
    assert float(local_row["total_distance_m"]) == 10.0
    assert report_artifacts["summary_markdown"].exists()
    assert report_artifacts["manifest"].exists()
    summary_text = report_artifacts["summary_markdown"].read_text(encoding="utf-8")
    assert "Best strategy: `local_csi_p10`" in summary_text
