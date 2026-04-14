import csv

from cocoon_sionna.mobile_ap_schedule_viz import _load_trajectory_csv, load_mobile_ap_schedule


def test_load_mobile_ap_schedule_groups_rows_by_window(tmp_path):
    schedule_path = tmp_path / "mobile_ap_schedule.csv"
    with schedule_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["window_index", "start_time_s", "end_time_s", "ap_id", "x_m", "y_m", "z_m", "source"])
        writer.writerow([0, 0.0, 10.0, "ap_a", 1.0, 2.0, 7.0, "wall"])
        writer.writerow([0, 0.0, 10.0, "ap_b", 3.0, 4.0, 7.0, "wall"])
        writer.writerow([1, 10.0, 20.0, "ap_a", 5.0, 6.0, 7.0, "wall"])

    windows = load_mobile_ap_schedule(schedule_path)

    assert len(windows) == 2
    assert windows[0].window_index == 0
    assert [site.site_id for site in windows[0].sites] == ["ap_a", "ap_b"]
    assert windows[1].start_time_s == 10.0
    assert windows[1].sites[0].x_m == 5.0


def test_load_trajectory_csv_reconstructs_positions_and_velocities(tmp_path):
    trajectory_path = tmp_path / "trajectory.csv"
    with trajectory_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_s", "ue_id", "x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps"])
        writer.writerow([0.0, "ue_001", 1.0, 2.0, 1.5, 0.1, 0.2, 0.0])
        writer.writerow([0.0, "ue_000", 3.0, 4.0, 1.5, 0.3, 0.4, 0.0])
        writer.writerow([1.0, "ue_001", 5.0, 6.0, 1.5, 0.5, 0.6, 0.0])
        writer.writerow([1.0, "ue_000", 7.0, 8.0, 1.5, 0.7, 0.8, 0.0])

    trajectory = _load_trajectory_csv(trajectory_path)

    assert trajectory.ue_ids == ["ue_000", "ue_001"]
    assert trajectory.times_s.tolist() == [0.0, 1.0]
    assert trajectory.positions_m[0, 0].tolist() == [3.0, 4.0, 1.5]
    assert trajectory.positions_m[1, 1].tolist() == [5.0, 6.0, 1.5]
    assert trajectory.velocities_mps[1, 0].tolist() == [0.7, 0.8, 0.0]
