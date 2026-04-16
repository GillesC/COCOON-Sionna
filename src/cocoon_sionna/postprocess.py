"""Postprocessing utilities for manuscript-oriented analysis."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from .config import load_scenario_config
from .mobility import Trajectory, load_graph_json
from .pipeline import _animate_scene, _plot_colored_trajectories, _plot_coverage, _plot_scene_layout
from .sites import load_candidate_sites

STRATEGY_ORDER = ("central_massive_mimo", "distributed_fixed", "distributed_movable")
STRATEGY_LABELS = {
    "central_massive_mimo": "Central massive MIMO",
    "distributed_fixed": "Distributed fixed",
    "distributed_movable": "Distributed movable",
}


def _assert_artifacts_exist(artifacts: dict[str, Path]) -> dict[str, Path]:
    missing = [str(path) for path in artifacts.values() if isinstance(path, Path) and not path.exists()]
    if missing:
        raise RuntimeError("Postprocessing did not produce the expected artifacts: " + ", ".join(sorted(missing)))
    return artifacts


def _ordered_strategies(names: Iterable[str]) -> list[str]:
    unique = {str(name) for name in names}
    ordered = [name for name in STRATEGY_ORDER if name in unique]
    ordered.extend(sorted(name for name in unique if name not in STRATEGY_ORDER))
    return ordered


def _label(name: str) -> str:
    return STRATEGY_LABELS.get(name, name)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv_rows(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return path


def _format_markdown_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return str(value)


def _write_markdown_table(path: Path, headers: Sequence[str], rows: Sequence[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(headers) + " |\n")
        handle.write("| " + " | ".join("---" for _ in headers) + " |\n")
        for row in rows:
            handle.write("| " + " | ".join(_format_markdown_value(row.get(header, "")) for header in headers) + " |\n")
    return path


def _analysis_dir(default_parent: Path, suffix: str, override: Path | None) -> Path:
    target = override if override is not None else default_parent / "postprocessing" / suffix
    target.mkdir(parents=True, exist_ok=True)
    return target


def resolve_output_dir_argument(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.suffix.lower() in {".yaml", ".yml"}:
        return load_scenario_config(candidate).outputs.output_dir
    return candidate


def _require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required output artifact is missing: {path}")
    return path


def _load_trajectory_csv(path: Path) -> Trajectory:
    rows = _load_csv_rows(path)
    if not rows:
        raise ValueError(f"Trajectory CSV is empty: {path}")
    times = sorted({float(row["time_s"]) for row in rows})
    ue_ids = sorted({str(row["ue_id"]) for row in rows})
    time_index = {value: idx for idx, value in enumerate(times)}
    ue_index = {value: idx for idx, value in enumerate(ue_ids)}
    positions = np.zeros((len(times), len(ue_ids), 3), dtype=float)
    velocities = np.zeros((len(times), len(ue_ids), 3), dtype=float)
    for row in rows:
        t_idx = time_index[float(row["time_s"])]
        u_idx = ue_index[str(row["ue_id"])]
        positions[t_idx, u_idx] = [float(row["x_m"]), float(row["y_m"]), float(row["z_m"])]
        velocities[t_idx, u_idx] = [float(row["vx_mps"]), float(row["vy_mps"]), float(row["vz_mps"])]
    return Trajectory(
        times_s=np.asarray(times, dtype=float),
        ue_ids=ue_ids,
        positions_m=positions,
        velocities_mps=velocities,
    )


def _summary_for_output(output_dir: Path) -> dict[str, Any]:
    return _load_json(_require_file(output_dir / "summary.json"))


def _scene_context_paths(output_dir: Path, summary: dict[str, Any]) -> tuple[Path | None, Path | None]:
    context = summary.get("scene_context", {}) if isinstance(summary.get("scene_context"), dict) else {}
    metadata_path = Path(context["scene_metadata_path"]) if context.get("scene_metadata_path") else output_dir / "scene_metadata.json"
    graph_path = Path(context["walk_graph_path"]) if context.get("walk_graph_path") else output_dir / "walk_graph.json"
    return (metadata_path if metadata_path.exists() else None, graph_path if graph_path.exists() else None)


def _load_relocation_event_times(output_dir: Path) -> np.ndarray:
    schedule_path = output_dir / "distributed_movable_schedule.csv"
    if not schedule_path.exists():
        return np.asarray([], dtype=float)
    rows = _load_schedule_rows(schedule_path)
    if not rows:
        return np.asarray([], dtype=float)
    event_times = sorted({float(row["start_time_s"]) for row in rows if int(row["window_index"]) > 0})
    return np.asarray(event_times, dtype=float)


def _analysis_windows(output_dir: Path, times_s: np.ndarray) -> list[dict[str, float | int]]:
    schedule_path = output_dir / "distributed_movable_schedule.csv"
    if schedule_path.exists():
        rows = _load_schedule_rows(schedule_path)
        windows: list[dict[str, float | int]] = []
        seen: set[int] = set()
        for row in rows:
            window_index = int(row["window_index"])
            if window_index in seen:
                continue
            seen.add(window_index)
            windows.append(
                {
                    "window_index": window_index,
                    "start_time_s": float(row["start_time_s"]),
                    "end_time_s": float(row["end_time_s"]),
                }
            )
        if windows:
            windows.sort(key=lambda row: int(row["window_index"]))
            return windows
    if times_s.size == 0:
        return []
    return [
        {
            "window_index": 0,
            "start_time_s": float(times_s[0]),
            "end_time_s": float(times_s[-1]),
        }
    ]


def _plot_relocation_markers(ax, relocation_times_s: np.ndarray) -> None:
    for time_s in np.asarray(relocation_times_s, dtype=float):
        ax.axvline(float(time_s), color="#6b7280", linestyle=":", linewidth=1.1, alpha=0.9)


def _strategy_rows_for_output(output_dir: Path) -> list[dict[str, Any]]:
    summary_path = _require_file(output_dir / "summary.json")
    comparison_path = _require_file(output_dir / "strategy_comparison.csv")
    summary = _load_json(summary_path)
    comparison_rows = _load_csv_rows(comparison_path)
    comparison_by_name = {row["strategy"]: row for row in comparison_rows}
    strategy_meta = summary.get("strategies", {})
    strategy_names = _ordered_strategies(set(comparison_by_name) | set(strategy_meta))
    rows: list[dict[str, Any]] = []
    for name in strategy_names:
        row = comparison_by_name.get(name, {})
        meta = strategy_meta.get(name, {})
        rows.append(
            {
                "scenario": summary.get("scenario", output_dir.name),
                "output_dir": str(output_dir),
                "strategy": name,
                "strategy_label": _label(name),
                "is_best_strategy": name == summary.get("best_strategy"),
                "is_baseline_strategy": name == summary.get("baseline_strategy"),
                "is_scene_animation_strategy": name == summary.get("scene_animation_strategy"),
                "score": float(row.get("score", meta.get("score", np.nan))),
                "outage": float(row.get("outage", meta.get("outage", np.nan))),
                "percentile_10_db": float(row.get("percentile_10_db", meta.get("percentile_10_db", np.nan))),
                "peer_tiebreak": float(row.get("peer_tiebreak", meta.get("peer_tiebreak", np.nan))),
                "capped": str(row.get("capped", meta.get("capped", False))).lower() in {"1", "true", "yes"},
                "evaluated_combinations": int(row.get("evaluated_combinations", meta.get("evaluated_combinations", 0))),
                "num_selected_sites": len(meta.get("selected_site_ids", [])),
                "num_movable_sites": len(meta.get("movable_site_ids", [])),
                "num_final_candidates": len(meta.get("final_candidate_ids", [])),
                "selected_site_ids": " ".join(meta.get("selected_site_ids", [])),
                "movable_site_ids": " ".join(meta.get("movable_site_ids", [])),
                "final_candidate_ids": " ".join(meta.get("final_candidate_ids", [])),
            }
        )
    return rows


def run_strategy_summary_analysis(output_dirs: Sequence[str | Path], analysis_dir: str | Path | None = None) -> dict[str, Path]:
    resolved_dirs = [Path(path) for path in output_dirs]
    if not resolved_dirs:
        raise ValueError("At least one output directory is required")
    target_dir = _analysis_dir(resolved_dirs[0], "strategy", Path(analysis_dir) if analysis_dir is not None else None)
    rows: list[dict[str, Any]] = []
    for output_dir in resolved_dirs:
        rows.extend(_strategy_rows_for_output(output_dir))
    rank = {name: index for index, name in enumerate(_ordered_strategies(row["strategy"] for row in rows))}
    rows.sort(key=lambda row: (row["scenario"], rank.get(row["strategy"], 999), row["strategy"]))

    csv_headers = [
        "scenario",
        "output_dir",
        "strategy",
        "strategy_label",
        "is_best_strategy",
        "is_baseline_strategy",
        "is_scene_animation_strategy",
        "score",
        "outage",
        "percentile_10_db",
        "peer_tiebreak",
        "capped",
        "evaluated_combinations",
        "num_selected_sites",
        "num_movable_sites",
        "num_final_candidates",
        "selected_site_ids",
        "movable_site_ids",
        "final_candidate_ids",
    ]
    markdown_headers = [
        "scenario",
        "strategy_label",
        "is_best_strategy",
        "score",
        "outage",
        "percentile_10_db",
        "capped",
        "evaluated_combinations",
    ]
    return _assert_artifacts_exist({
        "csv": _write_csv_rows(target_dir / "strategy_summary.csv", csv_headers, rows),
        "markdown": _write_markdown_table(target_dir / "strategy_summary.md", markdown_headers, rows),
    })


def _load_user_sinr_payload(output_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, dict[str, np.ndarray]]]:
    payload = np.load(_require_file(output_dir / "user_sinr_snapshots.npz"), allow_pickle=True)
    snapshot_index = np.asarray(payload["snapshot_index"], dtype=int)
    times_s = np.asarray(payload["times_s"], dtype=float)
    strategy_names = _ordered_strategies(payload["strategy_names"].tolist())
    strategies: dict[str, dict[str, np.ndarray]] = {}
    for name in strategy_names:
        sinr_db_key = f"{name}_sinr_db"
        if sinr_db_key not in payload.files:
            continue
        sinr_db = np.asarray(payload[sinr_db_key], dtype=float)
        sinr_linear_key = f"{name}_sinr_linear"
        strategy_payload: dict[str, np.ndarray] = {
            "sinr_db": sinr_db,
            "sinr_linear": (
                np.asarray(payload[sinr_linear_key], dtype=float)
                if sinr_linear_key in payload.files
                else np.power(10.0, sinr_db / 10.0)
            ),
        }
        for suffix in ("desired_power_w", "interference_power_w", "noise_power_w"):
            key = f"{name}_{suffix}"
            if key in payload.files:
                strategy_payload[suffix] = np.asarray(payload[key], dtype=float)
        strategies[name] = strategy_payload
    return snapshot_index, times_s, [str(value) for value in payload["ue_ids"].tolist()], strategies


def _cdf_points(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = np.asarray(values, dtype=float).reshape(-1)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    flat.sort()
    return flat, np.arange(1, flat.size + 1, dtype=float) / flat.size


def _save_empty_plot(path: Path, title: str, message: str) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def run_sinr_snapshot_analysis(
    output_dir: str | Path,
    analysis_dir: str | Path | None = None,
    threshold_min_db: float = -10.0,
    threshold_max_db: float = 20.0,
    threshold_step_db: float = 1.0,
    outage_threshold_db: float = 0.0,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    target_dir = _analysis_dir(output_dir, "sinr", Path(analysis_dir) if analysis_dir is not None else None)
    snapshot_index, times_s, ue_ids, strategy_payloads = _load_user_sinr_payload(output_dir)
    sinr = {name: payload["sinr_db"] for name, payload in strategy_payloads.items()}
    sinr_linear = {name: payload["sinr_linear"] for name, payload in strategy_payloads.items()}
    thresholds = np.arange(threshold_min_db, threshold_max_db + 0.5 * threshold_step_db, threshold_step_db, dtype=float)
    relocation_times_s = _load_relocation_event_times(output_dir)
    analysis_windows = _analysis_windows(output_dir, times_s)
    window_index_by_snapshot = np.full(len(times_s), -1, dtype=int)
    for window in analysis_windows:
        mask = (
            (times_s >= float(window["start_time_s"]) - 1e-9)
            & (times_s <= float(window["end_time_s"]) + 1e-9)
        )
        window_index_by_snapshot[mask] = int(window["window_index"])

    summary_rows: list[dict[str, Any]] = []
    per_user_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    esr_summary_rows: list[dict[str, Any]] = []
    esr_timeseries_rows: list[dict[str, Any]] = []
    esr_window_rows: list[dict[str, Any]] = []
    strategy_names = _ordered_strategies(sinr)
    esr_by_strategy: dict[str, np.ndarray] = {}
    esr_step_by_strategy: dict[str, np.ndarray] = {}
    for name in strategy_names:
        values = np.asarray(sinr[name], dtype=float)
        values_linear = np.asarray(sinr_linear[name], dtype=float)
        flat = values.reshape(-1)
        worst_user = np.min(values, axis=1)
        esr = np.sum(np.log2(1.0 + np.clip(values_linear, 0.0, None)), axis=1)
        esr_by_strategy[name] = esr
        summary_rows.append(
            {
                "strategy": name,
                "strategy_label": _label(name),
                "num_snapshots": values.shape[0],
                "num_users": values.shape[1],
                "mean_sinr_db": float(np.mean(flat)),
                "median_sinr_db": float(np.median(flat)),
                "percentile_10_db": float(np.percentile(flat, 10)),
                "percentile_90_db": float(np.percentile(flat, 90)),
                "min_sinr_db": float(np.min(flat)),
                "max_sinr_db": float(np.max(flat)),
                f"outage_at_{outage_threshold_db:g}db": float(np.mean(flat < outage_threshold_db)),
                "mean_worst_user_sinr_db": float(np.mean(worst_user)),
                "percentile_10_worst_user_sinr_db": float(np.percentile(worst_user, 10)),
            }
        )
        esr_summary_rows.append(
            {
                "strategy": name,
                "strategy_label": _label(name),
                "mean_esr_bps_hz": float(np.mean(esr)),
                "median_esr_bps_hz": float(np.median(esr)),
                "percentile_10_esr_bps_hz": float(np.percentile(esr, 10)),
                "percentile_90_esr_bps_hz": float(np.percentile(esr, 90)),
                "min_esr_bps_hz": float(np.min(esr)),
                "max_esr_bps_hz": float(np.max(esr)),
            }
        )
        per_user_mean = np.mean(values, axis=0)
        per_user_p10 = np.percentile(values, 10, axis=0)
        per_user_p90 = np.percentile(values, 90, axis=0)
        for ue_id, mean_value, p10_value, p90_value, user_values in zip(ue_ids, per_user_mean, per_user_p10, per_user_p90, values.T):
            per_user_rows.append(
                {
                    "strategy": name,
                    "strategy_label": _label(name),
                    "ue_id": ue_id,
                    "mean_sinr_db": float(mean_value),
                    "median_sinr_db": float(np.median(user_values)),
                    "percentile_10_db": float(p10_value),
                    "percentile_90_db": float(p90_value),
                    "min_sinr_db": float(np.min(user_values)),
                    "max_sinr_db": float(np.max(user_values)),
                    f"outage_at_{outage_threshold_db:g}db": float(np.mean(user_values < outage_threshold_db)),
                }
            )
        for threshold_db in thresholds:
            below = values < threshold_db
            threshold_rows.append(
                {
                    "strategy": name,
                    "strategy_label": _label(name),
                    "threshold_db": float(threshold_db),
                    "outage_fraction": float(np.mean(below)),
                    "mean_users_below_threshold": float(np.mean(np.sum(below, axis=1))),
                        "probability_any_user_below_threshold": float(np.mean(np.any(below, axis=1))),
                }
            )
        step_series = np.copy(esr)
        for window in analysis_windows:
            mask = window_index_by_snapshot == int(window["window_index"])
            if not np.any(mask):
                continue
            window_values = esr[mask]
            window_mean = float(np.mean(window_values))
            step_series[mask] = window_mean
            esr_window_rows.append(
                {
                    "strategy": name,
                    "strategy_label": _label(name),
                    "window_index": int(window["window_index"]),
                    "start_time_s": float(window["start_time_s"]),
                    "end_time_s": float(window["end_time_s"]),
                    "mean_esr_bps_hz": window_mean,
                    "median_esr_bps_hz": float(np.median(window_values)),
                    "percentile_10_esr_bps_hz": float(np.percentile(window_values, 10)),
                    "percentile_90_esr_bps_hz": float(np.percentile(window_values, 90)),
                }
            )
        esr_step_by_strategy[name] = step_series
        for snap_idx, time_s, window_index, value, step_value in zip(snapshot_index, times_s, window_index_by_snapshot, esr, step_series, strict=True):
            esr_timeseries_rows.append(
                {
                    "strategy": name,
                    "strategy_label": _label(name),
                    "snapshot_index": int(snap_idx),
                    "time_s": float(time_s),
                    "window_index": int(window_index),
                    "esr_bps_hz": float(value),
                    "window_mean_esr_bps_hz": float(step_value),
                }
            )

    artifacts = {
        "summary_csv": _write_csv_rows(
            target_dir / "sinr_snapshot_summary.csv",
            summary_rows[0].keys() if summary_rows else [],
            summary_rows,
        ),
        "per_user_csv": _write_csv_rows(
            target_dir / "per_user_sinr_summary.csv",
            per_user_rows[0].keys() if per_user_rows else [],
            per_user_rows,
        ),
        "threshold_csv": _write_csv_rows(
            target_dir / "sinr_threshold_sweep.csv",
            threshold_rows[0].keys() if threshold_rows else [],
            threshold_rows,
        ),
        "summary_markdown": _write_markdown_table(
            target_dir / "sinr_snapshot_summary.md",
            ["strategy_label", "mean_sinr_db", "percentile_10_db", "percentile_90_db", f"outage_at_{outage_threshold_db:g}db"],
            summary_rows,
        ),
        "esr_summary_csv": _write_csv_rows(
            target_dir / "esr_snapshot_summary.csv",
            esr_summary_rows[0].keys() if esr_summary_rows else [],
            esr_summary_rows,
        ),
        "esr_timeseries_csv": _write_csv_rows(
            target_dir / "esr_timeseries.csv",
            esr_timeseries_rows[0].keys() if esr_timeseries_rows else [],
            esr_timeseries_rows,
        ),
        "esr_window_csv": _write_csv_rows(
            target_dir / "esr_window_summary.csv",
            esr_window_rows[0].keys() if esr_window_rows else [],
            esr_window_rows,
        ),
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    for name in strategy_names:
        x_values, y_values = _cdf_points(sinr[name])
        if x_values.size:
            ax.step(x_values, y_values, where="post", linewidth=2.0, label=_label(name))
    ax.set_xlabel("SINR per user snapshot [dB]")
    ax.set_ylabel("CDF")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    artifacts["cdf_plot"] = target_dir / "sinr_cdf_publication.png"
    fig.savefig(artifacts["cdf_plot"], dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    for name in strategy_names:
        strategy_rows = [row for row in threshold_rows if row["strategy"] == name]
        ax.plot(
            [row["threshold_db"] for row in strategy_rows],
            [row["outage_fraction"] for row in strategy_rows],
            linewidth=2.0,
            label=_label(name),
        )
    ax.set_xlabel("SINR threshold [dB]")
    ax.set_ylabel("Fraction of user snapshots below threshold")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    artifacts["threshold_plot"] = target_dir / "sinr_threshold_sweep.png"
    fig.savefig(artifacts["threshold_plot"], dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for name in strategy_names:
        ax.plot(times_s, np.min(sinr[name], axis=1), linewidth=1.8, label=_label(name))
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Worst-user SINR [dB]")
    _plot_relocation_markers(ax, relocation_times_s)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    artifacts["worst_user_plot"] = target_dir / "worst_user_sinr_timeseries.png"
    fig.savefig(artifacts["worst_user_plot"], dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for name in strategy_names:
        ax.plot(
            times_s,
            np.sum(np.asarray(sinr[name]) < outage_threshold_db, axis=1),
            linewidth=1.8,
            label=_label(name),
        )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(f"Users below {outage_threshold_db:g} dB")
    _plot_relocation_markers(ax, relocation_times_s)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    artifacts["users_below_plot"] = target_dir / "users_below_threshold_timeseries.png"
    fig.savefig(artifacts["users_below_plot"], dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    box_data = [np.mean(sinr[name], axis=0) for name in strategy_names]
    if box_data:
        ax.boxplot(box_data, tick_labels=[_label(name) for name in strategy_names], showmeans=True)
        ax.set_ylabel("Per-user mean SINR [dB]")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        artifacts["boxplot"] = target_dir / "per_user_mean_sinr_boxplot.png"
        fig.savefig(artifacts["boxplot"], dpi=220)
        plt.close(fig)
    else:
        plt.close(fig)
        artifacts["boxplot"] = _save_empty_plot(
            target_dir / "per_user_mean_sinr_boxplot.png",
            "Per-user mean SINR",
            "No SINR samples available",
        )

    fig, ax = plt.subplots(figsize=(9, 5))
    for name in strategy_names:
        ax.plot(times_s, esr_by_strategy[name], linewidth=1.0, alpha=0.35)
        ax.step(times_s, esr_step_by_strategy[name], where="post", linewidth=2.2, label=_label(name))
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("ESR [bit/s/Hz]")
    _plot_relocation_markers(ax, relocation_times_s)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    artifacts["esr_timeseries_plot"] = target_dir / "esr_timeseries.png"
    fig.savefig(artifacts["esr_timeseries_plot"], dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    for name in strategy_names:
        x_values, y_values = _cdf_points(esr_by_strategy[name])
        if x_values.size:
            ax.step(x_values, y_values, where="post", linewidth=2.0, label=_label(name))
    ax.set_xlabel("ESR [bit/s/Hz]")
    ax.set_ylabel("CDF")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    artifacts["esr_cdf_plot"] = target_dir / "esr_cdf.png"
    fig.savefig(artifacts["esr_cdf_plot"], dpi=220)
    plt.close(fig)

    if strategy_names and analysis_windows:
        fig, axes = plt.subplots(len(strategy_names), 1, figsize=(8, max(4.5, 3.2 * len(strategy_names))), squeeze=False)
        for axis, name in zip(axes[:, 0], strategy_names, strict=True):
            plotted = False
            for window in analysis_windows:
                mask = window_index_by_snapshot == int(window["window_index"])
                if not np.any(mask):
                    continue
                x_values, y_values = _cdf_points(esr_by_strategy[name][mask])
                if x_values.size == 0:
                    continue
                axis.step(
                    x_values,
                    y_values,
                    where="post",
                    linewidth=1.8,
                    label=f"W{int(window['window_index'])}",
                )
                plotted = True
            axis.set_title(_label(name))
            axis.set_xlabel("ESR [bit/s/Hz]")
            axis.set_ylabel("CDF")
            axis.grid(True, alpha=0.3)
            if plotted:
                axis.legend(ncols=min(4, max(1, len(analysis_windows))))
        fig.tight_layout()
        artifacts["esr_window_cdf_plot"] = target_dir / "esr_time_conditioned_cdf.png"
        fig.savefig(artifacts["esr_window_cdf_plot"], dpi=220)
        plt.close(fig)
    else:
        artifacts["esr_window_cdf_plot"] = _save_empty_plot(
            target_dir / "esr_time_conditioned_cdf.png",
            "Time-conditioned ESR CDF",
            "No ESR windows available",
        )

    return _assert_artifacts_exist(artifacts)


def _load_schedule_rows(path: Path) -> list[dict[str, Any]]:
    rows = _load_csv_rows(path)
    parsed: list[dict[str, Any]] = []
    for row in rows:
        parsed.append(
            {
                "window_index": int(row["window_index"]),
                "start_time_s": float(row["start_time_s"]),
                "end_time_s": float(row["end_time_s"]),
                "ap_id": str(row["ap_id"]),
                "x_m": float(row["x_m"]),
                "y_m": float(row["y_m"]),
                "z_m": float(row["z_m"]),
                "source": str(row.get("source", "")),
            }
        )
    return parsed


def _schedule_strategy_files(output_dir: Path) -> list[tuple[str, Path]]:
    pairs: list[tuple[str, Path]] = []
    for path in sorted(output_dir.glob("*_schedule.csv")):
        strategy = path.stem.removesuffix("_schedule")
        if strategy == "mobile_ap":
            continue
        pairs.append((strategy, path))
    if not pairs:
        raise FileNotFoundError(f"No per-strategy schedule CSVs were found in {output_dir}")
    return pairs


def run_schedule_analysis(output_dir: str | Path, analysis_dir: str | Path | None = None) -> dict[str, Path]:
    output_dir = Path(output_dir)
    target_dir = _analysis_dir(output_dir, "schedule", Path(analysis_dir) if analysis_dir is not None else None)
    transition_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    strategies = _ordered_strategies(name for name, _ in _schedule_strategy_files(output_dir))

    for strategy in strategies:
        path = output_dir / f"{strategy}_schedule.csv"
        if not path.exists():
            continue
        rows = _load_schedule_rows(path)
        by_ap: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_ap.setdefault(row["ap_id"], []).append(row)
        for ap_id in by_ap:
            by_ap[ap_id].sort(key=lambda row: row["window_index"])

        move_distances: list[float] = []
        unique_positions = {
            (row["ap_id"], row["x_m"], row["y_m"], row["z_m"])
            for row in rows
        }
        for ap_id, sequence in by_ap.items():
            for previous, current in zip(sequence[:-1], sequence[1:]):
                distance_m = float(
                    np.linalg.norm(
                        np.asarray(
                            [
                                current["x_m"] - previous["x_m"],
                                current["y_m"] - previous["y_m"],
                                current["z_m"] - previous["z_m"],
                            ],
                            dtype=float,
                        )
                    )
                )
                moved = distance_m > 1e-9
                if moved:
                    move_distances.append(distance_m)
                transition_rows.append(
                    {
                        "strategy": strategy,
                        "strategy_label": _label(strategy),
                        "ap_id": ap_id,
                        "from_window_index": previous["window_index"],
                        "to_window_index": current["window_index"],
                        "from_start_time_s": previous["start_time_s"],
                        "to_start_time_s": current["start_time_s"],
                        "distance_m": distance_m,
                        "moved": moved,
                        "from_source": previous["source"],
                        "to_source": current["source"],
                    }
                )

        total_transitions = sum(max(0, len(sequence) - 1) for sequence in by_ap.values())
        summary_rows.append(
            {
                "strategy": strategy,
                "strategy_label": _label(strategy),
                "num_windows": len({row["window_index"] for row in rows}),
                "num_movable_aps": len(by_ap),
                "num_transitions": total_transitions,
                "num_moves": len(move_distances),
                "move_fraction": float(len(move_distances) / total_transitions) if total_transitions else 0.0,
                "total_distance_m": float(np.sum(move_distances)) if move_distances else 0.0,
                "mean_move_distance_m": float(np.mean(move_distances)) if move_distances else 0.0,
                "median_move_distance_m": float(np.median(move_distances)) if move_distances else 0.0,
                "max_move_distance_m": float(np.max(move_distances)) if move_distances else 0.0,
                "unique_positions_used": len(unique_positions),
            }
        )

    artifacts = {
        "summary_csv": _write_csv_rows(
            target_dir / "schedule_mobility_summary.csv",
            summary_rows[0].keys() if summary_rows else [],
            summary_rows,
        ),
        "transition_csv": _write_csv_rows(
            target_dir / "schedule_transition_distances.csv",
            transition_rows[0].keys() if transition_rows else [],
            transition_rows,
        ),
        "summary_markdown": _write_markdown_table(
            target_dir / "schedule_mobility_summary.md",
            ["strategy_label", "num_windows", "num_moves", "move_fraction", "total_distance_m", "mean_move_distance_m"],
            summary_rows,
        ),
    }

    move_rows = [row for row in transition_rows if row["moved"]]
    histogram_path = target_dir / "schedule_transition_distance_histogram.png"
    if move_rows:
        fig, ax = plt.subplots(figsize=(8, 6))
        for strategy in strategies:
            values = [row["distance_m"] for row in move_rows if row["strategy"] == strategy]
            if values:
                ax.hist(values, bins=min(12, max(4, len(values))), alpha=0.45, label=_label(strategy))
        ax.set_xlabel("Relocation distance [m]")
        ax.set_ylabel("Count")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(histogram_path, dpi=220)
        plt.close(fig)
    else:
        _save_empty_plot(histogram_path, "Relocation distances", "No AP relocations were observed")
    artifacts["histogram"] = histogram_path

    overview_path = target_dir / "schedule_strategy_overview.png"
    if summary_rows:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        labels = [_label(row["strategy"]) for row in summary_rows]
        axes[0].bar(labels, [row["total_distance_m"] for row in summary_rows], color="#4c78a8")
        axes[0].set_ylabel("Total relocation distance [m]")
        axes[0].tick_params(axis="x", rotation=15)
        axes[0].grid(True, axis="y", alpha=0.3)
        axes[1].bar(labels, [row["move_fraction"] for row in summary_rows], color="#f58518")
        axes[1].set_ylabel("Fraction of AP transitions that moved")
        axes[1].tick_params(axis="x", rotation=15)
        axes[1].set_ylim(0.0, 1.0)
        axes[1].grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(overview_path, dpi=220)
        plt.close(fig)
    else:
        _save_empty_plot(overview_path, "Schedule overview", "No schedule rows were available")
    artifacts["overview"] = overview_path

    return _assert_artifacts_exist(artifacts)


def _strategy_site_path(output_dir: Path, strategy: str) -> Path:
    if strategy == "central_massive_mimo":
        return output_dir / "central_massive_mimo_ap.csv"
    return output_dir / f"{strategy}_aps.csv"


def run_scene_visualization_postprocess(
    output_dir: str | Path,
    scene_animation_speedup: float | None = None,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    summary = _summary_for_output(output_dir)
    metadata_path, graph_path = _scene_context_paths(output_dir, summary)
    if graph_path is None:
        raise FileNotFoundError(f"Scene walk graph is missing from {output_dir}")

    metadata = _load_json(metadata_path) if metadata_path is not None else None
    graph = load_graph_json(graph_path)
    trajectory = _load_trajectory_csv(_require_file(output_dir / "trajectory.csv"))
    candidate_sites = load_candidate_sites(output_dir / "candidate_ap_positions.csv") if (output_dir / "candidate_ap_positions.csv").exists() else []
    rooftop_sites = (
        load_candidate_sites(output_dir / "central_ap_rooftop_candidates.csv")
        if (output_dir / "central_ap_rooftop_candidates.csv").exists()
        else []
    )

    best_strategy = str(summary.get("best_strategy", "distributed_fixed"))
    animation_strategy = str(summary.get("scene_animation_strategy", best_strategy))
    baseline_strategy = str(summary.get("baseline_strategy", "distributed_fixed"))
    best_sites = load_candidate_sites(_require_file(_strategy_site_path(output_dir, best_strategy)))
    animation_sites = load_candidate_sites(_require_file(_strategy_site_path(output_dir, animation_strategy)))
    baseline_sites = load_candidate_sites(_require_file(_strategy_site_path(output_dir, baseline_strategy)))
    central_sites = load_candidate_sites(_require_file(_strategy_site_path(output_dir, "central_massive_mimo")))
    animation_schedule_rows = _load_schedule_rows(_require_file(output_dir / f"{animation_strategy}_schedule.csv"))
    speedup = float(scene_animation_speedup if scene_animation_speedup is not None else summary.get("scene_animation_speedup", 1.0))

    artifacts: dict[str, Path] = {}
    _plot_scene_layout(
        metadata,
        graph,
        [*candidate_sites, *rooftop_sites],
        best_sites,
        trajectory,
        output_dir / "scene_layout.png",
    )
    artifacts["scene_layout"] = output_dir / "scene_layout.png"
    _plot_colored_trajectories(trajectory, output_dir / "trajectory_colormap.png")
    artifacts["trajectory_colormap"] = output_dir / "trajectory_colormap.png"

    animation_path = _animate_scene(
        metadata,
        graph,
        [*candidate_sites, *rooftop_sites],
        animation_sites,
        trajectory,
        output_dir / "scene_animation.mp4",
        speedup=speedup,
        schedule_rows=animation_schedule_rows,
    )
    if animation_path is not None:
        artifacts["scene_animation"] = animation_path
    comparison_animation_path = _animate_scene(
        metadata,
        graph,
        [*candidate_sites, *rooftop_sites],
        animation_sites,
        trajectory,
        output_dir / "scene_animation_with_central_massive_mimo.mp4",
        speedup=speedup,
        schedule_rows=animation_schedule_rows,
        reference_sites=central_sites,
        reference_label="Central massive-MIMO BS",
    )
    if comparison_animation_path is not None:
        artifacts["scene_animation_with_central"] = comparison_animation_path

    fixed_coverage_path = output_dir / "fixed_coverage_map.npz"
    if fixed_coverage_path.exists():
        with np.load(fixed_coverage_path, allow_pickle=True) as payload:
            _plot_coverage(
                np.asarray(payload["best_sinr_db"], dtype=float),
                np.asarray(payload["cell_centers"], dtype=float),
                baseline_sites,
                trajectory,
                output_dir / "fixed_coverage_map.png",
            )
        artifacts["fixed_coverage_plot"] = output_dir / "fixed_coverage_map.png"

    coverage_path = output_dir / "coverage_map.npz"
    if coverage_path.exists():
        with np.load(coverage_path, allow_pickle=True) as payload:
            _plot_coverage(
                np.asarray(payload["best_sinr_db"], dtype=float),
                np.asarray(payload["cell_centers"], dtype=float),
                best_sites,
                trajectory,
                output_dir / "coverage_map.png",
            )
        artifacts["coverage_plot"] = output_dir / "coverage_map.png"

    return _assert_artifacts_exist(artifacts)


def run_manuscript_report(
    output_dir: str | Path,
    analysis_dir: str | Path | None = None,
    threshold_min_db: float = -10.0,
    threshold_max_db: float = 20.0,
    threshold_step_db: float = 1.0,
    outage_threshold_db: float = 0.0,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    target_dir = _analysis_dir(output_dir, "", Path(analysis_dir) if analysis_dir is not None else output_dir / "postprocessing")
    strategy_artifacts = run_strategy_summary_analysis([output_dir], target_dir / "strategy")
    sinr_artifacts = run_sinr_snapshot_analysis(
        output_dir,
        target_dir / "sinr",
        threshold_min_db=threshold_min_db,
        threshold_max_db=threshold_max_db,
        threshold_step_db=threshold_step_db,
        outage_threshold_db=outage_threshold_db,
    )
    schedule_artifacts = run_schedule_analysis(output_dir, target_dir / "schedule")

    strategy_rows = _strategy_rows_for_output(output_dir)
    summary_rows = _load_csv_rows(sinr_artifacts["summary_csv"])
    schedule_rows = _load_csv_rows(schedule_artifacts["summary_csv"])
    best_row = next((row for row in strategy_rows if row["is_best_strategy"]), strategy_rows[0] if strategy_rows else None)
    baseline_row = next((row for row in strategy_rows if row["is_baseline_strategy"]), None)
    sinr_by_strategy = {row["strategy"]: row for row in summary_rows}
    schedule_by_strategy = {row["strategy"]: row for row in schedule_rows}

    summary_path = target_dir / "manuscript_summary.md"
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("# Manuscript Analysis Summary\n\n")
        handle.write(f"- Output directory: `{output_dir}`\n")
        if best_row is not None:
            handle.write(f"- Best strategy: `{best_row['strategy']}` ({best_row['strategy_label']})\n")
            handle.write(f"- Best strategy score: {best_row['score']:.3f}\n")
            handle.write(f"- Best strategy outage: {best_row['outage']:.3f}\n")
            handle.write(f"- Best strategy trajectory P10 SINR: {best_row['percentile_10_db']:.3f} dB\n")
        if best_row is not None and baseline_row is not None and best_row["strategy"] in sinr_by_strategy and baseline_row["strategy"] in sinr_by_strategy:
            best_sinr = sinr_by_strategy[best_row["strategy"]]
            baseline_sinr = sinr_by_strategy[baseline_row["strategy"]]
            delta_p10 = float(best_sinr["percentile_10_db"]) - float(baseline_sinr["percentile_10_db"])
            delta_mean = float(best_sinr["mean_sinr_db"]) - float(baseline_sinr["mean_sinr_db"])
            handle.write(f"- P10 SINR gain over baseline: {delta_p10:.3f} dB\n")
            handle.write(f"- Mean SINR gain over baseline: {delta_mean:.3f} dB\n")
        handle.write("\n## Produced Artifacts\n\n")
        for group_name, artifacts in (
            ("Strategy tables", strategy_artifacts),
            ("SINR analysis", sinr_artifacts),
            ("Schedule analysis", schedule_artifacts),
        ):
            handle.write(f"### {group_name}\n\n")
            for artifact_name, artifact_path in artifacts.items():
                handle.write(f"- `{artifact_name}`: `{artifact_path}`\n")
            handle.write("\n")
        if best_row is not None and best_row["strategy"] in schedule_by_strategy:
            schedule_row = schedule_by_strategy[best_row["strategy"]]
            handle.write("## Mobility Notes\n\n")
            handle.write(
                f"- Best strategy move fraction: {float(schedule_row['move_fraction']):.3f}\n"
            )
            handle.write(
                f"- Best strategy total relocation distance: {float(schedule_row['total_distance_m']):.3f} m\n"
            )

    manifest = {
        "strategy": {key: str(value) for key, value in strategy_artifacts.items()},
        "sinr": {key: str(value) for key, value in sinr_artifacts.items()},
        "schedule": {key: str(value) for key, value in schedule_artifacts.items()},
        "summary_markdown": str(summary_path),
    }
    manifest_path = target_dir / "report_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return _assert_artifacts_exist({
        **{f"strategy_{key}": value for key, value in strategy_artifacts.items()},
        **{f"sinr_{key}": value for key, value in sinr_artifacts.items()},
        **{f"schedule_{key}": value for key, value in schedule_artifacts.items()},
        "summary_markdown": summary_path,
        "manifest": manifest_path,
    })


def run_visualization_postprocess(
    output_dir: str | Path,
    analysis_dir: str | Path | None = None,
    threshold_min_db: float = -10.0,
    threshold_max_db: float = 20.0,
    threshold_step_db: float = 1.0,
    outage_threshold_db: float = 0.0,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    target_dir = _analysis_dir(output_dir, "", Path(analysis_dir) if analysis_dir is not None else output_dir / "postprocessing")
    scene_artifacts = run_scene_visualization_postprocess(output_dir)

    analysis_artifacts: dict[str, Path] = {}
    if (output_dir / "user_sinr_snapshots.npz").exists():
        analysis_artifacts = run_manuscript_report(
            output_dir,
            target_dir,
            threshold_min_db=threshold_min_db,
            threshold_max_db=threshold_max_db,
            threshold_step_db=threshold_step_db,
            outage_threshold_db=outage_threshold_db,
        )
    else:
        analysis_artifacts.update(
            {
                f"strategy_{key}": value
                for key, value in run_strategy_summary_analysis([output_dir], target_dir / "strategy").items()
            }
        )
        analysis_artifacts.update(
            {
                f"schedule_{key}": value
                for key, value in run_schedule_analysis(output_dir, target_dir / "schedule").items()
            }
        )

    manifest_path = target_dir / "visualization_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "scene": {key: str(value) for key, value in scene_artifacts.items()},
                "analysis": {key: str(value) for key, value in analysis_artifacts.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return _assert_artifacts_exist(
        {
            **{f"scene_{key}": value for key, value in scene_artifacts.items()},
            **{f"analysis_{key}": value for key, value in analysis_artifacts.items()},
            "manifest": manifest_path,
        }
    )


def _strategy_summary_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate strategy-level performance tables from one or more output directories")
    parser.add_argument("output_dirs", nargs="+", help="One or more scenario output directories")
    parser.add_argument("--analysis-dir", type=Path, default=None, help="Directory where the tables should be written")
    return parser


def _sinr_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create publication-oriented SINR analyses from user_sinr_snapshots.npz")
    parser.add_argument("output_dir", help="Scenario output directory")
    parser.add_argument("--analysis-dir", type=Path, default=None, help="Directory where the analysis files should be written")
    parser.add_argument("--threshold-min-db", type=float, default=-10.0)
    parser.add_argument("--threshold-max-db", type=float, default=20.0)
    parser.add_argument("--threshold-step-db", type=float, default=1.0)
    parser.add_argument("--outage-threshold-db", type=float, default=0.0)
    return parser


def _schedule_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze per-strategy AP relocation schedules")
    parser.add_argument("output_dir", help="Scenario output directory")
    parser.add_argument("--analysis-dir", type=Path, default=None, help="Directory where the analysis files should be written")
    return parser


def _manuscript_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run all manuscript-oriented postprocessing analyses for one output directory")
    parser.add_argument("output_dir", help="Scenario output directory")
    parser.add_argument("--analysis-dir", type=Path, default=None, help="Directory where the report files should be written")
    parser.add_argument("--threshold-min-db", type=float, default=-10.0)
    parser.add_argument("--threshold-max-db", type=float, default=20.0)
    parser.add_argument("--threshold-step-db", type=float, default=1.0)
    parser.add_argument("--outage-threshold-db", type=float, default=0.0)
    return parser


def _postprocess_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rebuild all visualizations and analyses from stored scenario outputs")
    parser.add_argument("target", help="Scenario YAML path or scenario output directory")
    parser.add_argument("--analysis-dir", type=Path, default=None, help="Directory where the postprocessing files should be written")
    parser.add_argument("--threshold-min-db", type=float, default=-10.0)
    parser.add_argument("--threshold-max-db", type=float, default=20.0)
    parser.add_argument("--threshold-step-db", type=float, default=1.0)
    parser.add_argument("--outage-threshold-db", type=float, default=0.0)
    return parser


def main_strategy_summary() -> None:
    args = _strategy_summary_parser().parse_args()
    artifacts = run_strategy_summary_analysis(args.output_dirs, args.analysis_dir)
    print(json.dumps({key: str(value) for key, value in artifacts.items()}, indent=2))


def main_sinr_analysis() -> None:
    args = _sinr_parser().parse_args()
    artifacts = run_sinr_snapshot_analysis(
        args.output_dir,
        args.analysis_dir,
        threshold_min_db=args.threshold_min_db,
        threshold_max_db=args.threshold_max_db,
        threshold_step_db=args.threshold_step_db,
        outage_threshold_db=args.outage_threshold_db,
    )
    print(json.dumps({key: str(value) for key, value in artifacts.items()}, indent=2))


def main_schedule_analysis() -> None:
    args = _schedule_parser().parse_args()
    artifacts = run_schedule_analysis(args.output_dir, args.analysis_dir)
    print(json.dumps({key: str(value) for key, value in artifacts.items()}, indent=2))


def main_manuscript_report() -> None:
    args = _manuscript_parser().parse_args()
    artifacts = run_manuscript_report(
        args.output_dir,
        args.analysis_dir,
        threshold_min_db=args.threshold_min_db,
        threshold_max_db=args.threshold_max_db,
        threshold_step_db=args.threshold_step_db,
        outage_threshold_db=args.outage_threshold_db,
    )
    print(json.dumps({key: str(value) for key, value in artifacts.items()}, indent=2))


def main_visualization_postprocess() -> None:
    args = _postprocess_parser().parse_args()
    output_dir = resolve_output_dir_argument(args.target)
    artifacts = run_visualization_postprocess(
        output_dir,
        args.analysis_dir,
        threshold_min_db=args.threshold_min_db,
        threshold_max_db=args.threshold_max_db,
        threshold_step_db=args.threshold_step_db,
        outage_threshold_db=args.outage_threshold_db,
    )
    print(json.dumps({key: str(value) for key, value in artifacts.items()}, indent=2))
