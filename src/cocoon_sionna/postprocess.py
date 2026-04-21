"""Postprocessing utilities for manuscript-oriented analysis."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
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

STRATEGY_ORDER = (
    "central_massive_mimo",
    "distributed_fixed",
    "distributed_movable",
    "distributed_movable_optimization_2",
    "distributed_movable_optimization_3",
)
STRATEGY_LABELS = {
    "central_massive_mimo": "Central massive MIMO",
    "distributed_fixed": "Distributed fixed",
    "distributed_movable": "Distributed movable (Opt. 1)",
    "distributed_movable_optimization_2": "Distributed movable (Opt. 2)",
    "distributed_movable_optimization_3": "Distributed movable (Opt. 3)",
}
STRATEGY_COLORS = {
    "central_massive_mimo": "#d4a017",
    "distributed_fixed": "#2f5d8a",
    "distributed_movable": "#cb3a2a",
    "distributed_movable_optimization_2": "#1b9e77",
    "distributed_movable_optimization_3": "#7570b3",
}
STRATEGY_TIKZ_COLOR_NAMES = {
    "central_massive_mimo": "CentralMassiveMimoColor",
    "distributed_fixed": "DistributedFixedColor",
    "distributed_movable": "DistributedMovableColor",
    "distributed_movable_optimization_2": "DistributedMovableOptTwoColor",
    "distributed_movable_optimization_3": "DistributedMovableOptThreeColor",
}
STRATEGY_TEX_MACROS = {
    "central_massive_mimo": "strategy1",
    "distributed_fixed": "strategy2",
    "distributed_movable": "strategy3",
    "distributed_movable_optimization_2": "strategy4",
    "distributed_movable_optimization_3": "strategy5",
}
STRATEGY_LINESTYLES = {
    "central_massive_mimo": "-",
    "distributed_fixed": "-",
    "distributed_movable": "-",
    "distributed_movable_optimization_2": "-",
    "distributed_movable_optimization_3": "-",
}
WINDOW_LINESTYLES = ("-",)
PGFPLOTS_LINESTYLES = {
    "-": "solid",
    "--": "dashed",
    ":": "dotted",
    "-.": "dash dot",
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


def _strategy_tex_label(name: str, *, short: bool = False) -> str:
    macro_name = STRATEGY_TEX_MACROS.get(name)
    if macro_name is None:
        return _label(name)
    return f"\\{macro_name}[short]" if short else f"\\{macro_name}"


def _strategy_color(name: str) -> str:
    return STRATEGY_COLORS.get(name, "#4c566a")


def _strategy_linestyle(name: str) -> str:
    return STRATEGY_LINESTYLES.get(name, "-")


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


def _tikz_companion_path(path: Path) -> Path:
    return path.with_suffix(".tex")


def _tikz_data_dir(base_dir: Path) -> Path:
    target = base_dir / "tikz_data"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_").lower()
    return slug or "data"


def _relative_posix_path(path: Path, start: Path) -> str:
    return os.path.relpath(path, start).replace("\\", "/")


def _write_tikz_file(path: Path, body_lines: Sequence[str], *, extra_comments: Sequence[str] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"% Auto-generated PGFPlots figure for {path.stem}",
        "% Requires: \\usepackage{pgfplots,pgfplotstable,xcolor}",
        "% Optional: \\usepgfplotslibrary{groupplots}",
        "\\providecommand{\\postprocessfigurewidth}{\\linewidth}",
        "\\pgfplotsset{compat=1.18}",
    ]
    defined_colors: set[str] = set()
    for strategy in _ordered_strategies(STRATEGY_TIKZ_COLOR_NAMES):
        color_name = STRATEGY_TIKZ_COLOR_NAMES[strategy]
        if color_name in defined_colors:
            continue
        defined_colors.add(color_name)
        color_value = _strategy_color(strategy).lstrip("#").upper()
        lines.append(f"\\definecolor{{{color_name}}}{{HTML}}{{{color_value}}}")
    if extra_comments:
        lines.extend(extra_comments)
    lines.extend(body_lines)
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _save_figure(fig, path: Path, dpi: int = 220) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    return path


def _add_plot_artifact(artifacts: dict[str, Path], key: str, path: Path, *, tikz_path: Path | None = None) -> None:
    artifacts[key] = path
    if tikz_path is not None:
        artifacts[f"{key}_tikz"] = tikz_path


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


def _relocation_schedule_path(output_dir: Path) -> Path | None:
    for strategy in STRATEGY_ORDER:
        if strategy in {"central_massive_mimo", "distributed_fixed"}:
            continue
        candidate = output_dir / f"{strategy}_schedule.csv"
        if candidate.exists():
            return candidate
    for candidate in sorted(output_dir.glob("*_schedule.csv")):
        if candidate.exists():
            return candidate
    return None


def _load_relocation_event_times(output_dir: Path) -> np.ndarray:
    schedule_path = _relocation_schedule_path(output_dir)
    if schedule_path is None:
        return np.asarray([], dtype=float)
    rows = _load_schedule_rows(schedule_path)
    if not rows:
        return np.asarray([], dtype=float)
    event_times = sorted({float(row["start_time_s"]) for row in rows if int(row["window_index"]) > 0})
    return np.asarray(event_times, dtype=float)


def _analysis_windows(output_dir: Path, times_s: np.ndarray) -> list[dict[str, float | int]]:
    schedule_path = _relocation_schedule_path(output_dir)
    if schedule_path is not None:
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
        spectral_efficiency_key = f"{name}_spectral_efficiency_bps_hz"
        if spectral_efficiency_key in payload.files:
            strategy_payload["spectral_efficiency_bps_hz"] = np.asarray(payload[spectral_efficiency_key], dtype=float)
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
    _save_figure(fig, path, dpi=200)
    plt.close(fig)
    return path


def _empty_tikz(path: Path, title: str, message: str) -> Path:
    return _write_tikz_file(
        path,
        [
            "\\begin{tikzpicture}",
            "  \\node[align=center] at (0,0) {%s\\\\%s};" % (title, message),
            "\\end{tikzpicture}",
        ],
    )


def _tikz_color_name(strategy: str) -> str:
    return STRATEGY_TIKZ_COLOR_NAMES.get(strategy, "black")


def _tikz_line_style(strategy: str) -> str:
    return PGFPLOTS_LINESTYLES.get(_strategy_linestyle(strategy), "solid")


def _y_limits(values: Sequence[float] | np.ndarray, margin_ratio: float = 0.06) -> tuple[float, float]:
    flat = np.asarray(values, dtype=float).reshape(-1)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return (0.0, 1.0)
    minimum = float(np.min(flat))
    maximum = float(np.max(flat))
    span = maximum - minimum
    if span <= 1e-9:
        pad = max(abs(maximum) * 0.1, 1.0)
        return (minimum - pad, maximum + pad)
    pad = margin_ratio * span
    return (minimum - pad, maximum + pad)


def _write_strategy_series_csvs(
    data_dir: Path,
    prefix: str,
    rows_by_strategy: dict[str, list[dict[str, Any]]],
    fieldnames: Sequence[str],
) -> dict[str, Path]:
    return {
        strategy: _write_csv_rows(data_dir / f"{prefix}_{_slugify(strategy)}.csv", fieldnames, rows)
        for strategy, rows in rows_by_strategy.items()
    }


def _write_cdf_tikz(
    path: Path,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    x_column: str,
    y_column: str,
    series_paths: dict[str, Path],
    strategy_names: Sequence[str],
    xmin: float | None = None,
    each_nth_point: int | None = None,
) -> Path:
    body = [
        "\\begin{tikzpicture}",
        "  \\begin{axis}[",
        "    width=\\postprocessfigurewidth,",
        "    height=0.62\\postprocessfigurewidth,",
        f"    xlabel={{{xlabel}}},",
        f"    ylabel={{{ylabel}}},",
        "    grid=major,",
        "    legend pos=south east,",
    ]
    if xmin is not None:
        body.append(f"    xmin={float(xmin):g},")
    body.append("  ]")
    for strategy in strategy_names:
        csv_path = series_paths.get(strategy)
        if csv_path is None:
            continue
        rel = _relative_posix_path(csv_path, path.parent)
        extra_style = f",each nth point={int(each_nth_point)}" if each_nth_point is not None else ""
        body.extend(
            [
                "    \\addplot[const plot mark right, no markers, line width=1.2pt, color=%s, %s%s] table[x=%s,y=%s,col sep=comma] {%s};"
                % (_tikz_color_name(strategy), _tikz_line_style(strategy), extra_style, x_column, y_column, rel),
                "    \\addlegendentry{%s}" % _strategy_tex_label(strategy),
            ]
        )
    body.extend(["  \\end{axis}", "\\end{tikzpicture}"])
    return _write_tikz_file(path, body)


def _write_timeseries_tikz(
    path: Path,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    x_column: str,
    y_column: str,
    series_paths: dict[str, Path],
    strategy_names: Sequence[str],
    relocation_times_s: np.ndarray | None = None,
    step: bool = False,
    faint_series_paths: dict[str, Path] | None = None,
    faint_y_column: str | None = None,
    y_limits: tuple[float, float] | None = None,
    include_primary_series: bool = True,
) -> Path:
    ymin, ymax = y_limits if y_limits is not None else (0.0, 1.0)
    body = [
        "\\begin{tikzpicture}",
        "  \\begin{axis}[",
        "    width=\\postprocessfigurewidth,",
        "    height=0.58\\postprocessfigurewidth,",
        f"    xlabel={{{xlabel}}},",
        f"    ylabel={{{ylabel}}},",
        "    grid=major,",
        "    legend pos=south east,",
        f"    ymin={ymin:.8f},",
        f"    ymax={ymax:.8f},",
        "  ]",
    ]
    if relocation_times_s is not None:
        for time_s in np.asarray(relocation_times_s, dtype=float):
            body.append(
                "    \\addplot[gray, densely dotted, line width=0.5pt, no markers, forget plot] coordinates {(%0.8f,%0.8f) (%0.8f,%0.8f)};"
                % (float(time_s), ymin, float(time_s), ymax)
            )
        if faint_series_paths is not None or include_primary_series:
            body.append("")
    if faint_series_paths is not None and faint_y_column is not None:
        for strategy in strategy_names:
            csv_path = faint_series_paths.get(strategy)
            if csv_path is None:
                continue
            rel = _relative_posix_path(csv_path, path.parent)
            body.append(
                "    \\addplot[no markers, line width=0.5pt, opacity=0.35, color=%s, %s] table[x=%s,y=%s,col sep=comma] {%s};"
                % (_tikz_color_name(strategy), _tikz_line_style(strategy), x_column, faint_y_column, rel)
            )
    if include_primary_series:
        plot_style = "const plot mark right" if step else "no markers"
        for strategy in strategy_names:
            csv_path = series_paths.get(strategy)
            if csv_path is None:
                continue
            rel = _relative_posix_path(csv_path, path.parent)
            body.extend(
                [
                    "    \\addplot[%s, line width=1.2pt, color=%s, %s] table[x=%s,y=%s,col sep=comma] {%s};"
                    % (plot_style, _tikz_color_name(strategy), _tikz_line_style(strategy), x_column, y_column, rel),
                    "    \\addlegendentry{%s}" % _strategy_tex_label(strategy),
                ]
            )
    body.extend(["  \\end{axis}", "\\end{tikzpicture}"])
    return _write_tikz_file(path, body)


def _write_boxplot_tikz(
    path: Path,
    *,
    title: str,
    xlabel: str | None,
    ylabel: str,
    y_column: str,
    series_paths: dict[str, Path],
    strategy_names: Sequence[str],
) -> Path:
    tick_labels = ",".join("{%s}" % _strategy_tex_label(strategy, short=True) for strategy in strategy_names)
    body = [
        "\\begin{tikzpicture}",
        "  \\begin{axis}[",
        "    width=\\postprocessfigurewidth,",
        "    height=0.62\\postprocessfigurewidth,",
        (f"    xlabel={{{xlabel}}}," if xlabel else ""),
        f"    ylabel={{{ylabel}}},",
        "    grid=major,",
        "    xtick={%s}," % ",".join(str(index) for index in range(1, len(strategy_names) + 1)),
        f"    xticklabels={{{tick_labels}}},",
        "    xmin=0,",
        "  ]",
    ]
    body = [line for line in body if line]
    for index, strategy in enumerate(strategy_names, start=1):
        csv_path = series_paths.get(strategy)
        if csv_path is None:
            continue
        rel = _relative_posix_path(csv_path, path.parent)
        body.append(
            "    \\addplot[boxplot, boxplot/draw position=%d, draw=%s, fill=%s!35] table[y=%s,col sep=comma] {%s};"
            % (index, _tikz_color_name(strategy), _tikz_color_name(strategy), y_column, rel)
        )
    body.extend(["  \\end{axis}", "\\end{tikzpicture}"])
    return _write_tikz_file(path, body)


def _write_histogram_tikz(
    path: Path,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    series_paths: dict[str, Path],
    strategy_names: Sequence[str],
) -> Path:
    plotted_strategies = [strategy for strategy in strategy_names if series_paths.get(strategy) is not None]
    body = [
        "\\begin{tikzpicture}",
        "  \\begin{axis}[",
        "    width=\\postprocessfigurewidth,",
        "    height=0.62\\postprocessfigurewidth,",
        f"    xlabel={{{xlabel}}},",
        f"    ylabel={{{ylabel}}},",
        "    grid=major,",
        "    ybar,",
        "    bar width=5pt,",
        "    legend pos=north east,",
        "  ]",
    ]
    for index, strategy in enumerate(plotted_strategies):
        csv_path = series_paths.get(strategy)
        if csv_path is None:
            continue
        rel = _relative_posix_path(csv_path, path.parent)
        bar_shift = (float(index) - 0.5 * float(len(plotted_strategies) - 1)) * 6.0
        body.extend(
            [
                "    \\addplot[draw=%s, fill=%s, fill opacity=0.55, bar shift=%0.1fpt] table[x=bin_center_m,y=count,col sep=comma] {%s};"
                % (_tikz_color_name(strategy), _tikz_color_name(strategy), bar_shift, rel),
                "    \\addlegendentry{%s}" % _strategy_tex_label(strategy, short=True),
            ]
        )
    body.extend(["  \\end{axis}", "\\end{tikzpicture}"])
    return _write_tikz_file(path, body)


def _write_schedule_overview_tikz(
    path: Path,
    *,
    series_paths: dict[str, Path],
    strategy_names: Sequence[str],
) -> Path:
    body = [
        "\\begin{tikzpicture}",
        "  \\begin{axis}[",
        "    name=leftplot,",
        "    width=0.48\\postprocessfigurewidth,",
        "    height=0.44\\postprocessfigurewidth,",
        "    ylabel={Total relocation distance [m]},",
        "    grid=major,",
        "    ybar,",
        "    bar width=12pt,",
        "    xtick=\\empty,",
        "  ]",
    ]
    for strategy in strategy_names:
        csv_path = series_paths.get(strategy)
        if csv_path is None:
            continue
        rel = _relative_posix_path(csv_path, path.parent)
        body.append(
            "    \\addplot[draw=none, fill=%s] table[x=index,y=total_distance_m,col sep=comma] {%s};"
            % (_tikz_color_name(strategy), rel)
        )
    body.extend(
        [
            "  \\end{axis}",
            "  \\begin{axis}[",
            "    at={(leftplot.outer north east)},",
            "    anchor=outer north west,",
            "    xshift=1.3cm,",
            "    width=0.48\\postprocessfigurewidth,",
            "    height=0.44\\postprocessfigurewidth,",
            "    ylabel={Fraction of AP transitions that moved},",
            "    ymin=0.0,",
            "    ymax=1.0,",
            "    grid=major,",
            "    ybar,",
            "    bar width=12pt,",
            "    xtick=\\empty,",
            "  ]",
        ]
    )
    for strategy in strategy_names:
        csv_path = series_paths.get(strategy)
        if csv_path is None:
            continue
        rel = _relative_posix_path(csv_path, path.parent)
        body.append(
            "    \\addplot[draw=none, fill=%s] table[x=index,y=move_fraction,col sep=comma] {%s};"
            % (_tikz_color_name(strategy), rel)
        )
    body.extend(["  \\end{axis}", "\\end{tikzpicture}"])
    return _write_tikz_file(path, body)


def _write_xy_path_csv(path: Path, segments: Sequence[np.ndarray]) -> Path:
    rows: list[dict[str, Any]] = []
    for segment in segments:
        data = np.asarray(segment, dtype=float)
        if data.ndim != 2 or data.shape[0] == 0:
            continue
        for point in data:
            rows.append({"x_m": float(point[0]), "y_m": float(point[1])})
        rows.append({"x_m": float("nan"), "y_m": float("nan")})
    return _write_csv_rows(path, ["x_m", "y_m"], rows)


def _scene_polylines(metadata: dict[str, Any] | None, graph) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    boundary_segments: list[np.ndarray] = []
    building_segments: list[np.ndarray] = []
    edge_segments: list[np.ndarray] = []
    if metadata is not None and "boundary_local" in metadata:
        boundary = np.asarray(metadata["boundary_local"], dtype=float)
        if boundary.ndim == 2 and len(boundary):
            boundary_segments.append(boundary[:, :2])
    if metadata is not None:
        for building in metadata.get("buildings", []):
            polygon = np.asarray(building.get("polygon_local", []), dtype=float)
            if polygon.ndim == 2 and len(polygon):
                building_segments.append(polygon[:, :2])
    for u, v in graph.edges():
        start = graph.nodes[u]
        end = graph.nodes[v]
        edge_segments.append(
            np.asarray(
                [
                    [float(start["x"]), float(start["y"])],
                    [float(end["x"]), float(end["y"])],
                ],
                dtype=float,
            )
        )
    return boundary_segments, building_segments, edge_segments


def _write_scene_layout_tikz(
    path: Path,
    *,
    boundary_csv: Path | None,
    buildings_csv: Path | None,
    edges_csv: Path | None,
    trajectory_csv: Path | None,
    start_csv: Path | None,
    end_csv: Path | None,
    candidate_csv: Path | None,
    selected_csv: Path | None,
    reference_csv: Path | None,
    selected_strategy: str,
    reference_label: str = "Central massive-MIMO BS",
) -> Path:
    body = [
        "\\begin{tikzpicture}",
        "  \\begin{axis}[",
        "    width=\\postprocessfigurewidth,",
        "    height=0.78\\postprocessfigurewidth,",
        "    xlabel={x [m]},",
        "    ylabel={y [m]},",
        "    axis equal image,",
        "    grid=major,",
        "    legend pos=north east,",
        "    unbounded coords=jump,",
        "  ]",
    ]
    for csv_path, style in (
        (boundary_csv, "black, line width=1.0pt"),
        (buildings_csv, "gray!70, line width=0.8pt"),
        (edges_csv, "gray!50, line width=0.5pt"),
        (trajectory_csv, "orange!85!black, line width=0.9pt"),
    ):
        if csv_path is None:
            continue
        rel = _relative_posix_path(csv_path, path.parent)
        body.append("    \\addplot[%s] table[x=x_m,y=y_m,col sep=comma] {%s};" % (style, rel))
    if candidate_csv is not None:
        body.extend(
            [
                "    \\addplot[only marks, mark=square*, mark size=2pt, color=DistributedFixedColor] table[x=x_m,y=y_m,col sep=comma] {%s};"
                % _relative_posix_path(candidate_csv, path.parent),
                "    \\addlegendentry{Candidate APs}",
            ]
        )
    if selected_csv is not None:
        body.extend(
            [
                "    \\addplot[only marks, mark=triangle*, mark size=3pt, color=%s] table[x=x_m,y=y_m,col sep=comma] {%s};"
                % (_tikz_color_name(selected_strategy), _relative_posix_path(selected_csv, path.parent)),
                "    \\addlegendentry{Selected APs}",
            ]
        )
    if reference_csv is not None:
        body.extend(
            [
                "    \\addplot[only marks, mark=star, mark size=4pt, color=CentralMassiveMimoColor, mark options={solid, fill=CentralMassiveMimoColor, draw=black}] table[x=x_m,y=y_m,col sep=comma] {%s};"
                % _relative_posix_path(reference_csv, path.parent),
                f"    \\addlegendentry{{{reference_label}}}",
            ]
        )
    if start_csv is not None:
        body.extend(
            [
                "    \\addplot[only marks, mark=*, mark size=2.5pt, color=green!50!black] table[x=x_m,y=y_m,col sep=comma] {%s};"
                % _relative_posix_path(start_csv, path.parent),
                "    \\addlegendentry{UE start}",
            ]
        )
    if end_csv is not None:
        body.extend(
            [
                "    \\addplot[only marks, mark=x, mark size=2.5pt, color=orange!85!black] table[x=x_m,y=y_m,col sep=comma] {%s};"
                % _relative_posix_path(end_csv, path.parent),
                "    \\addlegendentry{UE end}",
            ]
        )
    body.extend(["  \\end{axis}", "\\end{tikzpicture}"])
    return _write_tikz_file(path, body)


def _write_trajectory_colormap_tikz(path: Path, *, points_csv: Path) -> Path:
    rel = _relative_posix_path(points_csv, path.parent)
    body = [
        "\\begin{tikzpicture}",
        "  \\begin{axis}[",
        "    width=\\postprocessfigurewidth,",
        "    height=0.72\\postprocessfigurewidth,",
        "    xlabel={x [m]},",
        "    ylabel={y [m]},",
        "    axis equal image,",
        "    grid=major,",
        "    colorbar,",
        "    point meta min=0,",
        "  ]",
        "    \\addplot[scatter, only marks, mark=*, mark size=1.7pt, scatter src=explicit] table[x=x_m,y=y_m,meta=time_s,col sep=comma] {%s};"
        % rel,
        "  \\end{axis}",
        "\\end{tikzpicture}",
    ]
    return _write_tikz_file(path, body)


def _coverage_grid_rows(best_sinr_db: np.ndarray, cell_centers: np.ndarray) -> list[dict[str, Any]]:
    centers = np.asarray(cell_centers, dtype=float)
    values = np.asarray(best_sinr_db, dtype=float)
    rows: list[dict[str, Any]] = []
    if values.ndim != 2 or centers.ndim < 3:
        return rows
    for row_index in range(values.shape[0]):
        for col_index in range(values.shape[1]):
            rows.append(
                {
                    "x_m": float(centers[row_index, col_index, 0]),
                    "y_m": float(centers[row_index, col_index, 1]),
                    "best_sinr_db": float(values[row_index, col_index]),
                }
            )
    return rows


def _write_coverage_tikz(
    path: Path,
    *,
    grid_csv: Path,
    selected_csv: Path | None,
    trajectory_csv: Path | None,
    title: str,
    selected_strategy: str,
) -> Path:
    body = [
        "\\begin{tikzpicture}",
        "  \\begin{axis}[",
        "    width=\\postprocessfigurewidth,",
        "    height=0.72\\postprocessfigurewidth,",
        "    xlabel={x [m]},",
        "    ylabel={y [m]},",
        "    axis equal image,",
        "    grid=major,",
        "    colorbar,",
        "  ]",
        "    \\addplot[scatter, only marks, mark=square*, mark size=5pt, scatter src=explicit] table[x=x_m,y=y_m,meta=best_sinr_db,col sep=comma] {%s};"
        % _relative_posix_path(grid_csv, path.parent),
    ]
    if selected_csv is not None:
        body.extend(
            [
                "    \\addplot[only marks, mark=triangle*, mark size=3pt, color=%s] table[x=x_m,y=y_m,col sep=comma] {%s};"
                % (_tikz_color_name(selected_strategy), _relative_posix_path(selected_csv, path.parent)),
                "    \\addlegendentry{Selected APs}",
            ]
        )
    if trajectory_csv is not None:
        body.extend(
            [
                "    \\addplot[only marks, mark=*, mark size=1.0pt, color=black, opacity=0.35] table[x=x_m,y=y_m,col sep=comma] {%s};"
                % _relative_posix_path(trajectory_csv, path.parent),
                "    \\addlegendentry{UE trajectory}",
            ]
        )
    body.extend(["  \\end{axis}", "\\end{tikzpicture}"])
    return _write_tikz_file(path, body)


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
    spectral_efficiency = {
        name: (
            np.asarray(payload["spectral_efficiency_bps_hz"], dtype=float)
            if "spectral_efficiency_bps_hz" in payload
            else np.log2(1.0 + np.clip(np.asarray(payload["sinr_linear"], dtype=float), 0.0, None))
        )
        for name, payload in strategy_payloads.items()
    }
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
        values_rate = np.asarray(spectral_efficiency[name], dtype=float)
        flat = values.reshape(-1)
        worst_user = np.min(values, axis=1)
        esr = np.sum(values_rate, axis=1)
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
    tikz_data_dir = _tikz_data_dir(target_dir)
    cdf_series_paths = _write_strategy_series_csvs(
        tikz_data_dir,
        "sinr_cdf",
        {
            name: [
                {"value_db": float(x_value), "cdf": float(y_value)}
                for x_value, y_value in zip(*_cdf_points(sinr[name]), strict=True)
            ]
            for name in strategy_names
        },
        ["value_db", "cdf"],
    )
    threshold_series_paths = _write_strategy_series_csvs(
        tikz_data_dir,
        "sinr_threshold",
        {
            name: [row for row in threshold_rows if row["strategy"] == name]
            for name in strategy_names
        },
        ["threshold_db", "outage_fraction", "mean_users_below_threshold", "probability_any_user_below_threshold"],
    )
    worst_user_series_paths = _write_strategy_series_csvs(
        tikz_data_dir,
        "worst_user_sinr",
        {
            name: [
                {"time_s": float(time_s), "worst_user_sinr_db": float(value)}
                for time_s, value in zip(times_s, np.min(sinr[name], axis=1), strict=True)
            ]
            for name in strategy_names
        },
        ["time_s", "worst_user_sinr_db"],
    )
    users_below_series_paths = _write_strategy_series_csvs(
        tikz_data_dir,
        "users_below_threshold",
        {
            name: [
                {"time_s": float(time_s), "users_below_threshold": int(value)}
                for time_s, value in zip(times_s, np.sum(np.asarray(sinr[name]) < outage_threshold_db, axis=1), strict=True)
            ]
            for name in strategy_names
        },
        ["time_s", "users_below_threshold"],
    )
    boxplot_series_paths = _write_strategy_series_csvs(
        tikz_data_dir,
        "per_user_mean_sinr",
        {
            name: [
                {"mean_sinr_db": float(np.mean(user_values))}
                for user_values in np.asarray(sinr[name], dtype=float).T
            ]
            for name in strategy_names
        },
        ["mean_sinr_db"],
    )
    esr_timeseries_paths = _write_strategy_series_csvs(
        tikz_data_dir,
        "esr_timeseries",
        {
            name: [
                {
                    "time_s": float(time_s),
                    "esr_bps_hz": float(value),
                    "window_mean_esr_bps_hz": float(step_value),
                }
                for time_s, value, step_value in zip(times_s, esr_by_strategy[name], esr_step_by_strategy[name], strict=True)
            ]
            for name in strategy_names
        },
        ["time_s", "esr_bps_hz", "window_mean_esr_bps_hz"],
    )
    esr_cdf_series_paths = _write_strategy_series_csvs(
        tikz_data_dir,
        "esr_cdf",
        {
            name: [
                {"esr_bps_hz": float(x_value), "cdf": float(y_value)}
                for x_value, y_value in zip(*_cdf_points(esr_by_strategy[name]), strict=True)
            ]
            for name in strategy_names
        },
        ["esr_bps_hz", "cdf"],
    )
    esr_window_series_paths: dict[str, Path] = {}
    for strategy in strategy_names:
        for window_offset, window in enumerate(analysis_windows):
            mask = window_index_by_snapshot == int(window["window_index"])
            if not np.any(mask):
                continue
            x_values, y_values = _cdf_points(esr_by_strategy[strategy][mask])
            if x_values.size == 0:
                continue
            esr_window_series_paths[f"{strategy}:W{int(window['window_index'])}"] = _write_csv_rows(
                tikz_data_dir / f"esr_window_cdf_{_slugify(strategy)}_w{int(window['window_index'])}.csv",
                ["esr_bps_hz", "cdf", "window_index", "window_style"],
                [
                    {
                        "esr_bps_hz": float(x_value),
                        "cdf": float(y_value),
                        "window_index": int(window["window_index"]),
                        "window_style": PGFPLOTS_LINESTYLES[WINDOW_LINESTYLES[window_offset % len(WINDOW_LINESTYLES)]],
                    }
                    for x_value, y_value in zip(x_values, y_values, strict=True)
                ],
            )

    fig, ax = plt.subplots(figsize=(8, 6))
    for name in strategy_names:
        x_values, y_values = _cdf_points(sinr[name])
        if x_values.size:
            ax.step(
                x_values,
                y_values,
                where="post",
                linewidth=2.0,
                color=_strategy_color(name),
                linestyle=_strategy_linestyle(name),
                label=_label(name),
            )
    ax.set_xlabel("SINR per user snapshot [dB]")
    ax.set_ylabel("CDF")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _add_plot_artifact(
        artifacts,
        "cdf_plot",
        _save_figure(fig, target_dir / "sinr_cdf_publication.png"),
        tikz_path=_write_cdf_tikz(
            _tikz_companion_path(target_dir / "sinr_cdf_publication.png"),
            title="SINR CDF",
            xlabel="SINR per user snapshot [dB]",
            ylabel="CDF",
            x_column="value_db",
            y_column="cdf",
            series_paths=cdf_series_paths,
            strategy_names=strategy_names,
            xmin=-20.0,
            each_nth_point=5,
        ),
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    for name in strategy_names:
        strategy_rows = [row for row in threshold_rows if row["strategy"] == name]
        ax.plot(
            [row["threshold_db"] for row in strategy_rows],
            [row["outage_fraction"] for row in strategy_rows],
            linewidth=2.0,
            color=_strategy_color(name),
            linestyle=_strategy_linestyle(name),
            label=_label(name),
        )
    ax.set_xlabel("SINR threshold [dB]")
    ax.set_ylabel("Fraction of user snapshots below threshold")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _add_plot_artifact(
        artifacts,
        "threshold_plot",
        _save_figure(fig, target_dir / "sinr_threshold_sweep.png"),
        tikz_path=_write_timeseries_tikz(
            _tikz_companion_path(target_dir / "sinr_threshold_sweep.png"),
            title="SINR threshold sweep",
            xlabel="SINR threshold [dB]",
            ylabel="Fraction of user snapshots below threshold",
            x_column="threshold_db",
            y_column="outage_fraction",
            series_paths=threshold_series_paths,
            strategy_names=strategy_names,
            y_limits=(0.0, 1.0),
        ),
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for name in strategy_names:
        ax.plot(
            times_s,
            np.min(sinr[name], axis=1),
            linewidth=1.8,
            color=_strategy_color(name),
            linestyle=_strategy_linestyle(name),
            label=_label(name),
        )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Worst-user SINR [dB]")
    _plot_relocation_markers(ax, relocation_times_s)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _add_plot_artifact(
        artifacts,
        "worst_user_plot",
        _save_figure(fig, target_dir / "worst_user_sinr_timeseries.png"),
        tikz_path=_write_timeseries_tikz(
            _tikz_companion_path(target_dir / "worst_user_sinr_timeseries.png"),
            title="Worst-user SINR",
            xlabel="Time [s]",
            ylabel="Worst-user SINR [dB]",
            x_column="time_s",
            y_column="worst_user_sinr_db",
            series_paths=worst_user_series_paths,
            strategy_names=strategy_names,
            relocation_times_s=relocation_times_s,
            y_limits=_y_limits(np.concatenate([np.asarray(np.min(sinr[name], axis=1), dtype=float) for name in strategy_names]) if strategy_names else np.asarray([0.0])),
        ),
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for name in strategy_names:
        ax.plot(
            times_s,
            np.sum(np.asarray(sinr[name]) < outage_threshold_db, axis=1),
            linewidth=1.8,
            color=_strategy_color(name),
            linestyle=_strategy_linestyle(name),
            label=_label(name),
        )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(f"Users below {outage_threshold_db:g} dB")
    _plot_relocation_markers(ax, relocation_times_s)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _add_plot_artifact(
        artifacts,
        "users_below_plot",
        _save_figure(fig, target_dir / "users_below_threshold_timeseries.png"),
        tikz_path=_write_timeseries_tikz(
            _tikz_companion_path(target_dir / "users_below_threshold_timeseries.png"),
            title="Users below threshold",
            xlabel="Time [s]",
            ylabel=f"Users below {outage_threshold_db:g} dB",
            x_column="time_s",
            y_column="users_below_threshold",
            series_paths=users_below_series_paths,
            strategy_names=strategy_names,
            relocation_times_s=relocation_times_s,
            y_limits=_y_limits(
                np.concatenate([np.sum(np.asarray(sinr[name]) < outage_threshold_db, axis=1) for name in strategy_names])
                if strategy_names
                else np.asarray([0.0])
            ),
        ),
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    box_data = [np.mean(sinr[name], axis=0) for name in strategy_names]
    if box_data:
        boxplot = ax.boxplot(
            box_data,
            tick_labels=[_label(name) for name in strategy_names],
            showmeans=True,
            patch_artist=True,
        )
        for patch, name in zip(boxplot["boxes"], strategy_names, strict=True):
            patch.set_facecolor(_strategy_color(name))
            patch.set_alpha(0.4)
        for whisker, name in zip(boxplot["whiskers"], np.repeat(strategy_names, 2), strict=True):
            whisker.set_color(_strategy_color(str(name)))
        for cap, name in zip(boxplot["caps"], np.repeat(strategy_names, 2), strict=True):
            cap.set_color(_strategy_color(str(name)))
        for median, name in zip(boxplot["medians"], strategy_names, strict=True):
            median.set_color(_strategy_color(name))
            median.set_linewidth(1.8)
        for mean, name in zip(boxplot["means"], strategy_names, strict=True):
            mean.set_markerfacecolor(_strategy_color(name))
            mean.set_markeredgecolor("black")
        ax.set_ylabel("Per-user mean SINR [dB]")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        _add_plot_artifact(
            artifacts,
            "boxplot",
            _save_figure(fig, target_dir / "per_user_mean_sinr_boxplot.png"),
            tikz_path=_write_boxplot_tikz(
                _tikz_companion_path(target_dir / "per_user_mean_sinr_boxplot.png"),
                title="Per-user mean SINR",
                xlabel="Per-user mean SINR [dB]",
                ylabel="Deployment Strategy",
                y_column="mean_sinr_db",
                series_paths=boxplot_series_paths,
                strategy_names=strategy_names,
            ),
        )
        plt.close(fig)
    else:
        plt.close(fig)
        _add_plot_artifact(
            artifacts,
            "boxplot",
            _save_empty_plot(
                target_dir / "per_user_mean_sinr_boxplot.png",
                "Per-user mean SINR",
                "No SINR samples available",
            ),
            tikz_path=_empty_tikz(
                _tikz_companion_path(target_dir / "per_user_mean_sinr_boxplot.png"),
                "Per-user mean SINR",
                "No SINR samples available",
            ),
        )

    fig, ax = plt.subplots(figsize=(9, 5))
    for name in strategy_names:
        ax.plot(
            times_s,
            esr_by_strategy[name],
            linewidth=1.0,
            alpha=0.35,
            color=_strategy_color(name),
            linestyle=_strategy_linestyle(name),
        )
        ax.step(
            times_s,
            esr_step_by_strategy[name],
            where="post",
            linewidth=2.2,
            color=_strategy_color(name),
            linestyle=_strategy_linestyle(name),
            label=_label(name),
        )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("ESR [bit/s/Hz]")
    _plot_relocation_markers(ax, relocation_times_s)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _add_plot_artifact(
        artifacts,
        "esr_timeseries_plot",
        _save_figure(fig, target_dir / "esr_timeseries.png"),
        tikz_path=_write_timeseries_tikz(
            _tikz_companion_path(target_dir / "esr_timeseries.png"),
            title="ESR evolution",
            xlabel="Time [s]",
            ylabel="ESR [bit/s/Hz]",
            x_column="time_s",
            y_column="window_mean_esr_bps_hz",
            series_paths=esr_timeseries_paths,
            strategy_names=strategy_names,
            relocation_times_s=relocation_times_s,
            step=True,
            faint_series_paths=esr_timeseries_paths,
            faint_y_column="esr_bps_hz",
            y_limits=_y_limits(np.concatenate([np.asarray(esr_by_strategy[name], dtype=float) for name in strategy_names]) if strategy_names else np.asarray([0.0])),
            include_primary_series=False,
        ),
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    for name in strategy_names:
        x_values, y_values = _cdf_points(esr_by_strategy[name])
        if x_values.size:
            ax.step(
                x_values,
                y_values,
                where="post",
                linewidth=2.0,
                color=_strategy_color(name),
                linestyle=_strategy_linestyle(name),
                label=_label(name),
            )
    ax.set_xlabel("ESR [bit/s/Hz]")
    ax.set_ylabel("CDF")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _add_plot_artifact(
        artifacts,
        "esr_cdf_plot",
        _save_figure(fig, target_dir / "esr_cdf.png"),
        tikz_path=_write_cdf_tikz(
            _tikz_companion_path(target_dir / "esr_cdf.png"),
            title="ESR CDF",
            xlabel="ESR [bit/s/Hz]",
            ylabel="CDF",
            x_column="esr_bps_hz",
            y_column="cdf",
            series_paths=esr_cdf_series_paths,
            strategy_names=strategy_names,
            each_nth_point=5,
        ),
    )
    plt.close(fig)

    if strategy_names and analysis_windows:
        fig, axes = plt.subplots(len(strategy_names), 1, figsize=(8, max(4.5, 3.2 * len(strategy_names))), squeeze=False)
        for axis, name in zip(axes[:, 0], strategy_names, strict=True):
            plotted = False
            for window_offset, window in enumerate(analysis_windows):
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
                    color=_strategy_color(name),
                    linestyle=WINDOW_LINESTYLES[window_offset % len(WINDOW_LINESTYLES)],
                    alpha=max(0.35, 1.0 - 0.12 * window_offset),
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
        _add_plot_artifact(
            artifacts,
            "esr_window_cdf_plot",
            _save_figure(fig, target_dir / "esr_time_conditioned_cdf.png"),
            tikz_path=_write_tikz_file(
                _tikz_companion_path(target_dir / "esr_time_conditioned_cdf.png"),
                [
                    "\\begin{tikzpicture}",
                    "  \\begin{axis}[",
                    "    width=\\postprocessfigurewidth,",
                    "    height=0.66\\postprocessfigurewidth,",
                    "    xlabel={ESR [bit/s/Hz]},",
                    "    ylabel={CDF},",
                    "    grid=major,",
                    "    legend pos=south east,",
                    "  ]",
                    *[
                        line
                        for strategy in strategy_names
                        for window_offset, window in enumerate(analysis_windows)
                        for line in (
                            []
                            if f"{strategy}:W{int(window['window_index'])}" not in esr_window_series_paths
                            else [
                                "    \\addplot[const plot mark right, no markers, line width=1.0pt, color=%s, %s] table[x=esr_bps_hz,y=cdf,col sep=comma] {%s};"
                                % (
                                    _tikz_color_name(strategy),
                                    PGFPLOTS_LINESTYLES[WINDOW_LINESTYLES[window_offset % len(WINDOW_LINESTYLES)]],
                                    _relative_posix_path(
                                        esr_window_series_paths[f'{strategy}:W{int(window["window_index"])}'],
                                        _tikz_companion_path(target_dir / "esr_time_conditioned_cdf.png").parent,
                                    ),
                                ),
                                "    \\addlegendentry{%s W%d}" % (_strategy_tex_label(strategy), int(window["window_index"])),
                            ]
                        )
                    ],
                    "  \\end{axis}",
                    "\\end{tikzpicture}",
                ],
            ),
        )
        plt.close(fig)
    else:
        _add_plot_artifact(
            artifacts,
            "esr_window_cdf_plot",
            _save_empty_plot(
                target_dir / "esr_time_conditioned_cdf.png",
                "Time-conditioned ESR CDF",
                "No ESR windows available",
            ),
            tikz_path=_empty_tikz(
                _tikz_companion_path(target_dir / "esr_time_conditioned_cdf.png"),
                "Time-conditioned ESR CDF",
                "No ESR windows available",
            ),
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
    tikz_data_dir = _tikz_data_dir(target_dir)
    overview_series_paths = {
        str(row["strategy"]): _write_csv_rows(
            tikz_data_dir / f"schedule_overview_{_slugify(str(row['strategy']))}.csv",
            ["index", "strategy", "strategy_label", "total_distance_m", "move_fraction"],
            [
                {
                    "index": index,
                    "strategy": row["strategy"],
                    "strategy_label": row["strategy_label"],
                    "total_distance_m": row["total_distance_m"],
                    "move_fraction": row["move_fraction"],
                }
            ],
        )
        for index, row in enumerate(summary_rows)
    }

    move_rows = [row for row in transition_rows if row["moved"]]
    histogram_path = target_dir / "schedule_transition_distance_histogram.png"
    histogram_series_paths: dict[str, Path] = {}
    if move_rows:
        all_values = np.asarray([row["distance_m"] for row in move_rows], dtype=float)
        num_bins = min(12, max(4, len(all_values)))
        bin_edges = np.histogram_bin_edges(all_values, bins=num_bins)
        histogram_values_by_strategy = [
            np.asarray([row["distance_m"] for row in move_rows if row["strategy"] == strategy], dtype=float)
            for strategy in strategies
        ]
        nonempty_histogram_series = [
            (strategy, values) for strategy, values in zip(strategies, histogram_values_by_strategy, strict=True) if values.size
        ]

        fig, ax = plt.subplots(figsize=(8, 6))
        if nonempty_histogram_series:
            ax.hist(
                [values for _, values in nonempty_histogram_series],
                bins=bin_edges,
                alpha=0.75,
                color=[_strategy_color(strategy) for strategy, _ in nonempty_histogram_series],
                label=[_label(strategy) for strategy, _ in nonempty_histogram_series],
                histtype="bar",
            )
        for strategy in strategies:
            values = np.asarray([row["distance_m"] for row in move_rows if row["strategy"] == strategy], dtype=float)
            if values.size == 0:
                continue
            counts, _ = np.histogram(values, bins=bin_edges)
            histogram_series_paths[strategy] = _write_csv_rows(
                tikz_data_dir / f"schedule_histogram_{_slugify(strategy)}.csv",
                ["bin_center_m", "count"],
                [
                    {
                        "bin_center_m": float(0.5 * (bin_edges[index] + bin_edges[index + 1])),
                        "count": int(count),
                    }
                    for index, count in enumerate(counts)
                ],
            )
        ax.set_xlabel("Relocation distance [m]")
        ax.set_ylabel("Count")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        _save_figure(fig, histogram_path)
        plt.close(fig)
    else:
        _save_empty_plot(histogram_path, "Relocation distances", "No AP relocations were observed")
    _add_plot_artifact(
        artifacts,
        "histogram",
        histogram_path,
        tikz_path=(
            _write_histogram_tikz(
                _tikz_companion_path(histogram_path),
                title="Relocation distances",
                xlabel="Relocation distance [m]",
                ylabel="Count",
                series_paths=histogram_series_paths,
                strategy_names=strategies,
            )
            if histogram_series_paths
            else _empty_tikz(_tikz_companion_path(histogram_path), "Relocation distances", "No AP relocations were observed")
        ),
    )

    overview_path = target_dir / "schedule_strategy_overview.png"
    if summary_rows:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        labels = [_label(row["strategy"]) for row in summary_rows]
        colors = [_strategy_color(str(row["strategy"])) for row in summary_rows]
        axes[0].bar(labels, [row["total_distance_m"] for row in summary_rows], color=colors)
        axes[0].set_ylabel("Total relocation distance [m]")
        axes[0].tick_params(axis="x", rotation=15)
        axes[0].grid(True, axis="y", alpha=0.3)
        axes[1].bar(labels, [row["move_fraction"] for row in summary_rows], color=colors)
        axes[1].set_ylabel("Fraction of AP transitions that moved")
        axes[1].tick_params(axis="x", rotation=15)
        axes[1].set_ylim(0.0, 1.0)
        axes[1].grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        _save_figure(fig, overview_path)
        plt.close(fig)
    else:
        _save_empty_plot(overview_path, "Schedule overview", "No schedule rows were available")
    _add_plot_artifact(
        artifacts,
        "overview",
        overview_path,
        tikz_path=(
            _write_schedule_overview_tikz(
                _tikz_companion_path(overview_path),
                series_paths=overview_series_paths,
                strategy_names=strategies,
            )
            if summary_rows
            else _empty_tikz(_tikz_companion_path(overview_path), "Schedule overview", "No schedule rows were available")
        ),
    )

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
    tikz_data_dir = _tikz_data_dir(output_dir)
    boundary_segments, building_segments, edge_segments = _scene_polylines(metadata, graph)
    boundary_csv = _write_xy_path_csv(tikz_data_dir / "scene_layout_boundary.csv", boundary_segments) if boundary_segments else None
    buildings_csv = _write_xy_path_csv(tikz_data_dir / "scene_layout_buildings.csv", building_segments) if building_segments else None
    edges_csv = _write_xy_path_csv(tikz_data_dir / "scene_layout_walk_edges.csv", edge_segments) if edge_segments else None
    trajectory_segments = [np.asarray(trajectory.positions_m[:, u_idx, :2], dtype=float) for u_idx, _ in enumerate(trajectory.ue_ids)]
    trajectory_paths_csv = _write_xy_path_csv(tikz_data_dir / "scene_layout_trajectories.csv", trajectory_segments)
    trajectory_points_csv = _write_csv_rows(
        tikz_data_dir / "trajectory_colormap_points.csv",
        ["time_s", "x_m", "y_m"],
        [
            {
                "time_s": float(trajectory.times_s[t_idx]),
                "x_m": float(trajectory.positions_m[t_idx, u_idx, 0]),
                "y_m": float(trajectory.positions_m[t_idx, u_idx, 1]),
            }
            for t_idx in range(len(trajectory.times_s))
            for u_idx in range(len(trajectory.ue_ids))
        ],
    )
    ue_start_csv = _write_csv_rows(
        tikz_data_dir / "scene_layout_ue_start.csv",
        ["x_m", "y_m"],
        [
            {"x_m": float(trajectory.positions_m[0, u_idx, 0]), "y_m": float(trajectory.positions_m[0, u_idx, 1])}
            for u_idx in range(len(trajectory.ue_ids))
        ],
    )
    ue_end_csv = _write_csv_rows(
        tikz_data_dir / "scene_layout_ue_end.csv",
        ["x_m", "y_m"],
        [
            {"x_m": float(trajectory.positions_m[-1, u_idx, 0]), "y_m": float(trajectory.positions_m[-1, u_idx, 1])}
            for u_idx in range(len(trajectory.ue_ids))
        ],
    )
    candidate_csv = _write_csv_rows(
        tikz_data_dir / "scene_layout_candidate_sites.csv",
        ["x_m", "y_m"],
        [{"x_m": float(site.x_m), "y_m": float(site.y_m)} for site in [*candidate_sites, *rooftop_sites]],
    )
    selected_best_csv = _write_csv_rows(
        tikz_data_dir / "scene_layout_selected_sites.csv",
        ["x_m", "y_m"],
        [{"x_m": float(site.x_m), "y_m": float(site.y_m)} for site in best_sites],
    )
    reference_csv = _write_csv_rows(
        tikz_data_dir / "scene_layout_central_massive_mimo.csv",
        ["x_m", "y_m"],
        [{"x_m": float(site.x_m), "y_m": float(site.y_m)} for site in central_sites],
    )
    trajectory_overlay_csv = _write_csv_rows(
        tikz_data_dir / "coverage_trajectory_points.csv",
        ["x_m", "y_m"],
        [
            {"x_m": float(position[0]), "y_m": float(position[1])}
            for position in np.asarray(trajectory.positions_m[..., :2], dtype=float).reshape(-1, 2)
        ],
    )

    artifacts: dict[str, Path] = {}
    _plot_scene_layout(
        metadata,
        graph,
        [*candidate_sites, *rooftop_sites],
        best_sites,
        trajectory,
        output_dir / "scene_layout.png",
        reference_sites=central_sites,
        reference_label="Central massive-MIMO BS",
    )
    _add_plot_artifact(
        artifacts,
        "scene_layout",
        output_dir / "scene_layout.png",
        tikz_path=_write_scene_layout_tikz(
            _tikz_companion_path(output_dir / "scene_layout.png"),
            boundary_csv=boundary_csv,
            buildings_csv=buildings_csv,
            edges_csv=edges_csv,
            trajectory_csv=trajectory_paths_csv,
            start_csv=ue_start_csv,
            end_csv=ue_end_csv,
            candidate_csv=candidate_csv,
            selected_csv=selected_best_csv,
            reference_csv=reference_csv,
            selected_strategy=best_strategy,
            reference_label="Central massive-MIMO BS",
        ),
    )
    _plot_colored_trajectories(trajectory, output_dir / "trajectory_colormap.png")
    _add_plot_artifact(
        artifacts,
        "trajectory_colormap",
        output_dir / "trajectory_colormap.png",
        tikz_path=_write_trajectory_colormap_tikz(
            _tikz_companion_path(output_dir / "trajectory_colormap.png"),
            points_csv=trajectory_points_csv,
        ),
    )

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
            fixed_grid_csv = _write_csv_rows(
                tikz_data_dir / "fixed_coverage_grid.csv",
                ["x_m", "y_m", "best_sinr_db"],
                _coverage_grid_rows(
                    np.asarray(payload["best_sinr_db"], dtype=float),
                    np.asarray(payload["cell_centers"], dtype=float),
                ),
            )
            _plot_coverage(
                np.asarray(payload["best_sinr_db"], dtype=float),
                np.asarray(payload["cell_centers"], dtype=float),
                baseline_sites,
                trajectory,
                output_dir / "fixed_coverage_map.png",
            )
        fixed_selected_csv = _write_csv_rows(
            tikz_data_dir / "fixed_coverage_selected_sites.csv",
            ["x_m", "y_m"],
            [{"x_m": float(site.x_m), "y_m": float(site.y_m)} for site in baseline_sites],
        )
        _add_plot_artifact(
            artifacts,
            "fixed_coverage_plot",
            output_dir / "fixed_coverage_map.png",
            tikz_path=_write_coverage_tikz(
                _tikz_companion_path(output_dir / "fixed_coverage_map.png"),
                grid_csv=fixed_grid_csv,
                selected_csv=fixed_selected_csv,
                trajectory_csv=trajectory_overlay_csv,
                title="Fixed coverage map",
                selected_strategy=baseline_strategy,
            ),
        )

    coverage_path = output_dir / "coverage_map.npz"
    if coverage_path.exists():
        with np.load(coverage_path, allow_pickle=True) as payload:
            coverage_grid_csv = _write_csv_rows(
                tikz_data_dir / "coverage_grid.csv",
                ["x_m", "y_m", "best_sinr_db"],
                _coverage_grid_rows(
                    np.asarray(payload["best_sinr_db"], dtype=float),
                    np.asarray(payload["cell_centers"], dtype=float),
                ),
            )
            _plot_coverage(
                np.asarray(payload["best_sinr_db"], dtype=float),
                np.asarray(payload["cell_centers"], dtype=float),
                best_sites,
                trajectory,
                output_dir / "coverage_map.png",
            )
        coverage_selected_csv = _write_csv_rows(
            tikz_data_dir / "coverage_selected_sites.csv",
            ["x_m", "y_m"],
            [{"x_m": float(site.x_m), "y_m": float(site.y_m)} for site in best_sites],
        )
        _add_plot_artifact(
            artifacts,
            "coverage_plot",
            output_dir / "coverage_map.png",
            tikz_path=_write_coverage_tikz(
                _tikz_companion_path(output_dir / "coverage_map.png"),
                grid_csv=coverage_grid_csv,
                selected_csv=coverage_selected_csv,
                trajectory_csv=trajectory_overlay_csv,
                title="Coverage map",
                selected_strategy=best_strategy,
            ),
        )

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
