"""Deterministic greedy plus one-swap site optimization."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .config import OptimizationConfig
from .logging_utils import progress_bar

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PlacementScore:
    score: float
    outage: float
    percentile_10_db: float
    peer_tiebreak: float
    grid_outage: float
    trajectory_outage: float


def summarize_candidate_set(
    grid_best_sinr_db: np.ndarray,
    trajectory_best_sinr_db: np.ndarray,
    peer_need_weights: np.ndarray,
    cfg: OptimizationConfig,
) -> PlacementScore:
    threshold = cfg.sinr_threshold_db
    grid_values = np.asarray(grid_best_sinr_db, dtype=float).reshape(-1)
    traj_values = np.asarray(trajectory_best_sinr_db, dtype=float).reshape(-1)
    grid_values = grid_values[np.isfinite(grid_values)]
    traj_values = traj_values[np.isfinite(traj_values)]

    grid_outage = float(np.mean(grid_values < threshold)) if grid_values.size else 1.0
    trajectory_outage = float(np.mean(traj_values < threshold)) if traj_values.size else 1.0
    # Optimization decisions are driven by instantaneous AP-UE SINR samples,
    # not by radio-map SINR values.
    combined_outage = trajectory_outage

    grid_p10 = float(np.percentile(grid_values, 10)) if grid_values.size else -120.0
    traj_p10 = float(np.percentile(traj_values, 10)) if traj_values.size else -120.0
    combined_p10 = traj_p10

    weights = np.asarray(peer_need_weights, dtype=float).reshape(-1)
    if traj_values.size and weights.size == traj_values.size:
        normalized = weights / max(np.mean(weights), 1e-9)
        peer_tiebreak = float(np.mean(traj_values * normalized))
    else:
        peer_tiebreak = float(np.mean(traj_values)) if traj_values.size else -120.0

    score = (
        -cfg.outage_weight * combined_outage
        + cfg.percentile_weight * combined_p10
        + cfg.peer_tiebreak_weight * peer_tiebreak
    )
    return PlacementScore(
        score=score,
        outage=combined_outage,
        percentile_10_db=combined_p10,
        peer_tiebreak=peer_tiebreak,
        grid_outage=grid_outage,
        trajectory_outage=trajectory_outage,
    )


def greedy_one_swap(
    candidate_ids: list[str],
    select_count: int,
    evaluator: Callable[[tuple[str, ...]], PlacementScore],
) -> tuple[list[str], PlacementScore]:
    if select_count <= 0:
        raise ValueError("select_count must be positive")
    ordered_candidates = list(candidate_ids)
    chosen: list[str] = []
    best_score: PlacementScore | None = None
    logger.info("Greedy selection started with %d candidates", len(ordered_candidates))

    with progress_bar(
        total=min(select_count, len(ordered_candidates)),
        desc="Greedy AP selection",
        unit="site",
        leave=False,
    ) as selection_progress:
        while len(chosen) < min(select_count, len(ordered_candidates)):
            local_best: tuple[str, PlacementScore] | None = None
            for candidate in ordered_candidates:
                if candidate in chosen:
                    continue
                subset = tuple(sorted([*chosen, candidate]))
                score = evaluator(subset)
                if local_best is None or score.score > local_best[1].score:
                    local_best = (candidate, score)
            if local_best is None:
                break
            chosen.append(local_best[0])
            best_score = local_best[1]
            selection_progress.update(1)
            logger.info(
                "Selected %s (%d/%d), score=%.3f, outage=%.3f, p10=%.2f dB",
                local_best[0],
                len(chosen),
                min(select_count, len(ordered_candidates)),
                local_best[1].score,
                local_best[1].outage,
                local_best[1].percentile_10_db,
            )

    improved = True
    while improved and chosen:
        improved = False
        current = tuple(sorted(chosen))
        current_score = evaluator(current)
        for existing in list(chosen):
            for candidate in ordered_candidates:
                if candidate in chosen:
                    continue
                proposal = sorted([candidate if site == existing else site for site in chosen])
                subset = tuple(proposal)
                score = evaluator(subset)
                if score.score > current_score.score:
                    logger.info(
                        "Swap improvement: %s -> %s, score %.3f -> %.3f",
                        existing,
                        candidate,
                        current_score.score,
                        score.score,
                    )
                    chosen = list(subset)
                    best_score = score
                    improved = True
                    break
            if improved:
                break

    if best_score is None:
        best_score = evaluator(tuple(sorted(chosen)))
    logger.info("Optimization finished with sites: %s", ", ".join(sorted(chosen)))
    return sorted(chosen), best_score
