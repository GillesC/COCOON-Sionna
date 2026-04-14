"""Placement scoring and candidate-selection helpers."""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .config import PlacementConfig
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
    cfg: PlacementConfig,
) -> PlacementScore:
    threshold = cfg.sinr_threshold_db
    grid_values = np.asarray(grid_best_sinr_db, dtype=float).reshape(-1)
    traj_values = np.asarray(trajectory_best_sinr_db, dtype=float).reshape(-1)
    grid_values = grid_values[np.isfinite(grid_values)]
    traj_values = traj_values[np.isfinite(traj_values)]

    grid_outage = float(np.mean(grid_values < threshold)) if grid_values.size else 1.0
    trajectory_outage = float(np.mean(traj_values < threshold)) if traj_values.size else 1.0
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


def sample_random_candidates(
    candidate_ids: list[str],
    select_count: int,
    seed: int,
) -> list[str]:
    if select_count < 0:
        raise ValueError("select_count must be non-negative")
    if select_count > len(candidate_ids):
        raise ValueError(
            f"Requested {select_count} movable APs, but only {len(candidate_ids)} candidate AP positions exist"
        )
    if select_count == 0:
        return []
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(candidate_ids), size=select_count, replace=False))
    return [candidate_ids[index] for index in indices.tolist()]


def select_local_csi_candidates(
    candidate_ids: list[str],
    select_count: int,
    evaluator: Callable[[tuple[str, ...]], float],
) -> list[str]:
    if select_count < 0:
        raise ValueError("select_count must be non-negative")
    if select_count > len(candidate_ids):
        raise ValueError(
            f"Requested {select_count} movable APs, but only {len(candidate_ids)} candidate AP positions exist"
        )

    ordered = list(candidate_ids)
    chosen: list[str] = []
    with progress_bar(
        total=select_count,
        desc="Local CSI placement",
        unit="ap",
        leave=False,
    ) as selection_progress:
        while len(chosen) < select_count:
            local_best: tuple[str, float] | None = None
            for candidate in ordered:
                if candidate in chosen:
                    continue
                subset = tuple(sorted([*chosen, candidate]))
                local_score = float(evaluator(subset))
                if local_best is None or local_score > local_best[1]:
                    local_best = (candidate, local_score)
            if local_best is None:
                break
            chosen.append(local_best[0])
            selection_progress.update(1)
            logger.info(
                "Local CSI step %d/%d selected %s with local P90 %.2f dB",
                len(chosen),
                select_count,
                local_best[0],
                local_best[1],
            )
    return sorted(chosen)


def capped_exact_search(
    candidate_ids: list[str],
    select_count: int,
    evaluator: Callable[[tuple[str, ...]], PlacementScore],
    max_iterations: int,
) -> tuple[list[str], PlacementScore, bool, int]:
    if select_count < 0:
        raise ValueError("select_count must be non-negative")
    if select_count > len(candidate_ids):
        raise ValueError(
            f"Requested {select_count} movable APs, but only {len(candidate_ids)} candidate AP positions exist"
        )
    if select_count == 0:
        score = evaluator(tuple())
        return [], score, False, 1

    best_ids: list[str] | None = None
    best_score: PlacementScore | None = None
    evaluations = 0
    capped = False

    for subset in itertools.combinations(sorted(candidate_ids), select_count):
        if evaluations >= max(max_iterations, 1):
            capped = True
            break
        score = evaluator(tuple(subset))
        evaluations += 1
        if best_score is None or score.score > best_score.score:
            best_ids = list(subset)
            best_score = score

    if best_ids is None or best_score is None:
        raise RuntimeError("Exact search evaluated no candidate combinations")
    logger.info(
        "Exact search evaluated %d combinations%s",
        evaluations,
        " before hitting the cap" if capped else "",
    )
    return best_ids, best_score, capped, evaluations
