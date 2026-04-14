import numpy as np

from cocoon_sionna.config import PlacementConfig
from cocoon_sionna.optimization import (
    PlacementScore,
    capped_exact_search,
    sample_random_candidates,
    select_local_csi_candidates,
    summarize_candidate_set,
)


def test_sample_random_candidates_is_stable_for_seed():
    chosen = sample_random_candidates(["a", "b", "c", "d"], 2, seed=11)
    assert chosen == ["a", "d"]


def test_select_local_csi_candidates_prefers_best_pair():
    scores = {
        ("a",): 1.0,
        ("b",): 0.5,
        ("c",): 0.4,
        ("a", "b"): 2.5,
        ("a", "c"): 1.1,
        ("b", "c"): 1.0,
    }

    def evaluator(subset):
        return PlacementScore(
            score=scores[subset],
            outage=0.0,
            percentile_10_db=0.0,
            peer_tiebreak=0.0,
            grid_outage=0.0,
            trajectory_outage=0.0,
        )

    chosen = select_local_csi_candidates(["a", "b", "c"], 2, lambda subset: evaluator(subset).score)
    assert chosen == ["a", "b"]


def test_capped_exact_search_returns_best_subset():
    scores = {
        ("a", "b"): 1.5,
        ("a", "c"): 2.5,
        ("b", "c"): 1.0,
    }

    def evaluator(subset):
        return PlacementScore(
            score=scores[subset],
            outage=0.0,
            percentile_10_db=0.0,
            peer_tiebreak=0.0,
            grid_outage=0.0,
            trajectory_outage=0.0,
        )

    chosen, summary, capped, evaluations = capped_exact_search(["a", "b", "c"], 2, evaluator, max_iterations=10)
    assert chosen == ["a", "c"]
    assert summary.score == 2.5
    assert capped is False
    assert evaluations == 3


def test_summarize_candidate_set_uses_instantaneous_csi_sinr_for_score():
    cfg = PlacementConfig(
        sinr_threshold_db=0.0,
        outage_weight=1.0,
        percentile_weight=0.1,
        peer_tiebreak_weight=0.0,
    )

    score = summarize_candidate_set(
        grid_best_sinr_db=np.array([-100.0, -100.0, -100.0]),
        trajectory_best_sinr_db=np.array([10.0, 12.0, 14.0]),
        peer_need_weights=np.ones(3, dtype=float),
        cfg=cfg,
    )

    assert score.outage == 0.0
    assert score.percentile_10_db > 0.0
    assert score.grid_outage == 1.0
