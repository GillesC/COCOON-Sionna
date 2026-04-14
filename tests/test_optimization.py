import numpy as np

from cocoon_sionna.config import OptimizationConfig
from cocoon_sionna.optimization import PlacementScore, greedy_one_swap, summarize_candidate_set


def test_greedy_one_swap_prefers_best_pair():
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

    chosen, summary = greedy_one_swap(["a", "b", "c"], 2, evaluator)
    assert chosen == ["a", "b"]
    assert summary.score == 2.5


def test_summarize_candidate_set_uses_instantaneous_csi_sinr_for_score():
    cfg = OptimizationConfig(
        sinr_threshold_db=0.0,
        grid_weight=0.9,
        trajectory_weight=0.1,
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
