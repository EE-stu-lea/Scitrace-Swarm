import pytest

from examples.scitrace_research.workflow.route_b2_experiment import (
    convex_fusion,
    knn_predictions,
    rank_biased_overlap,
    rbo_disagreement,
    score_dispersion,
    simplex_grid,
)


def test_rank_biased_overlap_rewards_matching_ranked_lists():
    left = [("a", 3.0), ("b", 2.0), ("c", 1.0)]
    reordered = [("c", 3.0), ("b", 2.0), ("a", 1.0)]

    assert rank_biased_overlap(left, left, depth=3) == pytest.approx(1.0)
    assert rank_biased_overlap(left, reordered, depth=3) < 1.0
    assert rbo_disagreement([left, left], depth=3) == pytest.approx(0.0)


def test_convex_fusion_and_score_dispersion_use_score_magnitudes():
    first = [("a", 10.0), ("b", 0.0)]
    second = [("b", 10.0), ("a", 9.0)]

    assert convex_fusion([first, second], [1.0, 0.0])[0][0] == "a"
    assert convex_fusion([first, second], [0.0, 1.0])[0][0] == "b"
    assert score_dispersion([first, first]) == pytest.approx(0.0)
    assert score_dispersion([first, second]) > 0.0


def test_simplex_grid_contains_only_normalized_weight_vectors():
    grid = simplex_grid(3, units=4)

    assert len(grid) == 15
    assert all(sum(weights) == pytest.approx(1.0) for weights in grid)
    assert [0.5, 0.25, 0.25] in grid


def test_knn_predictions_are_monotone_and_follow_nearest_training_query():
    train_features = {"easy": [0.0], "hard": [10.0]}
    train_curves = {
        "easy": [0.0, 1.0, 1.0],
        "hard": [0.0, 0.0, 1.0],
    }
    predictions = knn_predictions(
        train_features,
        train_curves,
        {"near_easy": [0.1], "near_hard": [9.9]},
        neighbors=1,
        feature_indices=[0],
    )

    assert predictions["near_easy"] == pytest.approx(train_curves["easy"])
    assert predictions["near_hard"] == pytest.approx(train_curves["hard"])
    assert all(
        curve[index] <= curve[index + 1]
        for curve in predictions.values()
        for index in range(len(curve) - 1)
    )
