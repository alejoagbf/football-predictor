"""Unit tests for the Poisson prediction module."""

from __future__ import annotations

import numpy as np
import pytest

from src.prediction.poisson import (
    PoissonPrediction,
    build_score_matrix,
    matrix_to_probabilities,
    predict_from_lambdas,
)


class TestPoissonMatrix:
    def test_matrix_sums_to_one(self) -> None:
        matrix = build_score_matrix(1.5, 1.2)
        assert matrix.sum() == pytest.approx(1.0, abs=1e-6)

    def test_matrix_shape(self) -> None:
        matrix = build_score_matrix(1.5, 1.2, max_goals=5)
        assert matrix.shape == (6, 6)

    def test_all_probs_non_negative(self) -> None:
        matrix = build_score_matrix(2.0, 1.5)
        assert (matrix >= 0).all()

    def test_result_probs_sum_to_one(self) -> None:
        pred = predict_from_lambdas(1.5, 1.2)
        total = pred.home_win + pred.draw + pred.away_win
        assert total == pytest.approx(1.0, abs=1e-4)

    def test_high_home_lambda_favours_home_win(self) -> None:
        pred = predict_from_lambdas(3.0, 0.5)
        assert pred.home_win > pred.away_win
        assert pred.home_win > pred.draw

    def test_balanced_lambdas_give_symmetric_result(self) -> None:
        pred_h = predict_from_lambdas(1.5, 1.5)
        pred_a = predict_from_lambdas(1.5, 1.5)
        assert pred_h.home_win == pytest.approx(pred_a.away_win, abs=1e-4)

    def test_btts_calculation(self) -> None:
        # BTTS = P(home >=1) * P(away >= 1) under independence
        lh, la = 2.0, 1.5
        from scipy.stats import poisson
        p_home_scores = 1 - poisson.pmf(0, lh)
        p_away_scores = 1 - poisson.pmf(0, la)
        expected_btts = p_home_scores * p_away_scores
        pred = predict_from_lambdas(lh, la)
        assert pred.btts == pytest.approx(expected_btts, abs=0.02)

    def test_over_25_consistent(self) -> None:
        pred_low = predict_from_lambdas(0.5, 0.5)
        pred_high = predict_from_lambdas(3.0, 3.0)
        assert pred_high.over_2_5 > pred_low.over_2_5

    def test_over_markets_monotone(self) -> None:
        pred = predict_from_lambdas(1.8, 1.3)
        assert pred.over_0_5 >= pred.over_1_5 >= pred.over_2_5 >= pred.over_3_5

    def test_most_likely_score_format(self) -> None:
        pred = predict_from_lambdas(1.5, 1.2)
        parts = pred.most_likely_score.split("-")
        assert len(parts) == 2
        assert all(p.isdigit() for p in parts)

    def test_score_probs_are_valid(self) -> None:
        pred = predict_from_lambdas(1.5, 1.2)
        assert len(pred.score_probs) > 0
        total = sum(pred.score_probs.values())
        assert total <= 1.0 + 1e-4  # can be < 1 since low-prob scores are filtered

    def test_zero_lambda_handled(self) -> None:
        """Very low lambda should still produce valid matrix."""
        pred = predict_from_lambdas(0.05, 0.05)
        total = pred.home_win + pred.draw + pred.away_win
        assert total == pytest.approx(1.0, abs=1e-3)

    def test_matrix_to_probabilities_identity(self) -> None:
        """Calling build_score_matrix then matrix_to_probabilities should
        match calling predict_from_lambdas directly."""
        lh, la = 1.8, 1.2
        matrix = build_score_matrix(lh, la)
        pred_a = matrix_to_probabilities(matrix, lh, la)
        pred_b = predict_from_lambdas(lh, la)
        assert pred_a.home_win == pytest.approx(pred_b.home_win, abs=1e-6)
        assert pred_a.draw == pytest.approx(pred_b.draw, abs=1e-6)
        assert pred_a.away_win == pytest.approx(pred_b.away_win, abs=1e-6)
