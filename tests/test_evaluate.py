import numpy as np
import pandas as pd
import pytest

from football_predictor.evaluate import (
    accuracy, brier, calibration_table, evaluate, log_loss, market_probs, rps, time_split,
    walk_forward,
)
from conftest import simulate_league


def test_metrics_by_hand():
    p = np.array([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]])
    y = np.array([0, 1])
    assert log_loss(p, y) == pytest.approx(-(np.log(0.5) + np.log(0.3)) / 2)
    # brier: (0.25+0.09+0.04) and (0.04+0.49+0.25)
    assert brier(p, y) == pytest.approx((0.38 + 0.78) / 2)
    # rps: cum (0.5,0.8) vs (1,1) -> (0.25+0.04)/2 ; cum (0.2,0.5) vs (0,1) -> (0.04+0.25)/2
    assert rps(p, y) == pytest.approx((0.145 + 0.145) / 2)
    assert accuracy(p, y) == 0.5


def test_perfect_forecast_scores():
    y = np.array([0, 1, 2])
    p = np.eye(3)
    assert log_loss(p, y) == pytest.approx(0, abs=1e-12)
    assert brier(p, y) == rps(p, y) == 0
    # RPS rewards being close in order: predicting draw for a home win beats predicting away
    assert rps(np.array([[0, 1, 0]]), np.array([0])) < rps(np.array([[0, 0, 1]]), np.array([0]))


def test_paired_logloss_diff():
    from football_predictor.evaluate import paired_logloss_diff
    p = np.array([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]])
    y = np.array([0, 2])
    assert paired_logloss_diff(p, p, y) == (0.0, 0.0)
    d, se = paired_logloss_diff(p, np.full((2, 3), 1 / 3), y)
    assert d == pytest.approx(np.log(1 / 3) - np.log(0.5))


def test_market_probs_remove_margin():
    df = pd.DataFrame({"OddsH": [2.0], "OddsD": [3.4], "OddsA": [3.8]})
    p = market_probs(df)
    assert p.sum() == pytest.approx(1)
    assert p[0, 0] > p[0, 2]


@pytest.fixture(scope="module")
def league():
    return simulate_league(n_teams=8, n_seasons=4, seed=5)


def test_time_split_is_ordered(league):
    train, val, test = time_split(league)
    assert test == "2324" and val == "2223" and train == ["2021", "2122"]
    last_train = league.loc[league.Season.isin(train), "Date"].max()
    assert last_train < league.loc[league.Season == val, "Date"].min()
    assert league.loc[league.Season == val, "Date"].max() < league.loc[league.Season == test, "Date"].min()


def test_walk_forward_never_sees_the_future(league):
    target = league["Season"] == "2324"
    pred = walk_forward(league, target, xi=0.002)
    assert len(pred) == target.sum()
    assert (pred["train_max_date"] < pred["Date"]).all()
    assert np.allclose(pred[["pH", "pD", "pA"]].sum(axis=1), 1)

    # changing every result from a cut-off onwards must not change earlier predictions
    cut = pred["Date"].iloc[len(pred) // 2]
    tampered = league.copy()
    future = tampered["Date"] >= cut
    tampered.loc[future, ["FTHG", "FTAG"]] = tampered.loc[future, ["FTAG", "FTHG"]].to_numpy() + 3
    pred2 = walk_forward(tampered, target, xi=0.002)
    week_of_cut = cut - pd.Timedelta(days=cut.weekday())
    before = pred["Date"] < week_of_cut
    np.testing.assert_allclose(pred.loc[before, ["pH", "pD", "pA"]],
                               pred2.loc[before, ["pH", "pD", "pA"]])
    assert not np.allclose(pred.loc[~before, "pH"], pred2.loc[~before, "pH"])


def test_evaluate_runs_and_beats_naive(league):
    res = evaluate(league, xi=0.0)
    m = res.metrics.set_index("model")
    assert {"Always home win", "Class frequencies", "Poisson (no DC)", "Dixon-Coles"} <= set(m.index)
    assert m.loc["Dixon-Coles", "log_loss"] < m.loc["Always home win", "log_loss"]


def test_calibration_table():
    rng = np.random.default_rng(0)
    p = rng.dirichlet([2, 1, 1], size=5000)
    y = np.array([rng.choice(3, p=row) for row in p])
    cal = calibration_table(p, y)
    big = cal[cal["n"] > 200]
    # forecasts drawn from the true probabilities are calibrated
    assert ((big["predicted"] >= big["lo"] - 0.02) & (big["predicted"] <= big["hi"] + 0.02)).all()
