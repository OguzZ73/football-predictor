import numpy as np
import pandas as pd
import pytest

from football_predictor.model import DixonColes
from football_predictor.simulate import league_table, remaining_fixtures, simulate_season


def test_league_table_points():
    played = pd.DataFrame({"HomeTeam": ["A", "B", "C"], "AwayTeam": ["B", "C", "A"],
                           "FTHG": [2, 1, 0], "FTAG": [0, 1, 3]})
    t = league_table(played, ["A", "B", "C"])
    assert t.loc["A", "points"] == 6 and t.loc["B", "points"] == 1 and t.loc["C", "points"] == 1
    assert t.loc["A", "gf"] == 5 and t.loc["A", "ga"] == 0
    assert remaining_fixtures(played, ["A", "B", "C"]) == [("A", "C"), ("B", "A"), ("C", "B")]


@pytest.fixture(scope="module")
def setup(small_league):
    last = small_league["Season"].iloc[-1]
    season = small_league[small_league["Season"] == last]
    cut = season["Date"].iloc[len(season) // 2]
    model = DixonColes().fit(small_league, ref_date=cut)
    played = season[season["Date"] < cut]
    return model, played


def test_probabilities_are_consistent(setup):
    model, played = setup
    sim = simulate_season(model, played, n=2000, europe_spots=4, relegation_spots=2)
    s = sim.summary
    assert s["p_champion"].sum() == pytest.approx(1)
    assert s["p_europe"].sum() == pytest.approx(4)
    assert s["p_relegation"].sum() == pytest.approx(2)
    np.testing.assert_allclose(sim.positions.sum(axis=0), 1)
    np.testing.assert_allclose(sim.positions.sum(axis=1), 1)
    assert (s["exp_points"] >= s["points_now"]).all()


def test_seed_reproducible(setup):
    model, played = setup
    a = simulate_season(model, played, n=500, seed=7).summary
    b = simulate_season(model, played, n=500, seed=7).summary
    c = simulate_season(model, played, n=500, seed=8).summary
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)


def test_finished_season_is_deterministic(small_league):
    last = small_league[small_league["Season"] == small_league["Season"].iloc[-1]]
    model = DixonColes().fit(small_league)
    sim = simulate_season(model, last, n=50)
    assert sim.n_remaining == 0
    s = sim.summary
    assert (s["exp_points"] == s["points_now"]).all()
    # only teams level on the maximum points can be champion
    assert s.loc[s["points_now"] < s["points_now"].max(), "p_champion"].sum() == 0
    assert s["p_champion"].sum() == pytest.approx(1)
