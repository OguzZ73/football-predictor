import numpy as np
import pandas as pd
import pytest

from football_predictor.model import (
    DixonColes, btts_prob, fit_poisson_glm, outcome_probs, over_prob, score_matrix,
    time_weights, top_scores,
)
from conftest import simulate_league


@pytest.mark.parametrize("lam,mu,rho", [(1.5, 1.1, 0.0), (1.3, 0.9, -0.12), (2.8, 0.4, 0.1)])
def test_score_matrix_sums_to_one(lam, mu, rho):
    m = score_matrix(lam, mu, rho)
    assert m.shape == (8, 8)
    assert m.sum() == pytest.approx(1.0)
    assert (m >= 0).all()
    assert outcome_probs(m).sum() == pytest.approx(1.0)


@pytest.mark.parametrize("lam,mu", [(1.5, 1.1), (2.2, 0.8)])
def test_truncation_mass_is_small_for_typical_rates(lam, mu):
    # mass beyond 7 goals (removed by renormalising) is < 0.2% for typical rates
    assert score_matrix(lam, mu, -0.1, max_goals=30)[:8, :8].sum() > 1 - 2e-3


def test_symmetry_equal_rates():
    m = score_matrix(1.2, 1.2, -0.1)
    np.testing.assert_allclose(m, m.T)
    p = outcome_probs(m)
    assert p[0] == pytest.approx(p[2])
    # swapping rates transposes the matrix and swaps home/away probabilities
    a, b = score_matrix(1.6, 0.9, -0.05), score_matrix(0.9, 1.6, -0.05)
    np.testing.assert_allclose(a, b.T)
    assert outcome_probs(a)[0] == pytest.approx(outcome_probs(b)[2])


def test_rho_zero_is_independent_poisson():
    m = score_matrix(1.4, 1.0, 0.0)
    np.testing.assert_allclose(m, np.outer(m.sum(1), m.sum(0)), atol=1e-12)


def test_negative_rho_inflates_draws():
    assert outcome_probs(score_matrix(1.2, 1.0, -0.15))[1] > outcome_probs(score_matrix(1.2, 1.0, 0))[1]


def test_derived_markets():
    m = np.zeros((8, 8))
    m[1, 1], m[2, 0], m[0, 0], m[3, 2] = 0.4, 0.3, 0.2, 0.1
    assert over_prob(m) == pytest.approx(0.1)
    assert btts_prob(m) == pytest.approx(0.5)
    assert top_scores(m, 2) == [("1-1", 0.4), ("2-0", 0.3)]


def test_time_weights():
    d = pd.Series(pd.to_datetime(["2024-01-01", "2024-01-11", "2024-01-21"]))
    w = time_weights(d, pd.Timestamp("2024-01-11"), xi=0.1)
    np.testing.assert_allclose(w, [np.exp(-1), 1.0, 0.0])


def test_parameter_recovery_on_known_data():
    att = np.linspace(-0.4, 0.4, 8)
    dfn = np.linspace(0.3, -0.3, 8) + 0.1
    df = simulate_league(n_teams=8, n_seasons=150, attack=att, defence=dfn,
                         home=0.3, rho=-0.1, seed=3)
    m = DixonColes(xi=0.0, l2=0.0).fit(df)
    np.testing.assert_allclose(m.attack, att - att.mean(), atol=0.05)
    np.testing.assert_allclose(m.defence, dfn + att.mean(), atol=0.05)
    assert m.home == pytest.approx(0.3, abs=0.03)
    assert m.rho == pytest.approx(-0.1, abs=0.04)


def test_matches_statsmodels_glm(small_league):
    """With ρ = 0 and no penalty the custom likelihood equals the Poisson GLM."""
    df = small_league
    w = time_weights(df["Date"], df["Date"].max() + pd.Timedelta(days=1), 0.002)
    ours = DixonColes(xi=0.002, l2=0.0, dixon_coles=False).fit(df)
    glm, _ = fit_poisson_glm(df, w)
    lam_glm = glm.fittedvalues.to_numpy()[: len(df)]
    lam_ours = np.array([ours.rates(h, a)[0] for h, a in zip(df.HomeTeam, df.AwayTeam)])
    np.testing.assert_allclose(lam_ours, lam_glm, rtol=1e-4)
    assert ours.home == pytest.approx(glm.params["is_home"], abs=1e-4)


def test_fit_ignores_future_matches(small_league):
    cut = small_league["Date"].iloc[len(small_league) // 2]
    a = DixonColes().fit(small_league, ref_date=cut)
    b = DixonColes().fit(small_league[small_league["Date"] < cut], ref_date=cut)
    np.testing.assert_allclose(a.attack, b.attack)
    assert a.n_matches == (small_league["Date"] < cut).sum()


def test_unknown_team_and_roundtrip(tmp_path, small_league):
    m = DixonColes().fit(small_league)
    with pytest.raises(KeyError):
        m.predict("Nope FC", "Team01")
    assert m.predict("Nope FC", "Team01", allow_unknown=True).matrix.sum() == pytest.approx(1)
    m.save(tmp_path / "m.json")
    m2 = DixonColes.load(tmp_path / "m.json")
    np.testing.assert_allclose(m.predict("Team03", "Team07").matrix,
                               m2.predict("Team03", "Team07").matrix)


def test_effective_goals_blends_goals_and_shots():
    from football_predictor.model import effective_goals
    df = pd.DataFrame({"FTHG": [2, 0] * 30, "FTAG": [1, 1] * 30,
                       "SOTH": [4.0, 2.0] * 30, "SOTA": [2.0, 2.0] * 30})
    conv = (60 + 60) / (180 + 120)  # goals / shots on target = 0.4
    xe, ye = effective_goals(df, 0.5)
    assert xe[0] == pytest.approx(0.5 * 2 + 0.5 * conv * 4)
    assert ye[1] == pytest.approx(0.5 * 1 + 0.5 * conv * 2)
    np.testing.assert_allclose(effective_goals(df, 1.0)[0], df["FTHG"])  # α=1: goals only
    no_shots = df.drop(columns=["SOTH", "SOTA"])
    np.testing.assert_allclose(effective_goals(no_shots, 0.5)[0], df["FTHG"])


def test_value_prior_gradient_and_effect(small_league):
    teams = sorted(set(small_league.HomeTeam))
    # synthetic "squad values": stronger teams (higher index in conftest) are richer
    values = {t: 15 + 0.3 * i for i, t in enumerate(teams)}
    m = DixonColes(use_values=True, l2=5.0).fit(small_league, team_values=values)
    f = m.objective_
    rng = np.random.default_rng(0)
    n = 2 * len(teams) + 3
    theta = rng.normal(0, 0.2, n)
    theta[-3] = -0.05  # rho
    _, g = f(theta)
    for k in list(rng.choice(n, 8, replace=False)) + [n - 2, n - 1]:
        e = np.zeros(n)
        e[k] = 1e-6
        num = (f(theta + e)[0] - f(theta - e)[0]) / 2e-6
        assert g[k] == pytest.approx(num, rel=1e-4, abs=1e-4)
    ba, bd = m.value_coef
    assert ba > 0 and bd < 0  # richer teams attack more and concede less in this league
    plain = DixonColes(l2=5.0).fit(small_league, team_values=values)  # use_values off -> ignored
    assert plain.value_coef == (0.0, 0.0)
