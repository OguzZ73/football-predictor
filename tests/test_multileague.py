import numpy as np
import pandas as pd
import pytest

from football_predictor.model import score_matrix
from football_predictor.multileague import CrossLeagueModel, OTHER, team_leagues, walk_forward_uefa


def simulate_world(seed=0, n_seasons=6, gap=0.5):
    """Two leagues (B weaker by ``gap`` in attack and defence) + cross-league cup games
    + one 'outside' team that only plays cup games."""
    rng = np.random.default_rng(seed)
    teams = {f"A{i}": ("LA", 0.15 * (i - 3.5) / 3.5) for i in range(8)}
    teams |= {f"B{i}": ("LB", 0.15 * (i - 3.5) / 3.5) for i in range(8)}
    level = {"LA": (gap / 2, -gap / 2), "LB": (-gap / 2, gap / 2)}

    def rates(h, a, home):
        (lh, sh), (la, sa) = teams[h], teams[a]
        att_h, def_h = level[lh][0] + sh, level[lh][1] - sh
        att_a, def_a = level[la][0] + sa, level[la][1] - sa
        return np.exp(0.2 + att_h + def_a + home), np.exp(0.2 + att_a + def_h)

    teams_outside = {"X0": ("LB", -0.2)}  # behaves like a weak league-B side
    rows, day = [], pd.Timestamp("2019-08-01")
    for s in range(n_seasons):
        for lg in ("LA", "LB"):
            ts = [t for t, (l, _) in teams.items() if l == lg]
            for k, (h, a) in enumerate((h, a) for h in ts for a in ts if h != a):
                lam, mu = rates(h, a, 0.25)
                cell = rng.choice(64, p=score_matrix(lam, mu, -0.05).ravel())
                rows.append((day + pd.Timedelta(days=7 * (k // 4)), lg, False, h, a, *divmod(cell, 8)))
        cup = [(f"A{rng.integers(8)}", f"B{rng.integers(8)}") for _ in range(60)]
        cup += [(f"B{rng.integers(8)}", f"A{rng.integers(8)}") for _ in range(60)]
        for k, (h, a) in enumerate(cup):
            lam, mu = rates(h, a, 0.3)
            cell = rng.choice(64, p=score_matrix(lam, mu, -0.05).ravel())
            rows.append((day + pd.Timedelta(days=3 + 7 * (k % 30)), "CL", True, h, a, *divmod(cell, 8)))
        day += pd.Timedelta(days=365)
    df = pd.DataFrame(rows, columns=["Date", "competition", "is_uefa", "home_id", "away_id", "FTHG", "FTAG"])
    df["HomeTeam"], df["AwayTeam"] = df["home_id"], df["away_id"]
    return df.sort_values("Date", kind="stable").reset_index(drop=True)


@pytest.fixture(scope="module")
def world():
    return simulate_world()


def test_team_leagues(world):
    extra = pd.DataFrame([{"Date": pd.Timestamp("2030-01-01"), "competition": "CL", "is_uefa": True,
                           "home_id": "A0", "away_id": "KuPS", "FTHG": 1, "FTAG": 0}])
    lg = team_leagues(pd.concat([world, extra]))
    assert lg["A3"] == "LA" and lg["B5"] == "LB" and lg["KuPS"] == OTHER


def test_gradient_matches_finite_differences(world):
    small = world[world["Date"] < "2021-01-01"]
    m = CrossLeagueModel(xi=0.002).fit(small)
    f = m.objective_
    rng = np.random.default_rng(1)
    n = 1 + 2 * len(m.leagues) + 2 * len(m.teams) + len(m.H) + 1 + 2  # + rho, b_att, b_def
    theta = rng.normal(0, 0.2, n)
    theta[n - 3] = -0.05  # rho
    _, g = f(theta)
    eps = 1e-6
    for k in rng.choice(n - 2, 12, replace=False).tolist() + [0, n - 3]:
        e = np.zeros(n)
        e[k] = eps
        num = (f(theta + e)[0] - f(theta - e)[0]) / (2 * eps)
        assert g[k] == pytest.approx(num, rel=1e-4, abs=1e-4)


def test_recovers_league_gap(world):
    m = CrossLeagueModel(xi=0.0, l2_league=0.01).fit(world)
    lt = m.league_table().set_index("league")
    gap = lt.loc["LA", "strength"] - lt.loc["LB", "strength"]
    assert gap == pytest.approx(1.0, abs=0.3)  # attack gap 0.5 + defence gap 0.5


def test_unseen_team_gets_other_or_league_level(world):
    m = CrossLeagueModel().fit(world)
    lam, mu, mat = m.predict_matrix("A0", "NobodyFC")
    assert mat.sum() == pytest.approx(1)
    assert lam > 0 and mu > 0


def test_walk_forward_has_no_leakage(world):
    target = world["is_uefa"] & (world["Date"] >= world["Date"].max() - pd.Timedelta(days=120))
    pred = walk_forward_uefa(world, target, xi=0.003)
    assert len(pred) == target.sum()
    assert (pred["train_max_date"] < pred["Date"]).all()
    assert np.allclose(pred[["pH", "pD", "pA"]].sum(axis=1), 1)


def test_value_prior_gradient(world):
    small = world[world["Date"] < "2021-01-01"]
    teams = sorted(set(small.home_id) | set(small.away_id))
    values = {t: 16 + (0.5 if t.startswith("A") else 0) + 0.05 * int(t[1:]) for t in teams}
    m = CrossLeagueModel(xi=0.002, use_values=True).fit(small, team_values=values)
    f = m.objective_
    rng = np.random.default_rng(2)
    n = 1 + 2 * len(m.leagues) + 2 * len(m.teams) + len(m.H) + 3
    theta = rng.normal(0, 0.2, n)
    theta[n - 3] = -0.05
    _, g = f(theta)
    for k in rng.choice(n, 10, replace=False).tolist() + [n - 2, n - 1]:
        e = np.zeros(n)
        e[k] = 1e-6
        assert g[k] == pytest.approx((f(theta + e)[0] - f(theta - e)[0]) / 2e-6, rel=1e-4, abs=1e-4)
