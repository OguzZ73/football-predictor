import numpy as np
import pandas as pd
import pytest

from football_predictor.model import score_matrix


def simulate_league(n_teams=10, n_seasons=3, attack=None, defence=None, home=0.3, rho=-0.1,
                    seed=0, start="2020-08-01"):
    """Double round-robin seasons with scores drawn from a known Dixon-Coles model."""
    rng = np.random.default_rng(seed)
    teams = [f"Team{i:02d}" for i in range(n_teams)]
    if attack is None:
        attack = np.linspace(-0.4, 0.4, n_teams)
    if defence is None:
        defence = np.linspace(0.3, -0.3, n_teams) + 0.1
    rows = []
    day = pd.Timestamp(start)
    for s in range(n_seasons):
        fixtures = [(h, a) for h in range(n_teams) for a in range(n_teams) if h != a]
        rng.shuffle(fixtures)
        per_round = n_teams // 2
        for k, (h, a) in enumerate(fixtures):
            date = day + pd.Timedelta(days=7 * (k // per_round))
            lam = np.exp(attack[h] + defence[a] + home)
            mu = np.exp(attack[a] + defence[h])
            m = score_matrix(lam, mu, rho, max_goals=10)
            cell = rng.choice(m.size, p=m.ravel())
            x, y = divmod(cell, m.shape[1])
            rows.append((date, f"{20 + s}{21 + s}", teams[h], teams[a], x, y))
        day = date + pd.Timedelta(days=60)
    df = pd.DataFrame(rows, columns=["Date", "Season", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    return df.sort_values(["Date", "HomeTeam"], kind="stable").reset_index(drop=True)


@pytest.fixture(scope="session")
def small_league():
    return simulate_league()
