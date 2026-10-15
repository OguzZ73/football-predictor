"""Monte Carlo simulation of the rest of a season."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .model import DixonColes


def league_table(played: pd.DataFrame, teams: list[str]) -> pd.DataFrame:
    """Points, goals for/against for ``teams`` from played matches (3/1/0 points)."""
    t = pd.DataFrame(0, index=teams, columns=["played", "points", "gf", "ga"])
    for h, a, x, y in played[["HomeTeam", "AwayTeam", "FTHG", "FTAG"]].itertuples(index=False):
        t.loc[[h, a], "played"] += 1
        t.loc[h, ["gf", "ga"]] += [x, y]
        t.loc[a, ["gf", "ga"]] += [y, x]
        if x > y:
            t.loc[h, "points"] += 3
        elif x < y:
            t.loc[a, "points"] += 3
        else:
            t.loc[[h, a], "points"] += 1
    return t


def remaining_fixtures(played: pd.DataFrame, teams: list[str]) -> list[tuple[str, str]]:
    """Double round-robin fixtures (each ordered pair once) not yet played."""
    done = set(zip(played["HomeTeam"], played["AwayTeam"]))
    return [(h, a) for h in teams for a in teams if h != a and (h, a) not in done]


@dataclass
class SeasonSimulation:
    table: pd.DataFrame        # current table
    summary: pd.DataFrame      # per-team probabilities, sorted by expected points
    positions: pd.DataFrame    # P(final position = k), teams × positions
    n_remaining: int


def simulate_season(model: DixonColes, season: pd.DataFrame, n: int = 10_000,
                    europe_spots: int = 4, relegation_spots: int = 3,
                    seed: int = 42, teams: list[str] | None = None) -> SeasonSimulation:
    """Simulate remaining fixtures ``n`` times by sampling scores from the score matrices.

    ``season`` holds the matches of this season already played; ``teams`` lists the
    league's teams when some have not played yet (default: teams in ``season``). Ties on points are
    broken by goal difference, then goals scored, then at random (head-to-head
    rules are not modelled).
    """
    rng = np.random.default_rng(seed)
    if teams is None:
        teams = sorted(set(season["HomeTeam"]) | set(season["AwayTeam"]))
    idx = {t: i for i, t in enumerate(teams)}
    table = league_table(season, teams)
    fixtures = remaining_fixtures(season, teams)
    T = len(teams)

    pts = np.tile(table["points"].to_numpy(float), (n, 1))
    gd = np.tile((table["gf"] - table["ga"]).to_numpy(float), (n, 1))
    gf = np.tile(table["gf"].to_numpy(float), (n, 1))
    for h, a in fixtures:
        m = model.predict(h, a, allow_unknown=True).matrix
        cells = np.searchsorted(np.cumsum(m.ravel()), rng.random(n) * m.sum())
        cells = np.minimum(cells, m.size - 1)
        x, y = np.divmod(cells, m.shape[1])
        i, j = idx[h], idx[a]
        pts[:, i] += np.where(x > y, 3, np.where(x == y, 1, 0))
        pts[:, j] += np.where(y > x, 3, np.where(x == y, 1, 0))
        gd[:, i] += x - y
        gd[:, j] += y - x
        gf[:, i] += x
        gf[:, j] += y

    # Lexicographic sort key: points > goal difference > goals for > random.
    key = pts * 1e8 + (gd + 500) * 1e4 + gf + rng.random((n, T))
    order = np.argsort(-key, axis=1)           # order[s, r] = team at rank r
    rank = np.empty_like(order)
    rank[np.arange(n)[:, None], order] = np.arange(T)  # rank[s, team]

    pos = np.stack([(rank == r).mean(axis=0) for r in range(T)], axis=1)
    positions = pd.DataFrame(pos, index=teams, columns=range(1, T + 1))
    summary = pd.DataFrame({
        "points_now": table["points"].to_numpy(),
        "played": table["played"].to_numpy(),
        "exp_points": pts.mean(axis=0),
        "p_champion": positions[1].to_numpy(),
        "p_europe": positions.loc[:, 1:europe_spots].sum(axis=1).to_numpy(),
        "p_relegation": positions.loc[:, T - relegation_spots + 1:T].sum(axis=1).to_numpy(),
    }, index=teams).sort_values("exp_points", ascending=False)
    return SeasonSimulation(table, summary, positions.loc[summary.index], len(fixtures))
