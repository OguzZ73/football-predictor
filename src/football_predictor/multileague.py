"""Cross-league hierarchical Dixon-Coles model: all domestic leagues + UEFA club cups.

Team strength has two layers::

    attack(team)  = A[league] + a[team]        defence(team) = D[league] + d[team]

    log λ = c + attack(home) + defence(away) + H[venue]
    log μ = c + attack(away) + defence(home)

``a``/``d`` are shrunk towards 0 (the team's own league average) and ``A``/``D``
towards 0 (the overall average) by L2 penalties. Teams of one league are linked
through domestic matches; leagues are linked to each other only through UEFA
club competitions. Teams whose league is not in the data (e.g. Finnish clubs that
only appear in UEFA matches) share one "OTHER" league level. ``H`` is a
league-specific home advantage for domestic matches and a separate one for UEFA
matches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .data import (TRANSFERMARKT_AWARDED, TRANSFERMARKT_NAMES, normalize_team, team_key,
                   clean_matches, _read_csv_bytes)
from .model import MAX_GOALS, RHO_BOUNDS, dc_tau, score_matrix, time_weights

UEFA_COMPETITIONS = ("CL", "CLQ", "EL", "ELQ", "UCOL", "ECLQ", "USC")
OTHER = "OTHER"
UEFA = "UEFA"
# Short names users type -> Transfermarkt names
WORLD_ALIASES = {"kups": "Kuopion Palloseura", "psg": "Paris Saint-Germain", "inter": "Inter Milan",
                 "manutd": "Manchester United", "mancity": "Manchester City", "barca": "FC Barcelona"}
# Defaults chosen by walk-forward log-loss on UEFA club matches 2024-25
BEST_PARAMS = {"xi": 0.0005, "l2_team": 10.0, "l2_league": 0.1}
LEAGUE_NAMES = {
    "GB1": "England", "ES1": "Spain", "L1": "Germany", "IT1": "Italy", "FR1": "France",
    "PO1": "Portugal", "NL1": "Netherlands", "BE1": "Belgium", "TR1": "Türkiye", "SC1": "Scotland",
    "A1": "Austria", "C1": "Switzerland", "DK1": "Denmark", "GR1": "Greece", "NO1": "Norway",
    "PL1": "Poland", "RO1": "Romania", "RU1": "Russia", "SE1": "Sweden", "SER1": "Serbia",
    "TS1": "Czechia", "UKR1": "Ukraine", "KR1": "Croatia", "ARG1": "Argentina", "BRA1": "Brazil",
    "MLS1": "USA (MLS)", "MEX1": "Mexico", "JAP1": "Japan", "RSK1": "South Korea",
    "SA1": "Saudi Arabia", "AUS1": "Australia", OTHER: "Other (UEFA-only clubs)",
}
# openfootball current-season files merged on top of the Transfermarkt history
OPENFOOTBALL_LEAGUES = {"T1": "TR1", "E0": "GB1"}


def _display_name(name: str) -> str:
    return TRANSFERMARKT_NAMES.get(name, normalize_team(name))


def load_world(data_dir: Path | str = "data", since: str = "2021-07-01") -> pd.DataFrame:
    """Domestic league + UEFA matches from Transfermarkt (plus openfootball current seasons).

    Columns: Date, competition, is_uefa, home_id, away_id, HomeTeam, AwayTeam, FTHG, FTAG.
    Forfeits are dropped with the same rules as :func:`data.parse_transfermarkt_games`.
    """
    games_path = Path(data_dir) / "_transfermarkt" / "games.csv"
    if not games_path.exists():
        raise FileNotFoundError(
            f"{games_path} missing. Run: football-predictor download --league T1 --source transfermarkt")
    g = pd.read_csv(games_path, low_memory=False)
    g = g[(g["competition_type"] == "domestic_league") | g["competition_id"].isin(UEFA_COMPETITIONS)]
    g = g.dropna(subset=["home_club_goals", "away_club_goals", "home_club_id", "away_club_id"])
    g = g[pd.to_datetime(g["date"]) >= pd.Timestamp(since)]
    forfeit = (g["referee"].isna() & g["home_club_manager_name"].isna()
               & (((g["home_club_goals"] == 3) & (g["away_club_goals"] == 0))
                  | ((g["home_club_goals"] == 0) & (g["away_club_goals"] == 3))))
    g = g[~forfeit]
    df = pd.DataFrame({
        "Date": pd.to_datetime(g["date"]),
        "competition": g["competition_id"],
        "is_uefa": g["competition_id"].isin(UEFA_COMPETITIONS),
        "home_id": g["home_club_id"].astype(int).astype(str),
        "away_id": g["away_club_id"].astype(int).astype(str),
        "HomeTeam": g["home_club_name"].fillna("?").map(_display_name),
        "AwayTeam": g["away_club_name"].fillna("?").map(_display_name),
        "FTHG": g["home_club_goals"].astype(int),
        "FTAG": g["away_club_goals"].astype(int),
    })
    awarded = np.array([(d.strftime("%Y-%m-%d"), h, a) in TRANSFERMARKT_AWARDED
                        for d, h, a in zip(df["Date"], df["HomeTeam"], df["AwayTeam"])], dtype=bool)
    df = df.loc[~awarded]
    df = pd.concat([df, _openfootball_current(df, Path(data_dir))], ignore_index=True)
    df = pd.concat([df, _manual_results(df, Path(data_dir))], ignore_index=True)
    return df.sort_values(["Date", "home_id"], kind="stable").reset_index(drop=True)


def _manual_results(known: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Hand-entered results (``add-result``) not already present within ±3 days."""
    from .data import TRANSFERMARKT_COMPETITIONS, load_manual
    m = load_manual(data_dir)
    if m.empty:
        return known.iloc[:0]
    comp = m["competition"].map(lambda c: TRANSFERMARKT_COMPETITIONS.get(c, c))
    ids = dict(zip(known["HomeTeam"], known["home_id"])) | dict(zip(known["AwayTeam"], known["away_id"]))
    out = pd.DataFrame({
        "Date": m["Date"], "competition": comp, "is_uefa": comp.isin(UEFA_COMPETITIONS),
        "home_id": [ids.get(t, f"manual:{t}") for t in m["HomeTeam"]],
        "away_id": [ids.get(t, f"manual:{t}") for t in m["AwayTeam"]],
        "HomeTeam": m["HomeTeam"], "AwayTeam": m["AwayTeam"],
        "FTHG": m["FTHG"].astype(int), "FTAG": m["FTAG"].astype(int),
    })
    pairs = known.groupby(["home_id", "away_id"])["Date"].apply(list).to_dict()
    dup = [any(abs((d - k).days) <= 3 for k in pairs.get((h, a), []))
           for d, h, a in zip(out["Date"], out["home_id"], out["away_id"])]
    return out.loc[~np.array(dup, dtype=bool)]


def _openfootball_current(tm: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Matches after the Transfermarkt cut-off from data/<league>/*.csv (T1, E0)."""
    frames = []
    for league, comp in OPENFOOTBALL_LEAGUES.items():
        dom = tm[tm["competition"] == comp]
        if dom.empty or not (data_dir / league).exists():
            continue
        cutoff = dom["Date"].max()
        ids = dict(zip(dom["HomeTeam"], dom["home_id"]))  # latest name -> id
        for path in sorted((data_dir / league).glob("*.csv")):
            m = clean_matches(_read_csv_bytes(path.read_bytes()), season=path.stem)
            m = m[m["Date"] > cutoff]
            if m.empty:
                continue
            frames.append(pd.DataFrame({
                "Date": m["Date"], "competition": comp, "is_uefa": False,
                "home_id": [ids.get(t, f"{comp}:{t}") for t in m["HomeTeam"]],
                "away_id": [ids.get(t, f"{comp}:{t}") for t in m["AwayTeam"]],
                "HomeTeam": m["HomeTeam"], "AwayTeam": m["AwayTeam"],
                "FTHG": m["FTHG"], "FTAG": m["FTAG"],
            }))
    return pd.concat(frames, ignore_index=True) if frames else tm.iloc[:0]


def team_leagues(df: pd.DataFrame) -> dict[str, str]:
    """League of each team = competition of its most recent domestic match (else OTHER)."""
    dom = df[~df["is_uefa"]]
    long = pd.concat([
        dom[["Date", "home_id", "competition"]].rename(columns={"home_id": "id"}),
        dom[["Date", "away_id", "competition"]].rename(columns={"away_id": "id"}),
    ]).sort_values("Date", kind="stable")
    league = long.groupby("id")["competition"].last().to_dict()
    for t in set(df["home_id"]) | set(df["away_id"]):
        league.setdefault(t, OTHER)
    return league


def team_names(df: pd.DataFrame) -> dict[str, str]:
    """Latest display name of each team id."""
    long = pd.concat([df[["home_id", "HomeTeam"]].set_axis(["id", "name"], axis=1),
                      df[["away_id", "AwayTeam"]].set_axis(["id", "name"], axis=1)])
    return long.groupby("id")["name"].last().to_dict()


@dataclass
class CrossLeagueModel:
    """Hierarchical cross-league model (``hierarchical=False``: no league levels, pooled)."""

    xi: float = 0.003
    l2_team: float = 2.0
    l2_league: float = 0.5
    hierarchical: bool = True
    use_values: bool = False  # shrink team effects towards a squad-value prediction
    value_coef: tuple[float, float] = (0.0, 0.0)
    teams: list[str] = field(default_factory=list)
    leagues: list[str] = field(default_factory=list)
    team_league: dict[str, str] = field(default_factory=dict)
    names: dict[str, str] = field(default_factory=dict)
    c: float = 0.0
    A: np.ndarray = field(default_factory=lambda: np.zeros(0))
    D: np.ndarray = field(default_factory=lambda: np.zeros(0))
    a: np.ndarray = field(default_factory=lambda: np.zeros(0))
    d: np.ndarray = field(default_factory=lambda: np.zeros(0))
    H: dict[str, float] = field(default_factory=dict)
    rho: float = 0.0
    ref_date: pd.Timestamp | None = None
    n_matches: int = 0

    def fit(self, df: pd.DataFrame, ref_date: pd.Timestamp | None = None,
            init: "CrossLeagueModel | None" = None,
            team_values: dict[str, float] | None = None) -> "CrossLeagueModel":
        """Penalised weighted MLE on matches strictly before ``ref_date``.

        With ``use_values`` and ``team_values`` (log squad value by club id, known at
        ``ref_date``) team effects are shrunk towards b_a·z / b_d·z instead of 0,
        z being the standardised log value across all clubs.
        """
        if ref_date is None:
            ref_date = df["Date"].max() + pd.Timedelta(days=1)
        ref_date = pd.Timestamp(ref_date)
        df = df[df["Date"] < ref_date]
        league_of = team_leagues(df)
        teams = sorted(league_of)
        leagues = sorted(set(league_of.values()))
        venues = sorted(set(df.loc[~df["is_uefa"], "competition"])) + [UEFA]
        ti = {t: i for i, t in enumerate(teams)}
        li = {l: i for i, l in enumerate(leagues)}
        vi = {v: i for i, v in enumerate(venues)}
        T, L, V = len(teams), len(leagues), len(venues)

        hi = df["home_id"].map(ti).to_numpy()
        ai = df["away_id"].map(ti).to_numpy()
        tl = np.array([li[league_of[t]] for t in teams])
        lh, la = tl[hi], tl[ai]
        hv = np.where(df["is_uefa"], vi[UEFA], df["competition"].map(lambda c: vi.get(c, vi[UEFA])))
        hv = hv.astype(int)
        x = df["FTHG"].to_numpy()
        y = df["FTAG"].to_numpy()
        w = time_weights(df["Date"], ref_date, self.xi)

        from .values import standardized
        use_v = self.use_values and bool(team_values)
        z = standardized(team_values, teams) if use_v else np.zeros(T)
        # theta = [c, A(L), D(L), a(T), d(T), H(V), rho, b_a, b_d]
        sl = np.cumsum([0, 1, L, L, T, T, V, 1, 2])
        idx = lambda k: slice(sl[k], sl[k + 1])  # noqa: E731
        theta0 = np.zeros(sl[-1])
        theta0[0] = 0.2
        theta0[idx(5)] = 0.25
        if init is not None and init.teams:
            old_t = {t: i for i, t in enumerate(init.teams)}
            old_l = {l: i for i, l in enumerate(init.leagues)}
            theta0[0] = init.c
            theta0[idx(1)] = [init.A[old_l[l]] if l in old_l else 0.0 for l in leagues]
            theta0[idx(2)] = [init.D[old_l[l]] if l in old_l else 0.0 for l in leagues]
            theta0[idx(3)] = [init.a[old_t[t]] if t in old_t else 0.0 for t in teams]
            theta0[idx(4)] = [init.d[old_t[t]] if t in old_t else 0.0 for t in teams]
            theta0[idx(5)] = [init.H.get(v, 0.25) for v in venues]
            theta0[sl[6]] = init.rho
            theta0[sl[7]:sl[8]] = init.value_coef if use_v else (0.0, 0.0)
        kt, kl = self.l2_team, self.l2_league

        def nll(theta):
            c = theta[0]
            A, D, a, d, H = (theta[idx(k)] for k in range(1, 6))
            rho = theta[sl[6]]
            ba, bd = theta[sl[7]], theta[sl[7] + 1]
            ra, rd = a - ba * z, d - bd * z
            log_lam = np.clip(c + A[lh] + a[hi] + D[la] + d[ai] + H[hv], -15, 5)
            log_mu = np.clip(c + A[la] + a[ai] + D[lh] + d[hi], -15, 5)
            lam, mu = np.exp(log_lam), np.exp(log_mu)
            tau = np.maximum(dc_tau(x, y, lam, mu, rho), 1e-10)
            ll = x * log_lam - lam + y * log_mu - mu + np.log(tau)
            obj = -(w * ll).sum() + 0.5 * kt * (ra @ ra + rd @ rd) + 0.5 * kl * (A @ A + D @ D)

            g_lam, g_mu, g_rho = _dc_grad_terms(x, y, lam, mu, rho, tau)
            g_lam *= w
            g_mu *= w
            grad = np.zeros_like(theta)
            grad[0] = -(g_lam.sum() + g_mu.sum())
            if self.hierarchical:
                grad[idx(1)] = -(np.bincount(lh, g_lam, L) + np.bincount(la, g_mu, L)) + kl * A
                grad[idx(2)] = -(np.bincount(la, g_lam, L) + np.bincount(lh, g_mu, L)) + kl * D
            grad[idx(3)] = -(np.bincount(hi, g_lam, T) + np.bincount(ai, g_mu, T)) + kt * ra
            grad[idx(4)] = -(np.bincount(ai, g_lam, T) + np.bincount(hi, g_mu, T)) + kt * rd
            grad[idx(5)] = -np.bincount(hv, g_lam, V)
            grad[sl[6]] = -(w * g_rho).sum()
            grad[sl[7]] = -kt * (ra @ z)
            grad[sl[7] + 1] = -kt * (rd @ z)
            return obj, grad

        self.objective_ = nll  # exposed for gradient checks in the tests
        free = (None, None)
        lvl = free if self.hierarchical else (0.0, 0.0)
        coef = free if use_v else (0.0, 0.0)
        bounds = ([free] + [lvl] * (2 * L) + [free] * (2 * T + V) + [RHO_BOUNDS] + [coef] * 2)
        if not self.hierarchical:
            theta0[sl[1]:sl[3]] = 0.0
        res = minimize(nll, theta0, jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 5000, "gtol": 1e-5})
        th = res.x
        self.teams, self.leagues, self.team_league = teams, leagues, league_of
        self.names = team_names(df)
        self.c = float(th[0])
        self.A, self.D, self.a, self.d = (th[idx(k)].copy() for k in range(1, 5))
        self.H = dict(zip(venues, map(float, th[idx(5)])))
        self.rho = float(th[sl[6]])
        self.value_coef = (float(th[sl[7]]), float(th[sl[7] + 1]))
        self._z = dict(zip(teams, z))
        self.ref_date, self.n_matches = ref_date, len(df)
        return self

    # ------------------------------------------------------------- predicting
    def _strength(self, team: str, league: str | None = None) -> tuple[float, float]:
        """(attack, defence) on the common scale; unseen teams get their league level."""
        lg = self.team_league.get(team, league or OTHER)
        if lg not in self.leagues:
            lg = OTHER
        A = D = 0.0
        if lg in self.leagues:
            L = self.leagues.index(lg)
            A, D = self.A[L], self.D[L]
        if team in self.teams:
            i = self.teams.index(team)
            return float(A + self.a[i]), float(D + self.d[i])
        return float(A), float(D)

    def rates(self, home: str, away: str, venue: str = UEFA) -> tuple[float, float]:
        """Expected goals; ``venue`` is a league code (domestic), "UEFA" or "neutral"."""
        ah, dh = self._strength(home)
        aa, da = self._strength(away)
        h = 0.0 if venue == "neutral" else self.H.get(venue, self.H.get(UEFA, 0.0))
        return float(np.exp(self.c + ah + da + h)), float(np.exp(self.c + aa + dh))

    def predict_matrix(self, home: str, away: str, venue: str = UEFA,
                       max_goals: int = MAX_GOALS) -> tuple[float, float, np.ndarray]:
        lam, mu = self.rates(home, away, venue)
        return lam, mu, score_matrix(lam, mu, self.rho, max_goals)

    def predict_outcomes(self, df: pd.DataFrame) -> np.ndarray:
        """1-X-2 probabilities for rows of ``df`` (venue from competition)."""
        from .model import outcome_probs
        out = []
        for h, a, comp, uefa in zip(df["home_id"], df["away_id"], df["competition"], df["is_uefa"]):
            out.append(outcome_probs(self.predict_matrix(h, a, UEFA if uefa else comp)[2]))
        return np.array(out)

    def n_matches_of(self, df: pd.DataFrame, team: str) -> int:
        past = df[df["Date"] < self.ref_date]
        return int(((past["home_id"] == team) | (past["away_id"] == team)).sum())

    def team_table(self) -> pd.DataFrame:
        """All teams on one scale: strength = attack − defence (higher is better)."""
        L = {l: i for i, l in enumerate(self.leagues)}
        li = np.array([L[self.team_league[t]] for t in self.teams])
        att = self.A[li] + self.a
        dfn = self.D[li] + self.d
        out = pd.DataFrame({"id": self.teams, "team": [self.names.get(t, t) for t in self.teams],
                            "league": [self.team_league[t] for t in self.teams],
                            "attack": att, "defence": dfn, "strength": att - dfn})
        return out.sort_values("strength", ascending=False).reset_index(drop=True)

    def league_table(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        """League levels: strength = A − D (teams of the league on average).

        With ``df`` the number of UEFA matches played by the league's teams is added;
        leagues without any are not connected to the others and their level is only
        the prior (0).
        """
        out = pd.DataFrame({"league": self.leagues,
                            "name": [LEAGUE_NAMES.get(l, l) for l in self.leagues],
                            "attack": self.A, "defence": self.D, "strength": self.A - self.D,
                            "home_adv": [self.H.get(l, np.nan) for l in self.leagues]})
        if df is not None:
            u = df[df["is_uefa"] & (df["Date"] < self.ref_date)]
            ids = pd.concat([u["home_id"], u["away_id"]])
            counts = ids.map(self.team_league).value_counts()
            out["uefa_matches"] = out["league"].map(counts).fillna(0).astype(int)
        return out.sort_values("strength", ascending=False).reset_index(drop=True)

    def find_team(self, query: str) -> list[str]:
        """Team ids whose name matches ``query`` (accent/case-insensitive, common short names)."""
        k = team_key(WORLD_ALIASES.get(team_key(query), query))
        exact = [t for t in self.teams if team_key(self.names.get(t, t)) == k]
        if exact:
            return exact
        return [t for t in self.teams if k in team_key(self.names.get(t, t))]


def _dc_grad_terms(x, y, lam, mu, rho, tau):
    """d ll / d log λ, d ll / d log μ, d ll / d ρ for the Dixon-Coles likelihood."""
    g_lam = (x - lam).astype(float)
    g_mu = (y - mu).astype(float)
    g_rho = np.zeros_like(lam)
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)
    g_lam[m00] -= lam[m00] * mu[m00] * rho / tau[m00]
    g_mu[m00] -= lam[m00] * mu[m00] * rho / tau[m00]
    g_rho[m00] = -lam[m00] * mu[m00] / tau[m00]
    g_lam[m01] += lam[m01] * rho / tau[m01]
    g_rho[m01] = lam[m01] / tau[m01]
    g_mu[m10] += mu[m10] * rho / tau[m10]
    g_rho[m10] = mu[m10] / tau[m10]
    g_rho[m11] = -1 / tau[m11]
    return g_lam, g_mu, g_rho


def walk_forward_uefa(world: pd.DataFrame, target: pd.Series, **params) -> pd.DataFrame:
    """Predict the selected (UEFA) rows with weekly refits on strictly earlier matches."""
    from .data import outcome
    rows = world[target].copy()
    d = rows["Date"].dt.normalize()
    rows["week"] = d - pd.to_timedelta(d.dt.weekday, unit="D")
    preds, last = [], None
    values = params.pop("values", None)
    for week, group in rows.groupby("week", sort=True):
        tv = None
        if values is not None:
            past = world[world["Date"] < week]
            tv = values.log_values(week, sorted(set(past["home_id"]) | set(past["away_id"])))
        model = CrossLeagueModel(**params, use_values=values is not None).fit(
            world, ref_date=week, init=last, team_values=tv)
        last = model
        p = model.predict_outcomes(group)
        preds.append(pd.DataFrame(p, columns=["pH", "pD", "pA"], index=group.index)
                     .assign(train_max_date=world.loc[world["Date"] < week, "Date"].max()))
    out = rows.drop(columns="week").join(pd.concat(preds))
    out["y"] = outcome(out)
    return out
