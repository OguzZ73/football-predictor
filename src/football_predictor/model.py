"""Time-weighted Poisson / Dixon-Coles model for football scores.

For a match between home team ``h`` and away team ``a``::

    log λ = attack[h] + defence[a] + home      (expected home goals)
    log μ = attack[a] + defence[h]             (expected away goals)

    P(X=x, Y=y) = τ(x, y; λ, μ, ρ) · Pois(x; λ) · Pois(y; μ)

with the Dixon-Coles correction τ that only touches 0-0, 1-0, 0-1 and 1-1.
Parameters maximise the weighted log-likelihood Σ w_i log P_i, with weights
w_i = exp(-ξ · days_before_reference), a small L2 penalty for stability, and
the identifiability constraint Σ attack = 0. The penalty shrinks attack towards 0
and defence towards its mean, so teams with few matches stay finite.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

MAX_GOALS = 7  # score matrix covers 0..7 goals for each side (8x8)
RHO_BOUNDS = (-0.3, 0.3)


def time_weights(dates: pd.Series, ref_date: pd.Timestamp, xi: float) -> np.ndarray:
    """Exponential decay weights exp(-xi * age_in_days); matches after ref_date get 0."""
    age = (pd.Timestamp(ref_date) - pd.to_datetime(dates)).dt.days.to_numpy(dtype=float)
    w = np.exp(-xi * np.clip(age, 0, None))
    w[age < 0] = 0.0
    return w


def dc_tau(x: np.ndarray, y: np.ndarray, lam: np.ndarray, mu: np.ndarray,
           rho: float) -> np.ndarray:
    """Dixon-Coles low-score correction factor τ(x, y)."""
    tau = np.ones_like(lam, dtype=float)
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)
    tau[m00] = 1 - lam[m00] * mu[m00] * rho
    tau[m01] = 1 + lam[m01] * rho
    tau[m10] = 1 + mu[m10] * rho
    tau[m11] = 1 - rho
    return tau


def score_matrix(lam: float, mu: float, rho: float = 0.0,
                 max_goals: int = MAX_GOALS) -> np.ndarray:
    """(max_goals+1)² matrix P[home_goals, away_goals], renormalised to sum to 1.

    Mass beyond ``max_goals`` is tiny for realistic rates; renormalising keeps
    derived probabilities consistent.
    """
    g = np.arange(max_goals + 1)
    m = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
    m[0, 0] *= 1 - lam * mu * rho
    m[0, 1] *= 1 + lam * rho
    m[1, 0] *= 1 + mu * rho
    m[1, 1] *= 1 - rho
    return m / m.sum()


def outcome_probs(m: np.ndarray) -> np.ndarray:
    """[P(home win), P(draw), P(away win)] from a score matrix."""
    return np.array([np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()])


def over_prob(m: np.ndarray, line: float = 2.5) -> float:
    """P(total goals > line)."""
    i, j = np.indices(m.shape)
    return float(m[(i + j) > line].sum())


def btts_prob(m: np.ndarray) -> float:
    """P(both teams score)."""
    return float(m[1:, 1:].sum())


def top_scores(m: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
    """The k most likely scores as ("h-a", probability)."""
    flat = np.argsort(m, axis=None)[::-1][:k]
    return [(f"{i}-{j}", float(m[i, j])) for i, j in zip(*np.unravel_index(flat, m.shape))]


def effective_goals(df: pd.DataFrame, goal_weight: float) -> tuple[np.ndarray, np.ndarray]:
    """α·goals + (1−α)·(shots on target × goals-per-shot-on-target) for each side.

    The conversion rate is computed from ``df`` itself (the training matches), so
    no future information is used. Rows without shot data keep their goals.
    """
    x = df["FTHG"].to_numpy(dtype=float)
    y = df["FTAG"].to_numpy(dtype=float)
    if goal_weight >= 1.0 or not {"SOTH", "SOTA"} <= set(df.columns):
        return x, y
    sh = df["SOTH"].to_numpy(dtype=float)
    sa = df["SOTA"].to_numpy(dtype=float)
    ok = ~(np.isnan(sh) | np.isnan(sa))
    if ok.sum() < 50 or (sh[ok].sum() + sa[ok].sum()) == 0:
        return x, y
    conv = (x[ok].sum() + y[ok].sum()) / (sh[ok].sum() + sa[ok].sum())
    a = goal_weight
    xe = np.where(ok, a * x + (1 - a) * conv * np.nan_to_num(sh), x)
    ye = np.where(ok, a * y + (1 - a) * conv * np.nan_to_num(sa), y)
    return xe, ye


@dataclass
class MatchPrediction:
    """Everything derived from one score matrix."""

    home: str
    away: str
    lam: float
    mu: float
    matrix: np.ndarray

    @property
    def probs(self) -> np.ndarray:
        return outcome_probs(self.matrix)

    def summary(self) -> dict:
        p = self.probs
        return {
            "home": self.home, "away": self.away,
            "exp_home_goals": self.lam, "exp_away_goals": self.mu,
            "p_home": p[0], "p_draw": p[1], "p_away": p[2],
            "over_2_5": over_prob(self.matrix), "under_2_5": 1 - over_prob(self.matrix),
            "btts": btts_prob(self.matrix),
            "top_scores": top_scores(self.matrix),
        }


@dataclass
class DixonColes:
    """Time-weighted Dixon-Coles model. ``dixon_coles=False`` gives the plain Poisson model."""

    xi: float = 0.0019
    dixon_coles: bool = True
    l2: float = 2.0
    goal_weight: float = 1.0  # α: fit on α·goals + (1−α)·shots-on-target expected goals
    use_values: bool = False  # shrink team strength towards a squad-market-value prediction
    value_coef: tuple[float, float] = (0.0, 0.0)  # fitted (b_attack, b_defence) per z-score
    teams: list[str] = field(default_factory=list)
    attack: np.ndarray = field(default_factory=lambda: np.zeros(0))
    defence: np.ndarray = field(default_factory=lambda: np.zeros(0))
    home: float = 0.0
    rho: float = 0.0
    fallback: tuple[float, float] = (0.0, 0.0)
    ref_date: pd.Timestamp | None = None
    n_matches: int = 0

    # ---------------------------------------------------------------- fitting
    def fit(self, df: pd.DataFrame, ref_date: pd.Timestamp | None = None,
            init: "DixonColes | None" = None,
            team_values: dict[str, float] | None = None) -> "DixonColes":
        """Fit on matches strictly before ``ref_date`` (default: day after the last match).

        With ``use_values`` and ``team_values`` (log squad value per team, known at
        ``ref_date``) the L2 penalty pulls attack towards b_a·z and defence towards
        b_d·z, where z is the standardised log value and b_a, b_d are estimated.
        """
        if ref_date is None:
            ref_date = df["Date"].max() + pd.Timedelta(days=1)
        ref_date = pd.Timestamp(ref_date)
        df = df[df["Date"] < ref_date]
        if df.empty:
            raise ValueError("No matches before the reference date")
        teams = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
        idx = {t: i for i, t in enumerate(teams)}
        hi = df["HomeTeam"].map(idx).to_numpy()
        ai = df["AwayTeam"].map(idx).to_numpy()
        x = df["FTHG"].to_numpy()
        y = df["FTAG"].to_numpy()
        xe, ye = effective_goals(df, self.goal_weight)
        w = time_weights(df["Date"], ref_date, self.xi)
        n = len(teams)

        from .values import standardized
        use_v = self.use_values and bool(team_values)
        z = standardized(team_values, teams) if use_v else np.zeros(n)
        k = 2 * n - 1  # theta = attack[0..n-2], defence[0..n-1], home, rho, b_att, b_def
        theta0 = np.zeros(k + 4)
        theta0[k] = 0.25
        if init is not None and init.teams:
            old = {t: i for i, t in enumerate(init.teams)}
            att = np.array([init.attack[old[t]] if t in old else 0.0 for t in teams])
            att -= att.mean()
            theta0[: n - 1] = att[:-1]
            theta0[n - 1: 2 * n - 1] = [init.defence[old[t]] if t in old else 0.0 for t in teams]
            theta0[k] = init.home
            theta0[k + 1] = init.rho if self.dixon_coles else 0.0
            if use_v:
                theta0[k + 2], theta0[k + 3] = init.value_coef

        def unpack(theta):
            att = np.append(theta[: n - 1], -theta[: n - 1].sum())
            return att, theta[n - 1: k], theta[k], theta[k + 1]

        def nll(theta):
            att, dfn, home, rho = unpack(theta)
            ba, bd = theta[k + 2], theta[k + 3]
            if not self.dixon_coles:
                rho = 0.0
            # clip keeps exp() finite when the line search tries extreme steps
            log_lam = np.clip(att[hi] + dfn[ai] + home, -15, 5)
            log_mu = np.clip(att[ai] + dfn[hi], -15, 5)
            lam, mu = np.exp(log_lam), np.exp(log_mu)
            tau = dc_tau(x, y, lam, mu, rho)
            tau_safe = np.maximum(tau, 1e-10)
            ll = xe * log_lam - lam + ye * log_mu - mu + np.log(tau_safe)
            dc = dfn - dfn.mean()  # defence also carries the overall goal level: shrink to its mean
            ra, rd = att - ba * z, dc - bd * z  # residuals from the squad-value prediction
            obj = -(w * ll).sum() + 0.5 * self.l2 * (ra @ ra + rd @ rd)

            # analytic gradient of log τ with respect to log λ, log μ and ρ
            g_lam = xe - lam
            g_mu = ye - mu
            g_rho = np.zeros_like(lam)
            m00 = (x == 0) & (y == 0)
            m01 = (x == 0) & (y == 1)
            m10 = (x == 1) & (y == 0)
            m11 = (x == 1) & (y == 1)
            t = tau_safe
            g_lam[m00] -= lam[m00] * mu[m00] * rho / t[m00]
            g_mu[m00] -= lam[m00] * mu[m00] * rho / t[m00]
            g_rho[m00] = -lam[m00] * mu[m00] / t[m00]
            g_lam[m01] += lam[m01] * rho / t[m01]
            g_rho[m01] = lam[m01] / t[m01]
            g_mu[m10] += mu[m10] * rho / t[m10]
            g_rho[m10] = mu[m10] / t[m10]
            g_rho[m11] = -1 / t[m11]
            g_lam *= w
            g_mu *= w
            d_att = np.bincount(hi, g_lam, n) + np.bincount(ai, g_mu, n)
            d_def = np.bincount(ai, g_lam, n) + np.bincount(hi, g_mu, n)
            d_att = -d_att + self.l2 * ra
            d_def = -d_def + self.l2 * (rd - rd.mean())
            grad = np.empty_like(theta)
            grad[: n - 1] = d_att[:-1] - d_att[-1]  # chain rule for att[-1] = -Σ others
            grad[n - 1: k] = d_def
            grad[k] = -g_lam.sum()
            grad[k + 1] = -(w * g_rho).sum() if self.dixon_coles else 0.0
            grad[k + 2] = -self.l2 * (ra @ z)
            grad[k + 3] = -self.l2 * (rd @ z)
            return obj, grad

        fixed = (0.0, 0.0)
        bounds = ([(None, None)] * (2 * n) + [RHO_BOUNDS if self.dixon_coles else fixed]
                  + [(None, None) if use_v else fixed] * 2)
        self.objective_ = nll  # exposed for gradient checks in the tests
        res = minimize(nll, theta0, jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 2000, "gtol": 1e-6})
        att, dfn, home, rho = unpack(res.x)
        self.teams, self.attack, self.defence = teams, att, dfn
        self.home, self.rho = float(home), float(rho if self.dixon_coles else 0.0)
        self.value_coef = (float(res.x[k + 2]), float(res.x[k + 3]))
        self.ref_date, self.n_matches = ref_date, len(df)
        self.fallback = self._weak_team_params(df)
        return self

    def _weak_team_params(self, df: pd.DataFrame, k: int = 4) -> tuple[float, float]:
        """Mean parameters of the k weakest teams of the latest season, used for unseen
        (usually newly promoted) teams."""
        last = df[df["Season"] == df["Season"].iloc[-1]] if "Season" in df else df
        recent = sorted(set(last["HomeTeam"]) | set(last["AwayTeam"]))
        ids = [self.teams.index(t) for t in recent]
        strength = self.attack[ids] - self.defence[ids]
        weakest = np.array(ids)[np.argsort(strength)[:k]]
        return float(self.attack[weakest].mean()), float(self.defence[weakest].mean())

    # ------------------------------------------------------------- predicting
    def _params(self, team: str, allow_unknown: bool) -> tuple[float, float]:
        if team in self.teams:
            i = self.teams.index(team)
            return float(self.attack[i]), float(self.defence[i])
        if allow_unknown:
            return self.fallback
        raise KeyError(f"Unknown team {team!r}")

    def rates(self, home: str, away: str, allow_unknown: bool = False) -> tuple[float, float]:
        """Expected goals (λ, μ) for home and away team."""
        ah, dh = self._params(home, allow_unknown)
        aa, da = self._params(away, allow_unknown)
        return float(np.exp(ah + da + self.home)), float(np.exp(aa + dh))

    def predict(self, home: str, away: str, allow_unknown: bool = False,
                max_goals: int = MAX_GOALS) -> MatchPrediction:
        """Score matrix and derived probabilities for one match."""
        lam, mu = self.rates(home, away, allow_unknown)
        return MatchPrediction(home, away, lam, mu, score_matrix(lam, mu, self.rho, max_goals))

    def predict_outcomes(self, df: pd.DataFrame, allow_unknown: bool = True) -> np.ndarray:
        """(n, 3) array of 1-X-2 probabilities for the rows of ``df``."""
        return np.array([
            self.predict(h, a, allow_unknown).probs
            for h, a in zip(df["HomeTeam"], df["AwayTeam"])
        ])

    def ratings(self) -> pd.DataFrame:
        """Team parameters sorted by overall strength (attack − defence)."""
        out = pd.DataFrame({"team": self.teams, "attack": self.attack, "defence": self.defence})
        out["strength"] = out["attack"] - out["defence"]
        return out.sort_values("strength", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        return {
            "xi": self.xi, "dixon_coles": self.dixon_coles, "l2": self.l2,
            "goal_weight": self.goal_weight, "use_values": self.use_values,
            "value_coef": list(self.value_coef),
            "teams": self.teams, "attack": list(map(float, self.attack)),
            "defence": list(map(float, self.defence)), "home": self.home, "rho": self.rho,
            "fallback": list(self.fallback),
            "ref_date": None if self.ref_date is None else self.ref_date.isoformat(),
            "n_matches": self.n_matches,
        }

    def save(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path | str) -> "DixonColes":
        d = json.loads(Path(path).read_text())
        return cls(
            xi=d["xi"], dixon_coles=d["dixon_coles"], l2=d["l2"],
            goal_weight=d.get("goal_weight", 1.0), use_values=d.get("use_values", False),
            value_coef=tuple(d.get("value_coef", (0.0, 0.0))), teams=d["teams"],
            attack=np.array(d["attack"]), defence=np.array(d["defence"]),
            home=d["home"], rho=d["rho"], fallback=tuple(d["fallback"]),
            ref_date=None if d["ref_date"] is None else pd.Timestamp(d["ref_date"]),
            n_matches=d["n_matches"],
        )


def fit_poisson_glm(df: pd.DataFrame, weights: np.ndarray | None = None):
    """Independent-Poisson model as a statsmodels GLM (reference implementation).

    Each match gives two rows (home goals, away goals); regressors are team
    attack dummies, opponent defence dummies and a home indicator. Used to
    cross-check the hand-written likelihood with ρ = 0 and no penalty.
    """
    import statsmodels.api as sm

    teams = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
    n = len(df)
    w = np.ones(n) if weights is None else np.asarray(weights)
    long = pd.DataFrame({
        "goals": np.concatenate([df["FTHG"], df["FTAG"]]),
        "team": np.concatenate([df["HomeTeam"], df["AwayTeam"]]),
        "opp": np.concatenate([df["AwayTeam"], df["HomeTeam"]]),
        "is_home": np.r_[np.ones(n), np.zeros(n)],
        "w": np.r_[w, w],
    })
    team_cat = pd.Categorical(long["team"], categories=teams)
    opp_cat = pd.Categorical(long["opp"], categories=teams)
    X = pd.concat([
        pd.get_dummies(team_cat, prefix="att", dtype=float),
        pd.get_dummies(opp_cat, prefix="def", dtype=float).iloc[:, 1:],  # drop one level
        long[["is_home"]],
    ], axis=1)
    return sm.GLM(long["goals"], X, family=sm.families.Poisson(),
                  var_weights=long["w"]).fit(), teams
