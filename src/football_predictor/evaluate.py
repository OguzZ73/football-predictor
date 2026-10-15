"""Time-ordered evaluation: walk-forward predictions, metrics, baselines and calibration.

Every prediction for a match played in week W is made by a model fitted only on
matches played before the Monday that starts week W. There is no random split.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import outcome
from .model import DixonColes

XI_GRID = (0.0, 0.0005, 0.001, 0.0019, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02)
LABELS = ("home", "draw", "away")


# ------------------------------------------------------------------- metrics
def log_loss(p: np.ndarray, y: np.ndarray, eps: float = 1e-15) -> float:
    """Mean negative log probability of the observed outcome."""
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], eps, 1))))


def brier(p: np.ndarray, y: np.ndarray) -> float:
    """Multi-class Brier score: mean Σ_k (p_k − o_k)²."""
    return float(np.mean(((p - np.eye(3)[y]) ** 2).sum(axis=1)))


def rps(p: np.ndarray, y: np.ndarray) -> float:
    """Ranked Probability Score for the ordered outcomes home < draw < away."""
    cum_p = np.cumsum(p, axis=1)[:, :-1]
    cum_o = np.cumsum(np.eye(3)[y], axis=1)[:, :-1]
    return float(np.mean(((cum_p - cum_o) ** 2).sum(axis=1) / 2))


def accuracy(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(p.argmax(axis=1) == y))


def all_metrics(p: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {"n": len(y), "log_loss": log_loss(p, y), "rps": rps(p, y),
            "brier": brier(p, y), "accuracy": accuracy(p, y)}


def paired_logloss_diff(p_a: np.ndarray, p_b: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Mean per-match log-loss difference (a − b) and its standard error."""
    idx = np.arange(len(y))
    d = -np.log(np.clip(p_a[idx, y], 1e-15, 1)) + np.log(np.clip(p_b[idx, y], 1e-15, 1))
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


# ----------------------------------------------------------------- baselines
def market_probs(df: pd.DataFrame) -> np.ndarray:
    """Bookmaker probabilities 1/odds normalised to remove the margin (NaN if no odds)."""
    if not {"OddsH", "OddsD", "OddsA"} <= set(df.columns):
        return np.full((len(df), 3), np.nan)
    inv = 1.0 / df[["OddsH", "OddsD", "OddsA"]].to_numpy(dtype=float)
    return inv / inv.sum(axis=1, keepdims=True)


def frequency_probs(train: pd.DataFrame, n: int) -> np.ndarray:
    """Training-set class frequencies repeated for n matches."""
    freq = np.bincount(outcome(train), minlength=3) / len(train)
    return np.tile(freq, (n, 1))


def always_home_probs(n: int) -> np.ndarray:
    return np.tile([1.0, 0.0, 0.0], (n, 1))


# ------------------------------------------------------------------ splitting
def complete_seasons(df: pd.DataFrame, min_share: float = 0.95) -> list[str]:
    """Seasons with (almost) all n·(n−1) fixtures played, in chronological order."""
    out = []
    for season, g in df.groupby("Season", sort=False):
        n = len(set(g["HomeTeam"]) | set(g["AwayTeam"]))
        if len(g) >= min_share * n * (n - 1):
            out.append(season)
    return sorted(out, key=lambda s: df.loc[df["Season"] == s, "Date"].min())


def time_split(df: pd.DataFrame, test_season: str | None = None) -> tuple[list[str], str, str]:
    """(train seasons, validation season, test season), strictly ordered in time.

    Default test season is the latest complete season; validation is the season
    before it and everything earlier is training history.
    """
    order = sorted(df["Season"].unique(), key=lambda s: df.loc[df["Season"] == s, "Date"].min())
    if test_season is None:
        test_season = complete_seasons(df)[-1]
    i = order.index(test_season)
    if i < 2:
        raise ValueError("Need at least two seasons before the test season")
    return order[: i - 1], order[i - 1], test_season


# --------------------------------------------------------------- walk-forward
def _week_start(dates: pd.Series) -> pd.Series:
    d = pd.to_datetime(dates).dt.normalize()
    return d - pd.to_timedelta(d.dt.weekday, unit="D")


def walk_forward(df: pd.DataFrame, target: pd.Series, xi: float, dixon_coles: bool = True,
                 l2: float = 2.0, goal_weight: float = 1.0, values=None) -> pd.DataFrame:
    """Predict the rows selected by boolean ``target`` with weekly refits on past data only.

    Returns the target rows with columns pH, pD, pA and ``train_max_date`` (the
    latest match date the predicting model was allowed to see).
    """
    rows = df[target].copy()
    rows["week"] = _week_start(rows["Date"])
    preds, last = [], None
    for week, group in rows.groupby("week", sort=True):
        past = df[df["Date"] < week]
        tv = None
        if values is not None:  # squad values known before this week (values.SquadValues)
            tv = values.log_values(week, sorted(set(past["HomeTeam"]) | set(past["AwayTeam"])))
        model = DixonColes(xi=xi, dixon_coles=dixon_coles, l2=l2, goal_weight=goal_weight,
                           use_values=values is not None).fit(past, ref_date=week, init=last,
                                                              team_values=tv)
        last = model
        p = model.predict_outcomes(group, allow_unknown=True)
        preds.append(pd.DataFrame(p, columns=["pH", "pD", "pA"], index=group.index)
                     .assign(train_max_date=past["Date"].max()))
    out = rows.drop(columns="week").join(pd.concat(preds))
    out["y"] = outcome(out)
    return out


def probs_of(pred: pd.DataFrame) -> np.ndarray:
    return pred[["pH", "pD", "pA"]].to_numpy()


GOAL_WEIGHTS = (1.0, 0.75, 0.5, 0.25)  # α grid: 1 = goals only, lower = more weight on shots


def select_params(df: pd.DataFrame, val_season: str, grid=XI_GRID, goal_weights=GOAL_WEIGHTS,
                  verbose: bool = False) -> tuple[float, float, pd.DataFrame]:
    """Choose (ξ, α) by walk-forward log-loss on the validation season (data after it unused).

    α < 1 is only tried when shots-on-target data are available.
    """
    hist = df[df["Date"] <= df.loc[df["Season"] == val_season, "Date"].max()]
    target = hist["Season"] == val_season
    has_shots = "SOTH" in hist and hist.loc[target, "SOTH"].notna().mean() > 0.8
    alphas = goal_weights if has_shots else (1.0,)
    rows = []
    for a in alphas:
        for xi in grid:
            pred = walk_forward(hist, target, xi, goal_weight=a)
            m = all_metrics(probs_of(pred), pred["y"].to_numpy())
            rows.append({"goal_weight": a, "xi": xi, **m})
            if verbose:
                print(f"  α={a:<5} xi={xi:<7} log-loss={m['log_loss']:.4f}  rps={m['rps']:.4f}")
    table = pd.DataFrame(rows)
    best = table.loc[table["log_loss"].idxmin()]
    return float(best["xi"]), float(best["goal_weight"]), table


def select_xi(df: pd.DataFrame, val_season: str, grid=XI_GRID,
              verbose: bool = False) -> tuple[float, pd.DataFrame]:
    """Choose ξ alone (goals only) by walk-forward log-loss on the validation season."""
    xi, _, table = select_params(df, val_season, grid, (1.0,), verbose)
    return xi, table


@dataclass
class EvaluationResult:
    xi: float
    xi_table: pd.DataFrame
    metrics: pd.DataFrame
    predictions: pd.DataFrame  # test rows with DC, Poisson and market probabilities
    seasons: tuple[list[str], str, str]
    blend_weight: float | None = None
    goal_weight: float = 1.0
    use_values: bool = False
    l2: float = 2.0


VALUE_L2_GRID = (0.5, 1.0, 2.0, 5.0)


def select_value_prior(df: pd.DataFrame, val_season: str, xi: float, goal_weight: float, values,
                       base_loss: float, verbose: bool = False) -> tuple[bool, float]:
    """Use the squad-value prior only if it beats ``base_loss`` on the validation season.

    Returns (use_values, l2). The L2 strength is chosen on the same validation season.
    """
    hist = df[df["Date"] <= df.loc[df["Season"] == val_season, "Date"].max()]
    target = hist["Season"] == val_season
    losses = {}
    for l2 in VALUE_L2_GRID:
        pred = walk_forward(hist, target, xi, l2=l2, goal_weight=goal_weight, values=values)
        losses[l2] = log_loss(probs_of(pred), pred["y"].to_numpy())
        if verbose:
            print(f"  squad values, l2={l2:<4} log-loss={losses[l2]:.4f}  (without: {base_loss:.4f})")
    best = min(losses, key=losses.get)
    return (True, best) if losses[best] < base_loss else (False, 2.0)


def evaluate(df: pd.DataFrame, test_season: str | None = None, xi: float | None = None,
             verbose: bool = False, goal_weight: float | None = None,
             values=None) -> EvaluationResult:
    """Full evaluation on the test season against baselines (see module docstring)."""
    train_s, val_s, test_s = time_split(df, test_season)
    if xi is None:
        xi, alpha, xi_table = select_params(df, val_s, verbose=verbose)
        base_loss = float(xi_table["log_loss"].min())
    else:
        alpha, xi_table = (1.0 if goal_weight is None else goal_weight), pd.DataFrame()
        base_loss = np.inf if values is not None else 0.0
    use_v, l2 = (select_value_prior(df, val_s, xi, alpha, values, base_loss, verbose)
                 if values is not None else (False, 2.0))
    vals = values if use_v else None
    hist = df[df["Date"] <= df.loc[df["Season"] == test_s, "Date"].max()]
    target = hist["Season"] == test_s
    test_start = hist.loc[target, "Date"].min()

    dc = walk_forward(hist, target, xi, dixon_coles=True, goal_weight=alpha, l2=l2, values=vals)
    po = walk_forward(hist, target, xi, dixon_coles=False, goal_weight=alpha, l2=l2, values=vals)
    y = dc["y"].to_numpy()
    n = len(y)
    candidates = {
        "Always home win": always_home_probs(n),
        "Class frequencies": frequency_probs(hist[hist["Date"] < test_start], n),
    }
    if alpha < 1 or use_v:  # ablation: same ξ, goals only, no values
        candidates["Dixon-Coles, goals only (α=1)"] = probs_of(walk_forward(hist, target, xi))
    if alpha < 1 and use_v:  # ablation: shots, no values
        candidates[f"Dixon-Coles + shots (α={alpha:g}), no values"] = probs_of(
            walk_forward(hist, target, xi, goal_weight=alpha))
    suffix = (f" + shots (α={alpha:g})" if alpha < 1 else "") + (f" + squad value (l2={l2:g})" if use_v else "")
    candidates[f"Poisson (no DC){suffix}"] = probs_of(po)
    candidates[f"Dixon-Coles{suffix}"] = probs_of(dc)
    rows = [{"model": k, **all_metrics(p, y)} for k, p in candidates.items()]

    mk = market_probs(dc)
    has = ~np.isnan(mk).any(axis=1)
    blend_w = None
    if has.any():
        rows.append({"model": "Dixon-Coles (rows with odds)", **all_metrics(probs_of(dc)[has], y[has])})
        rows.append({"model": "Bookmaker odds", **all_metrics(mk[has], y[has])})
        blend_w = blend_weight(df, val_s, xi, alpha, l2=l2, values=vals)
        if blend_w is not None:
            bl = blend(probs_of(dc)[has], mk[has], blend_w)
            rows.append({"model": f"Blend {blend_w:.2f}·model + {1 - blend_w:.2f}·market",
                         **all_metrics(bl, y[has])})

    preds = dc.rename(columns={"pH": "dc_H", "pD": "dc_D", "pA": "dc_A"})
    preds[["po_H", "po_D", "po_A"]] = probs_of(po)
    preds[["mk_H", "mk_D", "mk_A"]] = mk
    return EvaluationResult(xi, xi_table, pd.DataFrame(rows), preds, (train_s, val_s, test_s),
                            blend_w, alpha, use_v, l2)


def blend(model: np.ndarray, market: np.ndarray, w: float) -> np.ndarray:
    """Linear opinion pool w·model + (1−w)·market."""
    return w * model + (1 - w) * market


def blend_weight(df: pd.DataFrame, val_season: str, xi: float, goal_weight: float = 1.0,
                 grid=np.round(np.arange(0, 1.0001, 0.05), 2), l2: float = 2.0,
                 values=None) -> float | None:
    """Blend weight minimising walk-forward log-loss on the validation season (None if no odds)."""
    hist = df[df["Date"] <= df.loc[df["Season"] == val_season, "Date"].max()]
    pred = walk_forward(hist, hist["Season"] == val_season, xi, goal_weight=goal_weight,
                        l2=l2, values=values)
    mk = market_probs(pred)
    has = ~np.isnan(mk).any(axis=1)
    if has.sum() < 50:
        return None
    p, y = probs_of(pred)[has], pred["y"].to_numpy()[has]
    losses = {w: log_loss(blend(p, mk[has], w), y) for w in grid}
    return float(min(losses, key=losses.get))


# ---------------------------------------------------------------- calibration
def calibration_table(p: np.ndarray, y: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Per outcome and probability bin: mean forecast, observed rate and 95% Wilson interval."""
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for k, label in enumerate(LABELS):
        hit = (y == k).astype(float)
        b = np.clip(np.digitize(p[:, k], edges) - 1, 0, bins - 1)
        for i in range(bins):
            sel = b == i
            n = int(sel.sum())
            if n == 0:
                continue
            obs = hit[sel].mean()
            z = 1.96
            centre = (obs + z * z / (2 * n)) / (1 + z * z / n)
            half = z * np.sqrt(obs * (1 - obs) / n + z * z / (4 * n * n)) / (1 + z * z / n)
            rows.append({"outcome": label, "bin": i, "n": n, "predicted": p[sel, k].mean(),
                         "observed": obs, "lo": centre - half, "hi": centre + half})
    return pd.DataFrame(rows)
