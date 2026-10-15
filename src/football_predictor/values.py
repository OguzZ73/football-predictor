"""Squad market values from Transfermarkt player valuations (used as a prior for team strength).

The squad value of a club on a date is the sum of each player's most recent
valuation before that date (at most ``window_days`` old), credited to the club the
player belonged to at that valuation. Only valuations dated before the reference
date are used, so a model fitted for a match never sees later information.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

VALUATIONS = Path("_transfermarkt") / "player_valuations.csv"
SNAPSHOTS = Path("_transfermarkt") / "squad_values.csv"


def build_snapshots(valuations: pd.DataFrame, start: str = "2018-01-01",
                    end: str | None = None, window_days: int = 365) -> pd.DataFrame:
    """Monthly snapshots (1st of month): club_id, value_eur, n_players."""
    v = valuations[["player_id", "date", "market_value_in_eur", "current_club_id"]].dropna()
    v = v.assign(date=pd.to_datetime(v["date"])).sort_values("date", kind="stable")
    end = pd.Timestamp(end) if end else v["date"].max() + pd.offsets.MonthBegin(1)
    rows = []
    for snap in pd.date_range(start, end, freq="MS"):
        w = v[(v["date"] < snap) & (v["date"] >= snap - pd.Timedelta(days=window_days))]
        last = w.groupby("player_id").tail(1)  # sorted by date: latest valuation per player
        agg = last.groupby("current_club_id")["market_value_in_eur"].agg(["sum", "size"])
        rows.append(pd.DataFrame({"snapshot": snap, "club_id": agg.index.astype(int).astype(str),
                                  "value_eur": agg["sum"].to_numpy(), "n_players": agg["size"].to_numpy()}))
    return pd.concat(rows, ignore_index=True)


class SquadValues:
    """Lookup of squad values by club id (and optionally by display name) at a date."""

    def __init__(self, snapshots: pd.DataFrame, min_players: int = 11):
        s = snapshots[snapshots["n_players"] >= min_players]
        self._snaps = {d: g.set_index("club_id")["value_eur"] for d, g in s.groupby("snapshot")}
        self._dates = np.array(sorted(self._snaps), dtype="datetime64[ns]")
        self.name_to_id: dict[str, str] = {}

    @classmethod
    def load(cls, data_dir: Path | str = "data", rebuild: bool = False) -> "SquadValues":
        """Load cached snapshots or build them from ``player_valuations.csv``."""
        cache = Path(data_dir) / SNAPSHOTS
        if cache.exists() and not rebuild:
            snaps = pd.read_csv(cache, dtype={"club_id": str}, parse_dates=["snapshot"])
        else:
            src = Path(data_dir) / VALUATIONS
            if not src.exists():
                raise FileNotFoundError(
                    f"{src} missing. Run: kaggle datasets download davidcariboo/player-scores "
                    f"-f player_valuations.csv -p {Path(data_dir) / '_transfermarkt'} --unzip")
            snaps = build_snapshots(pd.read_csv(src))
            snaps.to_csv(cache, index=False)
        return cls(snaps)

    def at(self, date: pd.Timestamp) -> pd.Series:
        """club_id -> squad value from the latest snapshot on or before ``date``."""
        i = np.searchsorted(self._dates, np.datetime64(pd.Timestamp(date)), side="right") - 1
        if i < 0:
            return pd.Series(dtype=float)
        return self._snaps[pd.Timestamp(self._dates[i])]

    def log_values(self, date: pd.Timestamp, teams: list[str]) -> dict[str, float]:
        """log(value) for teams given as club ids or as names in :attr:`name_to_id`."""
        snap = self.at(date)
        out = {}
        for t in teams:
            cid = self.name_to_id.get(t, t)
            if cid in snap.index and snap[cid] > 0:
                out[t] = float(np.log(snap[cid]))
        return out


def standardized(log_values: dict[str, float], teams: list[str]) -> np.ndarray:
    """z-scores of log squad value aligned to ``teams`` (0 for missing), centred over teams."""
    z = np.zeros(len(teams))
    vals = np.array([log_values[t] for t in teams if t in log_values])
    if len(vals) >= 3 and vals.std() > 0:
        for i, t in enumerate(teams):
            if t in log_values:
                z[i] = (log_values[t] - vals.mean()) / vals.std()
    return z - z.mean()


def league_name_map(data_dir: Path | str, competition: str) -> dict[str, str]:
    """Display name (as used by load_league) -> Transfermarkt club id for one league."""
    from .data import TRANSFERMARKT_NAMES, normalize_team
    g = pd.read_csv(Path(data_dir) / "_transfermarkt" / "games.csv", low_memory=False,
                    usecols=["competition_id", "home_club_id", "home_club_name"])
    g = g[g["competition_id"] == competition].dropna()
    names = g["home_club_name"].map(lambda n: TRANSFERMARKT_NAMES.get(n, normalize_team(n)))
    return dict(zip(names, g["home_club_id"].astype(int).astype(str)))
