"""Build the single-file web page ("Maç Masası") with the fitted models embedded.

The page runs entirely in the browser: model parameters are written into it as
JSON and app_core.js computes the score matrix, so it can be opened from a file,
GitHub Pages or any static host.
"""

from __future__ import annotations

import json
import math
from datetime import date
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as data_mod
from .model import DixonColes

LEAGUE_LABELS = {"T1": "Süper Lig", "E0": "Premier League"}


def _league_block(league: str, data_dir: Path, models_dir: Path) -> dict | None:
    path = models_dir / f"{league}.json"
    if not path.exists():
        return None
    model = DixonColes.load(path)
    df = data_mod.load_league(league, data_dir)
    current = df[df["Season"] == df["Season"].iloc[-1]]
    current_teams = sorted(set(current["HomeTeam"]) | set(current["AwayTeam"]))

    squad: dict[str, float] = {}
    sd = 1.0
    try:
        from .values import SquadValues, league_name_map
        sv = SquadValues.load(data_dir)
        sv.name_to_id = league_name_map(data_dir, data_mod.TRANSFERMARKT_COMPETITIONS[league])
        logs = sv.log_values(model.ref_date, model.teams)
        if len(logs) >= 3:
            sd = float(np.std(list(logs.values())))
        squad = {t: round(math.exp(v) / 1e6, 1) for t, v in logs.items()}
    except (FileNotFoundError, KeyError):
        pass

    teams = {}
    for t, a, d in zip(model.teams, model.attack, model.defence):
        teams[t] = {"attack": round(float(a), 5), "defence": round(float(d), 5),
                    "squad_m": squad.get(t)}
    for t in current_teams:  # promoted team without history: model's fallback strength
        teams.setdefault(t, {"attack": round(model.fallback[0], 5),
                             "defence": round(model.fallback[1], 5), "squad_m": squad.get(t)})
    return {
        "label": LEAGUE_LABELS.get(league, league),
        "home": round(model.home, 5), "rho": round(model.rho, 5),
        "use_values": bool(model.use_values),
        "value_coef": [round(c, 5) for c in model.value_coef] if model.use_values else None,
        "value_sd": round(sd, 4),
        "data_to": str((model.ref_date - pd.Timedelta(days=1)).date()),
        "n_matches": model.n_matches,
        "current": current_teams,
        "matches_this_season": {t: int(((current["HomeTeam"] == t) | (current["AwayTeam"] == t)).sum())
                                for t in current_teams},
        "teams": teams,
    }


def _world_block(data_dir: Path) -> dict | None:
    try:
        from .multileague import BEST_PARAMS, LEAGUE_NAMES, CrossLeagueModel, load_world
        world = load_world(data_dir)
    except FileNotFoundError:
        return None
    model = CrossLeagueModel(**BEST_PARAMS).fit(world)
    counts = pd.concat([world["home_id"], world["away_id"]]).value_counts()
    teams = {}
    for i, t in enumerate(model.teams):
        teams[t] = {"name": model.names.get(t, t), "league": model.team_league[t],
                    "a": round(float(model.a[i]), 5), "d": round(float(model.d[i]), 5),
                    "n": int(counts.get(t, 0))}
    return {
        "c": round(model.c, 5), "rho": round(model.rho, 5),
        "H": {k: round(v, 5) for k, v in model.H.items()},
        "leagues": {l: {"A": round(float(a), 5), "D": round(float(d), 5),
                        "name": LEAGUE_NAMES.get(l, l)}
                    for l, a, d in zip(model.leagues, model.A, model.D)},
        "teams": teams,
        "data_to": str(world["Date"].max().date()),
        "n_matches": int(len(world)),
    }


def build_app_data(data_dir: Path | str = "data", models_dir: Path | str = "models",
                   defaults: dict | None = None) -> dict:
    """Everything the page needs, as plain JSON-serialisable values.

    ``defaults`` ({"league", "home", "away"}) picks the match the page opens with.
    """
    data_dir, models_dir = Path(data_dir), Path(models_dir)
    leagues = {lg: b for lg in ("T1", "E0") if (b := _league_block(lg, data_dir, models_dir))}
    if not leagues:
        raise FileNotFoundError("No trained models. Run: football-predictor train --league T1")
    d = {"league": "T1", "home": "Galatasaray", "away": "Fenerbahçe", **(defaults or {})}
    lg = leagues.get(d["league"])
    if lg is None or d["home"] not in lg["current"] or d["away"] not in lg["current"]:
        raise ValueError(f"Default match {d} is not a current-season fixture of the models")
    return {"built": str(date.today()), "leagues": leagues, "world": _world_block(data_dir),
            "defaults": d}


def render_app(app_data: dict, standalone: bool = True) -> str:
    """HTML for the page. ``standalone`` adds the document skeleton (for files and GitHub Pages)."""
    pkg = resources.files("football_predictor")
    template = (pkg / "app_template.html").read_text(encoding="utf-8")
    core = (pkg / "app_core.js").read_text(encoding="utf-8")
    payload = json.dumps(app_data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    page = template.replace("/*__CORE__*/", core).replace("/*__DATA__*/null", payload)
    if standalone:
        page = ('<!doctype html>\n<html lang="tr">\n<head>\n<meta charset="utf-8">\n'
                '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
                + page.replace("</style>", "</style>\n</head>\n<body>", 1) + "\n</body>\n</html>\n")
    return page


def write_app(out: Path | str, data_dir: Path | str = "data", models_dir: Path | str = "models",
              standalone: bool = True, defaults: dict | None = None) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    page = render_app(build_app_data(data_dir, models_dir, defaults), standalone)
    out.write_text(page, encoding="utf-8")
    return out
