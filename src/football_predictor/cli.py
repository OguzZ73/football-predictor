"""Command line interface: download, train, predict, evaluate, simulate-season."""

from __future__ import annotations

import argparse
import difflib
import math
import sys
from pathlib import Path

import pandas as pd

from . import data as data_mod
from .evaluate import (all_metrics, brier, calibration_table, complete_seasons, evaluate,
                       paired_logloss_diff, rps, select_params, select_value_prior)
from .model import DixonColes

DISCLAIMER = "Educational use only — not betting advice. / Eğitim amaçlıdır, bahis tavsiyesi değildir."


def resolve_team(name: str, teams: list[str]) -> str:
    """Match user input to a known team name (case/accent-insensitive, unique prefix/substring)."""
    if name in teams:
        return name
    keys = {data_mod.team_key(t): t for t in teams}
    k = data_mod.team_key(name)
    if k in keys:
        return keys[k]
    hits = [t for kk, t in keys.items() if kk.startswith(k) or k in kk]
    if len(hits) == 1:
        return hits[0]
    close = difflib.get_close_matches(name, teams, n=5, cutoff=0.5)
    hint = hits or close
    raise SystemExit(f"Unknown team {name!r}." + (f" Did you mean: {', '.join(hint)}?" if hint else ""))


def _pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


def _model_path(args) -> Path:
    return Path(args.models_dir) / f"{args.league}.json"


# -------------------------------------------------------------------- commands
def cmd_download(args) -> None:
    seasons = data_mod.season_codes(args.seasons)
    print(f"Downloading {args.league} seasons {seasons[0]}..{seasons[-1]} from {args.source}")
    try:
        paths = data_mod.download(args.league, seasons, args.data_dir, source=args.source,
                                  force=args.force)
    except RuntimeError as exc:
        raise SystemExit(str(exc))
    print(f"{len(paths)}/{len(seasons)} files saved under {Path(args.data_dir) / args.league}")


def cmd_train(args) -> None:
    df = data_mod.load_league(args.league, args.data_dir)
    if args.xi == "auto":
        val = complete_seasons(df)[-1]
        print(f"Choosing xi and goal weight α by walk-forward log-loss on season {val} ...")
        xi, alpha, table = select_params(df, val, verbose=True)
        values = _league_values(args)
        use_v, l2 = (select_value_prior(df, val, xi, alpha, values, float(table["log_loss"].min()), True)
                     if values is not None else (False, 2.0))
    else:
        xi, alpha, use_v, l2, values = float(args.xi), float(args.goal_weight), False, 2.0, None
    tv = None
    if use_v:
        ref = df["Date"].max() + pd.Timedelta(days=1)
        tv = values.log_values(ref, sorted(set(df["HomeTeam"]) | set(df["AwayTeam"])))
    model = DixonColes(xi=xi, goal_weight=alpha, l2=l2, use_values=use_v).fit(df, team_values=tv)
    model.save(_model_path(args))
    print(f"\nTrained on {model.n_matches} matches up to {df['Date'].max():%Y-%m-%d}; "
          f"xi={xi}, α={alpha:g}, squad values={'on (l2=' + format(l2, 'g') + ')' if use_v else 'off'}, "
          f"home={model.home:.3f} (×{math.exp(model.home):.2f} goals), "
          f"rho={model.rho:.3f}")
    latest = df[df["Season"] == df["Season"].iloc[-1]]
    current = set(latest["HomeTeam"]) | set(latest["AwayTeam"])
    r = model.ratings()
    r = r[r["team"].isin(current)].reset_index(drop=True)
    r.index += 1
    print(f"\nTeam ratings (season {df['Season'].iloc[-1]} teams):")
    print(r.to_string(float_format=lambda v: f"{v:+.3f}"))
    print(f"\nModel saved to {_model_path(args)}")


def cmd_predict(args) -> None:
    from .plots import score_heatmap

    path = _model_path(args)
    if not path.exists():
        raise SystemExit(f"No model at {path}. Run: football-predictor train --league {args.league}")
    model = DixonColes.load(path)
    home, away = resolve_team(args.home, model.teams), resolve_team(args.away, model.teams)
    pred = model.predict(home, away)
    s = pred.summary()
    print(f"\n{home} (home) vs {away} (away)   [{args.league}, model data to "
          f"{(model.ref_date - pd.Timedelta(days=1)):%Y-%m-%d}]")
    print(f"Expected goals     {s['exp_home_goals']:.2f} - {s['exp_away_goals']:.2f}")
    print("-" * 38)
    print(f"Home win (1)       {_pct(s['p_home'])}")
    print(f"Draw (X)           {_pct(s['p_draw'])}")
    print(f"Away win (2)       {_pct(s['p_away'])}")
    print("-" * 38)
    print(f"Over 2.5 goals     {_pct(s['over_2_5'])}")
    print(f"Under 2.5 goals    {_pct(s['under_2_5'])}")
    print(f"Both teams score   {_pct(s['btts'])}")
    print("-" * 38)
    print("Most likely scores")
    for score, p in s["top_scores"]:
        print(f"  {score:<16} {_pct(p)}")
    slug = f"{data_mod.team_key(home)}_{data_mod.team_key(away)}"
    out = score_heatmap(pred.matrix, home, away, Path(args.out_dir) / f"heatmap_{slug}.png",
                        subtitle=f"λ={pred.lam:.2f}, μ={pred.mu:.2f}, ρ={model.rho:.3f}")
    print(f"\nHeatmap: {out}\n{DISCLAIMER}")


def _league_values(args):
    """Squad values mapped to this league's team names, or None if the data are missing."""
    comp = data_mod.TRANSFERMARKT_COMPETITIONS.get(args.league)
    if comp is None:
        return None
    try:
        from .values import SquadValues, league_name_map
        sv = SquadValues.load(args.data_dir)
        sv.name_to_id = league_name_map(args.data_dir, comp)
        return sv
    except FileNotFoundError:
        return None


def cmd_evaluate(args) -> None:
    from .plots import calibration_plot

    df = data_mod.load_league(args.league, args.data_dir)
    xi = None if args.xi == "auto" else float(args.xi)
    res = evaluate(df, test_season=args.test_season, xi=xi, verbose=True,
                   values=_league_values(args))
    train_s, val_s, test_s = res.seasons
    print(f"\nSplit: history {train_s[0]}..{train_s[-1]} | validation (xi) {val_s} | test {test_s}")
    print(f"Chosen xi = {res.xi}  (half-life ≈ {0.693 / res.xi:.0f} days)" if res.xi else
          "Chosen xi = 0 (no time decay)")
    print("Squad-value prior: " + (f"used (l2={res.l2:g}, better on validation)" if res.use_values
                                   else "not used (not better on validation or no data)"))
    print(f"Chosen goal weight α = {res.goal_weight:g}"
          + ("  (fit on goals and shots-on-target expected goals)" if res.goal_weight < 1 else "  (goals only)"))
    print("\nTest-season results (walk-forward, weekly refit, lower is better except accuracy):")
    print(res.metrics.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    pr = res.predictions
    diff, se = paired_logloss_diff(pr[["dc_H", "dc_D", "dc_A"]].to_numpy(),
                                   pr[["po_H", "po_D", "po_A"]].to_numpy(), pr["y"].to_numpy())
    print(f"\nDixon-Coles − Poisson log-loss: {diff:+.4f} (95% CI {diff - 1.96 * se:+.4f} "
          f"to {diff + 1.96 * se:+.4f}, paired over matches)")
    mk = pr[["mk_H", "mk_D", "mk_A"]].to_numpy()
    has = ~pd.isna(mk).any(axis=1)
    if has.any():
        dcp = pr[["dc_H", "dc_D", "dc_A"]].to_numpy()[has]
        yy = pr["y"].to_numpy()[has]
        d2, se2 = paired_logloss_diff(dcp, mk[has], yy)
        print(f"Dixon-Coles − bookmaker log-loss: {d2:+.4f} (95% CI {d2 - 1.96 * se2:+.4f} "
              f"to {d2 + 1.96 * se2:+.4f}; positive = market better), n={has.sum()}")
        if res.blend_weight is not None:
            from .evaluate import blend
            w = res.blend_weight
            d3, se3 = paired_logloss_diff(blend(dcp, mk[has], w), mk[has], yy)
            print(f"Blend (w={w:.2f}, chosen on {val_s}) − bookmaker log-loss: {d3:+.4f} "
                  f"(95% CI {d3 - 1.96 * se3:+.4f} to {d3 + 1.96 * se3:+.4f})")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res.metrics.to_csv(out / f"eval_{args.league}.csv", index=False)
    (out / f"eval_{args.league}.md").write_text(
        f"League {args.league}, test season {test_s}, validation {val_s}, xi={res.xi}, "
        f"goal weight α={res.goal_weight:g}\n\n"
        + _markdown_table(res.metrics) + "\n")
    if not res.xi_table.empty:
        res.xi_table.to_csv(out / f"xi_selection_{args.league}.csv", index=False)
    res.predictions.to_csv(out / f"predictions_{args.league}_{test_s}.csv", index=False)
    cal = calibration_table(res.predictions[["dc_H", "dc_D", "dc_A"]].to_numpy(),
                            res.predictions["y"].to_numpy())
    cal.to_csv(out / f"calibration_{args.league}.csv", index=False)
    png = calibration_plot(cal, out / f"calibration_{args.league}.png",
                           title=f"Dixon-Coles calibration — {args.league} {test_s} (walk-forward)")
    print(f"\nSaved metrics, predictions and calibration plot ({png}) to {out}/")


def _markdown_table(df: pd.DataFrame) -> str:
    """Minimal markdown table (avoids the optional `tabulate` dependency)."""
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)  # noqa: E731
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def cmd_simulate(args) -> None:
    from .plots import positions_heatmap
    from .simulate import simulate_season

    df = data_mod.load_league(args.league, args.data_dir)
    season = args.season or df["Season"].iloc[-1]
    if args.as_of:
        cut = pd.Timestamp(args.as_of)
    else:
        cut = df["Date"].max() + pd.Timedelta(days=1)
    history = df[df["Date"] < cut]
    season_all = df[df["Season"] == season]
    played = history[history["Season"] == season]
    if season_all.empty:
        raise SystemExit(f"No data for season {season}")
    league = data_mod.LEAGUES.get(args.league)
    europe = args.europe or (league.europe_spots if league else 4)
    releg = args.relegation or (league.relegation_spots if league else 3)

    saved = _model_path(args)
    xi = DixonColes.load(saved).xi if saved.exists() else 0.0019
    model = DixonColes(xi=xi).fit(history, ref_date=cut)
    # all teams of the season, including any without a match before the cut-off
    teams = sorted(set(season_all["HomeTeam"]) | set(season_all["AwayTeam"]))
    sim = simulate_season(model, played, n=args.n, europe_spots=europe,
                          relegation_spots=releg, seed=args.seed, teams=teams)

    print(f"\n{args.league} {season}: {len(played)} matches played up to "
          f"{played['Date'].max():%Y-%m-%d}, {sim.n_remaining} remaining; "
          f"{args.n} simulations (seed {args.seed}), xi={xi}")
    s = sim.summary.copy()
    s.insert(0, "pos", range(1, len(s) + 1))
    for c in ("p_champion", "p_europe", "p_relegation"):
        s[c] = s[c].map(lambda v: f"{100 * v:5.1f}%")
    s["exp_points"] = s["exp_points"].map(lambda v: f"{v:5.1f}")
    s = s.rename(columns={"p_europe": f"p_top{europe}",
                          "p_relegation": f"p_bottom{releg}"})
    print(s.to_string())
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sim.summary.to_csv(out / f"simulation_{args.league}_{season}.csv")
    png = positions_heatmap(sim.positions, out / f"positions_{args.league}_{season}.png",
                            title=f"{args.league} {season} — final position probabilities (%)")
    print(f"\nSaved table and {png}\n{DISCLAIMER}")


def _world(args):
    from .multileague import BEST_PARAMS, CrossLeagueModel, load_world
    try:
        world = load_world(args.data_dir)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))
    ref = pd.Timestamp(args.as_of) if getattr(args, "as_of", None) else world["Date"].max() + pd.Timedelta(days=1)
    model = CrossLeagueModel(**BEST_PARAMS).fit(world, ref_date=ref)
    return world, model


def _resolve_world_team(model, query: str) -> str:
    if query in model.teams:
        return query
    hits = model.find_team(query)
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise SystemExit(f"Unknown team {query!r}.")
    opts = ", ".join(f"{model.names[h]} [{model.team_league[h]}, id {h}]" for h in hits[:10])
    raise SystemExit(f"{query!r} is ambiguous: {opts}. Use a more specific name or the id.")


def cmd_world_predict(args) -> None:
    from .model import btts_prob, outcome_probs, over_prob, top_scores
    from .multileague import UEFA
    from .plots import score_heatmap

    world, model = _world(args)
    h, a = _resolve_world_team(model, args.home), _resolve_world_team(model, args.away)
    venue = {"uefa": UEFA, "neutral": "neutral", "domestic": model.team_league[h]}[args.venue]
    lam, mu, m = model.predict_matrix(h, a, venue)
    p = outcome_probs(m)
    hn, an = model.names[h], model.names[a]
    print(f"\n{hn} [{model.team_league[h]}] vs {an} [{model.team_league[a]}]   venue={args.venue}, "
          f"data to {(model.ref_date - pd.Timedelta(days=1)):%Y-%m-%d}")
    print(f"Expected goals     {lam:.2f} - {mu:.2f}")
    print("-" * 44)
    print(f"Home win (1)        {_pct(p[0])}\nDraw (X)            {_pct(p[1])}\nAway win (2)        {_pct(p[2])}")
    print("-" * 44)
    print(f"Over 2.5 goals      {_pct(over_prob(m))}\nBoth teams score    {_pct(btts_prob(m))}")
    print("Most likely scores")
    for score, q in top_scores(m):
        print(f"  {score:<16} {_pct(q)}")
    for t in (h, a):
        n = model.n_matches_of(world, t)
        if n < 40:
            print(f"! {model.names[t]}: only {n} matches in the data -> very uncertain rating")
    slug = f"{data_mod.team_key(hn)}_{data_mod.team_key(an)}"
    out = score_heatmap(m, hn, an, Path(args.out_dir) / f"heatmap_world_{slug}.png",
                        subtitle=f"cross-league model · λ={lam:.2f}, μ={mu:.2f}, ρ={model.rho:.3f}")
    print(f"\nHeatmap: {out}\n{DISCLAIMER}")


def cmd_world_rankings(args) -> None:
    from .plots import league_strength_plot

    world, model = _world(args)
    lt = model.league_table(world)
    lt.index += 1
    print("\nLeague strength (A − D; 0 = average; leagues with 0 UEFA matches are not connected):")
    print(lt.to_string(float_format=lambda v: f"{v:+.3f}"))
    tt = model.team_table()
    if args.league:
        tt = tt[tt["league"] == args.league]
    tt = tt.head(args.top).reset_index(drop=True)
    tt.index += 1
    print(f"\nTop {len(tt)} teams on one scale:")
    print(tt.drop(columns="id").to_string(float_format=lambda v: f"{v:+.3f}"))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.team_table().to_csv(out / "world_team_ratings.csv", index=False)
    lt.to_csv(out / "world_league_ratings.csv")
    png = league_strength_plot(lt, out / "world_league_strength.png")
    print(f"\nSaved ratings and {png}")


def cmd_world_evaluate(args) -> None:
    from .evaluate import frequency_probs
    from .multileague import BEST_PARAMS, load_world, walk_forward_uefa

    world = load_world(args.data_dir)
    start = pd.Timestamp(f"20{args.season[:2]}-07-01")
    test = world["is_uefa"] & (world["Date"] >= start) & (world["Date"] < start + pd.DateOffset(years=1))
    print(f"Walk-forward over {test.sum()} UEFA club matches of {args.season} (weekly refits) ...")
    h = walk_forward_uefa(world, test, **BEST_PARAMS)
    po = walk_forward_uefa(world, test, xi=BEST_PARAMS["xi"], l2_team=2.0, hierarchical=False)
    y = h["y"].to_numpy()
    ph, pp = h[["pH", "pD", "pA"]].to_numpy(), po[["pH", "pD", "pA"]].to_numpy()
    past = world[world["is_uefa"] & (world["Date"] < start)]
    rows = [{"model": "Class frequencies (past UEFA)", **all_metrics(frequency_probs(past, len(y)), y)},
            {"model": "Pooled (no league levels)", **all_metrics(pp, y)},
            {"model": "Cross-league hierarchical", **all_metrics(ph, y)}]
    res = pd.DataFrame(rows)
    print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    d, se = paired_logloss_diff(ph, pp, y)
    print(f"\nHierarchical − pooled log-loss: {d:+.4f} (95% CI {d - 1.96 * se:+.4f} to {d + 1.96 * se:+.4f})")


def score_prediction(record: dict, home_goals: int, away_goals: int) -> pd.DataFrame:
    """Score every model block of a recorded prediction against the final result."""
    import numpy as np
    y = np.array([0 if home_goals > away_goals else 1 if home_goals == away_goals else 2])
    rows = []
    for name, block in record.items():
        if not isinstance(block, dict) or not any(k.endswith("_win") for k in block):
            continue
        wins = [k for k in block if k.endswith("_win")]
        p_home, p_away = block[wins[0]], block[wins[1]]
        p = np.array([[p_home, block["p_draw"], p_away]])
        score = f"{home_goals}-{away_goals}"
        top = dict(block.get("top_scores", []))
        over = home_goals + away_goals > 2.5
        btts = home_goals > 0 and away_goals > 0
        rows.append({
            "model": name, "p_actual_outcome": float(p[0, y[0]]),
            "log_loss": log_loss_one(p, y), "rps": float(rps(p, y)), "brier": float(brier(p, y)),
            "most_likely_outcome_correct": bool(p.argmax() == y[0]),
            "exact_score_in_top5": score in top, "p_exact_score": top.get(score),
            "brier_over_2_5": (block["over_2_5"] - over) ** 2, "brier_btts": (block["btts"] - btts) ** 2,
        })
    return pd.DataFrame(rows)


def log_loss_one(p, y) -> float:
    from .evaluate import log_loss
    return float(log_loss(p, y))


def cmd_score_prediction(args) -> None:
    import hashlib
    import json

    path = Path(args.file)
    raw = path.read_bytes()
    print(f"{path.name}  sha256={hashlib.sha256(raw).hexdigest()}")
    record = json.loads(raw)
    hg, ag = (int(v) for v in args.result.split("-"))
    print(f"{record.get('match', '')}\nwritten at {record.get('prediction_written_at')}, "
          f"data to {record.get('data_cutoff')}; result {hg}-{ag}\n")
    table = score_prediction(record, hg, ag)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nReference (2025-26 UEFA test, cross-league model): mean log-loss 0.9474; "
          "uniform 1/3 guess = 1.0986. One match is noise; judge the model on many matches.")
    out = path.with_name(path.stem + "_scored.json")
    out.write_text(json.dumps({"prediction_file": path.name,
                               "prediction_sha256": hashlib.sha256(raw).hexdigest(),
                               "result": f"{hg}-{ag}", "scores": table.to_dict("records")},
                              indent=2, ensure_ascii=False))
    print(f"Saved {out}")


COMPETITION_ALIASES = {"T1": "TR1", "E0": "GB1", "SP1": "ES1", "D1": "L1", "I1": "IT1",
                       "CONFERENCE": "UCOL", "ECL": "UCOL", "UECL": "UCOL", "UEL": "EL", "UCL": "CL"}


def _known_team_names(data_dir: str) -> list[str]:
    """All team names the models know (Transfermarkt world + downloaded league files)."""
    names: set[str] = set()
    try:
        from .multileague import load_world
        w = load_world(data_dir)
        names |= set(w["HomeTeam"]) | set(w["AwayTeam"])
    except FileNotFoundError:
        pass
    for league in data_mod.LEAGUES:
        try:
            df = data_mod.load_league(league, data_dir)
            names |= set(df["HomeTeam"]) | set(df["AwayTeam"])
        except FileNotFoundError:
            continue
    return sorted(names)


def _resolve_known_name(query: str, names: list[str], allow_new: bool) -> str:
    from .multileague import WORLD_ALIASES
    k = data_mod.team_key(WORLD_ALIASES.get(data_mod.team_key(query), query))
    exact = [n for n in names if data_mod.team_key(n) == k]
    if len(exact) == 1:
        return exact[0]
    hits = exact or [n for n in names if k and k in data_mod.team_key(n)]
    if len(hits) == 1:
        return hits[0]
    if allow_new and not exact:
        return data_mod.normalize_team(query)
    hint = hits[:8] or difflib.get_close_matches(query, names, n=5, cutoff=0.5)
    raise SystemExit(f"Team {query!r} " + ("is ambiguous" if hits else "is unknown") + "."
                     + (f" Did you mean: {', '.join(hint)}?" if hint else "")
                     + ("" if hits else " For a club that is new to the data add --new-team."))


def cmd_add_result(args) -> None:
    hg, ag = (int(v) for v in args.score.split("-"))
    comp = COMPETITION_ALIASES.get(args.competition.upper(), args.competition.upper())
    names = _known_team_names(args.data_dir)
    home = _resolve_known_name(args.home, names, args.new_team)
    away = _resolve_known_name(args.away, names, args.new_team)
    try:
        status = data_mod.add_manual_result(args.data_dir, args.date, comp, home, away, hg, ag,
                                            replace=args.replace)
    except ValueError as exc:
        raise SystemExit(str(exc))
    print(f"{status}: {args.date} [{comp}] {home} {hg}-{ag} {away}  -> "
          f"{Path(args.data_dir) / data_mod.MANUAL_FILE}")
    print("Models use it on their next run (train / predict / world-* commands).")


def cmd_list_results(args) -> None:
    df = data_mod.load_manual(args.data_dir)
    if df.empty:
        print("No hand-entered results yet.")
        return
    print(df.assign(Date=df["Date"].dt.strftime("%Y-%m-%d")).to_string(index=False))


def cmd_build_app(args) -> None:
    from .app import write_app
    try:
        out = write_app(args.out, args.data_dir, args.models_dir, standalone=not args.fragment)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))
    print(f"Wrote {out} ({out.stat().st_size / 1024:.0f} KB). Open it in a browser.")


# ------------------------------------------------------------------------ main
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="football-predictor", description=(
        "Dixon-Coles score probabilities for football matches. " + DISCLAIMER))
    p.add_argument("--data-dir", default="data")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--out-dir", default="outputs")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("download", help="download season results (openfootball or football-data.co.uk)")
    d.add_argument("--source", choices=data_mod.SOURCES, default="openfootball",
                   help="openfootball: GitHub, results only (T1, E0); "
                        "football-data: football-data.co.uk, results + odds; "
                        "tff-github: TFF match lists on GitHub (T1 only); "
                        "transfermarkt: Kaggle Transfermarkt dataset (needs a Kaggle token)")
    d.add_argument("--force", action="store_true",
                   help="overwrite season files even if they hold more matches")
    d.add_argument("--league", default="T1", help=f"league code, e.g. {', '.join(data_mod.LEAGUES)}")
    d.add_argument("--seasons", default="2018-2025", help="year range, e.g. 2018-2025 = 2018/19..2024/25")
    d.set_defaults(func=cmd_download)

    t = sub.add_parser("train", help="fit the model on all downloaded data")
    t.add_argument("--league", default="T1")
    t.add_argument("--xi", default="auto", help="time-decay per day, or 'auto' (validation, also picks α)")
    t.add_argument("--goal-weight", default="1.0",
                   help="α when --xi is given: 1 = goals only, 0.5 = half goals, half shots on target")
    t.set_defaults(func=cmd_train)

    pr = sub.add_parser("predict", help="score probabilities for one match")
    pr.add_argument("home")
    pr.add_argument("away")
    pr.add_argument("--league", default="T1")
    pr.set_defaults(func=cmd_predict)

    e = sub.add_parser("evaluate", help="walk-forward evaluation against baselines")
    e.add_argument("--league", default="T1")
    e.add_argument("--test-season", default=None, help="e.g. 2425 (default: latest complete)")
    e.add_argument("--xi", default="auto")
    e.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("simulate-season", help="Monte Carlo of the remaining fixtures")
    s.add_argument("--league", default="T1")
    s.add_argument("--season", default=None, help="season code, default: latest")
    s.add_argument("--as-of", default=None, help="simulate from this date (YYYY-MM-DD)")
    s.add_argument("--n", type=int, default=10_000)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--europe", type=int, default=None)
    s.add_argument("--relegation", type=int, default=None)
    s.set_defaults(func=cmd_simulate)

    wp = sub.add_parser("world-predict", help="cross-league prediction (any two clubs in the data)")
    wp.add_argument("home")
    wp.add_argument("away")
    wp.add_argument("--venue", choices=["uefa", "neutral", "domestic"], default="uefa")
    wp.add_argument("--as-of", default=None, help="use only matches before this date")
    wp.set_defaults(func=cmd_world_predict)

    wr = sub.add_parser("world-rankings", help="league and team strength on one scale")
    wr.add_argument("--top", type=int, default=30)
    wr.add_argument("--league", default=None, help="Transfermarkt league code, e.g. TR1")
    wr.add_argument("--as-of", default=None)
    wr.set_defaults(func=cmd_world_rankings)

    we = sub.add_parser("world-evaluate", help="walk-forward test on one season of UEFA matches")
    we.add_argument("--season", default="2526")
    we.set_defaults(func=cmd_world_evaluate)

    ba = sub.add_parser("build-app", help="write the interactive web page with the trained models")
    ba.add_argument("--out", default="docs/app/index.html")
    ba.add_argument("--fragment", action="store_true",
                    help="omit <html>/<head>/<body> (for hosts that add their own skeleton)")
    ba.set_defaults(func=cmd_build_app)

    sp = sub.add_parser("score-prediction", help="score a recorded prediction once the result is known")
    sp.add_argument("file", help="JSON written before the match (e.g. outputs/prediction_*.json)")
    sp.add_argument("--result", required=True, help="final score home-away, e.g. 1-2")
    sp.set_defaults(func=cmd_score_prediction)

    ar = sub.add_parser("add-result", help="enter a finished match by hand (kept in data/manual_results.csv)")
    ar.add_argument("home")
    ar.add_argument("away")
    ar.add_argument("score", help="final score home-away, e.g. 2-1")
    ar.add_argument("--date", default=pd.Timestamp.now().strftime("%Y-%m-%d"),
                    help="match date YYYY-MM-DD (default: today)")
    ar.add_argument("--competition", required=True,
                    help="league or cup code: TR1/T1, GB1/E0, ES1, L1, IT1 … or CL, EL, UCOL (Conference)")
    ar.add_argument("--new-team", action="store_true", help="allow a club that is not in the data yet")
    ar.add_argument("--replace", action="store_true", help="overwrite a stored score for the same match")
    ar.set_defaults(func=cmd_add_result)

    lr = sub.add_parser("list-results", help="show hand-entered results")
    lr.set_defaults(func=cmd_list_results)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
