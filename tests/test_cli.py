from pathlib import Path

import numpy as np

from football_predictor.cli import main, resolve_team
from conftest import simulate_league
import pytest


def test_resolve_team():
    teams = ["Galatasaray", "Fenerbahce", "Besiktas", "Buyuksehyr"]
    assert resolve_team("galatasaray", teams) == "Galatasaray"
    assert resolve_team("Beşiktaş", teams) == "Besiktas"
    assert resolve_team("Fener", teams) == "Fenerbahce"
    with pytest.raises(SystemExit):
        resolve_team("Trabzonspor", teams)


def test_cli_end_to_end(tmp_path, capsys):
    df = simulate_league(n_teams=6, n_seasons=4, seed=2)
    d = tmp_path / "data" / "XX"
    d.mkdir(parents=True)
    for season, g in df.groupby("Season"):
        g = g.assign(Date=g["Date"].dt.strftime("%d/%m/%Y"))
        g[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]].to_csv(d / f"{season}.csv", index=False)
    common = ["--data-dir", str(tmp_path / "data"), "--models-dir", str(tmp_path / "models"),
              "--out-dir", str(tmp_path / "out")]
    main(common + ["train", "--league", "XX", "--xi", "0.002"])
    main(common + ["predict", "team00", "Team05", "--league", "XX"])
    assert "Most likely scores" in capsys.readouterr().out
    main(common + ["evaluate", "--league", "XX", "--xi", "0.002"])
    main(common + ["simulate-season", "--league", "XX", "--n", "200", "--as-of", "2023-10-01"])
    out = tmp_path / "out"
    assert (out / "heatmap_team00_team05.png").exists()
    assert (out / "calibration_XX.png").exists()
    assert (out / "eval_XX.csv").exists()
    assert list(out.glob("positions_XX_*.png"))


def test_world_cli_requires_transfermarkt(tmp_path):
    with pytest.raises(SystemExit):
        main(["--data-dir", str(tmp_path), "world-rankings"])


def test_score_prediction():
    from football_predictor.cli import score_prediction
    rec = {"match": "A vs B", "m": {"p_A_win": 0.2, "p_draw": 0.3, "p_B_win": 0.5,
                                   "over_2_5": 0.4, "btts": 0.5, "top_scores": [["0-1", 0.12]]}}
    t = score_prediction(rec, 0, 1).iloc[0]
    assert t["p_actual_outcome"] == pytest.approx(0.5)
    assert t["log_loss"] == pytest.approx(-np.log(0.5))
    assert t["most_likely_outcome_correct"] and t["exact_score_in_top5"]
    assert t["brier_over_2_5"] == pytest.approx(0.16)
    draw = score_prediction(rec, 1, 1).iloc[0]
    assert draw["p_actual_outcome"] == pytest.approx(0.3) and not draw["most_likely_outcome_correct"]
