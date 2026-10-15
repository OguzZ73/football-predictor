import pandas as pd
import pytest

from football_predictor.data import (
    _read_csv_bytes, clean_matches, outcome, season_codes, season_url, team_key,
)

RAW = b"""Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365H,B365D,B365A,PSCH,PSCD,PSCA
T1,09/08/2024,19:00,Galatasaray ,Hatayspor,2,1,H,1.2,6.5,12,1.22,6.8,13
T1,10/08/24,19:00,Fenerbahce,Adana  Demirspor,1,1,D,1.3,5,9,,,
T1,11/08/2024,19:00,Besiktas,Samsunspor,x,0,,1.5,4,6,1.5,4.1,6.2
T1,,19:00,Konyaspor,Rizespor,0,0,D,2,3,4,2,3,4
T1,12/08/2024,19:00,Kasimpasa,Kasimpasa,1,0,H,2,3,4,2,3,4
T1,13/08/2024,19:00,Sivasspor,Goztep,0,3,A,2.5,3.2,2.9,2.6,3.3,2.8
,,,,,,,,,,,,,
"""


def test_season_codes():
    assert season_codes("2018-2025") == ["1819", "1920", "2021", "2122", "2223", "2324", "2425"]
    assert season_codes("2024") == ["2425"]
    assert season_codes("1999-2001") == ["9900", "0001"]
    with pytest.raises(ValueError):
        season_codes("2025-2018")


def test_season_url():
    assert season_url("T1", "2425", "football-data") == \
        "https://www.football-data.co.uk/mmz4281/2425/T1.csv"
    assert season_url("T1", "2425").endswith("openfootball/europe/master/turkey/2024-25_tr1.txt")
    assert season_url("E0", "1819").endswith("openfootball/england/master/2018-19/1-premierleague.txt")
    with pytest.raises(ValueError):
        season_url("SP1", "2425")


def test_clean_matches_drops_broken_rows_and_normalizes():
    df = clean_matches(_read_csv_bytes(RAW), season="2425")
    # bad goals, missing date, self-match and empty row are removed
    assert list(df["HomeTeam"]) == ["Galatasaray", "Fenerbahce", "Sivasspor"]
    assert "Adana Demirspor" in set(df["AwayTeam"])  # whitespace collapsed
    assert df["Date"].iloc[1] == pd.Timestamp("2024-08-10")  # 2-digit year parsed
    assert df["FTHG"].dtype.kind == "i"
    assert (df["Season"] == "2425").all()
    # closing odds preferred; falls back to opening odds when closing are missing
    assert df["OddsH"].iloc[0] == pytest.approx(1.22)
    assert df["OddsH"].iloc[1] == pytest.approx(1.3)


def test_clean_matches_requires_columns():
    with pytest.raises(ValueError):
        clean_matches(pd.DataFrame({"Date": [], "HomeTeam": []}))


def test_normalize_team():
    from football_predictor.data import normalize_team
    assert normalize_team("Arsenal FC") == normalize_team("Arsenal") == "Arsenal"
    assert normalize_team("AFC Bournemouth") == "Bournemouth"
    assert normalize_team("Hull City AFC") == "Hull City"
    assert normalize_team("Atiker  Konyaspor") == "Konyaspor"
    assert normalize_team("Kasimpasa") == normalize_team("Kasımpaşa SK") == "Kasımpaşa"


def test_outcome_and_team_key():
    df = pd.DataFrame({"FTHG": [2, 1, 0], "FTAG": [0, 1, 3]})
    assert outcome(df).tolist() == [0, 1, 2]
    assert team_key("Beşiktaş") == team_key("besiktas") == "besiktas"
    assert team_key("İstanbul Başakşehir") == "istanbulbasaksehir"


TXT_A = """= Turkish Süper Lig 2024/25
# Teams 4

▪ Matchday 1
  Fri Aug 9 2024
    21:00  Galatasaray             v Hatayspor                2-1 (0-0)
  Sat Aug 10
           Fenerbahçe              v Adana Demirspor          1-0
  Sun Jan 12
    19:00  Göztepe                 v Beşiktaş
"""

TXT_B = """= English Premier League 2018/19
▪ Matchday 1
Fri Aug 10
  20:00  Manchester United        2-1 (1-0)  Leicester City
                  (Hugo EKITIKE 37', Cody GAKPO 49';
                   Antoine SEMENYO 64', 76')
Sat May 11 2019
         Fulham FC                0-2  Crystal Palace
Wed Jul 1
         Burnley FC               1-1  Everton FC
"""


def test_parse_football_txt_v_layout():
    from football_predictor.data import parse_football_txt
    df = parse_football_txt(TXT_A, "2425")
    assert df.values.tolist() == [
        [pd.Timestamp("2024-08-09"), "Galatasaray", "Hatayspor", 2, 1],
        [pd.Timestamp("2024-08-10"), "Fenerbahçe", "Adana Demirspor", 1, 0],
    ]  # unplayed fixture skipped


def test_parse_football_txt_score_between_layout():
    from football_predictor.data import parse_football_txt
    df = parse_football_txt(TXT_B, "1819")
    assert df.values.tolist() == [
        [pd.Timestamp("2018-08-10"), "Manchester United", "Leicester City", 2, 1],
        [pd.Timestamp("2019-05-11"), "Fulham FC", "Crystal Palace", 0, 2],
        [pd.Timestamp("2019-07-01"), "Burnley FC", "Everton FC", 1, 1],  # year carried forward
    ]


TFF = """Mac_kodu,TFF_mac_linki,Ev_sahibi,Ev_sahibi_gol,Misafir_takim,Misafir_takim_gol,Stat,Ana_hakem,Tarih_saat
1,x,GALATASARAY A.Ş.,2,ARABAM.COM KONYASPOR,1,Rams Park,A,05.08.2022 21:00
2,x,MEDİPOL BAŞAKŞEHİR FK,0,KASIMPAŞA A.Ş.,0,Fatih Terim,B,13.8.2021 - 20:30
3,x,GAZİANTEP FUTBOL KULÜBÜ A.Ş.,0,TRABZONSPOR A.Ş.,3,Hükmen,-,19.03.2023 00:00
""".encode()


def test_parse_tff_csv_drops_forfeits_and_maps_names():
    from football_predictor.data import parse_tff_csv
    df, n_forfeit = parse_tff_csv(TFF)
    assert n_forfeit == 1
    assert df.values.tolist() == [
        [pd.Timestamp("2021-08-13"), "İstanbul Başakşehir", "Kasımpaşa", 0, 0],
        [pd.Timestamp("2022-08-05"), "Galatasaray", "Konyaspor", 2, 1],
    ]
    with pytest.raises(ValueError):
        parse_tff_csv(TFF.replace("TRABZONSPOR A.Ş.,3,Hükmen".encode(), "NEW FC,3,Stad".encode()))


def _tm_games():
    cols = ["competition_id", "season", "date", "home_club_name", "away_club_name",
            "home_club_goals", "away_club_goals", "referee", "home_club_manager_name"]
    rows = [
        ("TR1", 2022, "2022-08-05", "Fenerbahce", "Basaksehir FK", 2, 1, "Ref A", "Coach"),
        ("TR1", 2022, "2023-03-04", "Gaziantep FK", "Galatasaray", 0, 3, None, None),  # forfeit
        ("TR1", 2022, "2023-03-05", "Konyaspor", "Kasimpasa", 3, 0, "Ref B", "Coach"),  # real 3-0
        ("TR1", 2023, "2023-08-11", "Galatasaray", "Fenerbahce", 1, 1, "Ref C", "Coach"),
        ("TR1", 2024, "2025-02-09", "Galatasaray", "Adana Demirspor", 3, 0, "Ref E", "Coach"),
        ("GB1", 2022, "2022-08-05", "Arsenal FC", "AFC Bournemouth", 3, 0, "Ref D", "Coach"),
    ]
    return pd.DataFrame(rows, columns=cols)


def test_parse_transfermarkt_games():
    from football_predictor.data import parse_transfermarkt_games
    df, n_forfeit = parse_transfermarkt_games(_tm_games(), "T1", "2223")
    assert n_forfeit == 1
    assert df[["HomeTeam", "AwayTeam", "FTHG", "FTAG"]].values.tolist() == [
        ["Fenerbahçe", "İstanbul Başakşehir", 2, 1],
        ["Konyaspor", "Kasımpaşa", 3, 0],
    ]
    awarded, n = parse_transfermarkt_games(_tm_games(), "T1", "2425")
    assert awarded.empty and n == 1  # walk-off awarded 3-0 listed in TRANSFERMARKT_AWARDED
    missing, n = parse_transfermarkt_games(_tm_games(), "T1", "2627")
    assert missing.empty and n == 0 and "Date" in missing.columns
    e0, _ = parse_transfermarkt_games(_tm_games(), "E0", "2223")
    assert e0[["HomeTeam", "AwayTeam"]].values.tolist() == [["Arsenal", "Bournemouth"]]


def test_download_does_not_overwrite_bigger_file(tmp_path):
    from football_predictor.data import _save_season
    big = pd.DataFrame({"Date": pd.to_datetime(["2025-08-01", "2025-08-02"]),
                        "HomeTeam": ["A", "B"], "AwayTeam": ["B", "A"], "FTHG": [1, 0], "FTAG": [0, 0]})
    path = tmp_path / "2526.csv"
    assert _save_season(path, big, force=False)
    assert not _save_season(path, big.iloc[:1], force=False)  # smaller download is ignored
    assert len(pd.read_csv(path)) == 2
    assert _save_season(path, big.iloc[:1], force=True)
    assert len(pd.read_csv(path)) == 1


def test_manual_results_roundtrip_and_league_merge(tmp_path):
    from football_predictor.data import add_manual_result, load_league, load_manual, season_of
    assert season_of(pd.Timestamp("2026-10-15")) == "2627"
    assert season_of(pd.Timestamp("2027-03-01")) == "2627"
    d = tmp_path / "T1"
    d.mkdir()
    (d / "2627.csv").write_text("Date,HomeTeam,AwayTeam,FTHG,FTAG\n14/08/2026,Galatasaray,Konyaspor,2,0\n")
    assert add_manual_result(tmp_path, "2026-10-18", "TR1", "Trabzonspor", "Fenerbahçe", 1, 1) == "added"
    assert add_manual_result(tmp_path, "2026-10-18", "TR1", "Trabzonspor", "Fenerbahçe", 1, 1) == "exists"
    with pytest.raises(ValueError):
        add_manual_result(tmp_path, "2026-10-18", "TR1", "Trabzonspor", "Fenerbahçe", 2, 1)
    assert add_manual_result(tmp_path, "2026-10-18", "TR1", "Trabzonspor", "Fenerbahçe", 2, 1,
                             replace=True) == "updated"
    # a fixture already in the downloaded file is not duplicated
    add_manual_result(tmp_path, "2026-08-14", "TR1", "Galatasaray", "Konyaspor", 2, 0)
    add_manual_result(tmp_path, "2026-10-15", "UCOL", "KuPS", "Trabzonspor", 0, 1)
    assert len(load_manual(tmp_path)) == 3
    df = load_league("T1", tmp_path)
    assert len(df) == 2  # downloaded match + one new league result; cup match ignored
    row = df[df["HomeTeam"] == "Trabzonspor"].iloc[0]
    assert (row["FTHG"], row["FTAG"], row["Season"]) == (2, 1, "2627")


def test_attach_odds_maps_names_and_dates(tmp_path):
    from football_predictor.data import attach_odds
    ours = pd.DataFrame({
        "Date": pd.to_datetime(["2025-08-01", "2025-08-08", "2025-08-15", "2025-08-22"]),
        "Season": "2526", "HomeTeam": ["Fenerbahçe", "Beşiktaş", "Fenerbahçe", "Beşiktaş"],
        "AwayTeam": ["Beşiktaş", "Fenerbahçe", "Beşiktaş", "Fenerbahçe"],
        "FTHG": [2, 1, 0, 3], "FTAG": [1, 1, 0, 2]})
    odds = pd.DataFrame({
        "Division": "T1", "MatchDate": ["2025-08-01", "2025-08-08", "2025-08-14", "2025-08-22"],
        "HomeTeam": ["Fenerbahce", "Besiktas", "Fenerbahce", "Besiktas"],
        "AwayTeam": ["Besiktas", "Fenerbahce", "Besiktas", "Fenerbahce"],
        "FTHome": [2, 1, 0, 3], "FTAway": [1, 1, 0, 2],
        "OddHome": [2.0, 2.5, 2.1, 2.4], "OddDraw": [3.4, 3.3, 3.3, 3.2], "OddAway": [3.6, 2.8, 3.5, 2.9]})
    (tmp_path / "_odds").mkdir()
    odds.to_csv(tmp_path / "_odds" / "Matches.csv", index=False)
    out = attach_odds(ours, "T1", tmp_path)
    assert out["OddsH"].tolist() == [2.0, 2.5, 2.1, 2.4]  # third row matched one day apart
    assert "OddsH" not in attach_odds(ours, "E0", tmp_path)  # no E0 rows in the odds file


def test_blend_is_convex_combination():
    import numpy as np
    from football_predictor.evaluate import blend
    a, b = np.array([[0.5, 0.3, 0.2]]), np.array([[0.2, 0.3, 0.5]])
    np.testing.assert_allclose(blend(a, b, 0.25), [[0.275, 0.3, 0.425]])
    np.testing.assert_allclose(blend(a, b, 1.0), a)
