import pandas as pd
import pytest

from football_predictor import wikipedia as wp

TR_MATCHES = """{{Şablon:x}}
|ÇFK-ÇYR = 
{{Kapanabilir futbol maçı kutusu
|tarih             = {{Başlangıç tarihi|2026|10|18}}
|takım1            = [[Çorum FK|Çorum]]
|sonuç             = 
|takım2            = [[Çaykur Rizespor]]
}}
|GAL-FNB = 
{{Kapanabilir futbol maçı kutusu
|tarih             = {{Başlangıç tarihi|2026|10|25}}
|takım1            = [[Galatasaray (futbol takımı)|Galatasaray]]
|sonuç             = 
|takım2            = [[Fenerbahçe (futbol takımı)|Fenerbahçe]]
}}
|KON-GEN = 
{{Kapanabilir futbol maçı kutusu
|tarih             = {{Başlangıç tarihi|2026|8|14}}
|takım1            = [[Konyaspor]]
|sonuç             = 1-1
|takım2            = [[Gençlerbirliği]]
}}
|AME-KON = 
{{Kapanabilir futbol maçı kutusu
|tarih             = {{Başlangıç tarihi|2026|11|1}}
|takım1            = [[Amed SK|Amed]]
|sonuç             = 
|takım2            = [[Konyaspor]]
}}"""
TR_STANDINGS = "|match_ÇFK_ÇYR=2-1 |match_GAL_FNB=[[Fenerbahçe-Galatasaray derbisi|2-1]] |match_KON_GEN=1-1"


def test_parse_tr_league_merges_templates():
    df = wp.parse_tr_league(TR_MATCHES, TR_STANDINGS)
    assert df[["HomeTeam", "AwayTeam", "FTHG", "FTAG"]].values.tolist() == [
        ["Konyaspor", "Gençlerbirliği", 1, 1],
        ["Çorum", "Çaykur Rizespor", 2, 1],
        ["Galatasaray", "Fenerbahçe", 2, 1],
    ]  # the unplayed AME-KON fixture is dropped
    assert df["Date"].iloc[2] == pd.Timestamp("2026-10-25")
    with pytest.raises(ValueError):  # the two templates disagree
        wp.parse_tr_league(TR_MATCHES, "|match_KON_GEN=2-0")


def test_parse_results_matrix_handles_several_cells_per_line():
    page = """|name_ARS=[[Arsenal F.C.|Arsenal]]
|name_CHE=[[Chelsea F.C.|Chelsea]] |name_LIV=[[Liverpool F.C.|Liverpool]]
|match_ARS_CHE=[[Arsenal F.C.–Chelsea F.C. rivalry|2–1]] |match_ARS_LIV=
|match_CHE_ARS=0–0 |match_LIV_ARS=3–1<ref>x</ref>"""
    df = wp.parse_results_matrix(page)
    assert sorted(map(tuple, df.values.tolist())) == [
        ("Arsenal", "Chelsea", 2, 1), ("Chelsea", "Arsenal", 0, 0), ("Liverpool", "Arsenal", 3, 1)]


def test_fixture_dates_include_unplayed_fixtures():
    txt = """= English Premier League 2026/27
▪ Matchday 1
  Fri Aug 21 2026
    20:00  Arsenal FC              v Coventry City FC         3-0 (2-0)
  Sat Aug 22
    12:30  Hull City AFC           v Manchester United FC
"""
    df = wp.fixture_dates(txt, "2627")
    assert df.values.tolist() == [[pd.Timestamp("2026-08-21"), "Arsenal FC", "Coventry City FC"],
                                  [pd.Timestamp("2026-08-22"), "Hull City AFC", "Manchester United FC"]]


def test_parse_football_boxes():
    page = """{{#invoke:Football box|main
|date       = {{Start date|2026|10|15|df=y}}
|team1      = [[Kuopion Palloseura|KuPS]] {{fbaicon|FIN}}
|score      = [[#x|2–4]]
|team2      = {{fbaicon|TUR}} [[Trabzonspor]]
}}
{{#invoke:Football box|main
|date       = {{Start date|2026|11|5|df=y}}
|team1      = [[Lille OSC|Lille]]
|score      = v
|team2      = [[Galatasaray S.K.|Galatasaray]]
}}"""
    df = wp.parse_football_boxes(page)
    assert df.values.tolist() == [[pd.Timestamp("2026-10-15"), "KuPS", "Trabzonspor", 2, 4]]


def test_map_names_uses_whole_words_and_aliases():
    known = ["Qarabağ FK", "Aarhus GF", "Associazione Sportiva Roma", "AC Milan", "Inter Milan",
             "Kuopion Palloseura", "Fenerbahçe"]
    mp = wp.map_names(["AGF", "Roma", "Milan", "KuPS", "Fenerbahce", "Nobody"], known, wp.UEFA_ALIASES)
    assert mp["AGF"] == "Aarhus GF"  # alias; never "Qarabağ FK"
    assert mp["Roma"] == "Associazione Sportiva Roma"  # every word contained
    assert mp["Milan"] == "AC Milan"  # alias resolves the two Milans
    assert mp["KuPS"] == "Kuopion Palloseura"
    assert mp["Fenerbahce"] == "Fenerbahçe"
    assert mp["Nobody"] is None
    assert wp.map_names(["AGF"], known)["AGF"] is None  # without alias: no false match
