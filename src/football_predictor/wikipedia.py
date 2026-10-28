"""Current-season results from Wikipedia (fresher than the other free sources).

* Süper Lig (T1): Turkish Wikipedia, the match-box template of the season (dates for
  every fixture, scores for some) plus the standings template's ``match_X_Y`` scores.
* Premier League (E0): English Wikipedia results matrix (scores, no dates); dates are
  taken from the openfootball fixture list already on disk.
* UEFA club competitions: English Wikipedia league-phase pages (one match box per game).

Wikipedia is edited by volunteers. Downloads are cross-checked against the matches we
already have and every disagreement is reported; nothing is written silently.
"""

from __future__ import annotations

import re
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from .data import parse_football_txt, team_key

UA = {"User-Agent": "football-predictor/0.1 (educational; github.com/OguzZ73/football-predictor)"}
SCORE = r"(\d+)\s*[-–]\s*(\d+)"

PAGES = {
    "T1": ("tr", ["Şablon:{y1}-{y2s} Süper Lig maçları/maçlar", "Şablon:{y1}-{y2s} Süper Lig puan durumu"]),
    "E0": ("en", ["{y1}–{y2s} Premier League"]),
}
UEFA_PAGES = {"CL": "{y1}–{y2s} UEFA Champions League league phase",
              "EL": "{y1}–{y2s} UEFA Europa League league phase",
              "UCOL": "{y1}–{y2s} UEFA Conference League league phase"}


def fetch(lang: str, title: str, timeout: float = 30.0) -> str:
    """Raw wikitext of a page."""
    url = (f"https://{lang}.wikipedia.org/w/index.php?"
           + urllib.parse.urlencode({"title": title, "action": "raw"}))
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read().decode("utf-8")


def _titles(season: str, pattern: str) -> str:
    y1 = 2000 + int(season[:2])
    return pattern.format(y1=y1, y2s=f"{(y1 + 1) % 100:02d}")


def _unlink(s: str) -> str:
    """Plain text of wiki markup: [[A|B]] -> B, drop templates and refs."""
    s = re.sub(r"<ref[^>]*?/>|<ref[^>]*>.*?</ref>", "", s, flags=re.S)
    s = re.sub(r"\{\{[^{}]*\}\}", "", s)
    s = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", s)
    return re.sub(r"\s+", " ", s).strip()


# ----------------------------------------------------------------- Süper Lig
def parse_tr_league(match_template: str, standings_template: str = "") -> pd.DataFrame:
    """Played Süper Lig matches from the Turkish match-box and standings templates."""
    parts = re.split(r"\n\|([^\s|=]+-[^\s|=]+) *= *\n", "\n" + match_template)
    rows = {}
    for key, body in zip(parts[1::2], parts[2::2]):
        d = re.search(r"tarihi\|(\d+)\|(\d+)\|(\d+)", body)
        h = re.search(r"\|takım1 *= *(.*)", body)
        a = re.search(r"\|takım2 *= *(.*)", body)
        if not (d and h and a):
            continue
        sc = re.search(r"\|sonuç *= *" + SCORE, body)
        rows[key] = {"Date": pd.Timestamp(int(d[1]), int(d[2]), int(d[3])),
                     "HomeTeam": _unlink(h[1]), "AwayTeam": _unlink(a[1]),
                     "FTHG": int(sc[1]) if sc else None, "FTAG": int(sc[2]) if sc else None}
    for hc, ac, g1, g2 in re.findall(r"match_([^\s_=|]+)_([^\s_=|]+) *= *(?:\[\[[^|\]]*\|)?" + SCORE,
                                     standings_template):
        r = rows.get(f"{hc}-{ac}")
        if r is None:
            continue
        if r["FTHG"] is not None and (r["FTHG"], r["FTAG"]) != (int(g1), int(g2)):
            raise ValueError(f"Wikipedia templates disagree on {hc}-{ac}: "
                             f"{r['FTHG']}-{r['FTAG']} vs {g1}-{g2}")
        r["FTHG"], r["FTAG"] = int(g1), int(g2)
    df = pd.DataFrame(rows.values())
    if df.empty:
        return df
    df = df.dropna(subset=["FTHG"]).astype({"FTHG": int, "FTAG": int})
    return df.sort_values("Date", kind="stable").reset_index(drop=True)


# -------------------------------------------------------------- Premier League
def parse_results_matrix(page: str) -> pd.DataFrame:
    """(HomeTeam, AwayTeam, FTHG, FTAG) from an English 'sports results' matrix."""
    cell = r"((?:\[\[[^\]]*\]\]|\{\{[^}]*\}\}|[^\n|])*)"  # stops at the next '|' outside links
    names = {c: _unlink(n) for c, n in re.findall(r"\|\s*name_([A-Z]{3})\s*=\s*" + cell, page)}
    rows = []
    for hc, ac, value in re.findall(r"\|\s*match_([A-Z]{3})_([A-Z]{3})\s*=\s*" + cell, page):
        m = re.search(SCORE, _unlink(value))
        if m and hc in names and ac in names:
            rows.append({"HomeTeam": names[hc], "AwayTeam": names[ac], "FTHG": int(m[1]), "FTAG": int(m[2])})
    return pd.DataFrame(rows)


def fixture_dates(fixture_text: str, season: str) -> pd.DataFrame:
    """Scheduled date of every fixture in an openfootball file (played or not)."""
    lines = [ln.rstrip() + " 0-0" if " v " in ln and not re.search(r"\d+\s*-\s*\d+", ln) else ln
             for ln in fixture_text.splitlines()]
    df = parse_football_txt("\n".join(lines), season)
    return df[["Date", "HomeTeam", "AwayTeam"]]


# ------------------------------------------------------------------------ UEFA
def parse_football_boxes(page: str) -> pd.DataFrame:
    """Played matches from English {{Football box}} templates."""
    rows = []
    for box in re.split(r"\{\{#invoke:Football box\|main|\{\{Football box", page)[1:]:
        d = re.search(r"\|\s*date\s*=\s*\{\{Start date\|(\d+)\|(\d+)\|(\d+)", box)
        h = re.search(r"\|\s*team1\s*=\s*(.*)", box)
        a = re.search(r"\|\s*team2\s*=\s*(.*)", box)
        s = re.search(r"\|\s*score\s*=\s*(.*)", box)
        if not (d and h and a and s):
            continue
        m = re.search(SCORE, _unlink(s[1]))
        if not m:
            continue
        rows.append({"Date": pd.Timestamp(int(d[1]), int(d[2]), int(d[3])),
                     "HomeTeam": _unlink(h[1]), "AwayTeam": _unlink(a[1]),
                     "FTHG": int(m[1]), "FTAG": int(m[2])})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- helpers
# Wikipedia (English) club names -> names in the Transfermarkt data, where the
# word-based rule below cannot find them. Checked by hand.
UEFA_ALIASES = {
    "AEK Athens": "Athlitiki Enosi Konstantinoupoleos", "AGF": "Aarhus GF", "AZ": "AZ Alkmaar",
    "Atlético Madrid": "Atlético de Madrid", "Barcelona": "FC Barcelona",
    "Bayer Leverkusen": "Bayer 04 Leverkusen", "Braga": "SC Braga", "CSKA Sofia": "CSKA-Sofia",
    "Celta Vigo": "Celta de Vigo", "Gent": "KAA Gent", "Jagiellonia Białystok": "Jagiellonia Bialystok",
    "KuPS": "Kuopion Palloseura", "Lille": "LOSC Lille", "Milan": "AC Milan",
    "Mjällby AIF": "Mjällby Almänna Idrottsförening", "NEC": "NEC Nijmegen",
    "Nordsjælland": "FC Nordsjaelland", "OFI": "Omilos Filathlon Irakliou",
    "Olympiacos": "Olympiakos Syndesmos Filathlon Peiraios", "Porto": "FC Porto",
    "Rennes": "Stade Rennais", "TSG Hoffenheim": "TSG 1899 Hoffenheim", "Viking": "Viking Fotballklubb",
    "Ferencváros": "Ferencvárosi TC", "Sint-Truiden": "Sint-Truidense VV",
}


def _tokens(name: str) -> set[str]:
    import unicodedata
    table = str.maketrans({"ı": "i", "İ": "i", "ş": "s", "ğ": "g", "ø": "o", "æ": "ae", "ł": "l"})
    s = unicodedata.normalize("NFKD", name.lower().translate(table))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return {t for t in re.split(r"[^a-z0-9]+", s) if t}


def map_names(names, known: list[str], aliases: dict[str, str] | None = None) -> dict[str, str | None]:
    """Map Wikipedia team names onto our names.

    Order: explicit alias, exact accent/case-insensitive key, then a unique known
    name that contains every word of the Wikipedia name ("Roma" -> "Associazione
    Sportiva Roma"). Substring matching inside words is not used ("AGF" must not
    match "Qarabağ FK"). Unresolved names map to None.
    """
    aliases = aliases or {}
    keys = {team_key(k): k for k in known}
    toks = {k: _tokens(k) for k in known}
    out = {}
    for n in names:
        target = aliases.get(n, n)
        if team_key(target) in keys:
            out[n] = keys[team_key(target)]
            continue
        want = _tokens(target)
        hits = [k for k, t in toks.items() if want and want <= t]
        out[n] = hits[0] if len(hits) == 1 else None
    return out


def download_league(league: str, season: str, data_dir: Path | str = "data") -> pd.DataFrame:
    """Played matches of the current season for T1 or E0, with our team names."""
    from .data import clean_matches, _read_csv_bytes
    data_dir = Path(data_dir)
    if league not in PAGES:
        raise ValueError(f"wikipedia source supports {sorted(PAGES)}")
    lang, titles = PAGES[league]
    texts = [fetch(lang, _titles(season, t)) for t in titles]
    existing_path = data_dir / league / f"{season}.csv"
    known = []
    if existing_path.exists():
        ex = clean_matches(_read_csv_bytes(existing_path.read_bytes()), season=season)
        known = sorted(set(ex["HomeTeam"]) | set(ex["AwayTeam"]))
    if league == "T1":
        df = parse_tr_league(texts[0], texts[1])
    else:
        df = parse_results_matrix(texts[0])
        raw = data_dir / league / "raw" / f"{season}.txt"
        if not raw.exists():
            raise FileNotFoundError(f"{raw} missing: run the openfootball download first (dates)")
        dates = fixture_dates(raw.read_text(encoding="utf-8"), season)
        known = known or sorted(set(dates["HomeTeam"]) | set(dates["AwayTeam"]))
        mp = map_names(set(df["HomeTeam"]) | set(df["AwayTeam"]), sorted(set(dates["HomeTeam"])))
        df = df.assign(H=df["HomeTeam"].map(mp), A=df["AwayTeam"].map(mp))
        df = df.merge(dates, left_on=["H", "A"], right_on=["HomeTeam", "AwayTeam"],
                      suffixes=("", "_fx")).drop(columns=["H", "A", "HomeTeam_fx", "AwayTeam_fx"])
    if known:
        mp = map_names(set(df["HomeTeam"]) | set(df["AwayTeam"]), known)
        missing = sorted(k for k, v in mp.items() if v is None)
        if missing:
            raise ValueError(f"Wikipedia team names not matched to our data: {missing}")
        df = df.assign(HomeTeam=df["HomeTeam"].map(mp), AwayTeam=df["AwayTeam"].map(mp))
    return df.sort_values("Date", kind="stable").reset_index(drop=True)


def download_uefa(season: str, known: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """League-phase results of the three UEFA club competitions.

    With ``known`` (club names of the cross-league data) names are mapped onto them;
    clubs that cannot be mapped keep their Wikipedia name and are returned as the
    second value so the caller can report them (they enter the model as new clubs).
    """
    frames = []
    for comp, pattern in UEFA_PAGES.items():
        df = parse_football_boxes(fetch("en", _titles(season, pattern)))
        frames.append(df.assign(competition=comp))
    df = pd.concat(frames, ignore_index=True)
    new: list[str] = []
    if known:
        mp = map_names(set(df["HomeTeam"]) | set(df["AwayTeam"]), known, UEFA_ALIASES)
        new = sorted(k for k, v in mp.items() if v is None)
        df = df.assign(HomeTeam=df["HomeTeam"].map(lambda n: mp.get(n) or n),
                       AwayTeam=df["AwayTeam"].map(lambda n: mp.get(n) or n))
    return df, new
