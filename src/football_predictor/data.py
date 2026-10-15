"""Downloading, loading and cleaning match results.

Two sources are supported:

* ``openfootball`` (default) — public-domain Football.TXT files on GitHub
  (github.com/openfootball). Results only, no betting odds.
* ``football-data`` — football-data.co.uk CSVs (results + bookmaker odds).
* ``transfermarkt`` — Kaggle dataset "Football Data from Transfermarkt" (CC0);
  complete 2018-2025 history for T1/E0/SP1/D1/I1, needs a Kaggle API token.
* ``tff-github`` — T1 only: TFF match lists collected in a public GitHub repo
  (used to fill the 2021-22 and 2022-23 gap of openfootball).

Both are stored as one cleaned CSV per season: ``data/<league>/<season>.csv``.
"""

from __future__ import annotations

import io
import re
import shutil
import subprocess
import sys
import unicodedata
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
OPENFOOTBALL_URL = "https://raw.githubusercontent.com/openfootball/{path}"
# {s} is the season as "2024-25"
OPENFOOTBALL_PATHS = {
    "T1": "europe/master/turkey/{s}_tr1.txt",
    "E0": "england/master/{s}/1-premierleague.txt",
}
# TFF match lists collected in github.com/betul-bostan/super-lig-match-outcome-prediction
# (no license stated: fine for personal/educational use, do not redistribute).
TFF_GITHUB_URL = ("https://raw.githubusercontent.com/betul-bostan/super-lig-match-outcome-prediction/"
                  "main/{y1}_{y2}_sezon_maclari.csv")
SOURCES = ("openfootball", "football-data", "tff-github", "transfermarkt")

# Kaggle "Football Data from Transfermarkt" (CC0, updated regularly).
TRANSFERMARKT_DATASET = "davidcariboo/player-scores"
TRANSFERMARKT_COMPETITIONS = {"T1": "TR1", "E0": "GB1", "SP1": "ES1", "D1": "L1", "I1": "IT1"}
# Results awarded by the federation that still carry referee/manager data, so the
# generic forfeit rule cannot see them: (date, home, away) after name mapping.
TRANSFERMARKT_AWARDED = {
    ("2025-02-09", "Galatasaray", "Adana Demirspor"),  # Adana Demirspor left the pitch at 1-1
}
# Transfermarkt names that differ from the other sources (derived by matching
# 2018-2025 results on date and score; FC/AFC are handled by normalize_team).
TRANSFERMARKT_NAMES: dict[str, str] = {
    "Akhisarspor": "Akhisar Belediyespor", "Altay SK": "Altay",
    "Basaksehir FK": "İstanbul Başakşehir", "Beşiktaş Jimnastik Kulübü": "Beşiktaş",
    "Caykur Rizespor": "Çaykur Rizespor", "Erzurumspor FK": "BB Erzurumspor",
    "Fenerbahce": "Fenerbahçe", "Gençlerbirliği Spor Kulübü": "Gençlerbirliği",
    "Istanbulspor": "İstanbulspor", "Kasimpasa": "Kasımpaşa",
}

# Official TFF (sponsor) names -> names used by the other sources.
TFF_NAMES: dict[str, str] = {
    "ADANA DEMİRSPOR A.Ş.": "Adana Demirspor", "ALTAY": "Altay",
    "ALANYASPOR": "Alanyaspor", "AYTEMİZ ALANYASPOR": "Alanyaspor",
    "ATAKAŞ HATAYSPOR": "Hatayspor", "BEŞİKTAŞ A.Ş.": "Beşiktaş",
    "DEMİR GRUP SİVASSPOR": "Sivasspor", "FENERBAHÇE A.Ş.": "Fenerbahçe",
    "FRAPORT TAV ANTALYASPOR": "Antalyaspor", "GALATASARAY A.Ş.": "Galatasaray",
    "GAZİANTEP FUTBOL KULÜBÜ A.Ş.": "Gaziantep FK",
    "GZT GİRESUNSPOR": "Giresunspor", "BITEXEN GİRESUNSPOR": "Giresunspor",
    "GÖZTEPE A.Ş.": "Göztepe", "KASIMPAŞA A.Ş.": "Kasımpaşa",
    "MEDİPOL BAŞAKŞEHİR FK": "İstanbul Başakşehir", "TRABZONSPOR A.Ş.": "Trabzonspor",
    "VAVACARS FATİH KARAGÜMRÜK": "Fatih Karagümrük", "YUKATEL KAYSERİSPOR": "Kayserispor",
    "ÇAYKUR RİZESPOR A.Ş.": "Çaykur Rizespor", "ÖZNUR KABLO YENİ MALATYASPOR": "Yeni Malatyaspor",
    "İTTİFAK HOLDİNG KONYASPOR": "Konyaspor", "ARABAM.COM KONYASPOR": "Konyaspor",
    "HANGİKREDİ ÜMRANİYESPOR": "Ümraniyespor", "MKE ANKARAGÜCÜ": "MKE Ankaragücü",
    "İSTANBULSPOR A.Ş.": "İstanbulspor",
    "BÜYÜKŞEHİR BELEDİYE ERZURUMSPOR": "BB Erzurumspor", "FATİH KARAGÜMRÜK A.Ş.": "Fatih Karagümrük",
    "FRAPORT-TAV ANTALYASPOR": "Antalyaspor", "GENÇLERBİRLİĞİ": "Gençlerbirliği",
    "HELENEX YENİ MALATYASPOR": "Yeni Malatyaspor", "YENİ MALATYASPOR": "Yeni Malatyaspor",
    "HES KABLO KAYSERİSPOR": "Kayserispor", "YUKATEL DENİZLİSPOR": "Denizlispor",
}


@dataclass(frozen=True)
class League:
    """Static information about a league used by the CLI and the simulator."""

    code: str
    name: str
    europe_spots: int
    relegation_spots: int


# Europe/relegation counts are approximations of recent seasons (cup winners and
# rule changes are ignored); both can be overridden from the CLI.
LEAGUES: dict[str, League] = {
    "T1": League("T1", "Türkiye Süper Lig", europe_spots=4, relegation_spots=3),
    "E0": League("E0", "English Premier League", europe_spots=5, relegation_spots=3),
    "SP1": League("SP1", "La Liga", europe_spots=6, relegation_spots=3),
    "D1": League("D1", "Bundesliga", europe_spots=6, relegation_spots=2),
    "I1": League("I1", "Serie A", europe_spots=6, relegation_spots=3),
}

REQUIRED_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]

# Closing odds are preferred (Pinnacle, then Bet365, then market average); opening
# odds are used only when no closing odds exist. Used for the market baseline only.
ODDS_PREFERENCE = [
    ("PSCH", "PSCD", "PSCA"),
    ("B365CH", "B365CD", "B365CA"),
    ("AvgCH", "AvgCD", "AvgCA"),
    ("PSH", "PSD", "PSA"),
    ("B365H", "B365D", "B365A"),
    ("AvgH", "AvgD", "AvgA"),
]

# Spelling/sponsor variants seen across seasons are mapped to one canonical name
# (applied after the generic FC/AFC suffix removal in normalize_team).
TEAM_ALIASES: dict[str, str] = {
    "Atiker Konyaspor": "Konyaspor",
    "Aytemiz Alanyaspor": "Alanyaspor",
    "Gazişehir Gaziantep FK": "Gaziantep FK",
    "Gençlerbirligi": "Gençlerbirliği",
    "Kasimpasa": "Kasımpaşa",
    "Kasımpaşa SK": "Kasımpaşa",
    "İstanbulspor AŞ": "İstanbulspor",
}


def season_codes(spec: str) -> list[str]:
    """Turn ``"2018-2025"`` into ``["1819", ..., "2425"]``.

    The range names the first and last calendar years covered, so ``2018-2025``
    is the seven seasons 2018/19 … 2024/25. A single year ``"2024"`` means 2024/25.
    """
    parts = spec.split("-")
    if len(parts) == 1:
        start = int(parts[0])
        end = start + 1
    elif len(parts) == 2:
        start, end = int(parts[0]), int(parts[1])
    else:
        raise ValueError(f"Invalid season range: {spec!r}")
    if end <= start:
        raise ValueError(f"Season range must be increasing: {spec!r}")
    return [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(start, end)]


def season_url(league: str, season: str, source: str = "openfootball") -> str:
    """URL of one season file (``season`` as ``"2425"``)."""
    if source == "football-data":
        return BASE_URL.format(season=season, league=league)
    if source == "tff-github":
        if league != "T1":
            raise ValueError("tff-github source only covers T1")
        return TFF_GITHUB_URL.format(y1=f"20{season[:2]}", y2=season[2:])
    if league not in OPENFOOTBALL_PATHS:
        raise ValueError(f"openfootball source supports {sorted(OPENFOOTBALL_PATHS)}; "
                         f"use --source football-data for {league}")
    long = f"20{season[:2]}-{season[2:]}"
    return OPENFOOTBALL_URL.format(path=OPENFOOTBALL_PATHS[league].format(s=long))


def _played_matches(path: Path) -> int:
    """Number of clean matches in an existing season file (0 if missing/unreadable)."""
    try:
        return len(clean_matches(_read_csv_bytes(path.read_bytes())))
    except Exception:  # noqa: BLE001
        return 0


def _save_season(path: Path, matches: pd.DataFrame, force: bool) -> bool:
    """Write a season CSV unless an existing file already holds more matches."""
    existing = _played_matches(path) if path.exists() else 0
    if existing > len(matches) and not force:
        print(f"  = {path}: kept existing file ({existing} matches > {len(matches)} new; "
              "use --force to overwrite)")
        return False
    matches.assign(Date=matches["Date"].dt.strftime("%d/%m/%Y")).to_csv(path, index=False)
    return True


def download(league: str, seasons: list[str], data_dir: Path | str = "data",
             source: str = "openfootball", timeout: float = 30.0,
             force: bool = False) -> list[Path]:
    """Download seasons to ``data_dir/<league>/<season>.csv``; returns the paths written.

    Non-CSV sources are kept under ``raw/`` and converted to the common columns.
    A season file that already holds more matches is not overwritten unless
    ``force`` is set, so sources can be combined (e.g. a complete history from
    Transfermarkt plus the current season from openfootball).
    """
    if source == "transfermarkt":
        return download_transfermarkt(league, seasons, data_dir, force=force)
    out_dir = Path(data_dir) / league
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for season in seasons:
        url = season_url(league, season, source)
        req = urllib.request.Request(url, headers={"User-Agent": "football-predictor/0.1"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                content = resp.read()
        except Exception as exc:  # noqa: BLE001 - report and continue with other seasons
            print(f"  ! {url}: {exc}")
            continue
        path = out_dir / f"{season}.csv"
        if source == "football-data":
            if b"HomeTeam" not in content[:2000]:
                # e.g. an ISP block page or an HTML error page served with status 200
                print(f"  ! {url}: response is not a football-data CSV (blocked or missing?)")
                continue
            n_new = len(clean_matches(_read_csv_bytes(content)))
            if path.exists() and _played_matches(path) > n_new and not force:
                print(f"  = {path}: kept existing file (more matches than the download)")
                continue
            path.write_bytes(content)
            note = ""
        else:
            raw_dir = out_dir / "raw"
            raw_dir.mkdir(exist_ok=True)
            if source == "tff-github":
                (raw_dir / f"{season}_tff.csv").write_bytes(content)
                matches, n_forfeit = parse_tff_csv(content)
                note = f", {len(matches)} played matches, {n_forfeit} forfeits dropped"
            else:
                (raw_dir / f"{season}.txt").write_bytes(content)
                matches = parse_football_txt(content.decode("utf-8"), season)
                note = f", {len(matches)} played matches"
            if matches.empty:
                print(f"  ! {url}: no played matches found")
                continue
            if not _save_season(path, matches, force):
                continue
        print(f"  {url} -> {path} ({len(content) / 1024:.0f} KB{note})")
        written.append(path)
    return written


def _kaggle_executable() -> str:
    """Path of the ``kaggle`` CLI (installed with ``pip install ".[kaggle]"``)."""
    candidates = [shutil.which("kaggle"), str(Path(sys.executable).with_name("kaggle"))]
    for c in candidates:
        if c and Path(c).exists():
            return c
    raise RuntimeError('The transfermarkt source needs the Kaggle CLI: pip install ".[kaggle]"')


def fetch_transfermarkt_games(data_dir: Path | str = "data", refresh: bool = False) -> Path:
    """Download (or reuse) ``games.csv`` of the Kaggle dataset into ``data/_transfermarkt``.

    Requires a Kaggle API token (``~/.kaggle/access_token`` or the
    ``KAGGLE_API_TOKEN`` environment variable).
    """
    cache = Path(data_dir) / "_transfermarkt"
    path = cache / "games.csv"
    if path.exists() and not refresh:
        return path
    cache.mkdir(parents=True, exist_ok=True)
    cmd = [_kaggle_executable(), "datasets", "download", TRANSFERMARKT_DATASET,
           "-f", "games.csv", "-p", str(cache), "--unzip", "-q"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not path.exists():
        raise RuntimeError("Kaggle download failed (is an API token configured? see "
                           "https://www.kaggle.com/settings -> API):\n"
                           + (res.stderr or res.stdout).strip()[-800:])
    return path


def parse_transfermarkt_games(games: pd.DataFrame, league: str, season: str) -> tuple[pd.DataFrame, int]:
    """Matches of one league season from Transfermarkt ``games.csv``.

    Forfeits (no referee, no managers, 3-0 or 0-3; e.g. the 2023 earthquake
    withdrawals) and the results listed in :data:`TRANSFERMARKT_AWARDED` are
    dropped. Returns (matches, number of dropped forfeits/awarded results).
    """
    comp = TRANSFERMARKT_COMPETITIONS[league]
    year = 2000 + int(season[:2])
    g = games[(games["competition_id"] == comp) & (games["season"] == year)]
    g = g.dropna(subset=["home_club_goals", "away_club_goals", "home_club_name", "away_club_name"])
    score = list(zip(g["home_club_goals"], g["away_club_goals"]))
    forfeit = (g["referee"].isna() & g["home_club_manager_name"].isna()
               & pd.Series([s in {(3, 0), (0, 3)} for s in score], index=g.index))
    g = g[~forfeit]
    name = lambda n: TRANSFERMARKT_NAMES.get(n, normalize_team(n))  # noqa: E731
    out = pd.DataFrame({
        "Date": pd.to_datetime(g["date"]),
        "HomeTeam": g["home_club_name"].map(name),
        "AwayTeam": g["away_club_name"].map(name),
        "FTHG": g["home_club_goals"].astype(int),
        "FTAG": g["away_club_goals"].astype(int),
    })
    awarded = np.array([(d.strftime("%Y-%m-%d"), h, a) in TRANSFERMARKT_AWARDED
                        for d, h, a in zip(out["Date"], out["HomeTeam"], out["AwayTeam"])],
                       dtype=bool)
    out = out.loc[~awarded]
    return (out.sort_values("Date", kind="stable").reset_index(drop=True),
            int(forfeit.sum() + awarded.sum()))


def download_transfermarkt(league: str, seasons: list[str], data_dir: Path | str = "data",
                           force: bool = False, refresh: bool = False) -> list[Path]:
    """Split the Transfermarkt games file into ``data/<league>/<season>.csv`` files."""
    if league not in TRANSFERMARKT_COMPETITIONS:
        raise ValueError(f"transfermarkt source supports {sorted(TRANSFERMARKT_COMPETITIONS)}")
    games_path = fetch_transfermarkt_games(data_dir, refresh=refresh)
    games = pd.read_csv(games_path, low_memory=False)
    out_dir = Path(data_dir) / league
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for season in seasons:
        matches, n_forfeit = parse_transfermarkt_games(games, league, season)
        if matches.empty:
            print(f"  ! {league} {season}: not in the Transfermarkt dataset")
            continue
        path = out_dir / f"{season}.csv"
        if _save_season(path, matches, force):
            print(f"  transfermarkt {league} {season} -> {path} ({len(matches)} played matches"
                  + (f", {n_forfeit} forfeits dropped)" if n_forfeit else ")"))
            written.append(path)
    return written


def parse_tff_csv(raw: bytes) -> tuple[pd.DataFrame, int]:
    """Convert a TFF match list to Date/HomeTeam/AwayTeam/FTHG/FTAG.

    Matches awarded by forfeit (``Stat == "Hükmen"``, e.g. after the 2023
    earthquake withdrawals) were never played and are dropped. Unknown team
    names raise so new sponsor names are noticed.
    """
    d = pd.read_csv(io.StringIO(raw.decode("utf-8-sig")))
    forfeit = d["Stat"].astype(str).str.strip().str.lower().eq("hükmen")
    d = d[~forfeit]
    names = set(d["Ev_sahibi"]) | set(d["Misafir_takim"])
    unknown = sorted(n for n in names if n.strip() not in TFF_NAMES)
    if unknown:
        raise ValueError(f"Add these TFF names to TFF_NAMES: {unknown}")
    out = pd.DataFrame({
        "Date": pd.to_datetime(d["Tarih_saat"].astype(str).str.extract(
            r"(\d{1,2}\.\d{1,2}\.\d{4})")[0], format="%d.%m.%Y"),
        "HomeTeam": d["Ev_sahibi"].str.strip().map(TFF_NAMES),
        "AwayTeam": d["Misafir_takim"].str.strip().map(TFF_NAMES),
        "FTHG": d["Ev_sahibi_gol"].astype(int),
        "FTAG": d["Misafir_takim_gol"].astype(int),
    })
    return out.sort_values("Date", kind="stable").reset_index(drop=True), int(forfeit.sum())


_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_DATE_RE = re.compile(
    r"^\s*(?:\[)?(?:mon|tue|wed|thu|fri|sat|sun)\w*\.?\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*"
    r"\.?\s+(\d{1,2})(?:\s+(\d{4}))?(?:\])?\s*(.*)$", re.IGNORECASE)
_MATCH_RE = re.compile(
    r"^\s*(?:\d{1,2}[:.]\d{2}\s+)?(?P<home>\S.*?)\s+v\s+(?P<away>\S.*?)"
    r"(?:\s+(?P<hg>\d+)\s*-\s*(?P<ag>\d+)(?:\s*\(.*\))?)?\s*$")
# alternative layout: "Home  2-1 (1-0)  Away"
_MATCH_RE2 = re.compile(
    r"^\s*(?:\d{1,2}[:.]\d{2}\s+)?(?P<home>\S.*?)\s+(?P<hg>\d+)-(?P<ag>\d+)"
    r"(?:\s*\([^)]*\))?\s+(?P<away>\D\S*.*?)\s*$")


def parse_football_txt(text: str, season: str) -> pd.DataFrame:
    """Parse an openfootball Football.TXT league file into played matches.

    Handles date lines such as ``Fri Aug 9 2024`` / ``Sat Aug 10`` (year inferred
    from the previous date line, so e.g. the July 2020 restart lands in 2020), optional
    kick-off times, half-time scores and both ``Home v Away 2-1`` and
    ``Home 2-1 Away`` layouts. Fixtures without a score and goal-scorer lines
    (starting with "(") are skipped.
    """
    y1 = 2000 + int(season[:2])
    date = None
    rows = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith(("#", "=", "▪", "»", "(")):
            continue
        m = _DATE_RE.match(line)
        if m:
            month, day = _MONTHS[m.group(1).lower()[:3]], int(m.group(2))
            if m.group(3):
                year = int(m.group(3))
            elif date is None:
                year = y1 if month >= 7 else y1 + 1
            else:
                # dates run forward: same year as the previous date unless that goes back
                year = date.year
                if pd.Timestamp(year=year, month=month, day=day) < date - pd.Timedelta(days=60):
                    year += 1
            date = pd.Timestamp(year=year, month=month, day=day)
            line = m.group(4)  # a match may follow on the same line
            if not line.strip():
                continue
        m = _MATCH_RE.match(line) if " v " in line else _MATCH_RE2.match(line)
        if m and date is not None and m.group("hg") is not None:
            rows.append((date, m.group("home").strip(), m.group("away").strip(),
                         int(m.group("hg")), int(m.group("ag"))))
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def team_key(name: str) -> str:
    """Accent/case-insensitive key used to match user input to team names."""
    table = str.maketrans({"ı": "i", "İ": "i", "ş": "s", "Ş": "s", "ğ": "g", "Ğ": "g"})
    s = unicodedata.normalize("NFKD", name.translate(table))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", s.lower())


def normalize_team(name: str) -> str:
    """Collapse whitespace, drop "FC"/"AFC" affixes, then apply :data:`TEAM_ALIASES`."""
    clean = re.sub(r"\s+", " ", str(name)).strip()
    clean = re.sub(r"^AFC (?=\S)|(?<=\S) A?FC$", "", clean)
    return TEAM_ALIASES.get(clean, clean)


def _read_csv_bytes(raw: bytes) -> pd.DataFrame:
    for enc in ("utf-8-sig", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    # Some files have trailing commas / ragged rows; skip lines that cannot be parsed.
    return pd.read_csv(io.StringIO(text), on_bad_lines="skip", engine="python")


def _parse_dates(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    out = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    short = out.isna()
    out[short] = pd.to_datetime(s[short], format="%d/%m/%y", errors="coerce")
    return out


def _pick_odds(df: pd.DataFrame) -> pd.DataFrame:
    """Return OddsH/OddsD/OddsA using the first available bookmaker per row."""
    odds = pd.DataFrame(np.nan, index=df.index, columns=["OddsH", "OddsD", "OddsA"])
    for cols in ODDS_PREFERENCE:
        if not all(c in df.columns for c in cols):
            continue
        cand = df[list(cols)].apply(pd.to_numeric, errors="coerce")
        cand.columns = odds.columns
        valid = (cand > 1.0).all(axis=1)
        fill = odds.isna().any(axis=1) & valid
        odds.loc[fill] = cand.loc[fill].to_numpy()
    return odds


def clean_matches(raw: pd.DataFrame, season: str | None = None) -> pd.DataFrame:
    """Keep the needed columns, drop broken rows and normalize team names.

    Output columns: Date, Season, HomeTeam, AwayTeam, FTHG, FTAG, OddsH, OddsD, OddsA.
    """
    raw = raw.rename(columns=lambda c: str(c).strip().lstrip("﻿"))
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = pd.DataFrame({
        "Date": _parse_dates(raw["Date"]),
        "HomeTeam": raw["HomeTeam"],
        "AwayTeam": raw["AwayTeam"],
        "FTHG": pd.to_numeric(raw["FTHG"], errors="coerce"),
        "FTAG": pd.to_numeric(raw["FTAG"], errors="coerce"),
    })
    df = pd.concat([df, _pick_odds(raw)], axis=1)
    df = df.dropna(subset=REQUIRED_COLUMNS)
    df = df[(df["FTHG"] >= 0) & (df["FTAG"] >= 0)
            & (df["FTHG"] == df["FTHG"].round()) & (df["FTAG"] == df["FTAG"].round())]
    df["HomeTeam"] = df["HomeTeam"].map(normalize_team)
    df["AwayTeam"] = df["AwayTeam"].map(normalize_team)
    df = df[(df["HomeTeam"] != "") & (df["AwayTeam"] != "") & (df["HomeTeam"] != df["AwayTeam"])]
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)
    df.insert(1, "Season", season if season is not None else "")
    df = df.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"])
    return df.sort_values(["Date", "HomeTeam"], kind="stable").reset_index(drop=True)


MANUAL_FILE = "manual_results.csv"
MANUAL_COLUMNS = ["Date", "competition", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]


def season_of(date: pd.Timestamp) -> str:
    """Season code of a date (July-June seasons), e.g. 2026-10-15 -> "2627"."""
    y = date.year if date.month >= 7 else date.year - 1
    return f"{y % 100:02d}{(y + 1) % 100:02d}"


def load_manual(data_dir: Path | str = "data") -> pd.DataFrame:
    """Results entered by hand with ``football-predictor add-result``."""
    path = Path(data_dir) / MANUAL_FILE
    if not path.exists():
        df = pd.DataFrame(columns=MANUAL_COLUMNS)
    else:
        df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def add_manual_result(data_dir: Path | str, date: str, competition: str, home: str, away: str,
                      home_goals: int, away_goals: int, replace: bool = False) -> str:
    """Append one result to ``data/manual_results.csv``; returns "added", "updated" or "exists".

    The same match (date, home, away) is stored once; a different score for it
    raises unless ``replace`` is set.
    """
    if home == away:
        raise ValueError("Home and away team are the same")
    if home_goals < 0 or away_goals < 0:
        raise ValueError("Goals must be non-negative")
    df = load_manual(data_dir)
    d = pd.Timestamp(date)
    same = (df["Date"] == d) & (df["HomeTeam"] == home) & (df["AwayTeam"] == away)
    if same.any():
        row = df[same].iloc[0]
        if (row["FTHG"], row["FTAG"]) == (home_goals, away_goals) and row["competition"] == competition:
            return "exists"
        if not replace:
            raise ValueError(f"{d:%Y-%m-%d} {home}-{away} is already stored as "
                             f"{row['FTHG']}-{row['FTAG']} ({row['competition']}); use --replace")
        df = df[~same]
        status = "updated"
    else:
        status = "added"
    new = pd.DataFrame([[d, competition, home, away, int(home_goals), int(away_goals)]],
                       columns=MANUAL_COLUMNS)
    df = pd.concat([df, new], ignore_index=True).sort_values("Date", kind="stable")
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    df.assign(Date=df["Date"].dt.strftime("%Y-%m-%d")).to_csv(Path(data_dir) / MANUAL_FILE, index=False)
    return status


def load_league(league: str, data_dir: Path | str = "data",
                seasons: list[str] | None = None) -> pd.DataFrame:
    """Load and clean every downloaded season of ``league`` sorted by date.

    Hand-entered results for this league (``add-result``) are added when the
    downloaded files do not already contain that fixture of the season.
    """
    league_dir = Path(data_dir) / league
    paths = sorted(league_dir.glob("*.csv"))
    if seasons is not None:
        paths = [p for p in paths if p.stem in seasons]
    if not paths:
        raise FileNotFoundError(
            f"No data in {league_dir}. Run: football-predictor download --league {league}")
    frames = [clean_matches(_read_csv_bytes(p.read_bytes()), season=p.stem) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    manual = load_manual(data_dir)
    manual = manual[manual["competition"].isin({league, TRANSFERMARKT_COMPETITIONS.get(league)})]
    if not manual.empty:
        manual = manual.assign(Season=manual["Date"].map(season_of))
        if seasons is not None:
            manual = manual[manual["Season"].isin(seasons)]
        have = set(zip(df["Season"], df["HomeTeam"], df["AwayTeam"]))
        keep = [k not in have for k in zip(manual["Season"], manual["HomeTeam"], manual["AwayTeam"])]
        df = pd.concat([df, manual.loc[keep, ["Date", "Season", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]]],
                       ignore_index=True)
    df = df.sort_values(["Date", "HomeTeam"], kind="stable").reset_index(drop=True)
    return attach_odds(df, league, data_dir)


ODDS_DATASET = "adamgbor/club-football-match-data-2000-2025"
ODDS_FILE = Path("_odds") / "Matches.csv"


def odds_name_map(ours: pd.DataFrame, odds: pd.DataFrame) -> dict[str, str]:
    """Map odds-file team names to ours by voting over matches with equal date and score."""
    from collections import Counter, defaultdict
    votes: dict[str, Counter] = defaultdict(Counter)
    for shift in (0, -1, 1):
        o = ours.assign(Date=ours["Date"] + pd.Timedelta(days=shift))
        m = odds.merge(o, left_on=["MatchDate", "FTHome", "FTAway"], right_on=["Date", "FTHG", "FTAG"],
                       suffixes=("_odds", ""))
        for a, b, c, d in zip(m["HomeTeam_odds"], m["HomeTeam"], m["AwayTeam_odds"], m["AwayTeam"]):
            votes[a][b] += 1
            votes[c][d] += 1
    out = {}
    for name, c in votes.items():
        top = c.most_common(2)
        best, n1 = top[0]
        n2 = top[1][1] if len(top) > 1 else 0
        if n1 >= 3 and n1 >= 3 * n2:  # clear winner only
            out[name] = best
    return out


def attach_odds(df: pd.DataFrame, league: str, data_dir: Path | str = "data") -> pd.DataFrame:
    """Fill OddsH/OddsD/OddsA (Bet365 pre-match) and shots on target SOTH/SOTA from the
    Kaggle football-data compilation where missing.

    Rows are matched on mapped team names and a date within ±3 days.
    """
    path = Path(data_dir) / ODDS_FILE
    if not path.exists():
        return df
    cols = ["Division", "MatchDate", "HomeTeam", "AwayTeam", "FTHome", "FTAway",
            "OddHome", "OddDraw", "OddAway", "HomeTarget", "AwayTarget"]
    o = pd.read_csv(path, usecols=lambda c: c in cols, low_memory=False)
    for c in cols:
        if c not in o:
            o[c] = np.nan
    o = o[o["Division"] == league].dropna(subset=["FTHome", "FTAway"])
    if o.empty:
        return df
    o["MatchDate"] = pd.to_datetime(o["MatchDate"])
    o = o[o["MatchDate"] >= df["Date"].min() - pd.Timedelta(days=3)]
    names = odds_name_map(df, o)
    o = o.assign(H=o["HomeTeam"].map(names), A=o["AwayTeam"].map(names)).dropna(subset=["H", "A"])
    base = df.reset_index().rename(columns={"index": "_row"})
    m = base[["_row", "Date", "HomeTeam", "AwayTeam"]].merge(
        o, left_on=["HomeTeam", "AwayTeam"], right_on=["H", "A"], suffixes=("", "_o"))
    m = m[(m["Date"] - m["MatchDate"]).abs() <= pd.Timedelta(days=3)]
    m = m.assign(gap=(m["Date"] - m["MatchDate"]).abs()).sort_values("gap").drop_duplicates("_row")
    out = df.copy()
    for c in ("OddsH", "OddsD", "OddsA", "SOTH", "SOTA"):
        if c not in out:
            out[c] = np.nan
    idx = m["_row"].to_numpy()
    odds = m[["OddHome", "OddDraw", "OddAway"]].to_numpy(dtype=float)
    valid = (odds > 1).all(axis=1)
    fill = out.loc[idx, ["OddsH", "OddsD", "OddsA"]].isna().any(axis=1).to_numpy() & valid
    out.loc[idx[fill], ["OddsH", "OddsD", "OddsA"]] = odds[fill]
    out.loc[idx, ["SOTH", "SOTA"]] = m[["HomeTarget", "AwayTarget"]].to_numpy(dtype=float)
    return out


def outcome(df: pd.DataFrame) -> np.ndarray:
    """Match result as an int array: 0 = home win, 1 = draw, 2 = away win."""
    diff = df["FTHG"].to_numpy() - df["FTAG"].to_numpy()
    return np.where(diff > 0, 0, np.where(diff == 0, 1, 2))
