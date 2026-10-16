import json
import shutil
import subprocess
from importlib import resources

import numpy as np
import pytest

from football_predictor.model import DixonColes, outcome_probs, score_matrix
from football_predictor.multileague import CrossLeagueModel
from conftest import simulate_league

node = shutil.which("node")
CORE = resources.files("football_predictor") / "app_core.js"


def run_js(expr: str):
    script = f"const c = require({json.dumps(str(CORE))}); console.log(JSON.stringify({expr}));"
    return json.loads(subprocess.run([node, "-e", script], capture_output=True, text=True, check=True).stdout)


@pytest.mark.skipif(node is None, reason="node not installed")
def test_js_score_matrix_matches_python():
    for lam, mu, rho in [(1.8, 1.4, -0.08), (2.3, 1.2, 0.05), (0.9, 0.7, -0.12)]:
        js = np.array(run_js(f"c.scoreMatrix({lam}, {mu}, {rho})"))
        np.testing.assert_allclose(js, score_matrix(lam, mu, rho), atol=1e-12)


@pytest.mark.skipif(node is None, reason="node not installed")
def test_js_league_rates_match_model(small_league):
    m = DixonColes(l2=2.0).fit(small_league)
    L = {"home": m.home, "rho": m.rho, "value_coef": None, "value_sd": 1.0,
         "teams": {t: {"attack": float(a), "defence": float(d)}
                   for t, a, d in zip(m.teams, m.attack, m.defence)}}
    r = run_js(f"c.leagueRates({json.dumps(L)}, 'Team02', 'Team07', {{}})")
    assert (r["lam"], r["mu"]) == pytest.approx(m.rates("Team02", "Team07"), rel=1e-12)
    s = run_js(f"c.summarize(c.scoreMatrix({r['lam']}, {r['mu']}, {r['rho']}))")
    np.testing.assert_allclose([s["home"], s["draw"], s["away"]],
                               m.predict("Team02", "Team07").probs, atol=1e-12)
    neutral = run_js(f"c.leagueRates({json.dumps(L)}, 'Team02', 'Team07', {{neutral: true}})")
    assert neutral["lam"] == pytest.approx(r["lam"] / np.exp(m.home))


@pytest.mark.skipif(node is None, reason="node not installed")
def test_js_missing_players_and_adjustments():
    # losing players reduces attack (b_a > 0) and raises goals conceded (b_d < 0)
    s = run_js("c.missingPlayerShift(100, 20, [0.3, -0.1], 0.9)")
    dz = np.log(80 / 100) / 0.9
    assert s["att"] == pytest.approx(0.3 * dz) and s["def"] == pytest.approx(-0.1 * dz)
    assert run_js("c.missingPlayerShift(100, 0, [0.3, -0.1], 0.9)") == {"att": 0, "def": 0}
    assert run_js("c.missingPlayerShift(NaN, 10, [0.3, -0.1], 0.9)") == {"att": 0, "def": 0}
    a = run_js("c.applyAdjust({lam: 2, mu: 1, rho: 0}, {homeAtt: 10, awayDef: -10, awayAtt: 20, homeDef: 0})")
    assert a["lam"] == pytest.approx(2 * 1.1 * 0.9) and a["mu"] == pytest.approx(1.2)


@pytest.mark.skipif(node is None, reason="node not installed")
def test_js_world_rates_match_model():
    from test_multileague import simulate_world
    w = simulate_world()
    m = CrossLeagueModel().fit(w)
    W = {"c": m.c, "rho": m.rho, "H": m.H,
         "leagues": {l: {"A": float(a), "D": float(d)} for l, a, d in zip(m.leagues, m.A, m.D)},
         "teams": {t: {"league": m.team_league[t], "a": float(m.a[i]), "d": float(m.d[i])}
                   for i, t in enumerate(m.teams)}}
    r = run_js(f"c.worldRates({json.dumps(W)}, 'A1', 'B5', {{venue: 'UEFA'}})")
    assert (r["lam"], r["mu"]) == pytest.approx(m.rates("A1", "B5", "UEFA"), rel=1e-12)


def test_render_app_embeds_data_and_core():
    from football_predictor.app import render_app
    data = {"built": "x", "leagues": {"T1": {"label": "</script>"}}, "world": None, "defaults": {}}
    page = render_app(data, standalone=True)
    assert page.startswith("<!doctype html>") and "function scoreMatrix" in page
    assert "<\\/script>" in page  # embedded JSON cannot close the script tag
    assert "<html" not in render_app(data, standalone=False)
