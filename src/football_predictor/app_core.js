// Pure prediction functions shared by the web page and the Node test (tests/test_app.py).
const MAXG = 7;

function poissonPmf(k, lam) {
  let p = Math.exp(-lam);
  for (let i = 1; i <= k; i++) p *= lam / i;
  return p;
}

// (MAXG+1)^2 score matrix with the Dixon-Coles correction, renormalised (as in model.py).
function scoreMatrix(lam, mu, rho) {
  const m = [];
  let total = 0;
  for (let x = 0; x <= MAXG; x++) {
    const row = [];
    for (let y = 0; y <= MAXG; y++) {
      let p = poissonPmf(x, lam) * poissonPmf(y, mu);
      if (x === 0 && y === 0) p *= 1 - lam * mu * rho;
      else if (x === 0 && y === 1) p *= 1 + lam * rho;
      else if (x === 1 && y === 0) p *= 1 + mu * rho;
      else if (x === 1 && y === 1) p *= 1 - rho;
      row.push(p);
      total += p;
    }
    m.push(row);
  }
  return m.map(r => r.map(p => p / total));
}

function summarize(m) {
  let home = 0, draw = 0, away = 0, over = 0, btts = 0;
  const cells = [];
  m.forEach((row, x) => row.forEach((p, y) => {
    if (x > y) home += p; else if (x === y) draw += p; else away += p;
    if (x + y > 2.5) over += p;
    if (x > 0 && y > 0) btts += p;
    cells.push([x, y, p]);
  }));
  cells.sort((a, b) => b[2] - a[2]);
  return { home, draw, away, over, btts, top: cells.slice(0, 6) };
}

// Strength change when players worth `missing` (EUR m) are out of a squad worth `squad`.
// Uses the league model's fitted squad-value coefficients: Δattack = b_a·Δz, Δdefence = b_d·Δz,
// with Δz = Δlog(value) / sd(log value of the league's teams).
function missingPlayerShift(squad, missing, coef, sd) {
  if (!(squad > 0) || !(missing > 0) || !coef || sd <= 0) return { att: 0, def: 0 };
  const left = Math.max(squad - missing, squad * 0.05);
  const dz = (Math.log(left) - Math.log(squad)) / sd;
  return { att: coef[0] * dz, def: coef[1] * dz };
}

// Expected goals for a league match: log λ = att_h + def_a + home, log μ = att_a + def_h.
function leagueRates(L, home, away, opts) {
  const h = L.teams[home], a = L.teams[away];
  const sh = missingPlayerShift(opts.homeSquad, opts.homeMissing, L.value_coef, L.value_sd);
  const sa = missingPlayerShift(opts.awaySquad, opts.awayMissing, L.value_coef, L.value_sd);
  const ha = opts.neutral ? 0 : L.home;
  const lam = Math.exp(h.attack + sh.att + a.defence + sa.def + ha);
  const mu = Math.exp(a.attack + sa.att + h.defence + sh.def);
  return { lam, mu, rho: L.rho, shifts: { home: sh, away: sa } };
}

// Cross-league model: strength = league level + team effect.
function worldRates(W, home, away, opts) {
  const h = W.teams[home], a = W.teams[away];
  const lv = id => W.leagues[W.teams[id].league] || { A: 0, D: 0 };
  const ah = lv(home).A + h.a, dh = lv(home).D + h.d;
  const aa = lv(away).A + a.a, da = lv(away).D + a.d;
  const venue = opts.neutral ? 0 : (opts.venue === "UEFA" ? W.H.UEFA : (W.H[h.league] ?? W.H.UEFA));
  return { lam: Math.exp(W.c + ah + da + venue), mu: Math.exp(W.c + aa + dh), rho: W.rho };
}

// User adjustments in percent: attack scales the team's own expected goals,
// defence scales the goals it concedes (+10 = concedes 10% more).
function applyAdjust(r, adj) {
  const f = v => 1 + (v || 0) / 100;
  return { ...r, lam: r.lam * f(adj.homeAtt) * f(adj.awayDef), mu: r.mu * f(adj.awayAtt) * f(adj.homeDef) };
}

if (typeof module !== "undefined") {
  module.exports = { poissonPmf, scoreMatrix, summarize, missingPlayerShift, leagueRates, worldRates, applyAdjust };
}
