#!/usr/bin/env bash
# Regenerates every number and figure used in the README (run from the repo root).
set -euo pipefail
# Complete 2018-2026 history from Transfermarkt (Kaggle token needed), then the
# season in progress from openfootball. Season files that already hold more
# matches are never overwritten, so the order does not matter.
# Without a Kaggle token: use openfootball for everything and add
#   football-predictor download --league T1 --source tff-github --seasons 2021-2023
for L in T1 E0; do
  football-predictor download --league "$L" --source transfermarkt --seasons 2018-2026
  football-predictor download --league "$L" --seasons 2026-2027
done
# Bet365 odds (football-data.co.uk compiled on Kaggle) for the market comparison
mkdir -p data/_odds
[ -f data/_odds/Matches.csv ] || kaggle datasets download adamgbor/club-football-match-data-2000-2025 \
  -f Matches.csv -p data/_odds --unzip -q
# Player market values for the squad-value prior (Transfermarkt, CC0)
[ -f data/_transfermarkt/player_valuations.csv ] || kaggle datasets download davidcariboo/player-scores \
  -f player_valuations.csv -p data/_transfermarkt --unzip -q
for L in T1 E0; do
  football-predictor evaluate --league "$L" | tee "outputs/evaluate_$L.txt"
  football-predictor train --league "$L" | tee "outputs/train_$L.txt"
  football-predictor simulate-season --league "$L" --n 10000 | tee "outputs/simulate_$L.txt"
done
football-predictor predict "Galatasaray" "Fenerbahçe" --league T1 | tee outputs/predict_gs_fb.txt
# cross-league model (all leagues + UEFA club cups)
football-predictor world-rankings --top 30 | tee outputs/world_rankings.txt
football-predictor world-evaluate --season 2526 | tee outputs/world_evaluate.txt
football-predictor world-predict "KuPS" "Trabzonspor" --as-of 2026-10-15 | tee outputs/world_predict_kups_trabzonspor.txt
football-predictor simulate-season --league E0 --season 2526 --as-of 2026-01-01 --n 10000 \
  | tee outputs/simulate_E0_2526_asof.txt
football-predictor build-app --out docs/app/index.html
cp outputs/calibration_T1.png outputs/calibration_E0.png outputs/heatmap_galatasaray_fenerbahce.png \
   outputs/positions_E0_2526.png outputs/world_league_strength.png docs/img/
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda_and_results.ipynb
