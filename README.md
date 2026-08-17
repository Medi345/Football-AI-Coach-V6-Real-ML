# Football AI Coach V7 — Complete Prediction Engine

## What this version actually does

This is a real ML application, not an if/else demo.

It trains and persists:
- Full-time 1X2 HistGradientBoostingClassifier
- First-half home/away goal regressors
- Second-half home/away goal regressors
- Elo replay
- Weighted Form (5/10/15, overall/home/away)
- Poisson score matrices
- mathematically compatible FT score distribution from FH + SH
- 1X2, double chance, totals, BTTS, team-goals, clean-sheet and win-to-nil markets
- fair odds
- model-only best picks
- time-based train/test evaluation

## Important source disclosure

The bootstrap dataset used here is a public CC-BY dataset on Hugging Face containing 673,966 fixtures and real half-time/full-time fields plus historical odds. Its own metadata states that its upstream sources include API-Football and football-data.co.uk.

Therefore V7 does NOT call those services as APIs, but it also cannot honestly be described as a dataset with those upstream sources excluded.

If your strict requirement is "no API-Football and no football-data.co.uk even as upstream data", replace the bootstrap source with a dataset whose provenance meets that requirement and preserves reliable HT/FT/odds fields.

## No invented odds

Current bookmaker odds are not fabricated. Historical odds in the training data are not presented as current odds. If current public-web odds cannot be verified, the app shows `NOT AVAILABLE` and does not calculate value.

## Continual learning

The trained model is persisted locally and loaded on later runs. A complete live continual-learning loop requires a verified feed of newly finished matches. V7 does not pretend that scraping arbitrary pages is a reliable label feed.

## Deploy

Upload `app.py` and `requirements.txt` to GitHub and deploy `app.py` on Streamlit Community Cloud.
