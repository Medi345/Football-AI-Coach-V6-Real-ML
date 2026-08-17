# Football AI Coach V6 — Real Persistent ML

This is the first version in this project line that is explicit about the distinction between:
1. a real ML model,
2. a persistent trained model,
3. continual learning,
4. prediction-time data collection.

## What V6 does

- Uses a real historical football dataset for bootstrap training.
- Builds chronological, leakage-safe features.
- Trains a real `HistGradientBoostingClassifier` for 1X2.
- Uses Elo, recent goals, points, home/away form and sample size as features.
- Saves the trained model and state with `joblib`.
- On subsequent app restarts it loads the saved model instead of retraining.
- Uses Poisson as a separate probabilistic model and transparently ensembles it with ML.
- Computes full-time score probabilities and fair odds.
- Never calls football-data.org, API-Football or football-data.co.uk APIs.
- Does not invent bookmaker odds.
- Does not invent first-half/second-half data.

## Bootstrap dataset

The initial dataset is the public `schochastics/football-data` results dataset. Its README documents about 1,237,935 games from 207 top-tier domestic leagues and 20 international tournaments, covering 1888–2023.

Source:
https://github.com/schochastics/football-data

The dataset is downloaded once and cached locally, then the model is trained and saved under `coach_state/models/`.

## Important honesty about "pre-trained"

Because the model is not shipped as a fabricated or synthetic artifact, the very first deployment performs a one-time bootstrap training from the real dataset. After that:
- the model is persisted,
- prediction does not retrain it,
- new labeled matches can be used for incremental updates.

That is the correct engineering meaning of "trained once, then learns continuously".

## Continual learning

The persistence layer is ready for incremental updates. A future ingestion job should append only newly completed matches, build the same feature schema, and call `partial_fit` on an incremental classifier or run a controlled scheduled refresh. It must not mix future match outcomes into pre-match features.

## First-half / second-half

The bootstrap results dataset is full-time oriented. V6 deliberately does not fabricate FH/SH values. A verified half-time source must be added before FH/SH predictions are enabled.

## Deployment

1. Put `app.py` and `requirements.txt` in GitHub.
2. Deploy with Streamlit Community Cloud.
3. The first startup downloads the real dataset and trains the model once.
4. Later restarts load the saved model from the persistent filesystem when persistence is available.

For production-grade persistence across Streamlit restarts, attach an external persistent store (for example a GitHub-backed artifact or database). The model must not be silently regenerated from scratch.
