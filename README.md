# Football AI Coach V6.3

Real ML football prediction application.

## What is real
- 1.3M+ historical rows in the bootstrap source dataset.
- Chronological, leakage-safe feature construction.
- HistGradientBoostingClassifier.
- Time-based train/test split.
- Accuracy, log loss and multiclass Brier score.
- Persisted trained model.
- Elo/form/goal features.
- Poisson score matrix.
- ML + Poisson ensemble.
- Broad model-derived markets and fair odds.

## Important limitations
- The bootstrap dataset contains final scores, not reliable first-half scores for every competition. FH/SH are therefore not fabricated.
- Bookmaker odds are not invented. Without a verified odds source, bookmaker value/edge is not calculated.
- Public-web fixture verification is only a safety check; failure to verify does not create a fake fixture.
- V6.3 does NOT falsely claim that automatic online learning is already implemented. The persisted model can be loaded without retraining, but a production continual-learning ingestion pipeline still needs to be connected to verified new match results.

## Deploy
Replace `app.py`, `requirements.txt`, and `README.md` in the GitHub repository, commit, then reboot the Streamlit app.
