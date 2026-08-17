# Football AI Coach V6.1 — Real ML

V6.1 fixes a critical team-resolution bug from V6. The previous resolver treated any substring as a strong match, so `Paris Saint-Germain` could incorrectly match `Aris`.

## What V6.1 does
- Real `HistGradientBoostingClassifier` trained on chronological football results.
- Bootstrap dataset: real historical match results from the documented open dataset.
- Persistent `joblib` model and state.
- Elo + recent form + home/away features.
- Time-ordered train/test split; no random split.
- Accuracy, Log Loss and multiclass Brier score.
- Strict team resolver: exact normalized names + explicit aliases + conservative fuzzy fallback.
- Rejects weak substring matches such as `Aris` for `Paris Saint-Germain`.
- No football-data.org/API-Football dependency.
- No invented fixtures or bookmaker odds.

## Important
`historical rows available` is the size of the source dataset. `training_rows` and `test_rows` in Model Health are the actual rows used to fit/evaluate the ML model.

The bootstrap dataset currently contains full-time results, not reliable first-half/second-half scores for every competition. Therefore FH/SH predictions are not fabricated.

Continual learning requires new labeled matches to be appended to the persistent knowledge base and then applied to the saved model; the app does not silently retrain from scratch for each prediction.
