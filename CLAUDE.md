# Pitch Sequencing Classifier — Project Context

## Project Overview

A next-pitch type predictor for MLB pitchers. Given a sequence of pitches from a particular pitcher in an at-bat, predict what pitch comes next (and optionally where it goes). Built as a baseball analytics portfolio project.

## Repository Structure

```
pitch_sequencing/
├── data/                     # cached pybaseball pulls (.parquet)            
├── models/                   # saved model artifacts
├── notebooks/                # notebooks for experimentation
├── src/
│   ├── __init__.py
│   ├── data.py               # data pulling and caching
│   ├── arsenal.py            # arsenal computation
│   ├── features.py           # feature engineering
│   ├── model.py              # model definition
│   └── evaluate.py           # metrics and visualization
├── train.py                  # entry point
└── config.py                 # hyperparameters and constants
```

## Data

- Source: `pybaseball` — `statcast()` for pitch-by-pitch data
- Cache raw pulls as `.parquet` in `data/raw/` — pybaseball pulls are slow

## Key Features

- Pitcher arsenal: computed by manually to allow for platoon splits, etc. Some statcast pitch categories are dropped / collapsed (i.e. curveball + slow curve).
- Lag features for previous pitches (stand-in for a future LSTM)
- Situation: pitcher / batter handedness, runners on base, inning, etc.

## Models

### XGBoost (baseline)

- `objective="multi:softprob"` for calibrated probability distributions
- Apply arsenal mask post-softmax: zero out pitch types not in pitcher's arsenal, renormalize
- Feature importance: use `importance_type="gain"` not default `"weight"`
- SHAP: use `shap.Explainer(model)` (newer unified API) — `TreeExplainer` has a known compatibility issue with XGBoost's scientific notation booster dump

### PyTorch (two-head)

Shared trunk → two output heads:
1. Pitch type classification (softmax, with pre-softmax arsenal masking)
2. Location regression (plate_x, plate_z)

Combined loss: `alpha * cross_entropy + beta * mse_loss`

When feeding pitch type probs into location head: use `.detach()` to stop gradients crossing between heads.

## Evaluation

- **Don't report accuracy alone** — fastball base rate is ~40-50%, naive baseline looks decent
- Top-1 and top-2 accuracy vs "always predict most common pitch" baseline
- Log-loss / cross-entropy for calibration quality
- Per-count accuracy breakdown (accuracy is highest on 3-0, lowest on even counts)
- Normalized entropy per pitcher for predictability ranking:
  - Raw entropy conflates unpredictability with arsenal size
  - Normalize: `avg_entropy / max_entropy(arsenal_size)` → 0 to 1 scale

## Jupyter / notebook workflow
- Notebook work goes through the `jupyter` MCP server (Datalayer). Prefer its
  tools (list/read/edit/execute cells) over shelling out to run notebooks in
  the terminal or editing `.ipynb` JSON by hand.
- Before editing, call the connect/list-files tool to attach to the target
  notebook rather than assuming one is active.
- After changing a cell, execute it and read the output back to confirm it ran
  clean before moving on. Don't batch many unverified edits.
- Kernel is Python. Plots may render as images (ALLOW_IMG_OUTPUT is on) — read
  them back when validating figures.
- I have JupyterLab open on the same notebook via RTC. Don't overwrite cells
  I'm actively editing; if a cell looks mid-edit, ask before replacing it.

## Conventions
- Notebooks are for exploration/eval; production logic belongs in the package.
  If a notebook cell grows reusable, propose moving it into the right module.