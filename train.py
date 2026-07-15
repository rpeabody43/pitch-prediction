import pandas as pd

from src.data import pull_statcast
from src.arsenal import compute_arsenal
from src.features import build_features

import config

def main():
    df = pull_statcast(config.SEASON)

    # maybe want to compute arsenal from previous season's data instead
    arsenal = compute_arsenal(df)

    df = build_features(df, arsenal)

    arsenal_cols = [
        f"{c}_pct" for c in arsenal.columns
        if c not in ["pitcher", "total_pitches"]
    ]
    lag_cols = [
        c for c in df.columns if c.startswith("prev_")
    ]
    feature_cols = config.SITUATION_COLS + lag_cols + arsenal_cols

    X = df[feature_cols]
    y = df["pitch_type_idx"]

    # TODO train test split, xgboost training, etc.


if __name__ == "__main__":
    main()