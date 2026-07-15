import pandas as pd
from pybaseball import chadwick_register
import config

def get_player_names() -> pd.Series:
    register = chadwick_register()
    # keep only players with an MLBAM id
    register = register[register["key_mlbam"].notna()]
    register["name"] = register["name_first"] + " " + register["name_last"]

    register = register.drop_duplicates(subset="key_mlbam")
    return register.set_index("key_mlbam")["name"]

def encode_pitches(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[~df["pitch_type"].isin(config.DISALLOWED_PITCHES)]
    # map "slow curve" to regular curveball
    df = df.map(lambda x: "CU" if x == "CS" else x)
    # map pitch types to indices
    df["pitch_type_idx"] = df["pitch_type"].map(config.PITCH_TO_IDX)
    df = df[df["pitch_type_idx"].notna()]
    df["pitch_type_idx"] = df["pitch_type_idx"].astype(int)
    # Sort by game -> at bat -> pitch number
    # get pitch number of ab
    df["pitch_num_in_ab"] = df.groupby(
        ["game_pk", "at_bat_number"]
    ).cumcount()
    df = df.sort_values(by=["game_pk", "at_bat_number", "pitch_num_in_ab"], ascending=True)

    return df

def add_situation_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    runners_on = (
        df["on_1b"].notna().astype(int) * 1 +
        df["on_2b"].notna().astype(int) * 2 +
        df["on_3b"].notna().astype(int) * 4
    )
    df["runners_on"] = runners_on

    stand_dummies = pd.get_dummies(df["stand"], prefix="stand", dtype=int)
    df = df.assign(**stand_dummies)

    throws_dummies = pd.get_dummies(df["p_throws"], prefix="throws", dtype=int)
    df = df.assign(**throws_dummies)

    df["pitch_num_in_ab"] = df.groupby(
        ["game_pk", "at_bat_number"]
    ).cumcount()
    df["is_first_pitch"] = (df["pitch_num_in_ab"] == 0).astype(int)

    return df

def add_lag_features(df: pd.DataFrame, n_lags: int = config.N_LAGS) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(
        by=["game_pk", "at_bat_number", "pitch_num_in_ab"], ascending=True
    )

    grp = df.groupby(["game_pk", "at_bat_number"])
    for lag in range(1, n_lags + 1):
        df[f"prev_speed_{lag}"] = grp["release_speed"].shift(lag).fillna(-1)
        df[f"prev_plate_x_{lag}"] = grp["plate_x"].shift(lag).fillna(config.SENTINEL)
        df[f"prev_plate_z_{lag}"] = grp["plate_z"].shift(lag).fillna(config.SENTINEL)
        df[f"prev_pfx_x_{lag}"] = grp["pfx_x"].shift(lag).fillna(config.SENTINEL)
        df[f"prev_pfx_z_{lag}"] = grp["pfx_z"].shift(lag).fillna(config.SENTINEL)
        df[f"prev_spin_{lag}"] = grp["release_spin_rate"].shift(lag).fillna(-1)

        # result booleans
        df[f"prev_whiff_{lag}"] = (
            grp["description"].shift(lag).
            isin(config.WHIFF_DESCRIPTIONS)
            .astype(int).fillna(0)
        )
        df[f"prev_foul_{lag}"] = (
            grp["description"].shift(lag).
            isin(config.FOUL_DESCRIPTIONS)
            .astype(int).fillna(0)
        )
        
    return df

def add_arsenal_features(df: pd.DataFrame, arsenal: pd.DataFrame) -> pd.DataFrame:
    arsenal = arsenal.rename(columns={
        col: f"{col}_pct" 
        for col in arsenal.columns 
        if col not in ["pitcher_name", "total_pitches"]
    })
    arsenal_feat_cols = [
        c for c in arsenal.columns
        if c not in ["pitcher", "total_pitches"]
    ]
    df = df.join(arsenal[arsenal_feat_cols], on="pitcher")
    missing = df["pitcher"].isin(arsenal.index) == False
    df = df[~missing]

    return df

def build_features (df: pd.DataFrame, arsenal: pd.DataFrame) -> pd.DataFrame:
    df = encode_pitches(df)
    df = add_situation_features(df)
    df = add_lag_features(df)
    df = add_arsenal_features(df, arsenal)
    return df

def get_feature_cols (df: pd.DataFrame, arsenal: pd.DataFrame) -> list[str]:
    arsenal_cols = [
    f"{c}_pct" for c in arsenal.columns
    if c not in ["pitcher", "total_pitches"]
    ]
    lag_cols = [
        c for c in df.columns if c.startswith("prev_")
    ]
    return config.SITUATION_COLS + lag_cols + arsenal_cols