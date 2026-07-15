DATA_DIR = "data"
MODEL_DIR = "models"

# data
SEASON = 2025
MIN_TOTAL_PITCHES = 100
MIN_PITCH_PCT = 0.01

# features
N_LAGS = 3
SENTINEL = -99 # for missing previous pitch location (x, z)

SITUATION_COLS = [
    "balls", "strikes",
    "inning", "outs_when_up", 
    "pitch_number",
    "stand_R", "stand_L",
    "throws_R", "throws_L",
    "runners_on"
]

DISALLOWED_PITCHES = [
    "PO", # pitchout
    "UN", # unknown
    "SC", # screwball
    "FA", # other
    "EP", # eephus
]

PITCH_TO_IDX = {
    "FF": 0,
    "SI": 1,
    "SL": 2,
    "CH": 3,
    "ST": 4,
    "FC": 5,
    "CU": 6,
    "FS": 7,
    "KC": 8,
    "SV": 9,
    "FO": 10,
    "KN": 11,
}

WHIFF_DESCRIPTIONS = {
    "swinging_strike", "swinging_strike_blocked", "missed_bunt"
}
FOUL_DESCRIPTIONS = {
    "foul", "foul_tip", "foul_bunt"
}

# model
RANDOM_SEED = 42
TEST_SIZE = 0.2
XGB_PARAMS = {
    "objective": "multi:softprob",
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 300,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}