import pandas as pd
import config


# The statcast data only includes 10 common pitch types, so we have to 
# calculate arenal ourselves to get forkballs, knuckle-curves, etc.
def compute_arsenal(
        df, 
        min_pct: float = config.MIN_PITCH_PCT, 
        min_total_pitches: int =config.MIN_TOTAL_PITCHES
    ) -> pd.DataFrame:
    """
    Compute per-pitcher arsenal frequencies from raw pitch data.
    min_pct filters out pitch types that appear too rarely
    to be considered part of the arsenal.
    """
    totals = (
        df.groupby("pitcher")["pitch_type"]
        .count()
        .reset_index(name="total_pitches")
    )
    qualified_pitchers = totals[totals["total_pitches"] >= min_total_pitches]["pitcher"]
    df_filtered = df[df["pitcher"].isin(qualified_pitchers)]

    counts = (
        df_filtered.groupby(["pitcher", "pitch_type"])
        .size()
        .reset_index(name="n")
    )
    # normalize to frequencies within each pitcher
    counts["usage"] = (
        counts.groupby("pitcher")["n"]
        .transform(lambda x: x / x.sum())
    )
    # filter noise — a pitch thrown <1% of the time isn"t part of the arsenal
    counts = counts[counts["usage"] >= min_pct]
    
    # renormalize after filtering — usages no longer sum to 1
    # once rare pitches are dropped
    counts["usage"] = (
        counts.groupby("pitcher")["n"]
        .transform(lambda x: x / x.sum())
    )

    # pivot to wide format: one row per pitcher, one col per pitch type
    arsenal = counts.pivot(
        index="pitcher",
        columns="pitch_type",
        values="usage"
    ).fillna(0)
    
    return arsenal