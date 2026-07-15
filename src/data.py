import os
import pandas as pd
from pybaseball import statcast, cache as pybaseball_cache
import config

def pull_statcast(
        season: int, force_refresh: bool = False, override_cache_path: str = None
    ) -> pd.DataFrame:
    """Pull statcast data for a full season, caching locally."""
    if override_cache_path:
        cache_path = override_cache_path
    else:
        cache_path = os.path.join(config.DATA_DIR, f"statcast_{season}.parquet")
    pybaseball_cache.enable() # just to be safe

    if os.path.exists(cache_path) and not force_refresh:
        print(f"Loading cached data from {cache_path}")
        return pd.read_parquet(cache_path)
    
    print(f"Pulling statcast data for {season}...")
    df = statcast(
        start_dt=f"{season}-03-01",
        end_dt=f"{season}-11-01"
    )
    
    os.makedirs(config.DATA_DIR, exist_ok=True)
    df.to_parquet(cache_path)
    print(f"Cached to {cache_path}")
    
    return df