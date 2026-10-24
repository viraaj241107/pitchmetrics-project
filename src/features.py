# Turns raw season stats into the numbers ("features") the model learns from.
# Used by step 3 (training) and step 4 (predicting).
#
# The key idea: counts like goals depend on how much someone played, so we use
# "per 90 minutes" rates. 5 goals in 900 minutes = 0.5 per 90.

import pandas as pd

from config import BIG_CLUBS

POSITION_CODE = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
FULL_SEASON_MINUTES = 38 * 90

# new column name -> raw column it is made from
PER_90 = {
    "goals_p90": "goals_scored",
    "assists_p90": "assists",
    "xg_p90": "expected_goals",
    "xa_p90": "expected_assists",
    "threat_p90": "threat",
    "creativity_p90": "creativity",
    "influence_p90": "influence",
    "bps_p90": "bps",
    "bonus_p90": "bonus",
    "clean_sheets_p90": "clean_sheets",
    "conceded_p90": "goals_conceded",
    "saves_p90": "saves",
    "points_p90": "total_points",
}

FEATURES = ["age", "pos_code", "minutes_share", "starts_share", "team_pos", "big_club", "year",
            "prev_minutes_share", "goals", "assists"] + list(PER_90.keys())


def make_features(df, possible_minutes=FULL_SEASON_MINUTES):
    """df needs the raw stat columns plus: age, year, prev_minutes."""
    X = pd.DataFrame(index=df.index)
    minutes = df["minutes"].clip(lower=1)   # avoid dividing by zero

    X["age"] = df["age"]
    X["pos_code"] = df["position"].map(POSITION_CODE)
    X["minutes_share"] = df["minutes"] / possible_minutes          # 1.0 = played every minute
    X["starts_share"] = df["starts"] * 90 / possible_minutes
    X["team_pos"] = df["team_pos"]
    X["big_club"] = df["team_name"].isin(BIG_CLUBS).astype(int)
    X["year"] = df["year"]                                          # prices go up over the years
    X["prev_minutes_share"] = df["prev_minutes"] / FULL_SEASON_MINUTES
    X["goals"] = df["goals_scored"]
    X["assists"] = df["assists"]
    for new_col, raw_col in PER_90.items():
        X[new_col] = df[raw_col] / minutes * 90

    return X[FEATURES].astype(float)


def add_age_year_prev(seasons):
    """Adds 'year', 'age' and 'prev_minutes' (minutes in the season before) to the season table."""
    seasons = seasons.copy()
    seasons["year"] = seasons["season"].str[:4].astype(int) + 1     # 2024-25 -> summer 2025
    birth = pd.to_datetime(seasons["birth_date"])
    seasons["age"] = (pd.to_datetime(seasons["year"].astype(str) + "-08-01") - birth).dt.days / 365.25

    # minutes in the previous season (0 if he was not in the league)
    previous = seasons[["code", "year", "minutes"]].copy()
    previous["year"] = previous["year"] + 1
    previous = previous.rename(columns={"minutes": "prev_minutes"})
    seasons = seasons.merge(previous, on=["code", "year"], how="left")
    seasons["prev_minutes"] = seasons["prev_minutes"].fillna(0)
    first_year = seasons["year"].min()
    seasons.loc[seasons["year"] == first_year, "prev_minutes"] = None   # no data before our first season
    return seasons
