# STEP 4 - Value every current Premier League player.
#
# The value uses THIS SEASON's stats only. After a few games the numbers are
# noisy (one lucky week can make anyone look elite), so each per-90 stat is
# pulled toward the average for the player's position:
#
#     weight = minutes / (minutes + 450)
#     stat   = weight * his stat + (1 - weight) * position average
#
# At 450 minutes a player counts 50% his own numbers; at 2,000 minutes about 80%.
# Goals and assists are projected to a full 38-game season.
#
# We also value each PAST season on its own (for the value-history chart), and
# explain every value with SHAP: how much each stat pushed it up or down.
#
# Output: data/processed/players.json

import json
import math

import joblib
import numpy as np
import pandas as pd
import shap

from config import (CURRENT_SEASON, LAST_SEASON, MIN_MINUTES_NOW, MIN_MINUTES_TRAIN, MODELS,
                    PROCESSED, RAW, SHRINK_MINUTES, TODAY, XGB_WEIGHT)
from features import FEATURES, PER_90, add_age_year_prev, make_features

xgb = joblib.load(MODELS / "xgb_model.joblib")
lin = joblib.load(MODELS / "linear_model.joblib")
metrics = json.load(open(MODELS / "metrics.json"))


def predict_log_value(X):
    return XGB_WEIGHT * xgb.predict(X) + (1 - XGB_WEIGHT) * lin.predict(X)


def clean(x, digits=2):
    """Round a number, and turn NaN into None so it can be saved as JSON."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return round(float(x), digits)


# ---------- this season ----------
seasons = add_age_year_prev(pd.read_csv(PROCESSED / "player_seasons.csv"))
fixtures = pd.read_csv(RAW / f"fix_{CURRENT_SEASON}.csv")
finished = fixtures[fixtures["finished"] == True]
if len(finished) == 0:
    raise SystemExit("No matches played yet this season - keeping the old website.")
gameweeks = int(finished["event"].max())

# players who left for another league are still in the FPL file, marked "u" with "Has joined ..."
fpl_now = pd.read_csv(RAW / f"fpl_{CURRENT_SEASON}.csv")
news = fpl_now["news"].fillna("")
left = fpl_now[(fpl_now["status"] == "u") & news.str.contains("joined|departed|released", case=False)]

now = seasons[(seasons["season"] == CURRENT_SEASON) & (~seasons["code"].isin(left["code"]))].copy()
last_minutes = seasons[seasons["season"] == LAST_SEASON].set_index("code")["minutes"]
now["prev_minutes"] = now["code"].map(last_minutes).fillna(0)
now["age"] = (pd.Timestamp(TODAY) - pd.to_datetime(now["birth_date"])).dt.days / 365.25
now["year"] = TODAY.year

X_raw = make_features(now, possible_minutes=gameweeks * 90)
X = X_raw.copy()
has_value = now["minutes"] >= MIN_MINUTES_NOW

# pull each per-90 stat toward the position average (only players with enough minutes form the average)
weight = now["minutes"] / (now["minutes"] + SHRINK_MINUTES)
for col in PER_90:
    position_average = X[has_value].groupby(now["position"][has_value])[col].mean()
    X[col] = weight * X[col] + (1 - weight) * now["position"].map(position_average)

# project goals and assists to a 38-game season
X["goals"] = X["goals_p90"] * X["minutes_share"] * 38
X["assists"] = X["assists_p90"] * X["minutes_share"] * 38

now["value"] = np.exp(predict_log_value(X))
print(f"gameweeks played: {gameweeks}, players: {len(now)}, with a value: {has_value.sum()}")


# ---------- explain each value ----------
# SHAP gives each feature's push on the XGBoost prediction; for the linear model
# the push is simply coefficient x (scaled feature). We mix them 60/40 like the value.
explainer = shap.TreeExplainer(xgb)
xgb_push = explainer.shap_values(X)
imputer, scaler, ridge = lin.named_steps.values()
lin_push = scaler.transform(imputer.transform(X)) * ridge.coef_
push = XGB_WEIGHT * xgb_push + (1 - XGB_WEIGHT) * lin_push
average_value = math.exp(XGB_WEIGHT * float(explainer.expected_value) + (1 - XGB_WEIGHT) * ridge.intercept_)

# group related features so the website can say "Goal scoring +40%" instead of 3 separate lines
GROUP = {
    "age": "Age", "pos_code": "Position", "year": "Market year",
    "minutes_share": "Playing time", "starts_share": "Playing time", "prev_minutes_share": "Playing time",
    "big_club": "Club", "team_pos": "Club",
    "goals": "Goal scoring", "goals_p90": "Goal scoring", "xg_p90": "Goal scoring",
    "assists": "Creating chances", "assists_p90": "Creating chances", "xa_p90": "Creating chances",
    "creativity_p90": "Creating chances", "threat_p90": "Attacking threat",
    "influence_p90": "Overall match impact", "bps_p90": "Overall match impact",
    "bonus_p90": "Overall match impact", "points_p90": "Overall match impact",
    "clean_sheets_p90": "Defending", "conceded_p90": "Defending", "saves_p90": "Defending",
}


def ordinal(n):
    n = int(n)
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    return f"{n}" + {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def describe(group, p):
    """Short text shown next to each reason, e.g. '5 goals, 2.8 xG'."""
    if group == "Age":
        return f"{p['age']:.0f} years old"
    if group == "Playing time":
        return f"{p['minutes']:.0f} of {gameweeks * 90} minutes"
    if group == "Club":
        return f"{p['team_name']}, {ordinal(p['team_pos'])} in the table"
    if group == "Goal scoring":
        return f"{p['goals_scored']:.0f} goals, {p['expected_goals']:.1f} xG"
    if group == "Creating chances":
        return f"{p['assists']:.0f} assists, {p['expected_assists']:.1f} xA"
    if group == "Defending":
        if p["position"] == "GK":
            return f"{p['saves']:.0f} saves, {p['clean_sheets']:.0f} clean sheets"
        return f"{p['clean_sheets']:.0f} clean sheets, {p['goals_conceded']:.0f} conceded"
    return ""


# ---------- percentiles against players in the same position ----------
PERCENTILE_STATS = {
    "FWD": ["goals_p90", "xg_p90", "assists_p90", "xa_p90", "threat_p90", "minutes_share"],
    "MID": ["goals_p90", "xa_p90", "creativity_p90", "threat_p90", "defence_p90", "minutes_share"],
    "DEF": ["defence_p90", "clean_sheets_p90", "conceded_p90", "xa_p90", "influence_p90", "minutes_share"],
    "GK": ["saves_p90", "clean_sheets_p90", "conceded_p90", "influence_p90", "bps_p90", "minutes_share"],
}
STAT_NAMES = {
    "goals_p90": "Goals per 90", "xg_p90": "xG per 90", "assists_p90": "Assists per 90",
    "xa_p90": "xA per 90", "threat_p90": "Attacking threat", "creativity_p90": "Chance creation",
    "defence_p90": "Defensive actions per 90", "clean_sheets_p90": "Clean sheets per 90",
    "conceded_p90": "Goals conceded per 90", "influence_p90": "Match influence",
    "bps_p90": "Match rating points", "saves_p90": "Saves per 90", "minutes_share": "Minutes played",
}
defensive_actions = now[["tackles", "clearances_blocks_interceptions", "recoveries"]].fillna(0).sum(axis=1)
X_raw["defence_p90"] = defensive_actions / now["minutes"].clip(lower=1) * 90

percentiles = pd.DataFrame(index=now.index)
for position, stats in PERCENTILE_STATS.items():
    group = has_value & (now["position"] == position)
    for stat in stats:
        rank = X_raw.loc[group, stat].rank(pct=True) * 100
        if stat == "conceded_p90":          # fewer goals conceded is better
            rank = 100 - rank + 100 / group.sum()
        percentiles.loc[group, stat] = rank.round()


# ---------- value for every past season (value history chart) ----------
seasons["model_value"] = np.exp(predict_log_value(make_features(seasons)))
seasons.loc[seasons["minutes"] < MIN_MINUTES_TRAIN, "model_value"] = None

training = pd.read_csv(PROCESSED / "training.csv")
transfermarkt = training.groupby("code").apply(
    lambda g: [{"year": int(y), "value": float(v)} for y, v in zip(g["tm_year"], g["market_value"])],
    include_groups=False).to_dict()

SHOW = ["minutes", "starts", "goals_scored", "assists", "expected_goals", "expected_assists",
        "clean_sheets", "goals_conceded", "saves", "penalties_saved", "tackles",
        "clearances_blocks_interceptions", "recoveries", "yellow_cards", "red_cards"]

# ---------- put it all together ----------
low, high = metrics["range_low"], metrics["range_high"]
players = []
for i, (index, p) in enumerate(now.iterrows()):
    valued = bool(has_value[index])

    reasons = {}
    for feature, amount in zip(FEATURES, push[i]):
        reasons[GROUP[feature]] = reasons.get(GROUP[feature], 0) + amount
    reasons = sorted(reasons.items(), key=lambda r: -abs(r[1]))[:7]

    history = []
    for _, s in seasons[seasons["code"] == p["code"]].sort_values("season").iterrows():
        row = {"season": s["season"], "team": s["team_name"], "team_pos": clean(s["team_pos"], 0)}
        row.update({col: clean(s[col]) for col in SHOW})
        if s["season"] == CURRENT_SEASON:
            row["model_value"] = clean(p["value"], 0) if valued else None
        else:
            row["model_value"] = clean(s["model_value"], 0)
        history.append(row)

    players.append({
        "id": int(p["code"]), "name": p["name"], "short": p["web_name"], "team": p["team_name"],
        "pos": p["position"], "age": clean(p["age"], 1),
        "born": None if pd.isna(p["birth_date"]) else str(p["birth_date"])[:10],
        "minutes": int(p["minutes"]),
        "value": clean(p["value"], 0) if valued else None,
        "low": clean(p["value"] * math.exp(low), 0) if valued else None,
        "high": clean(p["value"] * math.exp(high), 0) if valued else None,
        "why": [{"f": g, "pct": clean((math.exp(a) - 1) * 100, 1), "x": describe(g, p)} for g, a in reasons] if valued else [],
        "pct": [{"k": s, "f": STAT_NAMES[s], "p": clean(percentiles.loc[index, s], 0)} for s in PERCENTILE_STATS[p["position"]]] if valued else [],
        "calc": {f: clean(v, 4) for f, v in X.loc[index].items()},   # starting point for the value calculator
        "seasons": history,
        "tm": transfermarkt.get(p["code"], []),
    })

output = {"gameweeks": gameweeks, "average_value": round(average_value), "players": players}
json.dump(output, open(PROCESSED / "players.json", "w"), allow_nan=False)

top = sorted([p for p in players if p["value"]], key=lambda p: -p["value"])[:10]
for p in top:
    print(f"  {p['name']:<24} {p['team']:<12} €{p['value'] / 1e6:.1f}m")
