# STEP 2 - Build the dataset.
#
# a) Put every FPL season into one table: one row = one player in one season.
# b) Load Transfermarkt market values (the "answer" the model learns from).
# c) Match the two by player name + age, so we know e.g.
#    "Declan Rice's 2022-23 season"  ->  "worth €90m in summer 2023".
#
# Output: data/processed/player_seasons.csv  (all players, all seasons)
#         data/processed/training.csv        (only the rows that have a market value)

import re

import pandas as pd
from rapidfuzz import fuzz
from unidecode import unidecode

from config import ALL_SEASONS, FINAL_TABLES, POSITIONS, PROCESSED, RAW

STAT_COLUMNS = [
    "minutes", "starts", "goals_scored", "assists", "expected_goals", "expected_assists",
    "clean_sheets", "goals_conceded", "saves", "penalties_saved", "yellow_cards", "red_cards",
    "bonus", "bps", "influence", "creativity", "threat", "total_points",
    "tackles", "clearances_blocks_interceptions", "recoveries",   # only exist from 2025-26
]


def clean_name(name):
    """'Martin Ødegaard' -> 'martin odegaard' so names can be compared."""
    name = unidecode(str(name)).lower()
    name = re.sub(r"[^a-z ]", " ", name)
    return " ".join(name.split())


# ---------- a) FPL seasons ----------
seasons = []
for season in ALL_SEASONS:
    players = pd.read_csv(RAW / f"fpl_{season}.csv")
    teams = pd.read_csv(RAW / f"teams_{season}.csv")

    players = players[players["element_type"].isin(POSITIONS.keys())].copy()   # drop managers
    players["season"] = season
    players["position"] = players["element_type"].map(POSITIONS)
    players["team_name"] = players["team"].map(dict(zip(teams["id"], teams["name"])))

    # league position of the player's team
    if season in FINAL_TABLES:
        table = FINAL_TABLES[season]
        players["team_pos"] = players["team_name"].apply(lambda t: table.index(t) + 1)
    else:
        players["team_pos"] = players["team"].map(dict(zip(teams["id"], teams["position"])))

    players["name"] = players["first_name"] + " " + players["second_name"]
    if "known_name" in players:   # e.g. "João Pedro" instead of his full legal name
        players["name"] = players["known_name"].fillna(players["name"])
        players.loc[players["name"] == "", "name"] = players["first_name"] + " " + players["second_name"]

    for col in STAT_COLUMNS + ["birth_date"]:
        if col not in players:
            players[col] = None   # older seasons do not have every column

    keep = ["season", "code", "name", "web_name", "first_name", "second_name", "birth_date",
            "position", "team_name", "team_pos"] + STAT_COLUMNS
    seasons.append(players[keep])

all_seasons = pd.concat(seasons, ignore_index=True)

# Birth date is missing in old seasons - copy it from any season where the same player has it.
# ("code" is FPL's id for a player and never changes between seasons.)
all_seasons["birth_date"] = pd.to_datetime(all_seasons["birth_date"], errors="coerce")
birth_dates = all_seasons.dropna(subset=["birth_date"]).groupby("code")["birth_date"].first()
all_seasons["birth_date"] = all_seasons["code"].map(birth_dates)

all_seasons.to_csv(PROCESSED / "player_seasons.csv", index=False)
print("player-seasons:", len(all_seasons))


# ---------- b) Transfermarkt values ----------
tm_files = []
for file in sorted(RAW.glob("tm_*.csv")):
    if file.stat().st_size > 0:
        tm_files.append(pd.read_csv(file))
tm = pd.concat(tm_files, ignore_index=True)
tm = tm[(tm["window"] == "summer") & (tm["market_value"] > 0)]
tm = tm.drop_duplicates(["player_id", "season"])      # a player can appear as "in" and "out"
tm["season"] = tm["season"].astype(int)
tm["stats_season"] = tm["season"].apply(lambda y: f"{y - 1}-{str(y)[2:]}")   # summer 2025 -> 2024-25
tm = tm[tm["stats_season"].isin(ALL_SEASONS)]
print("Transfermarkt values:", len(tm))


# ---------- c) match players ----------
def find_player(tm_row):
    """Return the index of the FPL row for this Transfermarkt player, or None."""
    candidates = all_seasons[all_seasons["season"] == tm_row["stats_season"]]
    tm_name = clean_name(tm_row["player_name"])
    surname = tm_name.split()[-1]

    best_index, best_score = None, 0
    for index, player in candidates.iterrows():
        fpl_name = clean_name(player["name"])
        full_name = clean_name(f"{player['first_name']} {player['second_name']}")
        if surname not in fpl_name and surname not in full_name:
            continue
        score = max(fuzz.token_set_ratio(tm_name, fpl_name), fuzz.token_set_ratio(tm_name, full_name))

        # check the age if we know the birth date
        if pd.notna(player["birth_date"]):
            age = (pd.Timestamp(f"{tm_row['season']}-08-01") - player["birth_date"]).days / 365.25
            if abs(age - tm_row["age"]) > 1.2:
                continue

        if score > best_score:
            best_index, best_score = index, score

    return best_index if best_score >= 90 else None


rows = []
for _, tm_row in tm.iterrows():
    index = find_player(tm_row)
    if index is not None:
        row = all_seasons.loc[index].copy()
        row["market_value"] = tm_row["market_value"]
        row["tm_age"] = tm_row["age"]
        row["tm_player_id"] = tm_row["player_id"]
        row["tm_year"] = tm_row["season"]
        rows.append(row)

training = pd.DataFrame(rows).drop_duplicates(["code", "season"])
training.to_csv(PROCESSED / "training.csv", index=False)
print("matched rows with a market value:", len(training))
