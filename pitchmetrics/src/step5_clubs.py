# STEP 5 - Clubs, league table, lineups and match ratings.
#
# - League table worked out from the finished fixtures
# - Each club's results, next match and last starting XI (arranged by formation)
# - A match rating for every player in every game, made from FPL's
#   Bonus Points System (BPS), which scores every action in a match:
#       rating = 6.0 + 0.06 * BPS   (kept between 3 and 10)
# - Nationality and exact position (e.g. "Left-Back") from Transfermarkt, when
#   we can find the player there
#
# Output: data/processed/clubs.json, and players.json gets match logs added

import json
import re

import pandas as pd
from rapidfuzz import fuzz
from unidecode import unidecode

from config import CURRENT_SEASON, PROCESSED, RAW

players_file = json.load(open(PROCESSED / "players.json"))
players = players_file["players"]
by_code = {p["id"]: p for p in players}

fpl = pd.read_csv(RAW / f"fpl_{CURRENT_SEASON}.csv")
teams = pd.read_csv(RAW / f"teams_{CURRENT_SEASON}.csv")
fixtures = pd.read_csv(RAW / f"fix_{CURRENT_SEASON}.csv")
matches = pd.read_csv(RAW / f"gw_{CURRENT_SEASON}.csv")

team_name = dict(zip(teams["id"], teams["name"]))
team_short = dict(zip(teams["id"], teams["short_name"]))
matches["code"] = matches["element"].map(dict(zip(fpl["id"], fpl["code"])))   # FPL id -> permanent code


def rating(bps, minutes):
    if minutes == 0:
        return None
    return round(min(10.0, max(3.0, 6.0 + 0.06 * bps)), 1)


matches["rating"] = [rating(b, m) for b, m in zip(matches["bps"], matches["minutes"])]


# ---------- nationality + exact position from Transfermarkt ----------
def clean_name(name):
    return " ".join(re.sub(r"[^a-z ]", " ", unidecode(str(name)).lower()).split())


tm = pd.concat([pd.read_csv(f) for f in sorted(RAW.glob("tm_*.csv")) if f.stat().st_size > 0])
tm = tm.sort_values("season").drop_duplicates("player_id", keep="last")   # newest record per player
tm["clean"] = tm["player_name"].apply(clean_name)
tm["born_year"] = tm["season"] - tm["age"]

for p in players:
    p["nat"], p["detail"], p["dpos"] = None, None, None
    name = clean_name(p["name"])
    if len(name.split()) < 2:
        continue
    surname = name.split()[-1]
    options = tm[tm["clean"].str.contains(surname, regex=False)]
    best, best_score = None, 0
    for _, row in options.iterrows():
        if p["born"] and abs(row["born_year"] - int(p["born"][:4])) > 1.5:
            continue
        s = fuzz.token_sort_ratio(name, row["clean"])
        if s > best_score:
            best, best_score = row, s
    if best is not None and best_score >= 90:
        p["nat"], p["detail"], p["dpos"] = best["nationality"], best["position"], best["pos"]
print("nationality found:", sum(1 for p in players if p["nat"]), "of", len(players))


# ---------- every player's matches this season ----------
for p in players:
    games = matches[matches["code"] == p["id"]].sort_values("GW")
    p["matches"] = []
    for _, g in games.iterrows():
        p["matches"].append({
            "gw": int(g["GW"]), "opp": team_short.get(g["opponent_team"]), "home": bool(g["was_home"]),
            "score": f"{int(g['team_h_score'])}-{int(g['team_a_score'])}", "min": int(g["minutes"]),
            "g": int(g["goals_scored"]), "a": int(g["assists"]), "r": g["rating"] if g["minutes"] else None,
        })
    played = [m for m in p["matches"] if m["min"] > 0]
    p["apps"] = len(played)
    p["avg_r"] = round(sum(m["r"] for m in played) / len(played), 2) if played else None
    p["s_goals"] = sum(m["g"] for m in played)
    p["s_assists"] = sum(m["a"] for m in played)


# ---------- league table ----------
finished = fixtures[fixtures["finished"] == True].sort_values("kickoff_time")
table = {t: {"p": 0, "w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0, "pts": 0, "form": []} for t in teams["id"]}
for _, f in finished.iterrows():
    for team, scored, conceded in [(f["team_h"], f["team_h_score"], f["team_a_score"]),
                                   (f["team_a"], f["team_a_score"], f["team_h_score"])]:
        row = table[team]
        row["p"] += 1
        row["gf"] += int(scored)
        row["ga"] += int(conceded)
        if scored > conceded:
            row["w"] += 1; row["pts"] += 3; row["form"].append("W")
        elif scored == conceded:
            row["d"] += 1; row["pts"] += 1; row["form"].append("D")
        else:
            row["l"] += 1; row["form"].append("L")

order = sorted(table, key=lambda t: (-table[t]["pts"], -(table[t]["gf"] - table[t]["ga"]), -table[t]["gf"]))


# ---------- last starting XI arranged by formation ----------
# Left-to-right order inside a line comes from Transfermarkt's exact position.
SIDE = {"LB": 0, "LWB": 0, "LM": 0, "LW": 0, "RB": 2, "RWB": 2, "RM": 2, "RW": 2}
ATTACKING = {"AM", "LW", "RW", "LM", "RM", "SS"}


def last_lineup(team_id, fixture_id):
    starters = matches[(matches["fixture"] == fixture_id) & (matches["team"] == team_name[team_id]) & (matches["starts"] == 1)]
    lines = {"GK": [], "DEF": [], "DM": [], "AM": [], "FWD": []}
    for _, s in starters.iterrows():
        player = by_code.get(s["code"], {})
        exact = player.get("dpos")
        line = s["position"]
        if line == "MID":
            line = "AM" if exact in ATTACKING else "DM"
        lines[line].append({"id": int(s["code"]), "side": SIDE.get(exact, 1), "r": s["rating"],
                            "g": int(s["goals_scored"]), "a": int(s["assists"])})
    # two wingers behind one striker = a front three
    if len(lines["FWD"]) == 1 and len(lines["AM"]) == 2 and all(x["side"] != 1 for x in lines["AM"]):
        lines["FWD"] += lines["AM"]
        lines["AM"] = []
    rows = [sorted(lines[k], key=lambda x: x["side"]) for k in ["GK", "DEF", "DM", "AM", "FWD"] if lines[k]]
    formation = "-".join(str(len(r)) for r in rows[1:])
    return formation, rows


clubs = []
for position, team_id in enumerate(order, start=1):
    name = team_name[team_id]
    t = table[team_id]
    team_games = fixtures[(fixtures["team_h"] == team_id) | (fixtures["team_a"] == team_id)].sort_values("kickoff_time")
    done = team_games[team_games["finished"] == True]
    todo = team_games[team_games["finished"] != True]

    results = []
    for _, f in done.iterrows():
        home = f["team_h"] == team_id
        results.append({"gw": int(f["event"]), "opp": team_name[f["team_a"] if home else f["team_h"]], "home": bool(home),
                        "gf": int(f["team_h_score"] if home else f["team_a_score"]),
                        "ga": int(f["team_a_score"] if home else f["team_h_score"])})

    next_match = None
    if len(todo) > 0:
        f = todo.iloc[0]
        home = f["team_h"] == team_id
        next_match = {"gw": int(f["event"]), "opp": team_name[f["team_a"] if home else f["team_h"]],
                      "home": bool(home), "kickoff": f["kickoff_time"]}

    lineup = None
    if len(done) > 0:
        last = done.iloc[-1]
        home = last["team_h"] == team_id
        formation, rows = last_lineup(team_id, last["id"])
        lineup = {"gw": int(last["event"]), "opp": team_name[last["team_a"] if home else last["team_h"]],
                  "home": bool(home), "formation": formation, "lines": rows}

    squad = [p["id"] for p in players if p["team"] == name]
    clubs.append({
        "id": int(team_id), "name": name, "short": team_short[team_id], "pos": position,
        "p": t["p"], "w": t["w"], "d": t["d"], "l": t["l"], "gf": t["gf"], "ga": t["ga"], "pts": t["pts"],
        "form": t["form"][-5:], "results": results, "next": next_match, "xi": lineup, "squad": squad,
        "squad_value": round(sum(by_code[c]["value"] or 0 for c in squad)),
    })

fixture_list = []
for _, f in fixtures.sort_values("kickoff_time").iterrows():
    fixture_list.append({"gw": int(f["event"]), "h": team_name[f["team_h"]], "a": team_name[f["team_a"]],
                         "ko": f["kickoff_time"], "done": bool(f["finished"]),
                         "hs": None if pd.isna(f["team_h_score"]) else int(f["team_h_score"]),
                         "as": None if pd.isna(f["team_a_score"]) else int(f["team_a_score"])})

json.dump(players_file, open(PROCESSED / "players.json", "w"), allow_nan=False)
json.dump({"clubs": clubs, "fixtures": fixture_list}, open(PROCESSED / "clubs.json", "w"))
for c in clubs[:3]:
    print(c["pos"], c["name"], c["pts"], "pts", c["xi"]["formation"] if c["xi"] else "")
