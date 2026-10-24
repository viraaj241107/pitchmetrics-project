# STEP 6 - Player roles with clustering (unsupervised machine learning).
#
# Two players can both be "midfielders" but play completely differently: one
# wins the ball back, another creates chances. K-means clustering groups players
# with similar per-90 stat profiles, without us telling it what the groups are.
#
#   1. Take each player's last ~12 months (last season + this season)
#   2. Scale every stat (StandardScaler) so goals and tackles count equally
#   3. K-means -> k groups per position
#   4. Name each group by the stat it is strongest in
#   5. PCA squeezes the stats into 2 numbers per player -> the "role map" chart
#   6. Similar players = the 5 nearest players in the scaled stat space
#
# Output: data/processed/roles.json

import json

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import CURRENT_SEASON, LAST_SEASON, PROCESSED

MIN_MINUTES = 900   # about 10 full games

# stats used for each position, the number of groups, and the role name for each "strongest stat"
SETUP = {
    "FWD": {"k": 3, "stats": ["goals", "xg", "assists", "xa", "threat", "creativity"],
            "names": {"goals": "Finisher", "xg": "Penalty-box striker", "threat": "Penalty-box striker",
                      "assists": "Creative forward", "xa": "Creative forward", "creativity": "Creative forward"}},
    "MID": {"k": 4, "stats": ["goals", "xg", "xa", "creativity", "threat", "defending", "influence"],
            "names": {"goals": "Goal-scoring midfielder", "xg": "Goal-scoring midfielder", "threat": "Attacking midfielder",
                      "xa": "Playmaker", "creativity": "Playmaker", "defending": "Ball-winner", "influence": "Box-to-box"}},
    "DEF": {"k": 3, "stats": ["defending", "xa", "creativity", "threat", "clean_sheets", "influence"],
            "names": {"defending": "Stopper", "xa": "Attacking full-back", "creativity": "Attacking full-back",
                      "threat": "Set-piece threat", "clean_sheets": "Organiser", "influence": "Ball-playing defender"}},
    "GK": {"k": 2, "stats": ["saves", "clean_sheets", "conceded", "influence"],
           "names": {"saves": "Shot-stopper", "influence": "Shot-stopper", "clean_sheets": "Clean-sheet keeper",
                     "conceded": "Shot-stopper"}},
}
FALLBACK = {"FWD": "All-round forward", "MID": "Squad midfielder", "DEF": "Steady defender", "GK": "Back-up keeper"}

seasons = pd.read_csv(PROCESSED / "player_seasons.csv")
recent = seasons[seasons["season"].isin([LAST_SEASON, CURRENT_SEASON])]

# add up last season + this season for each player
totals = recent.groupby("code")[["minutes", "goals_scored", "expected_goals", "assists", "expected_assists",
                                 "threat", "creativity", "influence", "clean_sheets", "goals_conceded",
                                 "saves", "tackles", "clearances_blocks_interceptions", "recoveries"]].sum()
current = seasons[seasons["season"] == CURRENT_SEASON].set_index("code")
totals["position"] = current["position"]
totals = totals.dropna(subset=["position"])
# only players still in the league (step 4 already removed those who left)
current_players = [p["id"] for p in json.load(open(PROCESSED / "players.json"))["players"]]
totals = totals[totals.index.isin(current_players)]
totals = totals[totals["minutes"] >= MIN_MINUTES]

per90 = pd.DataFrame(index=totals.index)
m = totals["minutes"] / 90
per90["goals"] = totals["goals_scored"] / m
per90["xg"] = totals["expected_goals"] / m
per90["assists"] = totals["assists"] / m
per90["xa"] = totals["expected_assists"] / m
per90["threat"] = totals["threat"] / m
per90["creativity"] = totals["creativity"] / m
per90["influence"] = totals["influence"] / m
per90["clean_sheets"] = totals["clean_sheets"] / m
per90["conceded"] = -totals["goals_conceded"] / m          # minus, so higher = better
per90["saves"] = totals["saves"] / m
per90["defending"] = (totals["tackles"] + totals["clearances_blocks_interceptions"] + totals["recoveries"]) / m

output = {"players": {}, "roles": {}}
for position, setup in SETUP.items():
    group = per90[totals["position"] == position][setup["stats"]]
    scaled = StandardScaler().fit_transform(group)

    kmeans = KMeans(n_clusters=setup["k"], n_init=20, random_state=42).fit(scaled)
    map_2d = PCA(n_components=2, random_state=42).fit_transform(scaled)

    # name each cluster after the stat its centre is highest in (each name used once)
    centres = pd.DataFrame(kmeans.cluster_centers_, columns=setup["stats"])
    names, used = {}, set()
    for c in centres.max(axis=1).sort_values(ascending=False).index:
        name = FALLBACK[position]
        if centres.loc[c].max() > 0.2:
            for stat in centres.loc[c].sort_values(ascending=False).index:
                if setup["names"][stat] not in used:
                    name = setup["names"][stat]
                    break
        if name in used:
            name = name + " (2)"
        used.add(name)
        names[c] = name

    output["roles"][position] = []
    for c, name in names.items():
        strengths = centres.loc[c].sort_values(ascending=False)
        output["roles"][position].append({
            "name": name,
            "count": int((kmeans.labels_ == c).sum()),
            "strong": [s for s in strengths.index[:3] if strengths[s] > 0],
            "weak": [s for s in strengths.index[-2:] if strengths[s] < 0],
        })

    # 5 most similar players = smallest distance in the scaled stat space
    codes = group.index.tolist()
    for i, code in enumerate(codes):
        distance = np.sqrt(((scaled - scaled[i]) ** 2).sum(axis=1))
        nearest = [codes[j] for j in np.argsort(distance)[1:6]]
        output["players"][int(code)] = {
            "role": names[kmeans.labels_[i]],
            "x": round(float(map_2d[i, 0]), 3), "y": round(float(map_2d[i, 1]), 3),
            "similar": [int(n) for n in nearest],
        }

    print(position, {n: int((kmeans.labels_ == c).sum()) for c, n in names.items()})

json.dump(output, open(PROCESSED / "roles.json", "w"))
