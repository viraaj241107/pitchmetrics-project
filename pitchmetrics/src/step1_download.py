# STEP 1 - Download the raw data.
#
# 1. Fantasy Premier League (FPL) stats for every player, one file per season.
#    Archived on GitHub by the vaastav/Fantasy-Premier-League project.
# 2. Match-by-match stats and fixtures for the current season.
# 3. Transfermarkt market values for players who joined or left a Premier League
#    club each summer (eordo/transfermarkt-data on GitHub).

import requests

from config import ALL_SEASONS, CURRENT_SEASON, FPL_URL, RAW, TODAY, TRANSFERMARKT_URL


def download(url, filename):
    response = requests.get(url, timeout=60)
    if response.status_code != 200:
        print("  not found:", url)
        return False
    (RAW / filename).write_bytes(response.content)
    print("  saved", filename)
    return True


print("FPL season stats")
for season in ALL_SEASONS:
    download(f"{FPL_URL}/{season}/players_raw.csv", f"fpl_{season}.csv")
    download(f"{FPL_URL}/{season}/teams.csv", f"teams_{season}.csv")

print("This season's matches")
download(f"{FPL_URL}/{CURRENT_SEASON}/gws/merged_gw.csv", f"gw_{CURRENT_SEASON}.csv")
download(f"{FPL_URL}/{CURRENT_SEASON}/fixtures.csv", f"fix_{CURRENT_SEASON}.csv")

print("Transfermarkt values")
for year in range(2019, TODAY.year + 1):
    download(f"{TRANSFERMARKT_URL}/{year}.csv", f"tm_{year}.csv")
