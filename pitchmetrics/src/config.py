# Settings used by every step of the project.
# Change things here instead of inside the scripts.

from datetime import date
from pathlib import Path

# ---------- brand and about ----------
SITE_NAME = "Pitch Metrics"
TAGLINE = "Player value, measured on the pitch."
BUILT_BY = "Viraaj Bhatia"
BUILT_BY_ROLE = "B.Tech IT student · Python, data and machine learning"
GITHUB_URL = ""        # e.g. "https://github.com/your-username"   (leave "" to hide the button)
LINKEDIN_URL = ""      # e.g. "https://www.linkedin.com/in/your-name"
SITE_URL = ""          # after hosting, e.g. "https://your-username.github.io/pitchmetrics/"
                       # (needed for the link preview image on WhatsApp / LinkedIn)

# ---------- folders ----------
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"              # downloaded files
PROCESSED = ROOT / "data" / "processed"  # cleaned tables made by the scripts
MODELS = ROOT / "models"                 # trained models and scores
SITE = ROOT / "site"                     # website template and club crests
DOCS = ROOT / "docs"                     # finished website (GitHub Pages serves this folder)

for folder in [RAW, PROCESSED, MODELS, DOCS]:
    folder.mkdir(parents=True, exist_ok=True)

# ---------- seasons ----------
TODAY = date.today()
# A Premier League season starts in August, so before August we are still in last year's season.
start_year = TODAY.year if TODAY.month >= 8 else TODAY.year - 1
CURRENT_SEASON = f"{start_year}-{str(start_year + 1)[2:]}"      # e.g. "2026-27"
LAST_SEASON = f"{start_year - 1}-{str(start_year)[2:]}"          # e.g. "2025-26"
FIRST_SEASON = 2019                                              # oldest season we download
ALL_SEASONS = [f"{y}-{str(y + 1)[2:]}" for y in range(FIRST_SEASON, start_year + 1)]

# ---------- data sources (all free, all on GitHub) ----------
FPL_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
TRANSFERMARKT_URL = "https://raw.githubusercontent.com/eordo/transfermarkt-data/master/premier_league"

# ---------- model settings ----------
MIN_MINUTES_TRAIN = 300     # a season needs this many minutes to be used for training
MIN_MINUTES_NOW = 270       # three full matches this season before we give a value
SHRINK_MINUTES = 450        # how strongly early-season numbers are pulled to the average
XGB_WEIGHT = 0.6            # final value = 60% XGBoost + 40% linear regression
EUR_TO_INR = 107.0          # for the rupee switch on the website

BIG_CLUBS = ["Arsenal", "Chelsea", "Liverpool", "Man City", "Man Utd", "Spurs"]

# FPL position codes
POSITIONS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

# Final league tables for older seasons (FPL's files only keep positions for recent seasons)
FINAL_TABLES = {
    "2019-20": ["Liverpool", "Man City", "Man Utd", "Chelsea", "Leicester", "Spurs", "Wolves", "Arsenal",
                "Sheffield Utd", "Burnley", "Southampton", "Everton", "Newcastle", "Crystal Palace",
                "Brighton", "West Ham", "Aston Villa", "Bournemouth", "Watford", "Norwich"],
    "2020-21": ["Man City", "Man Utd", "Liverpool", "Chelsea", "Leicester", "West Ham", "Spurs", "Arsenal",
                "Leeds", "Everton", "Aston Villa", "Newcastle", "Wolves", "Crystal Palace", "Southampton",
                "Brighton", "Burnley", "Fulham", "West Brom", "Sheffield Utd"],
    "2021-22": ["Man City", "Liverpool", "Chelsea", "Spurs", "Arsenal", "Man Utd", "West Ham", "Leicester",
                "Brighton", "Wolves", "Newcastle", "Crystal Palace", "Brentford", "Aston Villa",
                "Southampton", "Everton", "Leeds", "Burnley", "Watford", "Norwich"],
    "2022-23": ["Man City", "Arsenal", "Man Utd", "Newcastle", "Liverpool", "Brighton", "Aston Villa",
                "Spurs", "Brentford", "Fulham", "Crystal Palace", "Chelsea", "Wolves", "West Ham",
                "Bournemouth", "Nott'm Forest", "Everton", "Leicester", "Leeds", "Southampton"],
    "2023-24": ["Man City", "Arsenal", "Liverpool", "Aston Villa", "Spurs", "Chelsea", "Newcastle",
                "Man Utd", "West Ham", "Crystal Palace", "Brighton", "Bournemouth", "Fulham", "Wolves",
                "Everton", "Brentford", "Nott'm Forest", "Luton", "Burnley", "Sheffield Utd"],
    "2024-25": ["Liverpool", "Arsenal", "Man City", "Chelsea", "Newcastle", "Aston Villa",
                "Nott'm Forest", "Brighton", "Bournemouth", "Brentford", "Fulham", "Crystal Palace",
                "Everton", "West Ham", "Man Utd", "Wolves", "Spurs", "Leicester", "Ipswich",
                "Southampton"],
}
