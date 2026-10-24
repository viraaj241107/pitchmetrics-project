# Pitch Metrics

![Pitch Metrics](site/brand/pitchmetrics-logo-on-light.png)

**Player value, measured on the pitch.**

A machine-learning project that predicts footballers' market values from how they are
playing *this season*, and shows them on a website with player pages, club lineups,
player roles and a live "what-if" value calculator.

## What it does

- **Predicts market values** with a model trained on 525 real Transfermarkt values (2019-2025)
- **Explains each value**: "Goal scoring +50%, Age +44%, Club +33%..." (SHAP)
- **Finds player roles** with K-means clustering ("Playmaker", "Ball-winner", "Finisher"...)
- **Value calculator**: move sliders (goals, minutes, age, club) and the same model re-prices the player live in the browser
- **Club pages**: league table, results, next match, last starting XI on a pitch with match ratings
- **Updates itself every week** with GitHub Actions

## Results

Tested on players the model had never seen:

| Test | R² (log value) | Typical error | Within ±50% |
|---|---|---|---|
| 5-fold cross-validation | 0.71 | 34% | 70% |
| Train on 2019-24, test on 2024-25 | 0.82 | 31% | 79% |

The blend of XGBoost + linear regression beats each model on its own.

## Project structure

```
src/
  config.py              settings (seasons, folders, model settings)
  step1_download.py      download stats and market values
  step2_build_dataset.py join stats with market values (name + age matching)
  features.py            turn stats into per-90 features
  step3_train.py         train + test XGBoost and linear regression, export model
  step4_predict.py       value every current player, explain with SHAP
  step5_clubs.py         league table, lineups, match ratings
  step6_roles.py         K-means player roles + PCA role map
  step7_build_site.py    build the website into docs/index.html
  run_all.py             runs all steps in order
site/template.html       the website (HTML/CSS/JavaScript)
site/crests/             club badges
site/brand/              Pitch Metrics logo (SVG/PNG), icon and link-preview image
docs/index.html          the finished website (served by GitHub Pages)
.github/workflows/       weekly automatic update
```

## Run it on your computer

```bash
pip install -r requirements.txt
cd src
python run_all.py
```

Then open `docs/index.html` in your browser. A full run takes about 2 minutes.

## Put it online (free) and keep it updating by itself

1. **Create a GitHub repository** (for example `pitchmetrics`) and upload this whole folder,
   including the hidden `.github` folder.
2. **Turn on GitHub Pages**: repo → *Settings* → *Pages* → Source: *Deploy from a branch* →
   Branch: `main`, Folder: `/docs` → Save.
   After a minute your site is live at `https://<your-username>.github.io/pitchmetrics/`.
3. **Allow the weekly job to save changes**: *Settings* → *Actions* → *General* →
   *Workflow permissions* → *Read and write permissions* → Save.
4. **Test it now**: *Actions* tab → *Update Pitch Metrics* → *Run workflow*.
   It downloads the newest stats, retrains, re-prices everyone and commits the new
   `docs/index.html`. GitHub Pages then publishes it automatically.

From then on it runs **every Tuesday at 10:00 IST**, after the weekend's matches.
The stats source (the FPL archive) updates after every gameweek, so values, ratings,
lineups and the table move on their own.

**Your details and link preview**: open `src/config.py` and fill in `GITHUB_URL`, `LINKEDIN_URL`
and `SITE_URL` (your GitHub Pages address). The GitHub/LinkedIn buttons appear on the "How it works"
page, and sharing the link on WhatsApp or LinkedIn shows the Pitch Metrics preview card.

**Player photos** load straight from the Premier League's image server when the site is
online. If a photo is missing, a grey silhouette is shown instead.

**New season**: nothing to change. `config.py` works out the season from today's date. Before
the first match of a season the job stops safely and the old site stays up. If a newly
promoted club has no badge, add its image to `site/crests/` (named like `Hull_City.webp`).

**Rupees**: the ₹ switch uses a fixed rate set in `config.py` (`EUR_TO_INR`). Update it now and then.

## Data sources

- Player stats: official Fantasy Premier League data, archived by
  [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League)
- Market values: Transfermarkt, via [eordo/transfermarkt-data](https://github.com/eordo/transfermarkt-data)
- Club badges: [luukhopman/football-logos](https://github.com/luukhopman/football-logos)

## Limits

- Superstar values run low: few €100m+ players changed clubs in the training years.
- Market values are only known for players who moved clubs, so training leans towards them.
- Free data has no dribbles or progressive passes; contracts and injuries are not included.
- Early in a season values move a lot week to week.

See **LEARN.md** for an explanation of every idea used in the project.
