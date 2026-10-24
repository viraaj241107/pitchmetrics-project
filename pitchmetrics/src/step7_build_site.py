# STEP 7 - Build the website.
#
# Puts all the data, the logo and the club crests into site/template.html and saves:
#   docs/index.html     - the real website (GitHub Pages serves the docs/ folder).
#                         Player photos load from the Premier League's image server.
#   docs/og-image.png   - the picture shown when the link is shared on WhatsApp / LinkedIn
#   site/preview.html   - same page with photos switched off, for previewing
#                         somewhere that blocks outside images.

import base64
import json
import shutil
from urllib.parse import quote

import pandas as pd

from config import (BUILT_BY, BUILT_BY_ROLE, CURRENT_SEASON, DOCS, EUR_TO_INR, GITHUB_URL, LINKEDIN_URL,
                    MODELS, PROCESSED, SITE, SITE_NAME, SITE_URL, TAGLINE, TODAY)

BRAND = SITE / "brand"

players = json.load(open(PROCESSED / "players.json"))
clubs = json.load(open(PROCESSED / "clubs.json"))
roles = json.load(open(PROCESSED / "roles.json"))
metrics = json.load(open(MODELS / "metrics.json"))
model = json.load(open(MODELS / "model_for_website.json"))
tests = pd.read_csv(MODELS / "test_predictions.csv")

# club crests are small images built straight into the page
crests = {}
for club in clubs["clubs"]:
    file = SITE / "crests" / (club["name"].replace("'", "").replace(" ", "_") + ".webp")
    if file.exists():
        crests[club["name"]] = "data:image/webp;base64," + base64.b64encode(file.read_bytes()).decode()
    else:
        print("no crest for", club["name"], "- add", file.name, "to site/crests")

data = {
    "meta": {"generated": str(TODAY), "season": CURRENT_SEASON, "gameweeks": players["gameweeks"],
             "average_value": players["average_value"], "eur_inr": EUR_TO_INR, "tagline": TAGLINE,
             "builder": {"name": BUILT_BY, "role": BUILT_BY_ROLE, "github": GITHUB_URL, "linkedin": LINKEDIN_URL}},
    "players": players["players"],
    "clubs": clubs["clubs"],
    "fixtures": clubs["fixtures"],
    "roles": roles,
    "metrics": metrics,
    "tests": [{"n": r.name, "s": r.season, "p": r.position, "a": round(r.market_value), "m": round(r.predicted)}
              for r in tests.itertuples()],
    "model": model,
    "crests": crests,
}

# ---------- logo, favicon and link preview ----------
logo_svg = (BRAND / "pitchmetrics-logo-on-dark.svg").read_text()      # header is dark
icon_svg = (BRAND / "pitchmetrics-icon.svg").read_text()
favicon = "data:image/svg+xml," + quote(icon_svg)

link_preview = ""
if SITE_URL:
    image = SITE_URL.rstrip("/") + "/og-image.png"
    link_preview = f'''<meta property="og:title" content="{SITE_NAME}">
<meta property="og:description" content="{TAGLINE} Premier League market values predicted by machine learning.">
<meta property="og:image" content="{image}">
<meta property="og:url" content="{SITE_URL}">
<meta name="twitter:card" content="summary_large_image">'''
shutil.copy(BRAND / "og-image.png", DOCS / "og-image.png")

template = (SITE / "template.html").read_text(encoding="utf-8")
template = (template.replace("/*LOGO*/", logo_svg).replace("/*ICON*/", icon_svg)
            .replace("/*FAVICON*/", favicon).replace("<!--OG-->", link_preview))

for remote_photos, path in [(True, DOCS / "index.html"), (False, SITE / "preview.html")]:
    data["meta"]["remote_photos"] = remote_photos
    text = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    page = template.replace("/*DATA*/", text)
    if not remote_photos:
        # the preview host adds its own <html>, <head> and <body>, so leave ours out
        for tag in ['<!doctype html>', '<html lang="en">', '<head>', '</head>', '<body>', '</body>', '</html>',
                    '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">']:
            page = page.replace(tag, "", 1)
    path.write_text(page, encoding="utf-8")
    print("saved", path.relative_to(SITE.parent), f"({path.stat().st_size // 1024} KB)")
