#!/usr/bin/env python3
"""
Fetch completed WC 2026 scores from The Odds API and write them into results.csv.
Run daily before build_html.py.
"""

import csv
import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY  = os.environ.get("ODDS_API_KEY", "")
SPORT    = "soccer_fifa_world_cup"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "archive-3", "results.csv")

# Odds API team names → CSV team names
NAME_MAP = {
    "USA":                      "United States",
    "Bosnia & Herzegovina":     "Bosnia and Herzegovina",
    "Korea Republic":           "South Korea",
    "DR Congo":                 "DR Congo",
    "Cote d'Ivoire":            "Ivory Coast",
    "Côte d'Ivoire":            "Ivory Coast",
    "Iran (Islamic Republic)":  "Iran",
}


def normalise(name: str) -> str:
    return NAME_MAP.get(name, name)


def fetch_scores(days_from: int = 3) -> list:
    url = f"https://api.the-odds-api.com/v4/sports/{SPORT}/scores/"
    r = requests.get(url, params={"apiKey": API_KEY, "daysFrom": days_from}, timeout=15)
    r.raise_for_status()
    return r.json()


def build_lookup(games: list) -> dict:
    """Return {(home, away): (home_score, away_score)} for completed games only."""
    lookup = {}
    for g in games:
        if not g.get("completed") or not g.get("scores"):
            continue
        home = normalise(g["home_team"])
        away = normalise(g["away_team"])
        score_map = {normalise(s["name"]): s["score"] for s in g["scores"]}
        hs = score_map.get(home)
        as_ = score_map.get(away)
        if hs is not None and as_ is not None:
            lookup[(home, away)] = (hs, as_)
    return lookup


def update_csv(lookup: dict) -> int:
    with open(CSV_PATH, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    updated = 0
    for row in rows:
        # Only touch rows with missing scores
        if row["home_score"] not in ("", "NA") and row["away_score"] not in ("", "NA"):
            continue
        key = (row["home_team"], row["away_team"])
        if key in lookup:
            row["home_score"], row["away_score"] = lookup[key]
            updated += 1
            print(f"  ✓  {row['home_team']} {lookup[key][0]}-{lookup[key][1]} {row['away_team']}  ({row['date']})")

    with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return updated


def main():
    if not API_KEY:
        print("ERROR: ODDS_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    print("Fetching WC 2026 scores…")
    games  = fetch_scores(days_from=3)
    lookup = build_lookup(games)
    print(f"  {len(lookup)} completed games found in API window")

    updated = update_csv(lookup)
    print(f"  {updated} new score(s) written to CSV")


if __name__ == "__main__":
    main()
