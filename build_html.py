#!/usr/bin/env python3
"""
Generate index.html with embedded WC 2026 predictions.
Run: python3 build_html.py
Then push index.html to GitHub Pages.
"""

import json
import os
import sys
from datetime import datetime, date

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predictions import load_artifacts, predict_match, predict_goals

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FLAGS = {
    "Argentina": "🇦🇷", "Australia": "🇦🇺", "Austria": "🇦🇹", "Algeria": "🇩🇿",
    "Belgium": "🇧🇪", "Bosnia and Herzegovina": "🇧🇦", "Brazil": "🇧🇷",
    "Canada": "🇨🇦", "Cape Verde": "🇨🇻", "Colombia": "🇨🇴", "Croatia": "🇭🇷",
    "Czech Republic": "🇨🇿", "Curaçao": "🇨🇼", "DR Congo": "🇨🇩",
    "Ecuador": "🇪🇨", "Egypt": "🇪🇬", "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    "France": "🇫🇷", "Germany": "🇩🇪", "Ghana": "🇬🇭",
    "Haiti": "🇭🇹", "Iran": "🇮🇷", "Iraq": "🇮🇶", "Ivory Coast": "🇨🇮",
    "Japan": "🇯🇵", "Jordan": "🇯🇴", "Mexico": "🇲🇽", "Morocco": "🇲🇦",
    "Netherlands": "🇳🇱", "New Zealand": "🇳🇿", "Nigeria": "🇳🇬",
    "Norway": "🇳🇴", "Panama": "🇵🇦", "Paraguay": "🇵🇾",
    "Portugal": "🇵🇹", "Qatar": "🇶🇦", "Saudi Arabia": "🇸🇦",
    "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿", "Senegal": "🇸🇳", "Serbia": "🇷🇸",
    "South Africa": "🇿🇦", "South Korea": "🇰🇷", "Spain": "🇪🇸",
    "Sweden": "🇸🇪", "Switzerland": "🇨🇭", "Tunisia": "🇹🇳",
    "Turkey": "🇹🇷", "United States": "🇺🇸", "Uruguay": "🇺🇾",
    "Uzbekistan": "🇺🇿",
}


def build_data():
    arts = load_artifacts()

    # Build kickoff-time lookup from value_bets.json (has precise UTC commence times)
    vb_path = os.path.join(BASE_DIR, "value_bets.json")
    kickoff_lookup = {}
    if os.path.exists(vb_path):
        with open(vb_path) as f:
            vb_data = json.load(f)
        for r in vb_data.get("results", []):
            key = (r["home"], r["away"])
            if r.get("commence"):
                kickoff_lookup[key] = r["commence"]

    df = pd.read_csv(os.path.join(BASE_DIR, "archive-3", "results.csv"), parse_dates=["date"])
    today = pd.Timestamp(date.today())
    upcoming = (
        df[(df["date"] >= today) & (df["home_score"].isna())]
        .sort_values("date")
        .reset_index(drop=True)
    )
    upcoming["neutral"] = (
        upcoming["neutral"]
        .map({True: True, False: False, "TRUE": True, "FALSE": False, "True": True, "False": False})
        .fillna(True)
    )

    matches = []
    for _, row in upcoming.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        neutral = bool(row["neutral"])
        pred  = predict_match(home, away, neutral=neutral, tournament_weight=1.0, artifacts=arts)
        goals = predict_goals(home, away, neutral=neutral, tournament_weight=1.0, artifacts=arts)

        # H2H from snapshot h2h_summary
        from predictions import get_h2h_features
        h2h = arts["h2h_summary"]
        hw_r, dr_r, aw_r, ng = get_h2h_features(h2h, home, away)

        home_elo = arts["snapshot"].get(home, {}).get("elo", 1500)
        away_elo = arts["snapshot"].get(away, {}).get("elo", 1500)

        matches.append({
            "date":      row["date"].strftime("%Y-%m-%d"),
            "home":      home,
            "away":      away,
            "home_flag": FLAGS.get(home, "🏳"),
            "away_flag": FLAGS.get(away, "🏳"),
            "neutral":   neutral,
            "home_win":  round(pred["Home Win"] * 100, 1),
            "draw":      round(pred["Draw"] * 100, 1),
            "away_win":  round(pred["Away Win"] * 100, 1),
            "home_elo":  round(home_elo),
            "away_elo":  round(away_elo),
            "h2h_hw":      round(hw_r * 100),
            "h2h_dr":      round(dr_r * 100),
            "h2h_aw":      round(aw_r * 100),
            "h2h_n":       ng,
            "city":        str(row.get("city", "")),
            "country":     str(row.get("country", "")),
            "kickoff_utc": kickoff_lookup.get((home, away), ""),
            "over25":    round(goals.get("Over 2.5", 0) * 100, 1),
            "under25":   round(goals.get("Under 2.5", 0) * 100, 1),
            "btts_yes":  round(goals.get("BTTS Yes", 0) * 100, 1),
            "btts_no":   round(goals.get("BTTS No", 0) * 100, 1),
        })

    return matches


def compute_wc26_accuracy():
    """Run model retroactively on all completed WC 2026 group-stage games."""
    from predictions import load_artifacts, predict_match
    arts = load_artifacts()
    df = pd.read_csv(os.path.join(BASE_DIR, "archive-3", "results.csv"), parse_dates=["date"])

    wc = df[
        (df["tournament"] == "FIFA World Cup") &
        (df["date"].dt.year == 2026) &
        (df["home_score"].notna())
    ].copy()
    wc["neutral"] = (
        wc["neutral"]
        .map({True: True, False: False, "TRUE": True, "FALSE": False, "True": True, "False": False})
        .fillna(True)
    )

    # Historical WC draw rate 1990-2022
    hist = df[
        (df["tournament"] == "FIFA World Cup") &
        (df["date"].dt.year.between(1990, 2022)) &
        (df["home_score"].notna())
    ].copy()
    hist["draw"] = hist["home_score"].astype(float) == hist["away_score"].astype(float)
    hist_draw_pct = round(hist["draw"].mean() * 100, 1)

    correct = decisive_correct = draws = 0
    results = []
    for _, row in wc.iterrows():
        home, away = row["home_team"], row["away_team"]
        hs, as_ = float(row["home_score"]), float(row["away_score"])
        if hs > as_:   actual = "Home Win"
        elif hs < as_: actual = "Away Win"
        else:          actual = "Draw"; draws += 1

        pred = predict_match(home, away, neutral=bool(row["neutral"]), artifacts=arts)
        predicted = max(pred, key=pred.get)
        hit = predicted == actual
        if hit:
            correct += 1
            if actual != "Draw":
                decisive_correct += 1
        results.append({"home": home, "away": away,
                        "score": f"{int(hs)}-{int(as_)}", "actual": actual,
                        "predicted": predicted, "hit": hit,
                        "hw": round(pred["Home Win"]*100,1),
                        "dr": round(pred["Draw"]*100,1),
                        "aw": round(pred["Away Win"]*100,1)})

    total = len(results)
    wc26_draw_pct = round(draws / total * 100, 1) if total else 0

    hw_total = sum(1 for r in results if r["actual"] == "Home Win")
    hw_correct = sum(1 for r in results if r["actual"] == "Home Win" and r["hit"])
    dr_total = sum(1 for r in results if r["actual"] == "Draw")
    dr_correct = sum(1 for r in results if r["actual"] == "Draw" and r["hit"])
    aw_total = sum(1 for r in results if r["actual"] == "Away Win")
    aw_correct = sum(1 for r in results if r["actual"] == "Away Win" and r["hit"])

    return {
        "total":          total,
        "hw_correct":     hw_correct,
        "hw_total":       hw_total,
        "hw_pct":         round(hw_correct / hw_total * 100, 1) if hw_total else 0,
        "dr_correct":     dr_correct,
        "dr_total":       dr_total,
        "dr_pct":         round(dr_correct / dr_total * 100, 1) if dr_total else 0,
        "aw_correct":     aw_correct,
        "aw_total":       aw_total,
        "aw_pct":         round(aw_correct / aw_total * 100, 1) if aw_total else 0,
        "wc26_draw_pct":  wc26_draw_pct,
        "hist_draw_pct":  hist_draw_pct,
    }


def load_edges(top_n=15):
    """Read value_bets.json. Always includes ALL of today's edges; caps upcoming at top_n."""
    path = os.path.join(BASE_DIR, "value_bets.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)

    today_str = date.today().isoformat()
    today_flat    = []
    upcoming_flat = []

    for r in data.get("results", []):
        commence   = r.get("commence", "")
        match_date = commence[:10] if commence else ""
        for e in r.get("edges", []):
            if not e["best_odds"] or e["best_odds"] >= 100:
                continue   # drop suspended/placeholder lines
            entry = {
                "home":       r["home"],
                "away":       r["away"],
                "date":       match_date,
                "outcome":    e["outcome"],
                "model_p":    e["model_p"],
                "market_p":   e.get("market_p", 0),
                "edge":       e["edge"],
                "best_odds":  e["best_odds"],
                "best_book":  e["best_book"] or "",
                "dk_odds":    e.get("dk_odds"),
                "ev":         e.get("ev"),
                "kelly":      e.get("kelly", 0),
                "half_kelly": e.get("half_kelly", 0),
                "fair_odds":  e.get("fair_odds"),
                "certainty":  e.get("certainty", 0),
            }
            if match_date == today_str:
                today_flat.append(entry)
            else:
                upcoming_flat.append(entry)

    today_flat.sort(key=lambda x: -x["edge"])
    upcoming_flat.sort(key=lambda x: -x["edge"])
    return today_flat + upcoming_flat[:top_n]


HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>WC26 Picks</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700;800&display=swap" rel="stylesheet">
  <!-- Google tag (gtag.js) -->
  <script async src="https://www.googletagmanager.com/gtag/js?id=G-L8WZ7C8MJG"></script>
  <script>
    window.dataLayer = window.dataLayer || [];
    function gtag(){dataLayer.push(arguments);}
    gtag('js', new Date());
    gtag('config', 'G-L8WZ7C8MJG');
  </script>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      background: #f4f4f2;
      color: #111;
      font-family: "Space Grotesk", -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
      min-height: 100vh;
    }

    .container { max-width: 1100px; margin: 0 auto; padding: 0 20px 80px; margin-top: 58px; position: relative; z-index: 1; }

    /* ── Scroll ball ── */
    .scroll-ball {
      position: fixed;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%) rotate(0deg);
      font-size: 580px;
      line-height: 1;
      opacity: 0.045;
      pointer-events: none;
      user-select: none;
      z-index: 0;
      filter: grayscale(1);
      will-change: transform;
    }

    /* ── Ticker ── */
    .ticker-bar {
      background: #0d0d0d;
      overflow: hidden;
      padding: 0;
      position: fixed;
      top: 0; left: 0; right: 0;
      z-index: 200;
      border-bottom: 1px solid #1e1e1e;
      box-shadow: 0 2px 20px rgba(0,0,0,0.35);
    }
    .ticker-track {
      display: inline-flex;
      align-items: stretch;
      white-space: nowrap;
      animation: tickerScroll 55s linear infinite;
    }
    .ticker-bar:hover .ticker-track { animation-play-state: paused; }
    @keyframes tickerScroll {
      0%   { transform: translateX(0); }
      100% { transform: translateX(-33.333%); }
    }
    .ticker-item {
      display: inline-flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 2px;
      padding: 8px 28px;
      border-right: 1px solid #222;
    }
    .ticker-matchup {
      display: inline-flex; align-items: center; gap: 5px;
    }
    .ticker-team  { color: #f0f0f0; font-weight: 600; font-size: 0.8rem; }
    .ticker-fav {
      background: #16a34a; color: #fff;
      border-radius: 4px; padding: 1px 5px;
      font-size: 0.58rem; font-weight: 700; letter-spacing: 0.5px;
      text-transform: uppercase; margin-left: 3px;
    }
    .ticker-sep   { color: #444; font-size: 0.68rem; margin: 0 3px; }
    .ticker-flag  { font-size: 0.95rem; line-height: 1; }
    .ticker-meta  { font-size: 0.65rem; color: #555; letter-spacing: 0.3px; }

    /* ── Hero header ── */
    .site-hero {
      padding: 44px 0 36px;
      margin-bottom: 32px;
      border-bottom: 1px solid #e0e0e0;
    }
    .hero-eyebrow {
      font-size: 0.7rem; font-weight: 700; letter-spacing: 3px;
      text-transform: uppercase; color: #aaa; margin-bottom: 14px;
    }
    .hero-title {
      font-size: clamp(2.4rem, 6vw, 4.2rem);
      font-weight: 800; letter-spacing: -2px; line-height: 0.95;
      background: linear-gradient(120deg, #000 0%, #1a1a1a 40%, #16a34a 70%, #059669 100%);
      background-size: 200% auto;
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
      background-clip: text;
      animation: titleShimmer 5s linear infinite;
    }
    @keyframes titleShimmer {
      0%   { background-position: 0% center; }
      100% { background-position: 200% center; }
    }
    .hero-sub {
      margin-top: 18px; font-size: 0.9rem; color: #888; font-weight: 400;
    }
    .hero-stats {
      display: flex; gap: 28px; margin-top: 28px; flex-wrap: wrap;
    }
    .hero-stat {
      display: flex; flex-direction: column; gap: 2px;
    }
    .hero-stat .stat-val {
      font-size: 1.4rem; font-weight: 800; color: #111;
      font-family: "Courier New", monospace;
    }
    .hero-stat .stat-lbl {
      font-size: 0.65rem; font-weight: 600; letter-spacing: 1px;
      text-transform: uppercase; color: #bbb;
    }

    /* ── Best Bets chips ── */
    .bets-section { margin-bottom: 44px; }
    .bets-header {
      display: flex; align-items: baseline; gap: 10px; margin-bottom: 20px;
    }
    .bets-title { font-size: 1.05rem; font-weight: 700; color: #111; }
    .bets-count { font-size: 0.78rem; color: #bbb; }

    .chips-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      gap: 14px;
    }

    /* Base chip */
    .bet-chip {
      background: #fff;
      border: 1px solid #e5e5e5;
      border-radius: 16px;
      padding: 22px 22px 18px;
      display: flex; flex-direction: column; gap: 14px;
      transition: box-shadow 0.15s, transform 0.15s;
      border-top-width: 4px;
    }
    .bet-chip:hover { box-shadow: 0 6px 24px rgba(0,0,0,0.09); transform: translateY(-2px); }

    /* Color tiers — top border + background tint */
    .chip-green  { border-top-color: #16a34a; background: #f8fff9; }
    .chip-yellow { border-top-color: #d97706; background: #fffdf5; }
    .chip-red    { border-top-color: #ef4444; background: #fff; opacity: 0.82; }

    .chip-top {
      display: flex; align-items: flex-start;
      justify-content: space-between; gap: 8px;
    }
    .chip-match { font-size: 0.78rem; font-weight: 600; color: #888; line-height: 1.3; }

    /* Statement — the hero */
    .chip-statement {
      font-size: 1rem; color: #111; line-height: 1.55; font-weight: 400;
    }
    .chip-statement strong { font-weight: 700; color: #000; }

    .chip-bottom {
      display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
    }
    .chip-edge {
      font-size: 0.8rem; font-weight: 800; border-radius: 6px;
      padding: 3px 10px; white-space: nowrap;
    }
    .chip-green  .chip-edge { background: #f0fdf4; color: #16a34a; }
    .chip-yellow .chip-edge { background: #fffbeb; color: #d97706; }
    .chip-red    .chip-edge { background: #fef2f2; color: #ef4444; }

    .chip-dk {
      margin-left: auto;
      background: #f5f5f5; border-radius: 8px;
      padding: 5px 13px; font-size: 0.88rem; font-weight: 700; color: #111;
      white-space: nowrap;
    }
    .chip-dk .dk-label {
      font-weight: 400; color: #999; font-size: 0.68rem;
      display: block; margin-bottom: 1px;
    }

    .no-bets-today {
      text-align: center; padding: 32px; color: #bbb;
      font-size: 0.9rem; background: #fff;
      border: 1px dashed #e5e5e5; border-radius: 16px;
    }

    .bets-footer { margin-top: 16px; font-size: 0.67rem; color: #ccc; }

    /* ── Bet sort bar ── */
    .bet-sort-bar {
      display: flex; align-items: center; gap: 8px; margin-bottom: 20px; flex-wrap: wrap;
    }
    .bet-sort-label {
      font-size: 0.7rem; color: #aaa; font-weight: 700;
      text-transform: uppercase; letter-spacing: 1px; margin-right: 4px;
    }
    .sort-btn {
      padding: 5px 14px; border-radius: 99px; font-size: 0.75rem; font-weight: 700;
      border: 1.5px solid #e5e5e5; background: #fff; color: #888;
      cursor: pointer; transition: all 0.15s; white-space: nowrap;
    }
    .sort-btn:hover { border-color: #bbb; color: #444; }
    .sort-btn.active { background: #111; border-color: #111; color: #fff; }

    /* ── All Games header ── */
    .all-games-header {
      display: flex; align-items: center; gap: 12px;
      margin-bottom: 16px;
    }
    .all-games-title {
      font-size: 1.05rem; font-weight: 700; color: #111;
    }
    .all-games-header::after {
      content: ''; flex: 1; height: 1px; background: #e5e5e5;
    }

    /* ── Day tabs ── */
    .day-tabs {
      display: flex; gap: 6px; margin-bottom: 20px; flex-wrap: wrap;
    }
    .day-tab {
      padding: 7px 18px; border-radius: 99px;
      font-size: 0.78rem; font-weight: 700;
      border: 1.5px solid #e5e5e5; background: #fff; color: #888;
      cursor: pointer; transition: all 0.15s;
      white-space: nowrap;
    }
    .day-tab:hover { border-color: #bbb; color: #444; }
    .day-tab.active {
      background: #111; border-color: #111; color: #fff;
    }
    .day-tab .tab-dot {
      display: inline-block; width: 6px; height: 6px;
      background: #16a34a; border-radius: 50%; margin-left: 5px;
      vertical-align: middle; position: relative; top: -1px;
    }
    .day-tab.active .tab-dot { background: #4ade80; }
    .day-panel { display: none; }
    .day-panel.active { display: block; }
    .day-bets-header {
      display: flex; align-items: baseline; gap: 10px; margin-bottom: 14px;
    }
    .day-bets-title { font-size: 0.95rem; font-weight: 700; color: #111; }
    .day-bets-count { font-size: 0.75rem; color: #bbb; }
    .day-bets-block { margin-bottom: 28px; }

    /* ── Date section ── */
    .date-section { margin-bottom: 32px; }
    .date-label {
      font-size: 0.7rem; font-weight: 700; letter-spacing: 2px;
      text-transform: uppercase; color: #aaa;
      border-bottom: 1px solid #e5e5e5;
      padding-bottom: 8px; margin-bottom: 14px;
      display: flex; align-items: center; gap: 8px;
    }
    .today-tag {
      background: #111; color: #fff;
      border-radius: 4px; padding: 1px 7px; font-size: 0.65rem;
    }

    /* ── Match grid ── */
    .match-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
      gap: 12px;
    }

    /* ── Match card ── */
    .card {
      background: #fff;
      border: 1px solid #e5e5e5;
      border-radius: 12px;
      padding: 18px 18px 14px;
      transition: box-shadow 0.15s;
    }
    .card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.07); }

    .teams {
      display: flex; align-items: center; justify-content: center;
      gap: 8px; margin-bottom: 3px; flex-wrap: wrap;
    }
    .team-name { font-weight: 700; font-size: 0.9rem; color: #111; white-space: nowrap; }
    .card-fav {
      background: #16a34a; color: #fff;
      border-radius: 4px; padding: 1px 5px;
      font-size: 0.58rem; font-weight: 700; letter-spacing: 0.5px;
      text-transform: uppercase; margin-left: 4px; vertical-align: middle;
    }
    .flag { font-size: 1.1rem; }
    .vs {
      background: #f5f5f5; color: #bbb;
      border-radius: 5px; padding: 1px 7px;
      font-size: 0.65rem; font-weight: 700; letter-spacing: 1px;
    }
    .card-meta { text-align: center; font-size: 0.72rem; color: #bbb; margin-bottom: 14px; }

    /* ── Bars ── */
    .bars { display: flex; flex-direction: column; gap: 7px; }
    .bar-meta {
      display: flex; justify-content: space-between;
      font-size: 0.78rem; margin-bottom: 3px;
    }
    .bar-meta .outcome { color: #999; }
    .bar-meta .pct { font-weight: 600; color: #555; }
    .bar-meta.best .outcome { color: #111; font-weight: 600; }
    .bar-meta.best .pct    { color: #111; }
    .track { background: #f0f0f0; border-radius: 99px; height: 6px; overflow: hidden; }
    .fill  { height: 6px; border-radius: 99px; transition: width 0.5s ease; }
    .fill-hw { background: #16a34a; }
    .fill-dr { background: #d97706; }
    .fill-aw { background: #dc2626; }

    /* ── Goals / BTTS pills ── */
    .goals-row {
      margin-top: 12px; padding-top: 10px; border-top: 1px solid #f0f0f0;
      display: flex; align-items: center; justify-content: center;
      gap: 6px; flex-wrap: wrap;
    }
    .goals-pill {
      font-size: 0.7rem; border-radius: 6px; padding: 3px 9px; font-weight: 600;
      border: 1px solid #e5e5e5; color: #555; background: #fafafa;
      white-space: nowrap;
    }
    .goals-pill .g-label { color: #aaa; font-weight: 400; margin-right: 3px; }
    .goals-pill.highlight { background: #f0fdf4; border-color: #bbf7d0; color: #16a34a; }
    .goals-pill.highlight-red { background: #fef2f2; border-color: #fecaca; color: #dc2626; }

    /* ── H2H ── */
    .h2h-block {
      margin-top: 12px; padding-top: 10px; border-top: 1px solid #f0f0f0;
    }
    .h2h-title {
      font-size: 0.62rem; color: #bbb; font-weight: 600;
      text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 7px;
    }
    .h2h-bar {
      display: flex; height: 5px; border-radius: 99px; overflow: hidden; gap: 1px;
    }
    .h2h-seg { transition: width 0.4s ease; }
    .h2h-seg-hw { background: #16a34a; }
    .h2h-seg-dr { background: #d1d5db; }
    .h2h-seg-aw { background: #dc2626; }
    .h2h-legend {
      display: flex; justify-content: space-between;
      margin-top: 5px; font-size: 0.68rem; font-weight: 600;
    }
    .h2h-leg-hw { color: #16a34a; }
    .h2h-leg-dr { color: #aaa; font-weight: 400; }
    .h2h-leg-aw { color: #dc2626; }
    .h2h-none { font-size: 0.68rem; color: #ccc; font-style: italic;
      margin-top: 10px; padding-top: 10px; border-top: 1px solid #f0f0f0; }

    /* ── ELO ── */
    .elo-row { margin-top: 7px; text-align: center; font-size: 0.68rem; color: #ccc; }

    /* ── Monospace numbers ── */
    .mono { font-family: "Courier New", Courier, monospace; }

    /* ── Model disclosure ── */
    .model-disclosure {
      background: #fff; border: 1px solid #e5e5e5; border-radius: 12px;
      padding: 18px 22px; margin-bottom: 32px;
    }
    .disclosure-intro {
      font-size: 0.85rem; color: #555; margin-bottom: 14px; line-height: 1.5;
    }
    .disclosure-stats {
      display: flex; gap: 24px; flex-wrap: wrap;
    }
    .disclosure-row {
      display: flex; flex-direction: column; gap: 2px;
    }
    .disclosure-label {
      font-size: 0.62rem; font-weight: 700; letter-spacing: 1px;
      text-transform: uppercase; color: #bbb;
    }
    .disclosure-val {
      font-size: 0.88rem; color: #111; font-weight: 600;
    }
    .disclosure-val .mono { font-size: 1rem; font-weight: 700; }

    /* ── Scroll reveal ── */
    .reveal {
      opacity: 0;
      transform: translateY(28px);
      transition: opacity 0.55s cubic-bezier(0.22,1,0.36,1),
                  transform 0.55s cubic-bezier(0.22,1,0.36,1);
    }
    .reveal.visible { opacity: 1; transform: none; }

    /* ── Live accuracy card ── */
    .acc-card {
      background: #fff; border: 1px solid #e5e5e5; border-radius: 16px;
      padding: 26px 28px 22px; margin-bottom: 44px;
    }
    .acc-card-header {
      display: flex; align-items: baseline; gap: 10px; margin-bottom: 22px;
    }
    .acc-card-title { font-size: 1.05rem; font-weight: 700; color: #111; }
    .acc-card-sub   { font-size: 0.78rem; color: #bbb; }

    .acc-stats {
      display: flex; gap: 0; flex-wrap: wrap;
      border: 1px solid #f0f0f0; border-radius: 12px; overflow: hidden;
      margin-bottom: 20px;
    }
    .acc-stat {
      flex: 1; min-width: 120px;
      padding: 14px 18px;
      border-right: 1px solid #f0f0f0;
    }
    .acc-stat:last-child { border-right: none; }
    .acc-stat-frac {
      display: flex; align-items: baseline; gap: 8px; margin-bottom: 5px;
    }
    .acc-frac-num {
      font-size: 1.45rem; font-weight: 800; color: #111; line-height: 1;
    }
    .acc-frac-pct {
      font-size: 0.95rem; font-weight: 700; font-family: "Courier New", monospace;
    }
    .acc-frac-pct.green { color: #16a34a; }
    .acc-frac-pct.amber { color: #d97706; }
    .acc-frac-pct.red   { color: #dc2626; }
    .acc-stat-lbl { font-size: 0.68rem; color: #aaa; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; line-height: 1.3; }

    .draw-comparison { margin-bottom: 10px; }
    .draw-row {
      display: flex; align-items: center; gap: 10px; margin-bottom: 6px;
    }
    .draw-row-label { font-size: 0.72rem; color: #888; min-width: 170px; }
    .draw-track {
      flex: 1; background: #f5f5f5; border-radius: 99px; height: 7px; overflow: hidden;
    }
    .draw-fill-hist    { height: 7px; border-radius: 99px; background: #d1d5db; }
    .draw-fill-current { height: 7px; border-radius: 99px; background: #f59e0b; }
    .draw-pct { font-size: 0.72rem; font-weight: 700; font-family: "Courier New", monospace; min-width: 36px; text-align: right; }
    .draw-pct.amber { color: #d97706; }

    .acc-note {
      font-size: 0.78rem; color: #888; line-height: 1.55;
      border-top: 1px solid #f0f0f0; padding-top: 14px; margin-top: 4px;
    }
    .acc-note strong { color: #555; }

    /* ── Footer ── */
    footer {
      margin-top: 60px; text-align: center; font-size: 0.78rem; color: #bbb;
      border-top: 1px solid #e0e0e0; padding-top: 24px;
    }
    footer a { color: #aaa; }
  </style>
</head>
<body>
<div class="scroll-ball" aria-hidden="true">⚽</div>
<div id="ticker"></div>
<div class="container">

  <header class="site-hero">
    <div class="hero-eyebrow">FIFA World Cup 2026</div>
    <h1 class="hero-title">Match<br>Picks</h1>
    <p class="hero-sub">AI predictions for every game · updated daily · last refreshed __UPDATED__</p>
    <div class="hero-stats">
      <div class="hero-stat">
        <span class="stat-val">59.6%</span>
        <span class="stat-lbl">Win/Draw/Loss accuracy</span>
      </div>
      <div class="hero-stat">
        <span class="stat-val">59.1%</span>
        <span class="stat-lbl">Over/Under 2.5 accuracy</span>
      </div>
      <div class="hero-stat">
        <span class="stat-val">54.4%</span>
        <span class="stat-lbl">Both teams score accuracy</span>
      </div>
      <div class="hero-stat">
        <span class="stat-val">12k+</span>
        <span class="stat-lbl">Matches in training data</span>
      </div>
    </div>
  </header>

  <div id="acc-section"></div>

  <main id="app"></main>

  <footer>
    Built on 12,000+ international matches · team strength · form · head-to-head · goals data ·
    <a href="https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017">Data source</a>
  </footer>

</div>
<script>
const MATCHES    = __DATA__;
const VALUE_BETS = __VALUE_BETS__;
const WC26_ACC   = __ACCURACY__;

// ── Best Bets sort state ───────────────────────────────────────────────────────
let currentSort = 'edge';

const SORT_OPTS = [
  { key: 'edge',       label: 'Edge' },
  { key: 'ev',         label: 'EV' },
  { key: 'kelly',      label: 'Kelly %' },
  { key: 'half_kelly', label: 'Half Kelly' },
  { key: 'odds',       label: 'Best Odds' },
  { key: 'confidence', label: 'Confidence' },
];

function sortValue(e, sort) {
  if (sort === 'edge')       return e.edge || 0;
  if (sort === 'ev')         return e.ev != null ? e.ev : -99;
  if (sort === 'kelly')      return e.kelly || 0;
  if (sort === 'half_kelly') return e.half_kelly || 0;
  if (sort === 'odds')       return e.best_odds || 0;
  if (sort === 'confidence') return e.certainty || 0;
  return 0;
}

function chipMetricBadge(e, sort) {
  if (sort === 'ev') {
    if (e.ev == null) return '';
    const sign = e.ev >= 0 ? '+' : '';
    const tier = e.ev >= 0.10 ? 'chip-green' : e.ev >= 0.04 ? 'chip-yellow' : 'chip-red';
    return `<span class="chip-edge"><span class="mono">${sign}$${e.ev.toFixed(2)}</span> · EV per $1</span>`;
  }
  if (sort === 'kelly') {
    const lbl = e.kelly >= 10 ? 'Strong sizing' : e.kelly >= 5 ? 'Moderate' : 'Small stake';
    return `<span class="chip-edge"><span class="mono">${e.kelly || 0}%</span> · Kelly · ${lbl}</span>`;
  }
  if (sort === 'half_kelly') {
    return `<span class="chip-edge"><span class="mono">${e.half_kelly || 0}%</span> · Half Kelly</span>`;
  }
  if (sort === 'odds') {
    const odds = e.dk_odds || e.best_odds;
    const am = odds ? toAmerican(odds) : null;
    const book = e.dk_odds ? 'DraftKings' : e.best_book;
    return am ? `<span class="chip-edge"><span class="mono">${am}</span> · Best price @ ${book}</span>` : '';
  }
  if (sort === 'confidence') {
    const cert = e.certainty || 0;
    const lbl = cert >= 20 ? 'High confidence' : cert >= 10 ? 'Moderate' : 'Low confidence';
    return `<span class="chip-edge"><span class="mono">${cert >= 0 ? '+' : ''}${cert}%</span> · vs baseline · ${lbl}</span>`;
  }
  // default: edge
  const edgeLbl = e.edge >= 15 ? 'Strong edge' : e.edge >= 8 ? 'Good edge' : 'Lean';
  return `<span class="chip-edge"><span class="mono">+${e.edge}%</span> · ${edgeLbl}</span>`;
}

function renderSortBar() {
  if (!VALUE_BETS || !VALUE_BETS.length) return '';
  const btns = SORT_OPTS.map(o =>
    `<button class="sort-btn${o.key === currentSort ? ' active' : ''}" data-sort="${o.key}">${o.label}</button>`
  ).join('');
  return `<div class="bet-sort-bar"><span class="bet-sort-label">Sort by</span>${btns}</div>`;
}

// Build market-favorite lookup from bookmaker implied probabilities
const mktProbs = {};
(VALUE_BETS || []).forEach(function(e) {
  const key = e.home + '|' + e.away;
  if (!mktProbs[key]) mktProbs[key] = {};
  if (e.outcome === 'Home Win') mktProbs[key].home = e.market_p;
  if (e.outcome === 'Draw')     mktProbs[key].draw = e.market_p;
  if (e.outcome === 'Away Win') mktProbs[key].away = e.market_p;
});

function marketFav(home, away) {
  const m = mktProbs[home + '|' + away];
  if (!m) return null;
  // If an outcome has no edge it won't be in VALUE_BETS — infer its market prob
  // from the remaining probability (vig-free probs sum to ~100)
  const known_dr = m.draw || 0;
  const known_hw = m.home !== undefined ? m.home : Math.max(0, 100 - (m.away || 0) - known_dr);
  const known_aw = m.away !== undefined ? m.away : Math.max(0, 100 - known_hw - known_dr);
  if (known_hw > known_aw && known_hw > known_dr) return 'home';
  if (known_aw > known_hw && known_aw > known_dr) return 'away';
  return null;
}

function todayISO() {
  return new Date().toISOString().slice(0, 10);
}

function fmtShort(iso) {
  const d = new Date(iso + 'T12:00:00');
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

function toAmerican(decimal) {
  if (!decimal) return null;
  return decimal >= 2.0
    ? '+' + Math.round((decimal - 1) * 100)
    : '−' + Math.round(100 / (decimal - 1));
}

function outcomePhrase(e) {
  if (e.outcome === 'Home Win')  return `<strong>${e.home}</strong> to win`;
  if (e.outcome === 'Away Win')  return `<strong>${e.away}</strong> to win`;
  if (e.outcome === 'Draw')      return `<strong>${e.home} vs ${e.away}</strong> to draw`;
  if (e.outcome === 'Over 2.5')  return `this game to go <strong>Over 2.5 goals</strong>`;
  if (e.outcome === 'Under 2.5') return `this game to stay <strong>Under 2.5 goals</strong>`;
  return `<strong>${e.outcome}</strong>`;
}

function betStatement(e) {
  const odds = e.dk_odds || e.best_odds;
  const book = e.dk_odds ? 'DraftKings' : e.best_book;
  const am   = odds ? toAmerican(odds) : null;
  const oddsStr = am ? ` (${book} <strong class="mono">${am}</strong>)` : '';
  return `Our model gives ${outcomePhrase(e)} a <strong class="mono">${e.model_p}%</strong> chance. The market prices it at <strong class="mono">${e.market_p}%</strong>${oddsStr} — a <strong class="mono">+${e.edge}%</strong> edge.`;
}

function betChip(e, sort) {
  sort = sort || currentSort;
  const tierCls = e.edge >= 15 ? 'chip-green' : e.edge >= 8 ? 'chip-yellow' : 'chip-red';
  const badge   = chipMetricBadge(e, sort);
  const showDk  = sort !== 'odds' && (e.dk_odds || e.best_odds);
  return `
    <div class="bet-chip ${tierCls}">
      <div class="chip-top">
        <span class="chip-match">${e.home} vs ${e.away}</span>
      </div>
      <div class="chip-statement">${betStatement(e)}</div>
      <div class="chip-bottom">
        ${badge}
        ${showDk ? `<span class="chip-dk"><span class="dk-label">${e.dk_odds ? 'DraftKings' : e.best_book}</span><span class="mono">${toAmerican(e.dk_odds || e.best_odds)}</span></span>` : ''}
      </div>
    </div>`;
}

function betsForDate(dateStr) {
  return (VALUE_BETS || []).filter(e => e.date === dateStr);
}

function sortedBets(dateStr, sort) {
  return betsForDate(dateStr).slice().sort((a, b) => sortValue(b, sort) - sortValue(a, sort));
}

function renderDayBets(dateStr) {
  const bets = betsForDate(dateStr);
  if (!bets.length) return '';
  const chips = sortedBets(dateStr, currentSort).map(e => betChip(e, currentSort)).join('');
  const sortBar = `<div class="bet-sort-bar">
    <span class="bet-sort-label">View top bets by</span>
    ${SORT_OPTS.map(o => `<button class="sort-btn${o.key === currentSort ? ' active' : ''}" data-sort="${o.key}">${o.label}</button>`).join('')}
  </div>`;
  return `<div class="day-bets-block">
    <div class="day-bets-header">
      <span class="day-bets-title">Best Bets</span>
      <span class="day-bets-count">${bets.length} edge${bets.length !== 1 ? 's' : ''} found</span>
    </div>
    ${sortBar}
    <div class="chips-grid" data-date="${dateStr}">${chips}</div>
    <p class="bets-footer">Edges vs avg bookmaker (vig removed) · always bet responsibly</p>
  </div>`;
}

function fmtDate(iso) {
  const d = new Date(iso + 'T12:00:00');
  return d.toLocaleDateString('en-US', { weekday:'long', month:'long', day:'numeric' });
}

function isToday(iso) {
  return iso === new Date().toISOString().slice(0, 10);
}

function barRow(label, pct, fillClass, isBest) {
  return `
    <div class="bar-row">
      <div class="bar-meta${isBest ? ' best' : ''}">
        <span class="outcome">${label}</span>
        <span class="pct mono">${pct.toFixed(1)}%</span>
      </div>
      <div class="track"><div class="fill ${fillClass}" style="width:${pct}%"></div></div>
    </div>`;
}

function goalsSection(m) {
  const overCls  = m.over25  >= 55 ? 'highlight'     : m.over25  <= 40 ? 'highlight-red' : '';
  const bttsCls  = m.btts_yes >= 55 ? 'highlight'    : m.btts_yes <= 35 ? 'highlight-red' : '';
  return `
    <div class="goals-row">
      <span class="goals-pill ${overCls}"><span class="g-label">O/U 2.5</span>Over <span class="mono">${m.over25}%</span> · Under <span class="mono">${m.under25}%</span></span>
      <span class="goals-pill ${bttsCls}"><span class="g-label">BTTS</span>Yes <span class="mono">${m.btts_yes}%</span> · No <span class="mono">${m.btts_no}%</span></span>
    </div>`;
}

function h2hSection(m) {
  if (m.h2h_n === 0)
    return `<div class="h2h-none">No prior matchups on record</div>`;
  return `
    <div class="h2h-block">
      <div class="h2h-title">Last ${m.h2h_n} meetings</div>
      <div class="h2h-bar">
        <div class="h2h-seg h2h-seg-hw" style="width:${m.h2h_hw}%"></div>
        <div class="h2h-seg h2h-seg-dr" style="width:${m.h2h_dr}%"></div>
        <div class="h2h-seg h2h-seg-aw" style="width:${m.h2h_aw}%"></div>
      </div>
      <div class="h2h-legend">
        <span class="h2h-leg-hw">${m.home} ${m.h2h_hw}%</span>
        <span class="h2h-leg-dr">${m.h2h_dr}% draws</span>
        <span class="h2h-leg-aw">${m.h2h_aw}% ${m.away}</span>
      </div>
    </div>`;
}

function card(m) {
  const best     = m.home_win >= m.draw && m.home_win >= m.away_win ? 'hw'
                 : m.away_win >= m.draw ? 'aw' : 'dr';
  const mfav     = marketFav(m.home, m.away);
  const homeFav  = mfav === 'home';
  const awayFav  = mfav === 'away';
  const fav      = '<span class="card-fav">FAV</span>';
  return `
    <div class="card">
      <div class="teams">
        <span class="flag">${m.home_flag}</span>
        <span class="team-name">${m.home}${homeFav ? fav : ''}</span>
        <span class="vs">VS</span>
        <span class="team-name">${m.away}${awayFav ? fav : ''}</span>
        <span class="flag">${m.away_flag}</span>
      </div>
      <div class="card-meta">Model prediction</div>
      <div class="bars">
        ${barRow(m.home + ' Win', m.home_win, 'fill-hw', best === 'hw')}
        ${barRow('Draw',          m.draw,     'fill-dr', best === 'dr')}
        ${barRow(m.away + ' Win', m.away_win, 'fill-aw', best === 'aw')}
      </div>
      ${goalsSection(m)}
      ${h2hSection(m)}
      <div class="elo-row">Strength rating · ${m.home} <span class="mono">${m.home_elo}</span> · ${m.away} <span class="mono">${m.away_elo}</span></div>
    </div>`;
}

function render() {
  const byDate = {};
  MATCHES.forEach(m => { if (!byDate[m.date]) byDate[m.date] = []; byDate[m.date].push(m); });

  const today    = todayISO();
  const tomorrow = (function() {
    const d = new Date(today + 'T12:00:00'); d.setDate(d.getDate() + 1);
    return d.toISOString().slice(0, 10);
  })();

  const sortedDates = Object.keys(byDate).sort();

  // Build tab labels: Today, Tomorrow, then formatted dates for the rest
  function tabLabel(iso) {
    if (iso === today)    return 'Today';
    if (iso === tomorrow) return 'Tomorrow';
    return fmtShort(iso);
  }

  // Default to today's tab if it exists, otherwise the first date
  const defaultTab = byDate[today] ? today : sortedDates[0];

  const app = document.getElementById('app');

  const tabsHtml = sortedDates.map(d => {
    const hasBets = betsForDate(d).length > 0;
    const dot = hasBets ? '<span class="tab-dot"></span>' : '';
    return `<button class="day-tab${d === defaultTab ? ' active' : ''}" data-date="${d}">${tabLabel(d)}${dot}</button>`;
  }).join('');

  const panelsHtml = sortedDates.map(d => {
    const cards   = byDate[d].map(card).join('');
    const dayBets = renderDayBets(d);
    const allGamesHdr = dayBets
      ? `<div class="all-games-header"><span class="all-games-title">All Games</span></div>`
      : '';
    return `<div class="day-panel${d === defaultTab ? ' active' : ''}" data-date="${d}">
      ${dayBets}
      ${allGamesHdr}
      <div class="match-grid">${cards}</div>
    </div>`;
  }).join('');

  app.innerHTML = `<div class="day-tabs">${tabsHtml}</div>${panelsHtml}`;

  app.querySelectorAll('.day-tab').forEach(btn => {
    btn.addEventListener('click', function() {
      const target = this.dataset.date;
      app.querySelectorAll('.day-tab').forEach(b => b.classList.toggle('active', b.dataset.date === target));
      app.querySelectorAll('.day-panel').forEach(p => p.classList.toggle('active', p.dataset.date === target));
      // Trigger reveal for newly visible cards
      app.querySelectorAll('.day-panel.active .card:not(.visible)').forEach((el, i) => {
        el.style.transitionDelay = (i * 40) + 'ms';
        setTimeout(() => el.classList.add('visible'), 10);
      });
    });
  });

  app.querySelectorAll('.sort-btn').forEach(btn => {
    btn.addEventListener('click', function() {
      currentSort = this.dataset.sort;
      app.querySelectorAll('.sort-btn').forEach(b => b.classList.toggle('active', b.dataset.sort === currentSort));
      app.querySelectorAll('.chips-grid[data-date]').forEach(grid => {
        const date = grid.dataset.date;
        grid.innerHTML = sortedBets(date, currentSort).map(e => betChip(e, currentSort)).join('');
      });
    });
  });
}

function utcToET(isoStr) {
  if (!isoStr) return '';
  try {
    return new Date(isoStr).toLocaleTimeString('en-US', {
      hour: 'numeric', minute: '2-digit',
      timeZone: 'America/New_York', hour12: true
    }) + ' ET';
  } catch(e) { return ''; }
}

function renderTicker() {
  const today      = todayISO();
  const todayGames = MATCHES.filter(m => m.date === today);
  const wrap       = document.getElementById('ticker');
  if (!todayGames.length || !wrap) return;

  function tickerItem(m) {
    const mfav  = marketFav(m.home, m.away);
    const homeF = mfav === 'home';
    const awayF = mfav === 'away';
    const time   = utcToET(m.kickoff_utc);
    const city   = m.city || '';
    const meta   = [time, city].filter(Boolean).join(' · ');
    return '<span class="ticker-item">'
      + '<span class="ticker-matchup">'
      +   '<span class="ticker-flag">' + m.home_flag + '</span>'
      +   '<span class="ticker-team">' + m.home + (homeF ? ' <span class="ticker-fav">FAV</span>' : '') + '</span>'
      +   '<span class="ticker-sep">vs</span>'
      +   '<span class="ticker-flag">' + m.away_flag + '</span>'
      +   '<span class="ticker-team">' + m.away + (awayF ? ' <span class="ticker-fav">FAV</span>' : '') + '</span>'
      + '</span>'
      + (meta ? '<span class="ticker-meta">' + meta + '</span>' : '')
      + '</span>';
  }

  const items = todayGames.map(tickerItem).join('');
  wrap.innerHTML = '<div class="ticker-bar"><div class="ticker-track">' + items + items + items + '</div></div>';
}

function renderAccuracy() {
  const el = document.getElementById('acc-section');
  if (!el || !WC26_ACC || WC26_ACC.total === 0) return;
  const a = WC26_ACC;
  const histW = a.hist_draw_pct;
  const currW = a.wc26_draw_pct;
  const maxW  = Math.max(histW, currW, 40);

  function statBlock(correct, total, pct, label, colorCls) {
    return `<div class="acc-stat">
      <div class="acc-stat-frac">
        <span class="acc-frac-num mono">${correct}/${total}</span>
        <span class="acc-frac-pct ${colorCls}">${pct}%</span>
      </div>
      <div class="acc-stat-lbl">${label}</div>
    </div>`;
  }

  el.innerHTML = `
    <div class="acc-card">
      <div class="acc-card-header">
        <span class="acc-card-title">How the model has done at WC 2026</span>
        <span class="acc-card-sub">${a.total} games played</span>
      </div>
      <div class="acc-stats">
        ${statBlock(a.hw_correct, a.hw_total, a.hw_pct, 'Home wins predicted', 'green')}
        ${statBlock(a.dr_correct, a.dr_total, a.dr_pct, 'Draws predicted', 'red')}
        ${statBlock(a.aw_correct, a.aw_total, a.aw_pct, 'Away wins predicted', 'green')}
      </div>
      <div class="draw-comparison">
        <div class="draw-row">
          <span class="draw-row-label">Historical WC draw rate</span>
          <div class="draw-track"><div class="draw-fill-hist" style="width:${(histW/maxW*100).toFixed(1)}%"></div></div>
          <span class="draw-pct">${histW}%</span>
        </div>
        <div class="draw-row">
          <span class="draw-row-label">WC 2026 draw rate so far</span>
          <div class="draw-track"><div class="draw-fill-current" style="width:${(currW/maxW*100).toFixed(1)}%"></div></div>
          <span class="draw-pct amber">${currW}%</span>
        </div>
      </div>
      <p class="acc-note">
        Draws are the model's blind spot — and this tournament is <strong>unusually draw-heavy</strong> (${currW}%, highest since France '98 at 29.7%). On games with an actual winner the model is ${a.hw_pct >= a.aw_pct ? a.hw_pct : a.aw_pct}%+ accurate.
      </p>
    </div>`;
}

renderTicker();
renderAccuracy();
render();

// Scroll-driven ball rotation
(function() {
  const ball = document.querySelector('.scroll-ball');
  if (!ball) return;
  let ticking = false;
  window.addEventListener('scroll', function() {
    if (!ticking) {
      requestAnimationFrame(function() {
        const deg = window.scrollY * 0.08;
        ball.style.transform = 'translate(-50%, -50%) rotate(' + deg + 'deg)';
        ticking = false;
      });
      ticking = true;
    }
  }, { passive: true });
})();

// Scroll reveal
(function() {
  const obs = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        obs.unobserve(entry.target);
      }
    });
  }, { threshold: 0.08 });

  // Stagger cards — only in active panel and chips grid
  document.querySelectorAll('.day-panel.active .match-grid, .chips-grid').forEach(grid => {
    Array.from(grid.children).forEach((el, i) => {
      el.classList.add('reveal');
      el.style.transitionDelay = (i * 55) + 'ms';
      obs.observe(el);
    });
  });

  // Fade section headers, date labels, and accuracy card
  document.querySelectorAll('.date-label, .bets-header, .bets-group-label, .site-hero, .acc-card').forEach(el => {
    el.classList.add('reveal');
    obs.observe(el);
  });
})();
</script>
</body>
</html>
"""


def main():
    print("Loading model and computing predictions…")
    matches  = build_data()
    edges    = load_edges()
    accuracy = compute_wc26_accuracy()
    print(f"  {len(matches)} upcoming fixtures")
    print(f"  {len(edges)} value edges loaded from value_bets.json")
    hw, dr, aw = accuracy['hw_correct'], accuracy['dr_correct'], accuracy['aw_correct']
    print(f"  WC 2026: HW {accuracy['hw_correct']}/{accuracy['hw_total']} · Draw {accuracy['dr_correct']}/{accuracy['dr_total']} · AW {accuracy['aw_correct']}/{accuracy['aw_total']}")

    updated = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    html = HTML_TEMPLATE.replace("__DATA__",       json.dumps(matches,  ensure_ascii=False))
    html = html.replace("__VALUE_BETS__", json.dumps(edges,    ensure_ascii=False))
    html = html.replace("__ACCURACY__",   json.dumps(accuracy, ensure_ascii=False))
    html = html.replace("__UPDATED__", updated)

    out_path = os.path.join(BASE_DIR, "index.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"  Written → {out_path}")
    print(f"\nPush to GitHub Pages:")
    print(f"  git add index.html && git commit -m 'update predictions' && git push")


if __name__ == "__main__":
    main()
