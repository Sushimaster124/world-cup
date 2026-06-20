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
from predictions import load_artifacts, predict_match

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
        pred = predict_match(home, away, neutral=neutral, tournament_weight=1.0, artifacts=arts)

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
            "h2h_hw":    round(hw_r * 100),
            "h2h_dr":    round(dr_r * 100),
            "h2h_aw":    round(aw_r * 100),
            "h2h_n":     ng,
        })

    return matches


HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>2026 World Cup Predictions</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      background: #0a0f1e;
      color: #e2e8f0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
      min-height: 100vh;
    }

    /* ── Layout ── */
    .container { max-width: 1080px; margin: 0 auto; padding: 40px 20px 80px; }

    /* ── Header ── */
    .site-header { text-align: center; margin-bottom: 48px; }
    .site-header h1 {
      font-size: clamp(1.6rem, 4vw, 2.4rem);
      font-weight: 800;
      letter-spacing: -0.5px;
      background: linear-gradient(135deg, #f8fafc 0%, #94a3b8 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }
    .site-header .sub {
      margin-top: 8px; color: #64748b; font-size: 0.9rem;
    }
    .updated-badge {
      display: inline-block; margin-top: 14px;
      background: #1e293b; border: 1px solid #334155;
      border-radius: 99px; padding: 4px 14px;
      font-size: 0.78rem; color: #94a3b8;
    }

    /* ── Date section ── */
    .date-section { margin-bottom: 36px; }
    .date-label {
      font-size: 0.75rem; font-weight: 700; letter-spacing: 2px;
      text-transform: uppercase; color: #475569;
      border-bottom: 1px solid #1e293b;
      padding-bottom: 8px; margin-bottom: 16px;
    }
    .date-label .today-tag {
      background: #1d4ed8; color: #bfdbfe;
      border-radius: 4px; padding: 2px 7px;
      font-size: 0.7rem; margin-left: 8px; vertical-align: middle;
    }

    /* ── Match grid ── */
    .match-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
      gap: 14px;
    }

    /* ── Match card ── */
    .card {
      background: #111827;
      border: 1px solid #1f2937;
      border-radius: 16px;
      padding: 20px 20px 16px;
      transition: border-color 0.15s, transform 0.15s;
    }
    .card:hover {
      border-color: #334155;
      transform: translateY(-2px);
    }

    .teams {
      display: flex; align-items: center; justify-content: center;
      gap: 10px; margin-bottom: 4px; flex-wrap: wrap;
    }
    .team-name {
      font-weight: 700; font-size: 0.95rem; white-space: nowrap;
    }
    .flag { font-size: 1.2rem; }
    .vs {
      background: #1f2937; color: #6b7280;
      border-radius: 6px; padding: 2px 8px;
      font-size: 0.7rem; font-weight: 700; letter-spacing: 1px;
    }

    .card-meta {
      text-align: center; font-size: 0.75rem; color: #4b5563;
      margin-bottom: 16px;
    }
    .neutral-dot {
      display: inline-block; width: 5px; height: 5px;
      background: #4b5563; border-radius: 50;
      vertical-align: middle; margin: 0 5px;
    }

    /* ── Probability bars ── */
    .bars { display: flex; flex-direction: column; gap: 8px; }
    .bar-row { }
    .bar-meta {
      display: flex; justify-content: space-between;
      font-size: 0.8rem; margin-bottom: 4px;
    }
    .bar-meta .outcome { color: #9ca3af; }
    .bar-meta .pct { font-weight: 700; }
    .bar-meta.best .outcome { color: #e2e8f0; font-weight: 600; }
    .track {
      background: #1f2937; border-radius: 99px; height: 8px; overflow: hidden;
    }
    .fill {
      height: 8px; border-radius: 99px;
      transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .fill-hw { background: linear-gradient(90deg, #16a34a, #22c55e); }
    .fill-dr { background: linear-gradient(90deg, #d97706, #f59e0b); }
    .fill-aw { background: linear-gradient(90deg, #dc2626, #ef4444); }

    /* ── H2H pill ── */
    .h2h-row {
      margin-top: 14px; padding-top: 12px;
      border-top: 1px solid #1f2937;
      display: flex; align-items: center; justify-content: center;
      gap: 6px; flex-wrap: wrap;
    }
    .h2h-label { font-size: 0.7rem; color: #4b5563; font-weight: 600; letter-spacing: 0.5px; }
    .h2h-pill {
      font-size: 0.72rem; border-radius: 99px; padding: 2px 9px; font-weight: 600;
    }
    .h2h-hw { background: #14532d; color: #86efac; }
    .h2h-dr { background: #451a03; color: #fcd34d; }
    .h2h-aw { background: #450a0a; color: #fca5a5; }
    .h2h-none { font-size: 0.72rem; color: #4b5563; font-style: italic; }

    /* ── ELO row ── */
    .elo-row {
      margin-top: 8px;
      text-align: center;
      font-size: 0.72rem; color: #374151;
    }

    /* ── Footer ── */
    footer {
      margin-top: 60px; text-align: center;
      font-size: 0.8rem; color: #334155;
      border-top: 1px solid #1e293b; padding-top: 24px;
    }
    footer a { color: #475569; }
  </style>
</head>
<body>
<div class="container">

  <header class="site-header">
    <h1>⚽ 2026 FIFA World Cup Predictions</h1>
    <p class="sub">XGBoost · trained on 12,426 competitive international matches · 59.6% test accuracy</p>
    <span class="updated-badge">Updated __UPDATED__</span>
  </header>

  <main id="app"></main>

  <footer>
    Model: XGBoost w/ rolling ELO, form, H2H &amp; goal differential features ·
    <a href="https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017">Data source</a>
  </footer>

</div>
<script>
const MATCHES = __DATA__;

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
        <span class="pct" style="color:${isBest ? '#e2e8f0' : '#6b7280'}">${pct.toFixed(1)}%</span>
      </div>
      <div class="track">
        <div class="fill ${fillClass}" style="width:${pct}%"></div>
      </div>
    </div>`;
}

function h2hSection(m) {
  if (m.h2h_n === 0) {
    return `<div class="h2h-row"><span class="h2h-none">No prior H2H in competitive play</span></div>`;
  }
  return `
    <div class="h2h-row">
      <span class="h2h-label">H2H (${m.h2h_n} games)</span>
      <span class="h2h-pill h2h-hw">${m.home} ${m.h2h_hw}%</span>
      <span class="h2h-pill h2h-dr">Draw ${m.h2h_dr}%</span>
      <span class="h2h-pill h2h-aw">${m.away} ${m.h2h_aw}%</span>
    </div>`;
}

function card(m) {
  const best = m.home_win >= m.draw && m.home_win >= m.away_win ? 'hw'
             : m.away_win >= m.draw ? 'aw' : 'dr';
  const venue = m.neutral ? 'Neutral venue' : `${m.home} home advantage`;

  return `
    <div class="card">
      <div class="teams">
        <span class="flag">${m.home_flag}</span>
        <span class="team-name">${m.home}</span>
        <span class="vs">VS</span>
        <span class="team-name">${m.away}</span>
        <span class="flag">${m.away_flag}</span>
      </div>
      <div class="card-meta">${venue}</div>
      <div class="bars">
        ${barRow(m.home + ' Win', m.home_win, 'fill-hw', best === 'hw')}
        ${barRow('Draw',          m.draw,     'fill-dr', best === 'dr')}
        ${barRow(m.away + ' Win', m.away_win, 'fill-aw', best === 'aw')}
      </div>
      ${h2hSection(m)}
      <div class="elo-row">ELO · ${m.home} ${m.home_elo} · ${m.away} ${m.away_elo}</div>
    </div>`;
}

function render() {
  const byDate = {};
  MATCHES.forEach(m => {
    if (!byDate[m.date]) byDate[m.date] = [];
    byDate[m.date].push(m);
  });

  const app = document.getElementById('app');
  Object.keys(byDate).sort().forEach(dateStr => {
    const today = isToday(dateStr);
    const label = fmtDate(dateStr);
    const todayTag = today ? '<span class="today-tag">TODAY</span>' : '';

    const cards = byDate[dateStr].map(card).join('');
    app.innerHTML += `
      <section class="date-section">
        <div class="date-label">${label}${todayTag}</div>
        <div class="match-grid">${cards}</div>
      </section>`;
  });
}

render();
</script>
</body>
</html>
"""


def main():
    print("Loading model and computing predictions…")
    matches = build_data()
    print(f"  {len(matches)} upcoming fixtures found")

    updated = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(matches, ensure_ascii=False))
    html = html.replace("__UPDATED__", updated)

    out_path = os.path.join(BASE_DIR, "index.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"  Written → {out_path}")
    print(f"\nPush to GitHub Pages:")
    print(f"  git add index.html && git commit -m 'update predictions' && git push")


if __name__ == "__main__":
    main()
