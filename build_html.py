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
            "h2h_hw":    round(hw_r * 100),
            "h2h_dr":    round(dr_r * 100),
            "h2h_aw":    round(aw_r * 100),
            "h2h_n":     ng,
            "over25":    round(goals.get("Over 2.5", 0) * 100, 1),
            "under25":   round(goals.get("Under 2.5", 0) * 100, 1),
            "btts_yes":  round(goals.get("BTTS Yes", 0) * 100, 1),
            "btts_no":   round(goals.get("BTTS No", 0) * 100, 1),
        })

    return matches


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
                "home":      r["home"],
                "away":      r["away"],
                "date":      match_date,
                "outcome":   e["outcome"],
                "model_p":   e["model_p"],
                "edge":      e["edge"],
                "best_odds": e["best_odds"],
                "best_book": e["best_book"] or "",
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
  <title>2026 World Cup Predictions</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      background: #f5f5f5;
      color: #111;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
      min-height: 100vh;
    }

    .container { max-width: 1080px; margin: 0 auto; padding: 48px 20px 80px; }

    /* ── Header ── */
    .site-header { margin-bottom: 40px; }
    .site-header h1 {
      font-size: clamp(1.4rem, 3vw, 2rem);
      font-weight: 700; letter-spacing: -0.5px; color: #111;
    }
    .site-header .sub { margin-top: 4px; color: #888; font-size: 0.85rem; }
    .updated-badge {
      display: inline-block; margin-top: 10px;
      background: #fff; border: 1px solid #e5e5e5;
      border-radius: 99px; padding: 3px 12px;
      font-size: 0.75rem; color: #999;
    }

    /* ── Value edges ── */
    .edges-card {
      background: #fff;
      border: 1px solid #e5e5e5;
      border-radius: 12px;
      padding: 20px 24px 16px;
      margin-bottom: 36px;
    }
    .edges-header {
      display: flex; align-items: center; gap: 8px;
      margin-bottom: 16px; flex-wrap: wrap;
    }
    .edges-title { font-size: 0.95rem; font-weight: 700; color: #111; }
    .edges-sub {
      margin-left: auto; font-size: 0.72rem; color: #16a34a;
      background: #f0fdf4; border: 1px solid #bbf7d0;
      border-radius: 99px; padding: 2px 10px; font-weight: 600;
    }
    .edges-scroll { overflow-x: auto; }
    .edges-table { width: 100%; border-collapse: collapse; font-size: 0.83rem; min-width: 560px; }
    .edges-table thead th {
      text-align: left; color: #999;
      font-size: 0.68rem; text-transform: uppercase; letter-spacing: 1px;
      padding: 0 14px 10px; border-bottom: 1px solid #e5e5e5; white-space: nowrap;
    }
    .edges-table tbody tr { border-bottom: 1px solid #f5f5f5; transition: background 0.1s; }
    .edges-table tbody tr:hover { background: #fafafa; }
    .edges-table td { padding: 10px 14px; vertical-align: middle; }
    .e-rank  { color: #ccc; font-size: 0.7rem; width: 28px; }
    .e-match { font-weight: 600; color: #111; }
    .e-vs    { color: #ccc; font-weight: 400; margin: 0 6px; }
    .e-wesay { color: #555; font-size: 0.82rem; }
    .e-edge  { font-weight: 700; white-space: nowrap; font-size: 0.88rem; }
    .e-odds  { color: #555; white-space: nowrap; font-size: 0.82rem; }
    .e-book  { color: #bbb; font-size: 0.75rem; }
    .obadge {
      border-radius: 6px; padding: 2px 8px;
      font-size: 0.7rem; font-weight: 600; white-space: nowrap;
    }
    .obadge-hw { background: #f0fdf4; color: #16a34a; border: 1px solid #bbf7d0; }
    .obadge-dr { background: #fffbeb; color: #d97706; border: 1px solid #fde68a; }
    .obadge-aw { background: #fef2f2; color: #dc2626; border: 1px solid #fecaca; }
    .edges-section-label {
      font-size: 0.68rem; font-weight: 700; letter-spacing: 1.5px;
      text-transform: uppercase; color: #aaa;
      margin-bottom: 10px; margin-top: 4px;
    }
    .edges-footer { margin-top: 12px; font-size: 0.68rem; color: #bbb; text-align: center; }

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
    .h2h-row {
      margin-top: 12px; padding-top: 10px; border-top: 1px solid #f0f0f0;
      display: flex; align-items: center; justify-content: center;
      gap: 5px; flex-wrap: wrap;
    }
    .h2h-label { font-size: 0.67rem; color: #bbb; font-weight: 600; letter-spacing: 0.5px; }
    .h2h-pill  { font-size: 0.68rem; border-radius: 99px; padding: 2px 8px; font-weight: 600; }
    .h2h-hw { background: #f0fdf4; color: #16a34a; }
    .h2h-dr { background: #fffbeb; color: #d97706; }
    .h2h-aw { background: #fef2f2; color: #dc2626; }
    .h2h-none { font-size: 0.68rem; color: #bbb; font-style: italic; }

    /* ── ELO ── */
    .elo-row { margin-top: 7px; text-align: center; font-size: 0.68rem; color: #ccc; }

    /* ── Footer ── */
    footer {
      margin-top: 60px; text-align: center; font-size: 0.75rem; color: #bbb;
      border-top: 1px solid #e5e5e5; padding-top: 20px;
    }
    footer a { color: #aaa; }
  </style>
</head>
<body>
<div class="container">

  <header class="site-header">
    <h1>⚽ 2026 World Cup Predictions</h1>
    <p class="sub">XGBoost · 12,426 competitive matches · 59.6% test accuracy</p>
    <span class="updated-badge">Updated __UPDATED__</span>
  </header>

  <section class="edges-card" id="edges-section"></section>

  <main id="app"></main>

  <footer>
    XGBoost w/ rolling ELO, form, H2H &amp; goal differential ·
    <a href="https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017">Data</a>
  </footer>

</div>
<script>
const MATCHES    = __DATA__;
const VALUE_BETS = __VALUE_BETS__;

function todayISO() {
  return new Date().toISOString().slice(0, 10);
}

function edgeRows(list, startIdx) {
  return list.map((e, i) => {
    const cls = e.outcome === 'Draw'     ? 'obadge-dr'
              : e.outcome === 'Home Win' ? 'obadge-hw' : 'obadge-aw';
    const edgeColor = e.edge >= 20 ? '#16a34a' : e.edge >= 10 ? '#2563eb' : '#555';
    const weSay = e.outcome === 'Draw'     ? `Draw ${e.model_p}%`
                : e.outcome === 'Home Win' ? `${e.home} Win ${e.model_p}%`
                :                           `${e.away} Win ${e.model_p}%`;
    const oddsStr = e.best_odds
      ? `${e.best_odds} <span class="e-book">@ ${e.best_book}</span>`
      : '—';
    return `<tr>
      <td class="e-rank">${startIdx + i + 1}</td>
      <td class="e-match">${e.home}<span class="e-vs">vs</span>${e.away}</td>
      <td><span class="obadge ${cls}">${e.outcome}</span></td>
      <td class="e-wesay">${weSay}</td>
      <td class="e-edge" style="color:${edgeColor}">+${e.edge}%</td>
      <td class="e-odds">${oddsStr}</td>
    </tr>`;
  }).join('');
}

function edgeTable(list, startIdx) {
  if (!list.length) return '';
  return `
    <div class="edges-scroll">
      <table class="edges-table">
        <thead>
          <tr><th>#</th><th>Match</th><th>Bet</th><th>We Say</th><th>Edge</th><th>Odds</th></tr>
        </thead>
        <tbody>${edgeRows(list, startIdx)}</tbody>
      </table>
    </div>`;
}

function renderEdges() {
  const sec = document.getElementById('edges-section');
  if (!VALUE_BETS || VALUE_BETS.length === 0) { sec.style.display = 'none'; return; }

  const today    = todayISO();
  const todayBets    = VALUE_BETS.filter(e => e.date === today);
  const upcomingBets = VALUE_BETS.filter(e => e.date !== today);

  let inner = `
    <div class="edges-header">
      <span>🔥</span>
      <span class="edges-title">Best Value Edges</span>
      <span class="edges-sub">${VALUE_BETS.length} bets vs market</span>
    </div>`;

  if (todayBets.length) {
    inner += `
    <div class="edges-section-label">Today's Best Edges</div>
    ${edgeTable(todayBets, 0)}`;
  }

  if (upcomingBets.length) {
    inner += `
    <div class="edges-section-label" style="margin-top:${todayBets.length ? '20px' : '0'}">Upcoming Edges</div>
    ${edgeTable(upcomingBets, todayBets.length)}`;
  }

  inner += `<div class="edges-footer">Model probability vs consensus bookmaker line · bet responsibly</div>`;
  sec.innerHTML = inner;
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
        <span class="pct">${pct.toFixed(1)}%</span>
      </div>
      <div class="track"><div class="fill ${fillClass}" style="width:${pct}%"></div></div>
    </div>`;
}

function goalsSection(m) {
  const overCls  = m.over25  >= 55 ? 'highlight'     : m.over25  <= 40 ? 'highlight-red' : '';
  const bttsCls  = m.btts_yes >= 55 ? 'highlight'    : m.btts_yes <= 35 ? 'highlight-red' : '';
  return `
    <div class="goals-row">
      <span class="goals-pill ${overCls}"><span class="g-label">O/U 2.5</span>Over ${m.over25}% · Under ${m.under25}%</span>
      <span class="goals-pill ${bttsCls}"><span class="g-label">BTTS</span>Yes ${m.btts_yes}% · No ${m.btts_no}%</span>
    </div>`;
}

function h2hSection(m) {
  if (m.h2h_n === 0)
    return `<div class="h2h-row"><span class="h2h-none">No prior H2H in competitive play</span></div>`;
  return `
    <div class="h2h-row">
      <span class="h2h-label">H2H (${m.h2h_n})</span>
      <span class="h2h-pill h2h-hw">${m.home} ${m.h2h_hw}%</span>
      <span class="h2h-pill h2h-dr">Draw ${m.h2h_dr}%</span>
      <span class="h2h-pill h2h-aw">${m.away} ${m.h2h_aw}%</span>
    </div>`;
}

function card(m) {
  const best  = m.home_win >= m.draw && m.home_win >= m.away_win ? 'hw'
              : m.away_win >= m.draw ? 'aw' : 'dr';
  const venue = m.neutral ? 'Neutral venue' : `${m.home} home`;
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
      ${goalsSection(m)}
      ${h2hSection(m)}
      <div class="elo-row">ELO · ${m.home} ${m.home_elo} · ${m.away} ${m.away_elo}</div>
    </div>`;
}

function render() {
  const byDate = {};
  MATCHES.forEach(m => { if (!byDate[m.date]) byDate[m.date] = []; byDate[m.date].push(m); });

  const app = document.getElementById('app');
  Object.keys(byDate).sort().forEach(dateStr => {
    const todayTag = isToday(dateStr) ? '<span class="today-tag">TODAY</span>' : '';
    const cards = byDate[dateStr].map(card).join('');
    app.innerHTML += `
      <section class="date-section">
        <div class="date-label">${fmtDate(dateStr)}${todayTag}</div>
        <div class="match-grid">${cards}</div>
      </section>`;
  });
}

renderEdges();
render();
</script>
</body>
</html>
"""


def main():
    print("Loading model and computing predictions…")
    matches = build_data()
    edges   = load_edges()
    print(f"  {len(matches)} upcoming fixtures")
    print(f"  {len(edges)} value edges loaded from value_bets.json")

    updated = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    html = HTML_TEMPLATE.replace("__DATA__",       json.dumps(matches, ensure_ascii=False))
    html = html.replace("__VALUE_BETS__", json.dumps(edges,   ensure_ascii=False))
    html = html.replace("__UPDATED__", updated)

    out_path = os.path.join(BASE_DIR, "index.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"  Written → {out_path}")
    print(f"\nPush to GitHub Pages:")
    print(f"  git add index.html && git commit -m 'update predictions' && git push")


if __name__ == "__main__":
    main()
