#!/usr/bin/env python3
"""
2026 World Cup Match Predictor — Streamlit UI
Run: streamlit run app.py
"""

import os
import sys
from datetime import date

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predictions import load_artifacts, predict_match

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WC 2026 Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    /* hide streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 2rem; max-width: 1100px; }

    .page-title {
        font-size: 2rem; font-weight: 800; letter-spacing: -0.5px;
        margin-bottom: 0;
    }
    .page-sub {
        color: #888; font-size: 0.95rem; margin-bottom: 2rem;
    }
    .date-header {
        font-size: 1.05rem; font-weight: 700; color: #aaa;
        text-transform: uppercase; letter-spacing: 1px;
        border-bottom: 1px solid #2a2a2a; padding-bottom: 6px;
        margin: 24px 0 14px;
    }
    .match-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 14px;
        padding: 18px 20px 14px;
        margin-bottom: 12px;
        height: 100%;
    }
    .teams-row {
        display: flex; align-items: center; justify-content: center;
        gap: 10px; font-size: 1.05rem; font-weight: 700;
        margin-bottom: 3px; text-align: center;
    }
    .vs-badge {
        background: #1f2937; color: #6b7280;
        border-radius: 6px; padding: 2px 7px; font-size: 0.75rem;
        font-weight: 600; flex-shrink: 0;
    }
    .match-meta {
        text-align: center; font-size: 0.78rem; color: #6b7280;
        margin-bottom: 16px;
    }
    .bar-label {
        display: flex; justify-content: space-between;
        font-size: 0.82rem; margin-bottom: 3px;
    }
    .bar-label .outcome { color: #d1d5db; }
    .bar-label .pct { font-weight: 700; }
    .bar-track {
        background: #1f2937; border-radius: 99px; height: 9px; margin-bottom: 9px;
    }
    .bar-fill { border-radius: 99px; height: 9px; }
    .tag-neutral {
        display:inline-block; background:#1f2937; color:#9ca3af;
        border-radius:4px; font-size:0.72rem; padding:2px 6px;
    }
</style>
""", unsafe_allow_html=True)

# ── Flags ─────────────────────────────────────────────────────────────────────
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


def flag(team: str) -> str:
    return FLAGS.get(team, "🏳")


# ── Data & model ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def get_artifacts():
    return load_artifacts()


@st.cache_data(show_spinner="Computing predictions…")
def get_upcoming_fixtures():
    df = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "archive-3", "results.csv"),
        parse_dates=["date"],
    )
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
    return upcoming


# ── Card renderer ─────────────────────────────────────────────────────────────
def bar_html(label: str, pct: float, color: str, is_best: bool) -> str:
    bold = "font-weight:800;" if is_best else ""
    return f"""
    <div class="bar-label">
        <span class="outcome" style="{bold}">{label}</span>
        <span class="pct" style="color:{color};{bold}">{pct:.1%}</span>
    </div>
    <div class="bar-track">
        <div class="bar-fill" style="width:{pct*100:.1f}%;background:{color};"></div>
    </div>
    """


def match_card(home: str, away: str, pred: dict, match_date, neutral: bool):
    hw, dr, aw = pred["Home Win"], pred["Draw"], pred["Away Win"]
    best = max(pred, key=pred.get)

    meta_parts = [match_date.strftime("%b %d") if hasattr(match_date, "strftime") else str(match_date)]
    if neutral:
        meta_parts.append("Neutral")

    bars = (
        bar_html("Home Win", hw, "#22c55e", best == "Home Win")
        + bar_html("Draw",     dr, "#f59e0b", best == "Draw")
        + bar_html("Away Win", aw, "#ef4444", best == "Away Win")
    )

    st.markdown(f"""
    <div class="match-card">
        <div class="teams-row">
            <span>{flag(home)} {home}</span>
            <span class="vs-badge">VS</span>
            <span>{away} {flag(away)}</span>
        </div>
        <div class="match-meta">{' · '.join(meta_parts)}</div>
        {bars}
    </div>
    """, unsafe_allow_html=True)


# ── App ───────────────────────────────────────────────────────────────────────
arts = get_artifacts()
upcoming = get_upcoming_fixtures()

st.markdown('<div class="page-title">⚽ 2026 World Cup Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">ML predictions for every fixture · XGBoost trained on 100+ years of international football</div>', unsafe_allow_html=True)

tab_fixtures, tab_custom = st.tabs(["📅  Upcoming Fixtures", "🔮  Custom Match"])

# ── Tab 1: Upcoming fixtures ──────────────────────────────────────────────────
with tab_fixtures:
    if upcoming.empty:
        st.info("No upcoming fixtures found in the dataset.")
    else:
        # Date filter
        all_dates = sorted(upcoming["date"].dt.date.unique())
        selected_dates = st.multiselect(
            "Filter by date",
            options=all_dates,
            default=all_dates,
            format_func=lambda d: d.strftime("%A, %B %d"),
        )

        filtered = upcoming[upcoming["date"].dt.date.isin(selected_dates)]

        for match_date in sorted(filtered["date"].dt.date.unique()):
            day_matches = filtered[filtered["date"].dt.date == match_date]
            label = match_date.strftime("%A, %B %d")
            if match_date == date.today():
                label += "  — Today"
            st.markdown(f'<div class="date-header">{label}</div>', unsafe_allow_html=True)

            cols = st.columns(2, gap="medium")
            for i, (_, row) in enumerate(day_matches.iterrows()):
                pred = predict_match(
                    home_team         = row["home_team"],
                    away_team         = row["away_team"],
                    neutral           = bool(row["neutral"]),
                    tournament_weight = 1.0,
                    artifacts         = arts,
                )
                with cols[i % 2]:
                    match_card(
                        home       = row["home_team"],
                        away       = row["away_team"],
                        pred       = pred,
                        match_date = row["date"],
                        neutral    = bool(row["neutral"]),
                    )

# ── Tab 2: Custom predictor ───────────────────────────────────────────────────
with tab_custom:
    all_teams = sorted(arts["snapshot"].keys())

    st.markdown("#### Pick any two national teams")
    c1, c2, c3 = st.columns([3, 3, 2])
    with c1:
        home_team = st.selectbox("Home team", all_teams, index=all_teams.index("Brazil") if "Brazil" in all_teams else 0)
    with c2:
        away_options = [t for t in all_teams if t != home_team]
        default_away = "Argentina" if "Argentina" in away_options else away_options[0]
        away_team = st.selectbox("Away team", away_options, index=away_options.index(default_away))
    with c3:
        neutral = st.toggle("Neutral venue", value=True)
        st.write("")
        predict_btn = st.button("Predict  →", use_container_width=True, type="primary")

    if predict_btn:
        pred = predict_match(home_team, away_team, neutral=neutral, artifacts=arts)
        hw, dr, aw = pred["Home Win"], pred["Draw"], pred["Away Win"]
        best = max(pred, key=pred.get)

        st.divider()
        r1, r2, r3 = st.columns(3)

        def outcome_metric(col, label, pct, color, best_label):
            is_best = label == best_label
            icon = "🏆" if is_best else ""
            col.markdown(f"""
            <div style="text-align:center;padding:20px 10px;background:#111827;
                        border-radius:12px;border:{'2px solid '+color if is_best else '1px solid #1f2937'};">
                <div style="font-size:0.85rem;color:#9ca3af;margin-bottom:8px;">{label} {icon}</div>
                <div style="font-size:2.4rem;font-weight:800;color:{color};">{pct:.1%}</div>
            </div>
            """, unsafe_allow_html=True)

        outcome_metric(r1, f"{home_team} Win", hw, "#22c55e", f"{home_team} Win")
        outcome_metric(r2, "Draw",             dr, "#f59e0b", "Draw")
        outcome_metric(r3, f"{away_team} Win", aw, "#ef4444", f"{away_team} Win")

        st.markdown("<br>", unsafe_allow_html=True)
        venue_str = "neutral venue" if neutral else f"{home_team} home advantage"
        st.caption(
            f"ELO — {home_team}: {arts['snapshot'].get(home_team, {}).get('elo', 1500):.0f}  |  "
            f"{away_team}: {arts['snapshot'].get(away_team, {}).get('elo', 1500):.0f}  |  "
            f"Venue: {venue_str}"
        )
