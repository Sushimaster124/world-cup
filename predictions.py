#!/usr/bin/env python3
"""
Predict international soccer match outcomes.
Usage: python predictions.py "Brazil" "Argentina"
       python predictions.py "Brazil" "Argentina" false   # Brazil has home advantage
Run pipeline.py first to generate trained_model.pkl.
"""

import os
import pickle
import sys
from datetime import date

import numpy as np

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_PATH = os.path.join(BASE_DIR, "trained_model.pkl")

DEFAULT_ELO  = 1500.0
DEFAULT_REST = 365


# ── Artifact loading ──────────────────────────────────────────────────────────
def load_artifacts(path: str = ARTIFACTS_PATH) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at {path}. Run pipeline.py first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


# ── H2H (mirrors pipeline.py logic exactly) ───────────────────────────────────
def get_h2h_features(h2h_summary: dict, home: str, away: str, n: int = 10):
    entries = []
    for d, r in h2h_summary.get((home, away), []):
        entries.append((d, r))
    for d, r in h2h_summary.get((away, home), []):
        entries.append((d, 1.0 - r if r != 0.5 else 0.5))

    entries.sort(key=lambda x: x[0])
    recent = [r for _, r in entries[-n:]]
    ng = len(recent)
    if ng == 0:
        return 0.0, 0.0, 0.0, 0
    hw = sum(r == 1.0 for r in recent) / ng
    dr = sum(r == 0.5 for r in recent) / ng
    aw = sum(r == 0.0 for r in recent) / ng
    return hw, dr, aw, ng


# ── Core prediction function ──────────────────────────────────────────────────
def predict_match(
    home_team: str,
    away_team: str,
    neutral: bool = True,
    tournament_weight: float = 1.0,
    artifacts: dict = None,
) -> dict:
    """
    Returns {"Home Win": float, "Draw": float, "Away Win": float} — probabilities sum to 1.
    """
    if artifacts is None:
        artifacts = load_artifacts()

    model       = artifacts["model"]
    features    = artifacts["features"]
    snapshot    = artifacts["snapshot"]
    h2h_summary = artifacts["h2h_summary"]
    today       = date.today()

    def team_stats(team: str):
        s = snapshot.get(team, {})
        elo         = s.get("elo", DEFAULT_ELO)
        form_5      = s.get("form_5", 0.5)
        form_10     = s.get("form_10", 0.5)
        goal_diff_5 = s.get("goal_diff_5", 0.0)
        last        = s.get("last_date")
        if last is not None:
            last_d    = last.date() if hasattr(last, "date") else last
            days_rest = min((today - last_d).days, 365)
        else:
            days_rest = DEFAULT_REST
        return elo, form_5, form_10, goal_diff_5, days_rest

    e_home, hf5, hf10, hgd5, h_rest = team_stats(home_team)
    e_away, af5, af10, agd5, a_rest  = team_stats(away_team)
    hw_r, dr_r, aw_r, ng = get_h2h_features(h2h_summary, home_team, away_team)

    # Build feature vector in the same order as FEATURES in pipeline.py
    vec = np.array([[
        e_home, e_away, e_home - e_away,
        hf5, hf10,
        af5, af10,
        hw_r, dr_r, aw_r, ng,
        int(neutral),
        hgd5, agd5,
        h_rest, a_rest,
        tournament_weight,
    ]], dtype=float)

    if vec.shape[1] != len(features):
        raise ValueError(
            f"Feature mismatch: built {vec.shape[1]} features, model expects {len(features)}"
        )

    probs = model.predict_proba(vec)[0]   # [away_win, draw, home_win]
    return {
        "Away Win": float(probs[0]),
        "Draw":     float(probs[1]),
        "Home Win": float(probs[2]),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 3:
        print("Usage: python predictions.py <home_team> <away_team> [neutral=true]")
        print('       python predictions.py "Brazil" "Argentina"')
        print('       python predictions.py "Germany" "France" false')
        sys.exit(1)

    home_team = sys.argv[1]
    away_team = sys.argv[2]
    neutral   = True if len(sys.argv) < 4 else sys.argv[3].lower() not in ("false", "0", "no")

    arts = load_artifacts()
    known_teams = sorted(arts["snapshot"].keys())

    for team in (home_team, away_team):
        if team not in arts["snapshot"]:
            close = [t for t in known_teams if team.lower() in t.lower()]
            print(f"Warning: '{team}' not in dataset.", end="")
            if close:
                print(f"  Closest matches: {close[:5]}", end="")
            print()

    result = predict_match(home_team, away_team, neutral=neutral, artifacts=arts)

    venue = "Neutral venue" if neutral else f"{home_team} home advantage"
    print(f"\n{'─' * 44}")
    print(f"  {home_team}  vs  {away_team}")
    print(f"  {venue}")
    print(f"{'─' * 44}")
    for outcome, prob in sorted(result.items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 32)
        print(f"  {outcome:<12}  {prob:>5.1%}  {bar}")
    print(f"{'─' * 44}\n")


if __name__ == "__main__":
    main()
