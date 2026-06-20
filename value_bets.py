#!/usr/bin/env python3
"""
Value bet finder: model predictions vs live bookmaker odds.
Get a free API key at https://the-odds-api.com (500 req/month free)

Usage:
  python3 value_bets.py --api-key YOUR_KEY
  python3 value_bets.py --api-key YOUR_KEY --edge 0.05 --regions us,uk,eu
  ODDS_API_KEY=YOUR_KEY python3 value_bets.py

Saves value_bets.json for use by build_html.py.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predictions import load_artifacts, predict_match, predict_goals

# Load .env from project directory if present
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ODDS_URL  = "https://api.the-odds-api.com/v4/sports/soccer_fifa_world_cup/odds/"

# Odds API team name → our model's team name
NAME_MAP = {
    "Côte d'Ivoire":                "Ivory Coast",
    "Cote d'Ivoire":                "Ivory Coast",
    "Curacao":                      "Curaçao",
    "Democratic Republic of Congo": "DR Congo",
    "Korea Republic":               "South Korea",
    "Republic of Korea":            "South Korea",
    "USA":                          "United States",
    "Bosnia & Herzegovina":         "Bosnia and Herzegovina",
    "United States of America":     "United States",
}

BOOKMAKER_LABELS = {
    "draftkings":    "DraftKings",
    "fanduel":       "FanDuel",
    "betmgm":        "BetMGM",
    "caesars":       "Caesars",
    "pinnacle":      "Pinnacle",
    "williamhill_us": "William Hill",
    "betonlineag":   "BetOnline",
    "mybookieag":    "MyBookie",
    "bovada":        "Bovada",
    "bet365":        "Bet365",
    "unibet_eu":     "Unibet",
    "betfair":       "Betfair",
    "pointsbetus":   "PointsBet",
    "betrivers":     "BetRivers",
    "espnbet":       "ESPN Bet",
}


# ── Odds API ──────────────────────────────────────────────────────────────────
def fetch_odds(api_key: str, regions: str = "us", markets: str = "h2h,totals") -> tuple:
    params = urlencode({
        "apiKey":      api_key,
        "regions":     regions,
        "markets":     markets,
        "oddsFormat":  "decimal",
        "dateFormat":  "iso",
    })
    req = Request(f"{ODDS_URL}?{params}", headers={"Accept": "application/json"})
    with urlopen(req, timeout=12) as resp:
        remaining = resp.headers.get("x-requests-remaining", "?")
        used      = resp.headers.get("x-requests-used", "?")
        data      = json.loads(resp.read())
    return data, remaining, used


# ── Probability math ──────────────────────────────────────────────────────────
def remove_vig(outcomes: list) -> dict:
    """
    Convert decimal odds → vig-free implied probabilities.
    Normalises so probabilities sum to 1.0.
    Returns {team_or_draw_name: probability}.
    """
    raw = {o["name"]: 1.0 / o["price"] for o in outcomes}
    total = sum(raw.values())
    return {name: p / total for name, p in raw.items()}


def totals_consensus_probs(bookmakers: list, point: float = 2.5) -> dict:
    """Average vig-free over/under probs across bookmakers for a given line."""
    all_probs: dict = {}
    counts:    dict = {}
    for bm in bookmakers:
        for mkt in bm.get("markets", []):
            if mkt["key"] != "totals":
                continue
            relevant = [o for o in mkt["outcomes"] if abs(o.get("point", 0) - point) < 0.01]
            if len(relevant) != 2:
                continue
            vf = remove_vig(relevant)
            for name, p in vf.items():
                all_probs[name] = all_probs.get(name, 0.0) + p
                counts[name]    = counts.get(name, 0) + 1
    if not counts:
        return {}
    return {name: all_probs[name] / counts[name] for name in all_probs}


def best_totals_odds(bookmakers: list, point: float = 2.5) -> dict:
    """Best decimal price for Over/Under at a given line. Returns {name: (price, book)}."""
    best: dict = {}
    for bm in bookmakers:
        label = BOOKMAKER_LABELS.get(bm["key"], bm.get("title", bm["key"]))
        for mkt in bm.get("markets", []):
            if mkt["key"] != "totals":
                continue
            for o in mkt["outcomes"]:
                if abs(o.get("point", 0) - point) >= 0.01:
                    continue
                name, price = o["name"], o["price"]
                if name not in best or price > best[name][0]:
                    best[name] = (price, label)
    return best


def best_odds_per_outcome(bookmakers: list) -> dict:
    """
    For each outcome (home / draw / away) find the bookmaker offering
    the highest decimal price. Returns {name: (price, bookmaker_title)}.
    """
    best: dict = {}
    for bm in bookmakers:
        label = BOOKMAKER_LABELS.get(bm["key"], bm.get("title", bm["key"]))
        for mkt in bm.get("markets", []):
            if mkt["key"] != "h2h":
                continue
            for o in mkt["outcomes"]:
                name, price = o["name"], o["price"]
                if name not in best or price > best[name][0]:
                    best[name] = (price, label)
    return best


def consensus_probs(bookmakers: list) -> dict:
    """
    Average vig-free implied probability across all bookmakers that offer h2h.
    Returns {name: avg_probability} or {} if none.
    """
    all_probs: dict = {}
    counts:    dict = {}
    for bm in bookmakers:
        for mkt in bm.get("markets", []):
            if mkt["key"] != "h2h":
                continue
            vf = remove_vig(mkt["outcomes"])
            for name, p in vf.items():
                all_probs[name] = all_probs.get(name, 0.0) + p
                counts[name]    = counts.get(name, 0) + 1
    if not counts:
        return {}
    return {name: all_probs[name] / counts[name] for name in all_probs}


# ── Name resolution ───────────────────────────────────────────────────────────
def resolve(api_name: str, snapshot_keys: set) -> str:
    """Map Odds API team name to our model's team name."""
    if api_name in snapshot_keys:
        return api_name
    mapped = NAME_MAP.get(api_name)
    if mapped and mapped in snapshot_keys:
        return mapped
    # fuzzy fallback: case-insensitive substring
    api_lower = api_name.lower()
    for k in snapshot_keys:
        if api_lower in k.lower() or k.lower() in api_lower:
            return k
    return api_name   # return as-is; model will use default ELO


# ── Core analysis ─────────────────────────────────────────────────────────────
def analyse(event: dict, arts: dict, min_edge: float):
    """
    Compare model probabilities vs consensus market probabilities.
    Returns a result dict if any value edge is found, else None.
    """
    bms = event.get("bookmakers", [])
    if not bms:
        return None

    market_probs = consensus_probs(bms)
    if len(market_probs) != 3:
        return None   # skip if market doesn't have all 3 outcomes

    snap = arts["snapshot"]
    home_api = event["home_team"]
    away_api = event["away_team"]
    home = resolve(home_api, snap.keys())
    away = resolve(away_api, snap.keys())

    model        = predict_match(home, away, neutral=True, tournament_weight=1.0, artifacts=arts)
    goals_model  = predict_goals(home, away, neutral=True, tournament_weight=1.0, artifacts=arts)

    # ── 1X2 edges ──
    market = {}
    for api_name, prob in market_probs.items():
        if api_name == "Draw":
            market["Draw"] = prob
        elif api_name == home_api or api_name == home:
            market["Home Win"] = prob
        else:
            market["Away Win"] = prob

    if len(market) != 3:
        return None

    best = best_odds_per_outcome(bms)
    best_mapped = {}
    for api_name, (price, book) in best.items():
        if api_name == "Draw":
            best_mapped["Draw"] = (price, book)
        elif api_name == home_api or api_name == home:
            best_mapped["Home Win"] = (price, book)
        else:
            best_mapped["Away Win"] = (price, book)

    edges = []
    for outcome in ("Home Win", "Draw", "Away Win"):
        model_p  = model.get(outcome, 0.0)
        market_p = market.get(outcome, 0.0)
        edge     = model_p - market_p
        if edge >= min_edge:
            price, book = best_mapped.get(outcome, (None, None))
            edges.append({
                "market":   "1x2",
                "outcome":  outcome,
                "model_p":  round(model_p * 100, 1),
                "market_p": round(market_p * 100, 1),
                "edge":     round(edge * 100, 1),
                "best_odds": round(price, 2) if price else None,
                "best_book": book,
            })

    # ── Totals O/U 2.5 edges ──
    totals_market = totals_consensus_probs(bms, point=2.5)
    if totals_market and goals_model:
        best_tot = best_totals_odds(bms, point=2.5)
        # API names are "Over" / "Under"; our model labels are "Over 2.5" / "Under 2.5"
        label_map = {"Over": "Over 2.5", "Under": "Under 2.5"}
        for api_name, market_p in totals_market.items():
            our_label = label_map.get(api_name)
            if not our_label:
                continue
            model_p = goals_model.get(our_label, 0.0)
            edge    = model_p - market_p
            if edge >= min_edge:
                price, book = best_tot.get(api_name, (None, None))
                if price and price >= 100:   # skip suspended lines
                    continue
                edges.append({
                    "market":   "totals",
                    "outcome":  our_label,
                    "model_p":  round(model_p * 100, 1),
                    "market_p": round(market_p * 100, 1),
                    "edge":     round(edge * 100, 1),
                    "best_odds": round(price, 2) if price else None,
                    "best_book": book,
                })

    edges.sort(key=lambda x: -x["edge"])

    return {
        "home":        home,
        "away":        away,
        "commence":    event.get("commence_time", ""),
        "n_books":     len(bms),
        "model":       {k: round(v * 100, 1) for k, v in model.items()},
        "goals_model": {k: round(v * 100, 1) for k, v in goals_model.items()},
        "market":      {k: round(v * 100, 1) for k, v in market.items()},
        "edges":       edges,
        "has_value":   len(edges) > 0,
    }


# ── Display ───────────────────────────────────────────────────────────────────
EDGE_COLOR = {True: "\033[92m", False: "\033[0m"}
RESET = "\033[0m"
BOLD  = "\033[1m"
DIM   = "\033[2m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
RED    = "\033[91m"


def fmt_time(iso: str) -> str:
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        local = dt.astimezone()
        return local.strftime("%a %b %d · %I:%M %p")
    except Exception:
        return iso


def print_result(r: dict):
    title = f"{r['home']}  vs  {r['away']}"
    print(f"\n{'─' * 54}")
    if r["has_value"]:
        print(f"  {BOLD}{YELLOW}⚡ VALUE  {title}{RESET}")
    else:
        print(f"  {DIM}{title}{RESET}")
    print(f"  {DIM}{fmt_time(r['commence'])}  ·  {r['n_books']} bookmakers{RESET}")
    print()

    # Three-column probability table
    labels = ["Home Win", "Draw", "Away Win"]
    header = f"  {'Outcome':<14}  {'Model':>7}  {'Market':>7}  {'Edge':>7}"
    print(header)
    print(f"  {'─'*46}")
    for label in labels:
        model_p  = r["model"].get(label, 0)
        market_p = r["market"].get(label, 0)
        edge     = model_p - market_p
        is_val   = edge >= 0
        edge_str = f"+{edge:.1f}%" if edge >= 0 else f"{edge:.1f}%"
        color    = GREEN if edge >= 5 else (YELLOW if edge >= 2 else (DIM if edge < 0 else ""))
        print(f"  {label:<14}  {model_p:>6.1f}%  {market_p:>6.1f}%  {color}{edge_str:>7}{RESET}")

    if r["has_value"]:
        print()
        for e in r["edges"]:
            odds_str = f"  Best odds: {BOLD}{e['best_odds']}{RESET} @ {e['best_book']}" if e["best_odds"] else ""
            print(f"  {GREEN}✅ {e['outcome']}  +{e['edge']:.1f}% edge{RESET}{odds_str}")


def print_summary(results: list, min_edge: float):
    value = [r for r in results if r["has_value"]]
    print(f"\n{'═' * 54}")
    print(f"  SUMMARY  ·  edge threshold ≥ {min_edge*100:.0f}%")
    print(f"{'═' * 54}")
    print(f"  Matches analysed : {len(results)}")
    print(f"  Value bets found : {GREEN}{len(value)}{RESET}")
    if value:
        print(f"\n  {BOLD}Best edges:{RESET}")
        flat = []
        for r in value:
            for e in r["edges"]:
                flat.append((r["home"], r["away"], e["outcome"], e["edge"],
                             e["best_odds"], e["best_book"]))
        flat.sort(key=lambda x: -x[3])
        for home, away, outcome, edge, odds, book in flat[:8]:
            odds_str = f"  →  {odds} @ {book}" if odds else ""
            print(f"    {home} vs {away}  ·  {outcome}  +{edge:.1f}%{odds_str}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Find value bets vs live bookmaker odds")
    parser.add_argument("--api-key",  default=os.getenv("ODDS_API_KEY"), help="The Odds API key")
    parser.add_argument("--edge",     type=float, default=0.05, help="Min edge to flag (default 0.05 = 5%%)")
    parser.add_argument("--regions",  default="us", help="Bookmaker regions: us,uk,eu,au (default: us)")
    parser.add_argument("--save",     default="value_bets.json", help="Output JSON path")
    args = parser.parse_args()

    if not args.api_key:
        print("Error: provide --api-key or set ODDS_API_KEY env var")
        print("Get a free key at https://the-odds-api.com")
        sys.exit(1)

    print(f"Loading model…")
    arts = load_artifacts()

    print(f"Fetching live odds (regions: {args.regions})…")
    try:
        events, remaining, used = fetch_odds(args.api_key, regions=args.regions)
    except HTTPError as e:
        print(f"Odds API error {e.code}: {e.reason}")
        if e.code == 401:
            print("  → Invalid API key")
        elif e.code == 422:
            print("  → No odds available yet (market may not be open)")
        sys.exit(1)

    print(f"  {len(events)} events  ·  {used} requests used  ·  {remaining} remaining\n")

    results = []
    for event in events:
        r = analyse(event, arts, args.edge)
        if r:
            results.append(r)
            print_result(r)

    if not results:
        print("No events with h2h markets found.")
        return

    print_summary(results, args.edge)

    # Save JSON for build_html.py or other tooling
    out = os.path.join(BASE_DIR, args.save)
    with open(out, "w") as f:
        json.dump({
            "generated": datetime.now().isoformat(),
            "min_edge":  args.edge,
            "regions":   args.regions,
            "results":   results,
        }, f, indent=2)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
