#!/bin/bash
# Refresh odds + rebuild site + push to GitHub
# Runs daily via launchd — see com.worldcup.refresh.plist

set -e
cd "$(dirname "$0")"

echo "[$(date)] Refreshing World Cup predictions…"

# Pull latest odds + find value bets
python3 value_bets.py --regions us,uk,eu --edge 0.04

# Rebuild static site with new predictions + edges
python3 build_html.py

# Push to GitHub Pages
git add index.html value_bets.json
git commit -m "daily refresh: $(date '+%Y-%m-%d')" || echo "Nothing to commit"
git push

echo "[$(date)] Done."
