#!/usr/bin/env python3
"""
World Cup match outcome prediction pipeline.
Predicts Win / Draw / Loss (3-class) for competitive international matches.
Outputs: trained_model.pkl, elo_ratings.csv, confusion_xgb.png
"""

import os
import pickle
import warnings
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_PATH      = os.path.join(BASE_DIR, "archive-3", "results.csv")
SHOOTOUT_PATH  = os.path.join(BASE_DIR, "archive-3", "shootouts.csv")
TRAIN_CUT      = pd.Timestamp("2018-01-01")   # test set starts here
VAL_CUT        = pd.Timestamp("2016-01-01")   # val set starts here (for early stopping)
DEFAULT_ELO    = 1500.0
DEFAULT_REST   = 365   # days — used when team has no prior competitive match

COMPETITIVE = {
    "FIFA World Cup",
    "FIFA World Cup qualification",
    "UEFA Euro",
    "Copa América",
    "AFC Asian Cup",
    "African Cup of Nations",
    "FIFA Confederations Cup",
    "Confederations Cup",
}

K_FACTOR = {
    "FIFA World Cup":            30,
    "FIFA Confederations Cup":   25,
    "Confederations Cup":        25,
    "UEFA Euro":                 25,
    "Copa América":              25,
    "AFC Asian Cup":             20,
    "African Cup of Nations":    20,
    "FIFA World Cup qualification": 20,
}
DEFAULT_K  = 15
FRIENDLY_K = 10   # K-factor for non-competitive matches (ELO-only pass)

TOURN_WEIGHT = {
    "FIFA World Cup":            1.0,
    "FIFA Confederations Cup":   0.9,
    "Confederations Cup":        0.9,
    "UEFA Euro":                 0.9,
    "Copa América":              0.9,
    "AFC Asian Cup":             0.8,
    "African Cup of Nations":    0.8,
    "FIFA World Cup qualification": 0.7,
}
DEFAULT_TW = 0.6

FEATURES = [
    "elo_home", "elo_away", "elo_diff",
    "home_form_5", "home_form_10",
    "away_form_5", "away_form_10",
    "h2h_home_win_rate", "h2h_draw_rate", "h2h_away_win_rate", "h2h_n_games",
    "is_neutral",
    "home_goal_diff_5", "away_goal_diff_5",
    "home_days_rest", "away_days_rest",
    "tournament_weight",
]


# ── ELO ───────────────────────────────────────────────────────────────────────
def _expected(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))


def update_elo(r_home: float, r_away: float, score_home: float, k: float):
    """Return updated (elo_home, elo_away). score_home ∈ [0, 1]."""
    e = _expected(r_home, r_away)
    return r_home + k * (score_home - e), r_away + k * ((1.0 - score_home) - (1.0 - e))


# ── Form / goal diff ──────────────────────────────────────────────────────────
def weighted_form(results: list, n: int) -> float:
    """Linearly weighted win rate over last n results (most recent = highest weight)."""
    if not results:
        return 0.5
    recent = results[-n:]
    w = np.arange(1, len(recent) + 1, dtype=float)
    w /= w.sum()
    return float(np.dot(recent, w))


def rolling_avg(values: list, n: int) -> float:
    if not values:
        return 0.0
    return float(np.mean(values[-n:]))


# ── H2H ───────────────────────────────────────────────────────────────────────
def get_h2h_features(h2h: dict, home: str, away: str, n: int = 10):
    """
    h2h[(A, B)] = list of (date, result_from_A_as_home) tuples, chronological.
    Returns (home_win_rate, draw_rate, away_win_rate, n_games) for the home/away matchup.
    """
    entries = []
    for d, r in h2h.get((home, away), []):
        entries.append((d, r))
    for d, r in h2h.get((away, home), []):
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


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data(results_path: str, shootout_path: str):
    """
    Returns (all_df, shootout_winners).

    all_df: every match with valid scores, sorted by date.
            ELO is updated on ALL rows; features are only extracted for competitive rows.
    shootout_winners: {(date, home_team, away_team): winning_team}
                      Used to give a small ELO signal to the shootout winner instead of
                      treating every penalty finish as a pure draw.
    """
    df = pd.read_csv(results_path, parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"])
    df["neutral"] = (
        df["neutral"]
        .map({True: 1, False: 0, "TRUE": 1, "FALSE": 0, "True": 1, "False": 0})
        .fillna(0)
        .astype(int)
    )
    df = df.sort_values("date").reset_index(drop=True)

    comp_df = df[df["tournament"].isin(COMPETITIVE)]
    print(f"Total matches loaded : {len(df):,}  "
          f"({df['date'].min().date()} → {df['date'].max().date()})")
    print(f"Competitive matches  : {len(comp_df):,}")
    counts = comp_df["tournament"].value_counts()
    print("Tournament breakdown:")
    for t, c in counts.items():
        print(f"  {t:<45} {c:>5,}")

    shootouts = pd.read_csv(shootout_path, parse_dates=["date"])
    shootout_winners = {
        (row["date"], row["home_team"], row["away_team"]): row["winner"]
        for _, row in shootouts.iterrows()
    }
    print(f"Shootout records     : {len(shootout_winners):,}")

    return df, shootout_winners


# ── Feature engineering ───────────────────────────────────────────────────────
def build_features(all_df: pd.DataFrame, shootout_winners: dict):
    """
    Single chronological pass over ALL matches.

    ELO is updated on every row (friendlies included, with FRIENDLY_K).
    Form, goal-diff, H2H, and feature rows are only tracked for competitive matches.

    Shootout adjustment: when a competitive match ends level AND has a shootout result,
    the ELO update uses score_home=0.75 (home wins shoot-out) or 0.25 (away wins) instead
    of 0.5, giving a small signal to the actual winner without over-weighting luck.
    Form/H2H always use the true 90-min result (draw for shoot-out matches).

    Returns (feature_df, elo_dict, team_snapshot, h2h_summary).
    """
    elo            = defaultdict(lambda: DEFAULT_ELO)
    form           = defaultdict(list)   # team -> [result, ...]  0/0.5/1
    goal_diff      = defaultdict(list)   # team -> [gd, ...]  from team perspective
    h2h            = defaultdict(list)   # (home, away) -> [(date, result), ...]
    last_comp_date : dict = {}           # date of last COMPETITIVE match per team

    rows = []
    for _, row in all_df.iterrows():
        home  = row["home_team"]
        away  = row["away_team"]
        date  = row["date"]
        tourn = row["tournament"]
        hs    = float(row["home_score"])
        as_   = float(row["away_score"])
        is_comp = tourn in COMPETITIVE

        r_home = elo[home]
        r_away = elo[away]

        # Record pre-match features only for competitive matches
        if is_comp:
            hw_r, dr_r, aw_r, ng = get_h2h_features(h2h, home, away)
            h_rest = min((date - last_comp_date[home]).days, 365) if home in last_comp_date else DEFAULT_REST
            a_rest = min((date - last_comp_date[away]).days, 365) if away in last_comp_date else DEFAULT_REST

            rows.append({
                "date":               date,
                "home_team":          home,
                "away_team":          away,
                "elo_home":           r_home,
                "elo_away":           r_away,
                "elo_diff":           r_home - r_away,
                "home_form_5":        weighted_form(form[home], 5),
                "home_form_10":       weighted_form(form[home], 10),
                "away_form_5":        weighted_form(form[away], 5),
                "away_form_10":       weighted_form(form[away], 10),
                "h2h_home_win_rate":  hw_r,
                "h2h_draw_rate":      dr_r,
                "h2h_away_win_rate":  aw_r,
                "h2h_n_games":        ng,
                "is_neutral":         row["neutral"],
                "home_goal_diff_5":   rolling_avg(goal_diff[home], 5),
                "away_goal_diff_5":   rolling_avg(goal_diff[away], 5),
                "home_days_rest":     h_rest,
                "away_days_rest":     a_rest,
                "tournament_weight":  TOURN_WEIGHT.get(tourn, DEFAULT_TW),
                "home_score":         hs,
                "away_score":         as_,
                "tournament":         tourn,
            })

        # True 90-min result (used for form, H2H, and target)
        score_h_true = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)

        # ELO score: adjust for shootout winner when match finished level
        score_h_elo = score_h_true
        if hs == as_:
            winner = shootout_winners.get((date, home, away))
            if winner == home:
                score_h_elo = 0.75   # home team won penalties — slight boost
            elif winner == away:
                score_h_elo = 0.25   # away team won penalties — slight boost

        k = K_FACTOR.get(tourn, FRIENDLY_K)
        elo[home], elo[away] = update_elo(r_home, r_away, score_h_elo, k)

        # Update form / H2H / last_date only for competitive matches
        if is_comp:
            form[home].append(score_h_true)
            form[away].append(1.0 - score_h_true if score_h_true != 0.5 else 0.5)
            goal_diff[home].append(hs - as_)
            goal_diff[away].append(as_ - hs)
            h2h[(home, away)].append((date, score_h_true))
            last_comp_date[home] = date
            last_comp_date[away] = date

    feat_df = pd.DataFrame(rows)

    # Target: 0 = Away Win, 1 = Draw, 2 = Home Win
    feat_df["target"] = np.where(
        feat_df["home_score"] > feat_df["away_score"], 2,
        np.where(feat_df["home_score"] == feat_df["away_score"], 1, 0),
    )

    # Team snapshot — final state (for predictions.py)
    snapshot = {
        team: {
            "elo":         elo[team],
            "form_5":      weighted_form(form[team], 5),
            "form_10":     weighted_form(form[team], 10),
            "goal_diff_5": rolling_avg(goal_diff[team], 5),
            "last_date":   last_comp_date.get(team),
        }
        for team in elo
    }

    # H2H summary — keep last 20 per ordered pair (enough for any n=10 query)
    h2h_summary = {
        pair: entries[-20:]
        for pair, entries in h2h.items()
        if entries
    }

    return feat_df, dict(elo), snapshot, h2h_summary


# ── Training ──────────────────────────────────────────────────────────────────
def train_models(feat_df: pd.DataFrame):
    train_df = feat_df[feat_df["date"] < VAL_CUT]
    val_df   = feat_df[(feat_df["date"] >= VAL_CUT) & (feat_df["date"] < TRAIN_CUT)]
    test_df  = feat_df[feat_df["date"] >= TRAIN_CUT]

    X_train, y_train = train_df[FEATURES].values, train_df["target"].values.astype(int)
    X_val,   y_val   = val_df[FEATURES].values,   val_df["target"].values.astype(int)
    X_test,  y_test  = test_df[FEATURES].values,  test_df["target"].values.astype(int)

    print(f"\nDataset split — train: {len(X_train):,}  val: {len(X_val):,}  test: {len(X_test):,}")
    print(f"Target dist (test) — Away:{(y_test==0).mean():.1%}  "
          f"Draw:{(y_test==1).mean():.1%}  Home:{(y_test==2).mean():.1%}")

    w_train = compute_sample_weight("balanced", y_train)

    # ── XGBoost ──
    xgb_model = xgb.XGBClassifier(
        objective             = "multi:softprob",
        num_class             = 3,
        n_estimators          = 500,
        max_depth             = 4,
        learning_rate         = 0.05,
        subsample             = 0.8,
        colsample_bytree      = 0.8,
        min_child_weight      = 3,
        gamma                 = 0.1,
        reg_alpha             = 0.1,
        reg_lambda            = 1.0,
        random_state          = 42,
        eval_metric           = "mlogloss",
        early_stopping_rounds = 30,   # XGBoost 2.x: constructor param
        verbosity             = 0,
    )
    xgb_model.fit(
        X_train, y_train,
        sample_weight = w_train,
        eval_set      = [(X_val, y_val)],
        verbose       = False,
    )
    best = xgb_model.best_iteration if hasattr(xgb_model, "best_iteration") else "n/a"
    print(f"XGBoost best iteration: {best}")

    # ── Logistic Regression baseline ──
    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_train)
    X_te_sc  = scaler.transform(X_test)

    lr_model = LogisticRegression(
        max_iter     = 2000,
        C            = 1.0,
        multi_class  = "multinomial",
        class_weight = "balanced",
        random_state = 42,
    )
    lr_model.fit(X_tr_sc, y_train)

    return xgb_model, lr_model, scaler, X_test, y_test, X_te_sc


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(name: str, y_true, y_pred, y_prob):
    print(f"\n{'─' * 52}")
    print(f"  {name}")
    print(f"{'─' * 52}")
    print(f"  Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"  Log-loss : {log_loss(y_true, y_prob):.4f}")
    print(classification_report(
        y_true, y_pred,
        target_names=["Away Win", "Draw", "Home Win"],
        digits=3,
    ))


def plot_confusion_matrix(y_true, y_pred, title: str, path: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Away Win", "Draw", "Home Win"],
        yticklabels=["Away Win", "Draw", "Home Win"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ── Save artifacts ────────────────────────────────────────────────────────────
def save_artifacts(xgb_model, elo_dict: dict, snapshot: dict, h2h_summary: dict):
    # trained_model.pkl — model + everything predictions.py needs
    artifacts = {
        "model":       xgb_model,
        "features":    FEATURES,
        "snapshot":    snapshot,
        "h2h_summary": h2h_summary,
    }
    model_path = os.path.join(BASE_DIR, "trained_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(artifacts, f)
    print(f"Saved: {model_path}")

    # elo_ratings.csv — human-readable ranking
    elo_df = (
        pd.DataFrame({"team": list(elo_dict.keys()), "elo": list(elo_dict.values())})
        .sort_values("elo", ascending=False)
        .reset_index(drop=True)
    )
    elo_path = os.path.join(BASE_DIR, "elo_ratings.csv")
    elo_df.to_csv(elo_path, index=False)
    print(f"Saved: {elo_path}  ({len(elo_df)} teams)")
    print("\nTop 10 ELO ratings:")
    print(elo_df.head(10).to_string(index=False))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Soccer Match Outcome Prediction — Pipeline")
    print("=" * 60)

    all_df, shootout_winners = load_data(DATA_PATH, SHOOTOUT_PATH)
    feat_df, elo_dict, snapshot, h2h_summary = build_features(all_df, shootout_winners)

    print(f"\nFeature matrix: {feat_df.shape[0]:,} rows × {len(FEATURES)} features")

    xgb_model, lr_model, scaler, X_test, y_test, X_te_sc = train_models(feat_df)

    xgb_pred = xgb_model.predict(X_test)
    xgb_prob = xgb_model.predict_proba(X_test)
    evaluate("XGBoost", y_test, xgb_pred, xgb_prob)
    plot_confusion_matrix(
        y_test, xgb_pred,
        "XGBoost — Test Set (2018+)",
        os.path.join(BASE_DIR, "confusion_xgb.png"),
    )

    lr_pred = lr_model.predict(X_te_sc)
    lr_prob = lr_model.predict_proba(X_te_sc)
    evaluate("Logistic Regression (baseline)", y_test, lr_pred, lr_prob)

    importances = (
        pd.Series(xgb_model.feature_importances_, index=FEATURES)
        .sort_values(ascending=False)
    )
    print("\nFeature importances (XGBoost):")
    for feat, imp in importances.items():
        bar = "█" * int(imp * 200)
        print(f"  {feat:<25} {imp:.4f}  {bar}")

    save_artifacts(xgb_model, elo_dict, snapshot, h2h_summary)

    print("\nDone.")


if __name__ == "__main__":
    main()
