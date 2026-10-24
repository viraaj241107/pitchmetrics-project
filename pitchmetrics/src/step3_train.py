# STEP 3 - Train and test the value model.
#
# Target: log(market value). Values go from €0.3m to €120m, so we predict the
# log; then a 20% miss on a cheap player counts the same as on an expensive one.
#
# Two models:
#   - XGBoost: gradient-boosted decision trees, good at non-linear patterns
#     (e.g. value rises with age until ~25 and then falls)
#   - Linear regression (Ridge): simple, and unlike trees it can go higher than
#     the most expensive player it has seen
# Final prediction = 60% XGBoost + 40% linear (on the log scale).
#
# Testing:
#   1. 5-fold cross-validation, grouped by player (the same player is never in
#      both the training part and the test part)
#   2. "Time test": train on 2019-20 to 2023-24, predict 2024-25
#
# Output: models/xgb_model.joblib, models/linear_model.joblib,
#         models/metrics.json, models/test_predictions.csv,
#         models/model_for_website.json (used by the value calculator)

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from config import MIN_MINUTES_TRAIN, MODELS, PROCESSED, XGB_WEIGHT
from features import FEATURES, add_age_year_prev, make_features


def new_xgboost():
    return XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=3, subsample=0.8,
                        colsample_bytree=0.8, min_child_weight=10, reg_lambda=2.0, random_state=42)


def new_linear():
    # fill missing values -> scale every column to mean 0 / std 1 -> ridge regression
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         RidgeCV(alphas=np.logspace(-2, 3, 20)))


def score(true_log, pred_log):
    """Easy-to-read accuracy numbers."""
    true, pred = np.exp(true_log), np.exp(pred_log)
    ratio = pred / true
    r2 = 1 - np.sum((true_log - pred_log) ** 2) / np.sum((true_log - np.mean(true_log)) ** 2)
    return {
        "r2_log": round(float(r2), 3),
        "median_pct_error": round(float(np.median(np.abs(ratio - 1)) * 100), 1),
        "within_50pct": round(float(np.mean((ratio > 0.5) & (ratio < 1.5)) * 100), 1),
        "n": int(len(true)),
    }


# ---------- load data ----------
seasons = add_age_year_prev(pd.read_csv(PROCESSED / "player_seasons.csv"))
training = pd.read_csv(PROCESSED / "training.csv")
training = training.merge(seasons[["code", "season", "year", "prev_minutes"]], on=["code", "season"])
training["age"] = training["tm_age"]                       # Transfermarkt's age at the window
training = training[training["minutes"] >= MIN_MINUTES_TRAIN].reset_index(drop=True)

X = make_features(training)
y = np.log(training["market_value"])
print("training rows:", len(X))

# ---------- test 1: cross-validation ----------
pred_xgb = np.zeros(len(y))
pred_lin = np.zeros(len(y))
for train_idx, test_idx in GroupKFold(n_splits=5).split(X, y, groups=training["tm_player_id"]):
    xgb = new_xgboost().fit(X.iloc[train_idx], y.iloc[train_idx])
    lin = new_linear().fit(X.iloc[train_idx], y.iloc[train_idx])
    pred_xgb[test_idx] = xgb.predict(X.iloc[test_idx])
    pred_lin[test_idx] = lin.predict(X.iloc[test_idx])
pred_blend = XGB_WEIGHT * pred_xgb + (1 - XGB_WEIGHT) * pred_lin

cv_scores = {"blend": score(y, pred_blend), "xgboost": score(y, pred_xgb), "linear_baseline": score(y, pred_lin)}

# ---------- test 2: train on the past, predict the newest season ----------
newest = training["season"].max()
past = training["season"] < newest
future = training["season"] == newest
xgb = new_xgboost().fit(X[past], y[past])
lin = new_linear().fit(X[past], y[past])
p_xgb, p_lin = xgb.predict(X[future]), lin.predict(X[future])
time_scores = {"blend": score(y[future], XGB_WEIGHT * p_xgb + (1 - XGB_WEIGHT) * p_lin),
               "xgboost": score(y[future], p_xgb), "linear_baseline": score(y[future], p_lin)}

for name, s in [("cross-validation", cv_scores), (f"time test ({newest})", time_scores)]:
    print(name)
    for model, numbers in s.items():
        print(f"   {model:<16} R2={numbers['r2_log']}  typical error={numbers['median_pct_error']}%  within 50%={numbers['within_50pct']}%")

# ---------- final models on all data ----------
xgb = new_xgboost().fit(X, y)
lin = new_linear().fit(X, y)
joblib.dump(xgb, MODELS / "xgb_model.joblib")
joblib.dump(lin, MODELS / "linear_model.joblib")

# the "likely range" shown on the website comes from how wrong we were in cross-validation
errors = y - pred_blend
importance = pd.Series(xgb.get_booster().get_score(importance_type="gain")).reindex(FEATURES).fillna(0)
importance = (importance / importance.sum()).sort_values(ascending=False)

metrics = {
    "cross_validation": cv_scores, "time_test": time_scores, "time_test_season": newest,
    "range_low": float(np.quantile(errors, 0.10)), "range_high": float(np.quantile(errors, 0.90)),
    "n_train": len(X), "seasons": sorted(training["season"].unique().tolist()),
    "importance": importance.round(4).to_dict(),
}
json.dump(metrics, open(MODELS / "metrics.json", "w"), indent=2)

test_predictions = training[["name", "season", "position", "market_value"]].copy()
test_predictions["predicted"] = np.exp(pred_blend)
test_predictions.to_csv(MODELS / "test_predictions.csv", index=False)

# ---------- export the model so the website can run it in the browser ----------
# XGBoost can write its trees as JSON; the linear model is just numbers.
imputer, scaler, ridge = lin.named_steps.values()
model_json = {
    "features": FEATURES,
    "xgb_weight": XGB_WEIGHT,
    "xgb_base": float(json.loads(xgb.get_booster().save_config())["learner"]["learner_model_param"]["base_score"].strip("[]")),
    "xgb_trees": [json.loads(t) for t in xgb.get_booster().get_dump(dump_format="json")],
    "lin_fill": imputer.statistics_.tolist(),
    "lin_mean": scaler.mean_.tolist(),
    "lin_std": scaler.scale_.tolist(),
    "lin_coef": ridge.coef_.tolist(),
    "lin_intercept": float(ridge.intercept_),
}
json.dump(model_json, open(MODELS / "model_for_website.json", "w"))
print("saved models")
