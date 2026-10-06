import argparse
import re
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

# Anything not listed here becomes "Other" (all one-hot columns = 0).
CATS = {
    "NAME_INCOME_TYPE": ["Working", "Commercial associate", "Pensioner", "State servant"],
    "NAME_HOUSING_TYPE": ["House / apartment", "Rented apartment", "With parents"],
    "NAME_EDUCATION_TYPE": ["Secondary / secondary special", "Higher education",
                            "Incomplete higher", "Lower secondary"],
}
# Only scale-free ratios and counts, so the model works in any currency.
NUMERIC = ["AGE_YEARS", "EMPLOY_YEARS", "EMPLOY_TO_AGE", "CREDIT_INCOME_PERCENT",
           "ANNUITY_INCOME_PERCENT", "CREDIT_TERM", "EXT_SOURCE_2", "EXT_SOURCE_3",
           "PREV_APP_COUNT", "PREV_APPROVED_COUNT", "PREV_REFUSED_COUNT"]


def col_name(prefix, value):
    return re.sub(r"\W+", "_", f"{prefix}__{value}").strip("_")


FEATURE_COLS = NUMERIC + [col_name(p, v) for p, vals in CATS.items() for v in vals]


def build_features(df):
    out = pd.DataFrame(index=df.index)
    out["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
    emp = df["DAYS_EMPLOYED"].where(df["DAYS_EMPLOYED"] != 365243)   # 365243 = "not employed" flag
    out["EMPLOY_YEARS"] = -emp / 365.25
    out["EMPLOY_TO_AGE"] = out["EMPLOY_YEARS"] / out["AGE_YEARS"]
    out["CREDIT_INCOME_PERCENT"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1)
    out["ANNUITY_INCOME_PERCENT"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1)
    out["CREDIT_TERM"] = df["AMT_ANNUITY"] / (df["AMT_CREDIT"] + 1)
    for c in ["EXT_SOURCE_2", "EXT_SOURCE_3", "PREV_APP_COUNT", "PREV_APPROVED_COUNT",
              "PREV_REFUSED_COUNT"]:
        out[c] = df[c]
    for prefix, vals in CATS.items():
        for v in vals:
            out[col_name(prefix, v)] = (df[prefix] == v).astype(float)
    return out[FEATURE_COLS].astype(float)


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(here / "data" if (here / "data").exists() else here.parent / "data"))
    ap.add_argument("--out", default=str(here / "app"))
    ap.add_argument("--margin", type=float, default=0.15, help="net lifetime margin on a good loan")
    ap.add_argument("--lgd", type=float, default=0.45, help="loss given default")
    a = ap.parse_args()
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    cols = ["SK_ID_CURR", "TARGET", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "DAYS_BIRTH",
            "DAYS_EMPLOYED", "EXT_SOURCE_2", "EXT_SOURCE_3"] + list(CATS)
    train = pd.read_csv(data / "application_train.csv", usecols=cols)
    prev = pd.read_csv(data / "previous_application.csv",
                       usecols=["SK_ID_PREV", "SK_ID_CURR", "NAME_CONTRACT_STATUS"])
    prev["ok"] = (prev["NAME_CONTRACT_STATUS"] == "Approved").astype(int)
    prev["no"] = (prev["NAME_CONTRACT_STATUS"] == "Refused").astype(int)
    agg = prev.groupby("SK_ID_CURR").agg(PREV_APP_COUNT=("SK_ID_PREV", "size"),
                                         PREV_APPROVED_COUNT=("ok", "sum"),
                                         PREV_REFUSED_COUNT=("no", "sum")).reset_index()
    df = train.merge(agg, on="SK_ID_CURR", how="left")
    for c in ["PREV_APP_COUNT", "PREV_APPROVED_COUNT", "PREV_REFUSED_COUNT"]:
        df[c] = df[c].fillna(0)

    X, y = build_features(df), df["TARGET"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.18, stratify=y, random_state=42)
    X_fit, X_va, y_fit, y_va = train_test_split(X_tr, y_tr, test_size=0.15, stratify=y_tr, random_state=1)

    params = dict(learning_rate=0.03, num_leaves=31, min_child_samples=200, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, random_state=42,
                  n_jobs=-1, verbose=-1)          # NOTE: no scale_pos_weight / is_unbalance
    probe = lgb.LGBMClassifier(n_estimators=2000, **params)
    probe.fit(X_fit, y_fit, eval_set=[(X_va, y_va)], eval_metric="auc",
              callbacks=[lgb.early_stopping(100, verbose=False)])
    model = lgb.LGBMClassifier(n_estimators=int(probe.best_iteration_), **params).fit(X_tr, y_tr)

    p = model.predict_proba(X_te)[:, 1]
    print(f"Base default rate : {y.mean():.2%}   |   mean predicted PD (test): {p.mean():.2%}")
    print(f"ROC-AUC {roc_auc_score(y_te, p):.4f} | PR-AUC {average_precision_score(y_te, p):.4f} "
          f"| Brier {brier_score_loss(y_te, p):.4f}")

    bands = pd.cut(p, [0, .03, .05, .08, .12, .20, 1.0])
    cal = (pd.DataFrame({"band": bands, "pred": p, "actual": y_te}).groupby("band", observed=True)
           .agg(share=("pred", lambda s: len(s) / len(p)), mean_pred=("pred", "mean"),
                actual_default=("actual", "mean")))
    print("\nCalibration on held-out applicants (mean_pred should match actual_default):")
    print(cal.round(3).to_string())

    break_even = a.margin / (a.margin + a.lgd)       # (1-PD)*margin = PD*LGD
    decline = round(break_even * 0.5, 2)             # 50% safety buffer
    approve = round(decline / 2, 2)
    sh = lambda m: f"{m.mean():.0%} of applicants, actual default {y_te[m].mean():.1%}" if m.any() else "none"
    print(f"\nBreak-even PD = {break_even:.1%}. Suggested: approve <= {approve}, decline > {decline}")
    print(f"  approve : {sh(p <= approve)}")
    print(f"  review  : {sh((p > approve) & (p <= decline))}")
    print(f"  decline : {sh(p > decline)}")

    joblib.dump(model, out / "lightgbm_model.pkl")
    joblib.dump(FEATURE_COLS, out / "feature_cols.pkl")
    joblib.dump({"cats": CATS, "base_rate": float(y.mean()), "approve_pd": approve,
                 "decline_pd": decline}, out / "model_meta.pkl")
    print(f"\nSaved artifacts to {out}")


if __name__ == "__main__":
    main()