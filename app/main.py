import math
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

st.set_page_config(page_title="Credit Risk Engine", layout="wide")

CURRENCY = "₸"                       # label only; change to "$" etc. (and the ranges below)
LGD = 0.45
SCORE_BASE, SCORE_BASE_ODDS, SCORE_PDO = 600, 50, 20
MIN_LOAN = 50_000
BASE_DIR = Path(__file__).resolve().parent

INCOME_TYPES = ["Working", "Commercial associate", "State servant", "Pensioner", "Other"]
HOUSING_TYPES = ["House / apartment", "Rented apartment", "With parents", "Other"]
EDUCATION_TYPES = ["Secondary / secondary special", "Higher education",
                   "Incomplete higher", "Lower secondary", "Other"]


def col_name(prefix, value):         # must match train_model.py
    return re.sub(r"\W+", "_", f"{prefix}__{value}").strip("_")


def find_artifact(name):
    for folder in (BASE_DIR / "app", BASE_DIR):
        if (folder / name).exists():
            return folder / name
    raise FileNotFoundError(f"'{name}' not found in {BASE_DIR / 'app'} or {BASE_DIR}. "
                            "Run train_model.py first.")


@st.cache_resource
def load_artifacts():
    return (joblib.load(find_artifact("lightgbm_model.pkl")),
            list(joblib.load(find_artifact("feature_cols.pkl"))),
            joblib.load(find_artifact("model_meta.pkl")))


@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)


try:
    model, feature_cols, meta = load_artifacts()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()


def monthly_payment(principal, apr, months):
    r = apr / 12
    return principal / months if r == 0 else principal * r / (1 - (1 + r) ** -months)


def max_principal(payment, apr, months):
    r = apr / 12
    return payment * months if r == 0 else payment * (1 - (1 + r) ** -months) / r


def pd_to_score(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    factor = SCORE_PDO / math.log(2)
    offset = SCORE_BASE - factor * math.log(SCORE_BASE_ODDS)
    return int(round(min(850, max(300, offset + factor * math.log((1 - p) / p)))))


def money(x):
    return f"{CURRENCY}{x:,.0f}"


# ---------------- Sidebar: credit policy ----------------
with st.sidebar:
    st.header("Credit Policy")
    approve_cut = st.slider("Auto-approve if PD ≤", 0.01, 0.20, float(meta.get("approve_pd", 0.06)), 0.01)
    review_cut = st.slider("Decline if PD >", 0.05, 0.40, float(meta.get("decline_pd", 0.12)), 0.01)
    max_dsr = st.slider("Max total debt-service ratio (DSR)", 0.20, 0.70, 0.50, 0.05)
    max_lti = st.slider("Max loan-to-annual-income", 0.5, 10.0, 3.0, 0.5)
    min_age = st.number_input("Minimum age", 18, 25, 21)
    max_age_maturity = st.number_input("Max age at loan maturity", 55, 80, 65)
    min_income = st.number_input(f"Minimum annual income ({CURRENCY}) - placeholder", 0, 10_000_000,
                                 600_000, 50_000, help="Set to 12 x the current official subsistence minimum.")
    min_emp_years = st.number_input("Minimum employment (years)", 0.0, 5.0, 0.5, 0.5)
    st.caption(f"Model base default rate: {meta.get('base_rate', 0):.1%}. Cut-offs default to values "
               "from train_model.py (break-even analysis); recalibrate on your own portfolio.")

# ---------------- Inputs ----------------
st.title("💳 Microfinance Credit Risk & Decision Engine")
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Applicant Information")
    age = st.number_input("Age", 18, 100, 38)
    income = st.number_input(f"Annual Income ({CURRENCY})", 100_000, 500_000_000, 6_000_000, 100_000)
    max_emp = float(max(age - 16, 0))
    emp_years = st.number_input("Years in current employment", 0.0, max_emp, min(3.0, max_emp), 0.5)
    existing_debt = st.number_input(f"Existing monthly debt payments ({CURRENCY})", 0, 10_000_000, 0, 10_000)

    st.markdown("**Requested Facility**")
    credit = st.number_input(f"Requested Loan Amount ({CURRENCY})", MIN_LOAN, 200_000_000, 1_500_000, 50_000)
    term = st.number_input("Loan term (months)", 6, 120, 36)
    apr = st.number_input("Annual interest rate (APR %)", 0.0, 60.0, 12.0, 0.5) / 100
    annuity = monthly_payment(credit, apr, term)
    st.caption(f"Calculated monthly instalment: **{money(annuity)}**")

    st.markdown("**External Credit Scores** (from bureau, 0–1)")
    no_ext2 = st.checkbox("No bureau score / no credit history (Score 2)")
    ext_source_2 = np.nan if no_ext2 else st.slider("External Score 2", 0.0, 1.0, 0.65)
    no_ext3 = st.checkbox("No bureau score / no credit history (Score 3)")
    ext_source_3 = np.nan if no_ext3 else st.slider("External Score 3", 0.0, 1.0, 0.65)

    income_type = st.selectbox("Income Type", INCOME_TYPES)
    housing_type = st.selectbox("Housing Type", HOUSING_TYPES)
    education = st.selectbox("Education", EDUCATION_TYPES)

    prev_count = st.number_input("Previous Applications Count", 0, 20, 3)
    approved_count = st.number_input("Previous Approved Count", 0, 20, 3)
    refused_count = st.number_input("Previous Refused Count", 0, 20, 0)

if approved_count + refused_count > prev_count:
    col1.error("Approved + refused cannot exceed total previous applications.")
    st.stop()

# ---------------- Features (same definitions as train_model.py) ----------------
is_pensioner = income_type == "Pensioner"          # training data has no tenure for pensioners
emp_model = np.nan if is_pensioner else float(emp_years)
row = {c: 0.0 for c in feature_cols}
row.update({
    "AGE_YEARS": float(age),
    "EMPLOY_YEARS": emp_model,
    "EMPLOY_TO_AGE": emp_model / age if not is_pensioner else np.nan,
    "CREDIT_INCOME_PERCENT": credit / (income + 1),
    "ANNUITY_INCOME_PERCENT": annuity / (income + 1),
    "CREDIT_TERM": annuity / (credit + 1),
    "EXT_SOURCE_2": ext_source_2,
    "EXT_SOURCE_3": ext_source_3,
    "PREV_APP_COUNT": float(prev_count),
    "PREV_APPROVED_COUNT": float(approved_count),
    "PREV_REFUSED_COUNT": float(refused_count),
})
for prefix, val in {"NAME_INCOME_TYPE": income_type, "NAME_HOUSING_TYPE": housing_type,
                    "NAME_EDUCATION_TYPE": education}.items():
    c = col_name(prefix, val)
    if c in row:                                   # "Other" -> all zeros
        row[c] = 1.0
X = pd.DataFrame([row])[feature_cols].astype(float)

pd_score = float(model.predict_proba(X)[0, 1])
credit_score = pd_to_score(pd_score)
expected_loss = pd_score * LGD * credit

# ---------------- Affordability, policy, decision ----------------
monthly_income = income / 12
dsr = (annuity + existing_debt) / monthly_income
lti = credit / income
age_at_maturity = age + term / 12
max_pmt = max_dsr * monthly_income - existing_debt
max_affordable = max_principal(max_pmt, apr, term) if max_pmt > 0 else 0.0

hard_declines, referrals = [], []
if age < min_age:
    hard_declines.append(f"Age {age} is below the minimum of {min_age}.")
if age_at_maturity > max_age_maturity:
    hard_declines.append(f"Age at maturity ({age_at_maturity:.0f}) exceeds {max_age_maturity}.")
if income < min_income:
    hard_declines.append(f"Income is below the minimum of {money(min_income)}.")
if dsr > max_dsr:
    hard_declines.append(f"Total DSR {dsr:.0%} exceeds the {max_dsr:.0%} limit (unaffordable).")
if lti > max_lti:
    hard_declines.append(f"Loan is {lti:.1f}x annual income (limit {max_lti:.1f}x).")

if not is_pensioner and emp_years < min_emp_years:
    referrals.append(f"Employment history under {min_emp_years:g} years.")
if max_dsr * 0.85 < dsr <= max_dsr:
    referrals.append(f"DSR {dsr:.0%} is close to the {max_dsr:.0%} limit.")
if refused_count >= 2:
    referrals.append(f"{refused_count} previous refusals.")
if no_ext2 or no_ext3:
    referrals.append("No bureau score on file (thin-file applicant).")

if hard_declines:
    badge = "🔴 DECLINED – POLICY"
elif pd_score > review_cut:
    badge = "🔴 DECLINED – RISK"
elif pd_score > approve_cut or referrals:
    badge = "🟡 MANUAL REVIEW"
else:
    badge = "🟢 APPROVED"

# ---------------- Output ----------------
with col2:
    st.subheader("Decision & Underwriting Analysis")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Probability of Default", f"{pd_score:.2%}")
    m2.metric("Credit Score (300–850)", credit_score)
    m3.metric("Expected Loss (PD×LGD×EAD)", money(expected_loss))
    m4.metric("Decision", badge)

    st.markdown("**Affordability**")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Monthly instalment", money(annuity))
    a2.metric("Total DSR", f"{dsr:.0%}", help=f"Policy limit {max_dsr:.0%}")
    a3.metric("Loan / income", f"{lti:.1f}x", help=f"Policy limit {max_lti:.1f}x")
    a4.metric("Max affordable loan", money(max_affordable))

    for msg in hard_declines:
        st.error(msg)
    for msg in referrals:
        st.warning(msg)
    if not hard_declines and pd_score > review_cut:
        st.error(f"Model PD {pd_score:.1%} is above the {review_cut:.0%} decline cut-off.")
    elif not hard_declines and pd_score > approve_cut:
        st.warning(f"Model PD {pd_score:.1%} is in the review band ({approve_cut:.0%}–{review_cut:.0%}).")
    if badge.startswith("🔴") and max_affordable >= MIN_LOAN:
        st.info(f"Counter-offer: up to {money(max_affordable)} would fit the DSR limit.")
    if is_pensioner:
        st.caption("Pensioner: employment tenure is not used by the model.")

    st.markdown("---")
    st.subheader("Top Decision Drivers (SHAP)")
    try:
        sv = get_explainer(model)(X)
        vals, base = sv.values, sv.base_values
        if vals.ndim == 3:
            vals, base = vals[:, :, 1], base[:, 1]
        vals, base = vals[0], float(np.ravel(base)[0])
        exp = shap.Explanation(values=vals, base_values=base, data=X.values[0], feature_names=feature_cols)
        shap.plots.waterfall(exp, max_display=8, show=False)
        st.pyplot(plt.gcf(), clear_figure=True)
        order = np.argsort(vals)
        up = [feature_cols[i] for i in order[::-1][:3] if vals[i] > 0]
        down = [feature_cols[i] for i in order[:3] if vals[i] < 0]
        if up:
            st.markdown("**Main factors increasing risk (reason codes):** " + ", ".join(up))
        if down:
            st.markdown("**Main factors reducing risk:** " + ", ".join(down))
    except Exception as e:
        st.warning(f"SHAP explanation unavailable: {e}")

st.caption("Decision-support tool only. Final credit decisions require human oversight, "
           "model validation and compliance review.")