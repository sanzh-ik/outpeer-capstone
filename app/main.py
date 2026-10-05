import math
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from scipy.sparse import csr_matrix, hstack

st.set_page_config(page_title="Credit Risk Engine", layout="wide")

# ----------------------------------------------------------------------------
# Constants (tune these to your institution / regulator)
# ----------------------------------------------------------------------------
LGD = 0.45                                   # Loss given default (Basel unsecured benchmark)
SCORE_BASE, SCORE_BASE_ODDS, SCORE_PDO = 600, 50, 20   # 600 pts = 50:1 good:bad, +20 pts doubles odds
BASE_DIR = Path(__file__).resolve().parent

INCOME_TYPES = ["State servant", "Commercial associate", "Working", "Pensioner"]
HOUSING_TYPES = ["House / apartment", "Rented apartment", "With parents"]
ORGANIZATIONS = ["Government", "Business Entity Type 3", "Self-employed", "School",
                 "Medicine", "Trade: type 7", "Other"]
PURPOSES = ["consumer goods", "home repairs", "car purchase", "education"]


# ----------------------------------------------------------------------------
# Artifact loading (paths are relative to this file, not the terminal's cwd)
# ----------------------------------------------------------------------------
def find_artifact(name: str) -> Path:
    for folder in (BASE_DIR / "app", BASE_DIR):
        p = folder / name
        if p.exists():
            return p
    raise FileNotFoundError(f"'{name}' not found in {BASE_DIR / 'app'} or {BASE_DIR}")


@st.cache_resource
def load_artifacts():
    tfidf = joblib.load(find_artifact("tfidf_vectorizer.pkl"))
    model = joblib.load(find_artifact("lightgbm_model.pkl"))
    feature_cols = list(joblib.load(find_artifact("feature_cols.pkl")))
    return tfidf, model, feature_cols


@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)


try:
    tfidf, model, feature_cols = load_artifacts()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()


# ----------------------------------------------------------------------------
# Finance helpers
# ----------------------------------------------------------------------------
def monthly_payment(principal, apr, months):
    r = apr / 12
    return principal / months if r == 0 else principal * r / (1 - (1 + r) ** -months)


def max_principal(payment, apr, months):
    r = apr / 12
    return payment * months if r == 0 else payment * (1 - (1 + r) ** -months) / r


def pd_to_score(pd_):
    """Standard scorecard scaling: score = offset + factor * ln(good:bad odds)."""
    pd_ = min(max(pd_, 1e-6), 1 - 1e-6)
    factor = SCORE_PDO / math.log(2)
    offset = SCORE_BASE - factor * math.log(SCORE_BASE_ODDS)
    return int(round(min(850, max(300, offset + factor * math.log((1 - pd_) / pd_)))))


def norm(s):
    return re.sub(r"\W+", "_", str(s)).strip("_").lower()


# ----------------------------------------------------------------------------
# Sidebar: credit policy (editable risk appetite)
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Credit Policy")
    approve_cut = st.slider("Auto-approve if PD ≤", 0.01, 0.20, 0.08, 0.01, format="%.2f")
    review_cut = st.slider("Decline if PD >", 0.05, 0.40, 0.15, 0.01, format="%.2f")
    max_dsr = st.slider("Max total debt-service ratio (DSR)", 0.20, 0.70, 0.45, 0.05)
    max_lti = st.slider("Max loan-to-annual-income", 1.0, 10.0, 5.0, 0.5)
    min_age = st.number_input("Minimum age", 18, 25, 21)
    max_age_maturity = st.number_input("Max age at loan maturity", 55, 80, 65)
    min_income = st.number_input("Minimum annual income ($)", 0, 50000, 6000)
    min_emp_years = st.number_input("Minimum employment (years)", 0.0, 5.0, 0.5, 0.5)
    st.caption("Placeholders. Calibrate cut-offs on your own validation data "
               "and align them with your regulator's guidelines.")

# ----------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------
st.title("💳 Microfinance Credit Risk & Decision Engine")
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Applicant Information")
    age = st.number_input("Age", 18, 100, 38)
    income = st.number_input("Annual Income ($)", 1000, 500000, 60000)
    max_emp = float(max(age - 16, 0))
    emp_years = st.number_input("Years in current employment", 0.0, max_emp,
                                min(3.0, max_emp), 0.5)
    existing_debt = st.number_input("Existing monthly debt payments ($)", 0, 50000, 0)

    st.markdown("**Requested Facility**")
    credit = st.number_input("Requested Loan Amount ($)", 500, 100000, 10000)
    term = st.number_input("Loan term (months)", 6, 120, 36)
    apr = st.number_input("Annual interest rate (APR %)", 0.0, 60.0, 12.0, 0.5) / 100
    annuity = monthly_payment(credit, apr, term)
    st.caption(f"Calculated monthly instalment: **${annuity:,.2f}**")

    st.markdown("**External Credit Scores** (from bureau, 0–1)")
    no_ext2 = st.checkbox("No bureau score / no credit history (Score 2)")
    ext_source_2 = np.nan if no_ext2 else st.slider("External Score 2", 0.0, 1.0, 0.65)
    no_ext3 = st.checkbox("No bureau score / no credit history (Score 3)")
    ext_source_3 = np.nan if no_ext3 else st.slider("External Score 3", 0.0, 1.0, 0.65)

    income_type = st.selectbox("Income Type", INCOME_TYPES)
    housing_type = st.selectbox("Housing Type", HOUSING_TYPES)
    organization = st.selectbox("Organization Type", ORGANIZATIONS)
    purpose = st.selectbox("Loan Purpose", PURPOSES)

    prev_count = st.number_input("Previous Applications Count", 0, 20, 3)
    approved_count = st.number_input("Previous Approved Count", 0, 20, 3)
    refused_count = st.number_input("Previous Refused Count", 0, 20, 0)

if approved_count + refused_count > prev_count:
    col1.error("Approved + refused cannot exceed total previous applications.")
    st.stop()

# Narrative: keep the SAME template your TF-IDF was trained on.
narrative = (
    f"Applicant is a {age}-year-old {income_type.lower()} worker in {organization.lower()}, "
    f"residing in {housing_type.lower()}. "
    f"Requesting a credit amount of {credit:,.0f} USD with an annual reported income of {income:,.0f} USD. "
    f"Holds {prev_count} previous applications ({approved_count} approved, {refused_count} refused), "
    f"primarily requested for {purpose.lower()}."
)

# ----------------------------------------------------------------------------
# Feature construction
# ----------------------------------------------------------------------------
days_birth = -age * 365.25
days_emp = -emp_years * 365.25

numeric = {
    "AMT_INCOME_TOTAL": income,
    "AMT_CREDIT": credit,
    "AMT_ANNUITY": annuity,
    "DAYS_BIRTH": days_birth,
    "DAYS_EMPLOYED": days_emp,
    "EXT_SOURCE_2": ext_source_2,
    "EXT_SOURCE_3": ext_source_3,
    "PREV_APP_COUNT": prev_count,
    "PREV_APPROVED_COUNT": approved_count,
    "PREV_REFUSED_COUNT": refused_count,
    "CREDIT_INCOME_PERCENT": credit / (income + 1),
    "ANNUITY_INCOME_PERCENT": annuity / (income + 1),
    "CREDIT_TERM": annuity / (credit + 1),
    "DAYS_EMPLOYED_PERCENT": days_emp / days_birth,
}
categorical = {
    "NAME_INCOME_TYPE": income_type,
    "NAME_HOUSING_TYPE": housing_type,
    "ORGANIZATION_TYPE": organization,
}

row = {c: np.nan for c in feature_cols}   # unknown stays missing (LightGBM handles NaN natively)
for c, v in numeric.items():
    if c in row:
        row[c] = float(v)
for prefix, val in categorical.items():   # fill one-hot columns if the model uses them
    for c in feature_cols:
        if c.startswith(prefix + "_"):
            row[c] = 1.0 if norm(c) == norm(f"{prefix}_{val}") else 0.0

X_tab = pd.DataFrame([row])[feature_cols].astype(float).values
coverage = int(np.sum(~np.isnan(X_tab)))
X_hybrid = hstack([csr_matrix(X_tab), tfidf.transform([narrative])]).tocsr()

pd_score = float(model.predict_proba(X_hybrid)[0, 1])
credit_score = pd_to_score(pd_score)
expected_loss = pd_score * LGD * credit

# ----------------------------------------------------------------------------
# Affordability + policy rules + decision
# ----------------------------------------------------------------------------
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
    hard_declines.append(f"Income is below the minimum of ${min_income:,.0f}.")
if dsr > max_dsr:
    hard_declines.append(f"Total DSR {dsr:.0%} exceeds the {max_dsr:.0%} limit (unaffordable).")
if lti > max_lti:
    hard_declines.append(f"Loan is {lti:.1f}x annual income (limit {max_lti:.1f}x).")

if emp_years < min_emp_years:
    referrals.append(f"Employment history under {min_emp_years:g} years.")
if no_ext2 or no_ext3:
    referrals.append("No bureau score on file (thin-file applicant).")
if dsr > 0.85 * max_dsr and dsr <= max_dsr:
    referrals.append(f"DSR {dsr:.0%} is close to the {max_dsr:.0%} limit.")
if refused_count >= 2:
    referrals.append(f"{refused_count} previous refusals.")
if coverage < 0.5 * len(feature_cols):
    referrals.append("Model has too few real inputs (most features are missing).")

if hard_declines:
    decision, badge = "DECLINED (policy)", "🔴 DECLINED – POLICY"
elif pd_score > review_cut:
    decision, badge = "DECLINED (risk)", "🔴 DECLINED – RISK"
elif pd_score > approve_cut or referrals:
    decision, badge = "REFER", "🟡 MANUAL REVIEW"
else:
    decision, badge = "APPROVED", "🟢 APPROVED"

# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------
with col2:
    st.subheader("Decision & Underwriting Analysis")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Probability of Default", f"{pd_score:.2%}")
    m2.metric("Credit Score (300–850)", credit_score)
    m3.metric("Expected Loss (PD×LGD×EAD)", f"${expected_loss:,.0f}")
    m4.metric("Decision", badge)

    st.markdown("**Affordability**")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Monthly instalment", f"${annuity:,.0f}")
    a2.metric("Total DSR", f"{dsr:.0%}", help=f"Policy limit {max_dsr:.0%}")
    a3.metric("Loan / income", f"{lti:.1f}x", help=f"Policy limit {max_lti:.1f}x")
    a4.metric("Max affordable loan", f"${max_affordable:,.0f}")

    for msg in hard_declines:
        st.error(msg)
    for msg in referrals:
        st.warning(msg)
    if decision == "DECLINED (risk)":
        st.error(f"Model PD {pd_score:.1%} is above the {review_cut:.0%} decline cut-off.")
    elif decision == "REFER" and pd_score > approve_cut:
        st.warning(f"Model PD {pd_score:.1%} is in the review band ({approve_cut:.0%}–{review_cut:.0%}).")
    if decision.startswith("DECLINED") and max_affordable >= 500:
        st.info(f"Counter-offer: up to ${max_affordable:,.0f} would fit the DSR limit.")

    st.caption(f"Model inputs populated: {coverage} of {len(feature_cols)} tabular features "
               "(the rest are treated as missing).")
    with st.expander("Narrative sent to the text model"):
        st.write(narrative)

    st.markdown("---")
    st.subheader("Top Decision Drivers (SHAP)")
    try:
        n_tab = len(feature_cols)
        X_dense = X_hybrid.toarray()
        sv = get_explainer(model)(X_dense)
        vals, base = sv.values, sv.base_values
        if vals.ndim == 3:                       # some SHAP versions return both classes
            vals, base = vals[:, :, 1], base[:, 1]
        vals, base = vals[0], float(np.ravel(base)[0])

        tab_vals = vals[:n_tab]
        text_val = vals[n_tab:].sum()            # collapse TF-IDF tokens into one bar
        names = feature_cols + ["TEXT_NARRATIVE (all tokens)"]
        exp = shap.Explanation(
            values=np.append(tab_vals, text_val),
            base_values=base,
            data=np.append(X_dense[0, :n_tab], np.nan),
            feature_names=names,
        )
        shap.plots.waterfall(exp, max_display=8, show=False)
        st.pyplot(plt.gcf(), clear_figure=True)

        order = np.argsort(tab_vals)
        up = [feature_cols[i] for i in order[::-1][:3] if tab_vals[i] > 0]
        down = [feature_cols[i] for i in order[:3] if tab_vals[i] < 0]
        if up:
            st.markdown("**Main factors increasing risk (reason codes):** " + ", ".join(up))
        if down:
            st.markdown("**Main factors reducing risk:** " + ", ".join(down))
    except Exception as e:  # e.g. model is a calibrated wrapper, not a raw tree model
        st.warning(f"SHAP explanation unavailable: {e}")

st.caption("Decision-support tool only. Final credit decisions require human oversight, "
           "model validation and compliance review.")