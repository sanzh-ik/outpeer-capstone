import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from scipy.sparse import hstack

st.set_page_config(page_title="Credit Risk Engine", layout="wide")

@st.cache_resource
def load_artifacts():
    tfidf = joblib.load('app/tfidf_vectorizer.pkl')
    model = joblib.load('app/lightgbm_model.pkl')
    feature_cols = joblib.load('app/feature_cols.pkl')
    return tfidf, model, feature_cols

tfidf, model, feature_cols = load_artifacts()

st.title("Microfinance Credit Risk & Decision Engine")
st.markdown("Automated underwriting platform with SHAP model explainability.")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Applicant Information")
    age = st.number_input("Age", 18, 100, 34)
    income = st.number_input("Annual Income ($)", 1000, 500000, 50000)
    credit = st.number_input("Requested Loan Amount ($)", 500, 100000, 15000)
    annuity = st.number_input("Loan Annuity ($)", 100, 20000, 1200)
    
    income_type = st.selectbox("Income Type", ["Working", "Commercial associate", "Pensioner", "State servant"])
    housing_type = st.selectbox("Housing Type", ["House / apartment", "Rented apartment", "With parents"])
    organization = st.text_input("Organization Type", "Business Entity Type 3")
    purpose = st.selectbox("Loan Purpose", ["home repairs", "car purchase", "education", "consumer goods"])
    
    prev_count = st.number_input("Previous Applications Count", 0, 20, 2)
    approved_count = st.number_input("Previous Approved Count", 0, 20, 2)
    refused_count = st.number_input("Previous Refused Count", 0, 20, 0)

narrative = (
    f"Applicant is a {age}-year-old {income_type.lower()} worker in {organization.lower()}, residing in {housing_type.lower()}. "
    f"Requesting a credit amount of {credit:,.0f} USD with an annual reported income of {income:,.0f} USD. "
    f"Holds {prev_count} previous applications ({approved_count} approved, {refused_count} refused), primarily requested for {purpose.lower()}."
)

with col2:
    st.subheader("Decision & Underwriting Analysis")
    
    input_df = pd.DataFrame([{
        'AMT_INCOME_TOTAL': income,
        'AMT_CREDIT': credit,
        'AMT_ANNUITY': annuity,
        'DAYS_BIRTH': -int(age * 365.25),
        'DAYS_EMPLOYED': -2000,
        'PREV_APP_COUNT': prev_count,
        'PREV_APPROVED_COUNT': approved_count,
        'PREV_REFUSED_COUNT': refused_count,
        'CREDIT_INCOME_PERCENT': credit / (income + 1),
        'ANNUITY_INCOME_PERCENT': annuity / (income + 1),
        'CREDIT_TERM': annuity / (credit + 1),
        'DAYS_EMPLOYED_PERCENT': -2000 / (-int(age * 365.25) + 1)
    }])
    
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0
            
    X_tab_input = input_df[feature_cols].values
    X_text_input = tfidf.transform([narrative])
    X_hybrid_input = hstack([X_tab_input, X_text_input]).tocsr()
    
    pd_score = model.predict_proba(X_hybrid_input)[0, 1]
    credit_score = int(850 - (pd_score * 550))
    
    if pd_score < 0.20:
        badge = "APPROVED"
    elif pd_score < 0.45:
        badge = "MANUAL REVIEW"
    else:
        badge = "DENIED"
        
    m1, m2, m3 = st.columns(3)
    m1.metric("Probability of Default (PD)", f"{pd_score:.2%}")
    m2.metric("Credit Score", credit_score)
    m3.metric("Decision Badge", badge)
    
    st.markdown("---")
    st.subheader("Top Decision Drivers (SHAP Explanation)")
    
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer(X_hybrid_input.toarray())
    feature_names = list(feature_cols) + list(tfidf.get_feature_names_out())
    
    fig, ax = plt.subplots(figsize=(8, 4))
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_vals.values[0],
            base_values=shap_vals.base_values[0],
            data=X_hybrid_input.toarray()[0],
            feature_names=feature_names
        ),
        max_display=7,
        show=False
    )
    st.pyplot(fig)