import pandas as pd
import numpy as np

def generate_applicant_narrative(row: pd.Series) -> str:
  
    days_birth = row.get('DAYS_BIRTH', np.nan)
    age = int(abs(days_birth) // 365.25) if pd.notnull(days_birth) else "Unknown age"
    
    income = row.get('AMT_INCOME_TOTAL', 0)
    credit = row.get('AMT_CREDIT', 0)
    income_type = str(row.get('NAME_INCOME_TYPE', 'Unspecified')).lower()
    housing_type = str(row.get('NAME_HOUSING_TYPE', 'Unspecified housing')).lower()
    organization = str(row.get('ORGANIZATION_TYPE', 'unspecified industry')).lower()
    
    prev_count = int(row.get('PREV_APP_COUNT', 0))
    approved_count = int(row.get('PREV_APPROVED_COUNT', 0))
    refused_count = int(row.get('PREV_REFUSED_COUNT', 0))
    purpose = str(row.get('PRIMARY_PURPOSE', 'general purposes')).lower()

    if prev_count == 0:
        history_str = "First-time borrower with no recorded credit history in previous applications."
    else:
        history_str = (
            f"Holds {prev_count} previous applications ({approved_count} approved, {refused_count} refused), "
            f"primarily requested for {purpose}."
        )

    narrative = (
        f"Applicant is a {age}-year-old {income_type} worker in {organization}, residing in {housing_type}. "
        f"Requesting a credit amount of {credit:,.0f} USD with an annual reported income of {income:,.0f} USD. "
        f"{history_str}"
    )
    
    return narrative

def batch_generate_narratives(df: pd.DataFrame) -> pd.Series:
    
    return df.apply(generate_applicant_narrative, axis=1)
