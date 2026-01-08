import pandas as pd


def clean_mmm_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Data engineering step for MMM:
    - Rename columns
    - Enforce types
    - Handle missing values
    - Aggregate to NPI x Month
    """

    df = df.copy()

    # ----------------------------
    # 1. Standardize column names
    # ----------------------------
    df = df.rename(columns={
    "YM_Date": "month",
    "Call Activity": "calls",
    "AMOUNT_SPENT_USD": "media_spend",
    "IMPRESSIONS": "impressions",
    "Digital RX impressions": "digital_rx_impressions",
    "EMAIL_OPEN_COUNT": "email_open",
    "EMAIL_CLICK_COUNT": "email_click",
    "HCP Interactioncs (Imperssion)": "hcp_interactions",
    "VUMEDI Data": "vumedi"
    })

    # ----------------------------
    # 2. Type enforcement
    # ----------------------------
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df["NPI"] = df["NPI"].astype(str)

    numeric_cols = [
        "sales",
        "Units",
        "calls",
        "media_spend",
        "digital_rx_impressions",
        "REACH",
        "LINK_CLICKS",
        "impressions",
        "hcp_interactions",
        "vumedi",
        "email_open",
        "email_click",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ----------------------------
    # 3. Missing value handling
    # ----------------------------
    existing_numeric_cols = [c for c in numeric_cols if c in df.columns]
    df[existing_numeric_cols] = df[existing_numeric_cols].fillna(0)

    df = df.dropna(subset=["NPI", "month"])

    # ----------------------------
    # 4. Aggregate to NPI x Month
    # ----------------------------
    agg_dict = {col: "sum" for col in existing_numeric_cols}

    df = (
        df
        .groupby(["NPI", "month"], as_index=False)
        .agg(agg_dict)
        .sort_values(["NPI", "month"])
    )

    # ----------------------------
    # 5. Sanity checks
    # ----------------------------
    if "sales" in df.columns:
        df["sales"] = df["sales"].clip(lower=0)

    if "media_spend" in df.columns:
        df["media_spend"] = df["media_spend"].clip(lower=0)

    return df
