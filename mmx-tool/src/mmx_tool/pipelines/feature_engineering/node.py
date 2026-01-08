import numpy as np
import pandas as pd


# ----------------------------
# Adstock (carryover)
# ----------------------------
def adstock(x: pd.Series, lam: float) -> np.ndarray:
    out = np.zeros(len(x))
    for i in range(len(x)):
        out[i] = x.iloc[i] if i == 0 else x.iloc[i] + lam * out[i - 1]
    return out


# ----------------------------
# Saturation (diminishing returns)
# ----------------------------
def saturation(x: pd.Series, alpha: float = 1.0) -> pd.Series:
    return np.log1p(alpha * x)


# ----------------------------
# Feature creation
# ----------------------------
def create_features(
    df: pd.DataFrame,
    adstock_lambda: dict,
    saturation_alpha: float
) -> pd.DataFrame:

    df = df.copy()

    # IMPORTANT: must match data engineering output
    df = df.sort_values(["NPI", "month"])

    channel_cols = [
        "calls",
        "media_spend",
        "impressions",
        "email_open"
    ]

    for col in channel_cols:
        # Adstock
        df[f"{col}_adstock"] = (
            df.groupby("NPI")[col]
              .transform(lambda x: adstock(x, adstock_lambda[col]))
        )

        # Saturation
        df[f"{col}_sat"] = saturation(
            df[f"{col}_adstock"],
            saturation_alpha
        )

    return df
