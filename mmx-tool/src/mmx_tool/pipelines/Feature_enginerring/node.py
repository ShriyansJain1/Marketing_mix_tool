import numpy as np
import pandas as pd

def adstock(x, lam=0.5):
    out = np.zeros(len(x))
    for i in range(len(x)):
        out[i] = x.iloc[i] if i == 0 else x.iloc[i] + lam * out[i-1]
    return out

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["NPI", "YM_Date"])

    for col in ["calls", "media_spend", "impressions", "email_open"]:
        df[f"{col}_adstock"] = (
            df.groupby("NPI")[col]
              .transform(lambda x: adstock(x))
        )

    return df
