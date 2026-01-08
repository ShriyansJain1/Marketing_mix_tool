import pandas as pd
import numpy as np


def compute_roi(feature_df: pd.DataFrame, model):

    channels = {
        "calls": "calls_sat",
        "media_spend": "media_spend_sat",
        "impressions": "impressions_sat",
        "email_open": "email_open_sat"
    }

    coefs = model.params
    rows = []

    for ch, feat in channels.items():
        incremental_sales = (feature_df[feat] * coefs[feat]).sum()

        if ch == "calls":
            spend = feature_df["calls"].sum()
        elif ch == "media_spend":
            spend = feature_df["media_spend"].sum()
        else:
            spend = feature_df[feat].sum()  # proxy for digital

        rows.append({
            "channel": ch,
            "incremental_sales": incremental_sales,
            "spend": spend,
            "ROI": incremental_sales / spend if spend > 0 else np.nan
        })

    return pd.DataFrame(rows)


def compute_marginal_roi(feature_df: pd.DataFrame, model, saturation_alpha: float):

    channels = {
        "calls": "calls_sat",
        "media_spend": "media_spend_sat",
        "impressions": "impressions_sat",
        "email_open": "email_open_sat"
    }

    coefs = model.params
    rows = []

    for ch, feat in channels.items():
        avg_feat = feature_df[feat].mean()
        marginal_roi = coefs[feat] * saturation_alpha / (1 + saturation_alpha * avg_feat)

        rows.append({
            "channel": ch,
            "marginal_roi": marginal_roi
        })

    return pd.DataFrame(rows)
