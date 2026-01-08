import numpy as np
import pandas as pd
from scipy.optimize import minimize


def optimize_budget(feature_df: pd.DataFrame, model):

    coef = model.params

    total_budget = feature_df["media_spend"].sum()

    def objective(x):
        return -(
            coef["calls_sat"] * x[0] +
            coef["media_spend_sat"] * x[1] +
            coef["impressions_sat"] * x[2] +
            coef["email_open_sat"] * x[3]
        )

    constraints = ({
        "type": "eq",
        "fun": lambda x: np.sum(x) - total_budget
    })

    bounds = [(0, total_budget)] * 4
    initial = [total_budget / 4] * 4

    result = minimize(objective, initial, bounds=bounds, constraints=constraints)

    return pd.DataFrame({
        "channel": ["calls", "media_spend", "impressions", "email_open"],
        "optimized_budget": result.x
    })


def compare_budget(feature_df: pd.DataFrame, optimized_budget: pd.DataFrame):

    current = pd.DataFrame({
        "channel": ["calls", "media_spend", "impressions", "email_open"],
        "current_budget": [
            feature_df["calls"].sum(),
            feature_df["media_spend"].sum(),
            feature_df["impressions"].sum(),
            feature_df["email_open"].sum()
        ]
    })

    return current.merge(optimized_budget, on="channel")
