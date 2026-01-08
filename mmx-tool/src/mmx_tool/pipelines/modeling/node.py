import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


# ----------------------------
# Train / validation split
# ----------------------------
def train_valid_split(
    df: pd.DataFrame,
    split_date: str = "2025-01-01"
):
    train_df = df[df["month"] < split_date]
    valid_df = df[df["month"] >= split_date]

    return train_df, valid_df


# ----------------------------
# Train LME model
# ----------------------------
def train_lme_model(train_df: pd.DataFrame):

    formula = """
    sales ~
    calls_sat +
    media_spend_sat +
    impressions_sat +
    email_open_sat
    """

    model = smf.mixedlm(
        formula=formula,
        data=train_df,
        groups=train_df["NPI"]            # random intercept
    )

    result = model.fit(method="lbfgs")

    return result


# ----------------------------
# Predict on validation
# ----------------------------
def predict_lme_model(model, valid_df: pd.DataFrame):

    df = valid_df.copy()
    df["predicted_sales"] = model.predict(df)

    return df

# ----------------------------
# Extract MMx coefficients
# ----------------------------
def extract_model_results(model):

    coef_df = (
        model.params
        .reset_index()
        .rename(columns={"index": "feature", 0: "coefficient"})
    )

    ci = model.conf_int().reset_index()
    ci.columns = ["feature", "ci_lower", "ci_upper"]

    results_df = coef_df.merge(ci, on="feature")

    return results_df


# ----------------------------
# Evaluate model
# ----------------------------

def evaluate_model(df: pd.DataFrame):

    y_true = df["sales"]
    y_pred = df["predicted_sales"]

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    metrics = pd.DataFrame({
        "metric": ["RMSE", "R2"],
        "value": [rmse, r2]
    })

    return metrics


# ----------------------------
# Channel contribution (MMx core)
# ----------------------------
def compute_channel_contribution(df: pd.DataFrame, model):

    channels = [
        "calls_sat",
        "media_spend_sat",
        "impressions_sat",
        "email_open_sat"
    ]

    coefs = model.params

    contrib_df = df[["month", "NPI"]].copy()

    for ch in channels:
        contrib_df[ch.replace("_sat", "_contribution")] = df[ch] * coefs[ch]

    contrib_df["total_contribution"] = contrib_df.filter(
        like="_contribution"
    ).sum(axis=1)

    return contrib_df
