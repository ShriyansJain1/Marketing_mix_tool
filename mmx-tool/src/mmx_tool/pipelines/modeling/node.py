import pandas as pd
import statsmodels.formula.api as smf


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
