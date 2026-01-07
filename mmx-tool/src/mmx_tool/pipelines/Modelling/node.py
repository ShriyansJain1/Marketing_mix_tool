import statsmodels.formula.api as smf

def train_lme(df):

    formula = """
    sales ~
    calls_adstock +
    media_spend_adstock +
    impressions_adstock +
    email_open_adstock
    """

    model = smf.mixedlm(
        formula=formula,
        data=df,
        groups=df["NPI"]   # Random intercept per HCP
    )

    result = model.fit(method="lbfgs")
    return result
