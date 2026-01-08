import pandas as pd


def extract_npi_random_effects(model):

    re = model.random_effects

    df = (
        pd.DataFrame(re)
        .T
        .reset_index()
        .rename(columns={"index": "NPI"})
    )

    return df
