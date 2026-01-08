from kedro.pipeline import Pipeline, node
from .node import extract_npi_random_effects


def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=extract_npi_random_effects,
            inputs="lme_model",
            outputs="npi_random_effects",
            name="extract_npi_random_effects"
        )
    ])
