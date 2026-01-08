from kedro.pipeline import Pipeline, node
from .node import create_features


def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=create_features,
            inputs=[
                "clean_mmm_data",
                "params:adstock_lambda",
                "params:saturation_alpha"
            ],
            outputs="feature_mmm_data",
            name="feature_engineering"
        )
    ])
