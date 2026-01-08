from kedro.pipeline import Pipeline, node
from .node import (
    train_valid_split,
    train_lme_model,
    predict_lme_model
)


def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=train_valid_split,
            inputs="feature_mmm_data",
            outputs=["train_data", "valid_data"],
            name="train_valid_split"
        ),
        node(
            func=train_lme_model,
            inputs="train_data",
            outputs="lme_model",
            name="train_lme_model"
        ),
        node(
            func=predict_lme_model,
            inputs=["lme_model", "valid_data"],
            outputs="validation_predictions",
            name="predict_lme_model"
        )
    ])
