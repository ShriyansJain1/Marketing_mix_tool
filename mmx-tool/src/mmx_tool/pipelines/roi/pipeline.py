from kedro.pipeline import Pipeline, node
from .node import compute_roi, compute_marginal_roi


def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=compute_roi,
            inputs=["feature_mmm_data", "lme_model"],
            outputs="roi_table",
            name="compute_roi"
        ),
        node(
            func=compute_marginal_roi,
            inputs=["feature_mmm_data", "lme_model", "params:saturation_alpha"],
            outputs="marginal_roi",
            name="compute_marginal_roi"
        )
    ])
