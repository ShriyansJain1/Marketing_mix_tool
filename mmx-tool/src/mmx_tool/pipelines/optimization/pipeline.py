from kedro.pipeline import Pipeline, node
from .node import optimize_budget, compare_budget


def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=optimize_budget,
            inputs=["feature_mmm_data", "lme_model"],
            outputs="optimized_budget",
            name="optimize_budget"
        ),
        node(
            func=compare_budget,
            inputs=["feature_mmm_data", "optimized_budget"],
            outputs="budget_comparison",
            name="compare_budget"
        )
    ])
