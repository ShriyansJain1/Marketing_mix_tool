from kedro.pipeline import Pipeline, node
from .node import clean_mmm_data

def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=clean_mmm_data,
            inputs="raw_mmm_data",
            outputs="clean_mmm_data",
            name="clean_data"
        )
    ])
