"""Project pipelines."""
from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from pharma_mmm.pipelines import (
    data_engineering,
    Feature_engineering,
    Modeling
)

def register_pipelines():
    return {
        "__default__": pipeline([
            data_engineering.create_pipeline(),
            feature_engineering.create_pipeline(),
            modeling.create_pipeline()
        ])
    }
