"""Project pipelines."""
from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from mmx_tool.pipelines import (
    data_engineering,
    feature_engineering,
    modeling
)

def register_pipelines():
    return {
        "__default__": Pipeline([
            data_engineering.create_pipeline(),
            feature_engineering.create_pipeline(),
            modeling.create_pipeline()
        ])
    }
