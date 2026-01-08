"""Project pipelines."""
from __future__ import annotations

from mmx_tool.pipelines.data_engineering.pipeline import create_pipeline as de_pipeline
from mmx_tool.pipelines.feature_engineering.pipeline import create_pipeline as fe_pipeline
from mmx_tool.pipelines.modeling.pipeline import create_pipeline as modeling_pipeline
from mmx_tool.pipelines.roi.pipeline import create_pipeline as roi_pipeline
from mmx_tool.pipelines.optimization.pipeline import create_pipeline as optimization_pipeline
from mmx_tool.pipelines.npi_analysis.pipeline import create_pipeline as npi_pipeline


def register_pipelines():
    data_engineering = de_pipeline()
    feature_engineering = fe_pipeline()
    modeling = modeling_pipeline()
    roi = roi_pipeline()
    optimization = optimization_pipeline()
    npi_analysis = npi_pipeline()

    return {
        # Individual pipelines
        "data_engineering": data_engineering,
        "feature_engineering": feature_engineering,
        "modeling": modeling,
        "roi": roi,
        "optimization": optimization,
        "npi_analysis": npi_analysis,

        # Full end-to-end pipeline
        "__default__": (
            data_engineering
            + feature_engineering
            + modeling
            + roi
            + optimization
            + npi_analysis
        ),
    }
