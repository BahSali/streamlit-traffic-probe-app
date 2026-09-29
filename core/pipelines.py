"""Backward-compatible import path for the Ixelles-Etterbeek pipeline.

The code now lives in cities/ixelles_etterbeek/pipeline.py. This module is kept
so existing imports of ``core.pipelines`` keep working.
"""
from cities.ixelles_etterbeek.pipeline import load_results_dict, run_estimation_pipeline

__all__ = ["load_results_dict", "run_estimation_pipeline"]
