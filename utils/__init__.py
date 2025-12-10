"""
Utility helpers for the option-pricing notebooks.

This package centralizes reusable plumbing (sampling helpers, IAE runners,
three-qubit setup, convergence helpers) so the notebooks can stay lean.
"""

from .utils import (  # noqa: F401
    SamplerWithShots,
    sampler_with_forced_shots,
    CountingSampler,
    QueryCountingSamplerV2,
    build_3q_components,
    make_3q_problem,
    run_iae_experiment,
    direct_sample_problem_price,
    quantum_discretized_baseline,
    load_jsonl_robust,
    summarize_rmse,
    fit_loglog_slope,
    slope_curve,
    wilson_interval,
)
