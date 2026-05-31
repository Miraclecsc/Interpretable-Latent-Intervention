"""Interpretable Latent Intervention (ILI).

Training-free, decode-time interventions on latent reasoning trajectories, as
described in "Unlocking the Black Box of Latent Reasoning: An
Interpretability-Guided Approach to Intervention".

The public surface is intentionally small:

    from ili import load_model, evaluate, InterventionConfig

Heavy entry points (``load_model``, ``evaluate``) are imported lazily so that
the pure-geometry modules (:mod:`ili.geometry`, :mod:`ili.interventions`,
:mod:`ili.probes`) can be used without the ``transformers`` dependency.
"""

from .config import INTERVENTIONS, InterventionConfig

__all__ = ["load_model", "LoadedModel", "InterventionConfig", "INTERVENTIONS", "evaluate"]


def __getattr__(name):
    if name in ("load_model", "LoadedModel"):
        from . import model_utils
        return getattr(model_utils, name)
    if name == "evaluate":
        from .runner import evaluate
        return evaluate
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
