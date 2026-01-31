"""Prompts for the orchestrator LLM agents.

Two-stage approach following Anderson & Gerbing (1988):
1. Latent Model - theoretical constructs + causal edges, NO DATA
2. Measurement Model - operationalize constructs into indicators, WITH DATA
"""

from . import latent_model, measurement_model, prior_elicitation

# Re-export with legacy names for backwards compatibility
LATENT_MODEL_SYSTEM = latent_model.SYSTEM
LATENT_MODEL_USER = latent_model.USER
LATENT_MODEL_REVIEW = latent_model.REVIEW

MEASUREMENT_MODEL_SYSTEM = measurement_model.SYSTEM
MEASUREMENT_MODEL_USER = measurement_model.USER
MEASUREMENT_MODEL_REVIEW = measurement_model.REVIEW
PROXY_REQUEST_SYSTEM = measurement_model.PROXY_SYSTEM
PROXY_REQUEST_USER = measurement_model.PROXY_USER

PRIOR_ELICITATION_SYSTEM = prior_elicitation.SYSTEM
PRIOR_ELICITATION_USER = prior_elicitation.USER
PRIOR_ELICITATION_PARAPHRASE = prior_elicitation.PARAPHRASE

__all__ = [
    # Modules
    "latent_model",
    "measurement_model",
    "prior_elicitation",
    # Legacy exports
    "LATENT_MODEL_SYSTEM",
    "LATENT_MODEL_USER",
    "LATENT_MODEL_REVIEW",
    "MEASUREMENT_MODEL_SYSTEM",
    "MEASUREMENT_MODEL_USER",
    "MEASUREMENT_MODEL_REVIEW",
    "PROXY_REQUEST_SYSTEM",
    "PROXY_REQUEST_USER",
    "PRIOR_ELICITATION_SYSTEM",
    "PRIOR_ELICITATION_USER",
    "PRIOR_ELICITATION_PARAPHRASE",
]
