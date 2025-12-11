"""Configuration loader for the causal agent pipeline."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for an LLM model."""

    model: str


@dataclass(frozen=True)
class DataConfig:
    """Configuration for data processing."""

    sample_chunks: int
    chunk_size: int


@dataclass(frozen=True)
class PipelineConfig:
    """Full pipeline configuration."""

    structure_proposal: ModelConfig
    workers: ModelConfig
    prior_elicitation: ModelConfig
    data: DataConfig


def _find_config_path() -> Path:
    """Find config.yaml by walking up from this file to the project root."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        config_path = parent / "config.yaml"
        if config_path.exists():
            return config_path
    raise FileNotFoundError("config.yaml not found in any parent directory")


@lru_cache(maxsize=1)
def load_config() -> PipelineConfig:
    """Load and parse the pipeline configuration.

    Returns cached config on subsequent calls.
    """
    config_path = _find_config_path()

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    return PipelineConfig(
        structure_proposal=ModelConfig(**raw["structure_proposal"]),
        workers=ModelConfig(**raw["workers"]),
        prior_elicitation=ModelConfig(**raw["prior_elicitation"]),
        data=DataConfig(**raw["data"]),
    )


def get_config() -> PipelineConfig:
    """Get the pipeline configuration."""
    return load_config()
