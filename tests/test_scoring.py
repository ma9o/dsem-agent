"""Tests for DSPy scoring function."""

import json
from dataclasses import dataclass

import pytest

from causal_agent.orchestrator.scoring import (
    _count_rule_points,
    score_structure_proposal,
    score_structure_proposal_normalized,
)
from causal_agent.orchestrator.schemas import (
    CausalEdge,
    Dimension,
    DSEMStructure,
    VariableType,
)


@dataclass
class MockPrediction:
    """Mock DSPy prediction object."""

    structure: str


class TestScoreStructureProposal:
    """Tests for the main scoring function."""

    def test_invalid_json_returns_zero(self):
        """Invalid JSON should return 0."""
        pred = MockPrediction(structure="not valid json")
        assert score_structure_proposal(None, pred) == 0.0

    def test_missing_structure_field_returns_zero(self):
        """Missing structure field should return 0."""

        class BadPred:
            pass

        assert score_structure_proposal(None, BadPred()) == 0.0

    def test_schema_validation_failure_returns_zero(self):
        """Schema validation failure should return 0."""
        # Missing required fields
        pred = MockPrediction(structure='{"dimensions": [], "edges": []}')
        # This is actually valid (empty structure)
        assert score_structure_proposal(None, pred) == 0.0  # No points for empty

        # Invalid: outcome without causal_granularity
        invalid = {
            "dimensions": [
                {
                    "name": "mood",
                    "description": "test",
                    "variable_type": "outcome",
                    "base_dtype": "continuous",
                    # missing causal_granularity
                }
            ],
            "edges": [],
        }
        pred = MockPrediction(structure=json.dumps(invalid))
        assert score_structure_proposal(None, pred) == 0.0

    def test_invalid_edge_returns_zero(self):
        """Edge referencing non-existent variable should return 0."""
        invalid = {
            "dimensions": [
                {
                    "name": "mood",
                    "description": "test",
                    "variable_type": "outcome",
                    "causal_granularity": "daily",
                    "base_dtype": "continuous",
                    "aggregation": "mean",
                }
            ],
            "edges": [{"cause": "nonexistent", "effect": "mood", "description": "Test"}],
        }
        pred = MockPrediction(structure=json.dumps(invalid))
        assert score_structure_proposal(None, pred) == 0.0

    def test_valid_simple_structure_scores_positive(self):
        """Valid simple structure should score > 0."""
        valid = {
            "dimensions": [
                {
                    "name": "stress",
                    "description": "Daily stress",
                    "variable_type": "input",
                    "causal_granularity": "daily",
                    "base_dtype": "continuous",
                    "aggregation": "mean",
                },
                {
                    "name": "mood",
                    "description": "Daily mood",
                    "variable_type": "outcome",
                    "causal_granularity": "daily",
                    "base_dtype": "continuous",
                    "aggregation": "mean",
                },
            ],
            "edges": [{"cause": "stress", "effect": "mood", "description": "Stress affects mood", "lagged": False}],
        }
        pred = MockPrediction(structure=json.dumps(valid))
        score = score_structure_proposal(None, pred)
        assert score > 0

    def test_complex_structure_scores_higher(self):
        """More complex valid structure should score higher."""
        simple = {
            "dimensions": [
                {
                    "name": "X",
                    "description": "input",
                    "variable_type": "input",
                    "causal_granularity": "daily",
                    "base_dtype": "continuous",
                    "aggregation": "mean",
                },
                {
                    "name": "Y",
                    "description": "outcome",
                    "variable_type": "outcome",
                    "causal_granularity": "daily",
                    "base_dtype": "continuous",
                    "aggregation": "mean",
                },
            ],
            "edges": [{"cause": "X", "effect": "Y", "description": "X causes Y"}],
        }

        complex_struct = {
            "dimensions": [
                {
                    "name": "stress",
                    "description": "hourly stress",
                    "variable_type": "input",
                    "causal_granularity": "hourly",
                    "base_dtype": "continuous",
                    "aggregation": "mean",
                },
                {
                    "name": "sleep",
                    "description": "daily sleep",
                    "variable_type": "outcome",
                    "causal_granularity": "daily",
                    "base_dtype": "continuous",
                    "aggregation": "mean",
                },
                {
                    "name": "mood",
                    "description": "daily mood",
                    "variable_type": "outcome",
                    "causal_granularity": "daily",
                    "base_dtype": "continuous",
                    "aggregation": "mean",
                },
                {
                    "name": "age",
                    "description": "participant age",
                    "variable_type": "covariate",
                    "base_dtype": "continuous",
                },
                {
                    "name": "intercept",
                    "description": "person baseline",
                    "variable_type": "random_effect",
                    "base_dtype": "continuous",
                },
            ],
            "edges": [
                {"cause": "stress", "effect": "mood", "description": "Stress affects mood", "aggregation": "mean"},
                {"cause": "sleep", "effect": "mood", "description": "Sleep affects mood", "lagged": False},
                {"cause": "mood", "effect": "sleep", "description": "Mood affects sleep"},
            ],
        }

        simple_score = score_structure_proposal(None, MockPrediction(json.dumps(simple)))
        complex_score = score_structure_proposal(None, MockPrediction(json.dumps(complex_struct)))

        assert complex_score > simple_score


class TestCountRulePoints:
    """Tests for the internal point counting function."""

    def _make_structure(self, dims, edges):
        """Helper to create DSEMStructure."""
        return DSEMStructure(dimensions=dims, edges=edges)

    def test_outcome_dimension_points(self):
        """Outcome dimension with all correct fields scores points."""
        structure = self._make_structure(
            dims=[
                Dimension(
                    name="mood",
                    description="test",
                    variable_type=VariableType.OUTCOME,
                    causal_granularity="daily",
                    base_dtype="continuous",
                    aggregation="mean",
                )
            ],
            edges=[],
        )
        points = _count_rule_points(structure)
        # +1 valid type, +1 granularity present, +1 valid granularity,
        # +1 aggregation present, +1 valid aggregation, +1 valid dtype
        assert points >= 6

    def test_covariate_dimension_points(self):
        """Covariate dimension scores points for correct constraints."""
        structure = self._make_structure(
            dims=[
                Dimension(
                    name="age",
                    description="test",
                    variable_type=VariableType.COVARIATE,
                    base_dtype="continuous",
                )
            ],
            edges=[],
        )
        points = _count_rule_points(structure)
        # +1 valid type, +1 no granularity, +1 no aggregation, +1 valid dtype
        assert points >= 4

    def test_random_effect_bonus(self):
        """Random effect gets bonus point for modeling heterogeneity."""
        structure = self._make_structure(
            dims=[
                Dimension(
                    name="intercept",
                    description="test",
                    variable_type=VariableType.RANDOM_EFFECT,
                    base_dtype="continuous",
                )
            ],
            edges=[],
        )
        points = _count_rule_points(structure)
        # +1 valid type, +1 no granularity, +1 no aggregation, +1 valid dtype, +1 bonus
        assert points >= 5

    def test_edge_points(self):
        """Edge between valid dimensions scores points."""
        structure = self._make_structure(
            dims=[
                Dimension(
                    name="X",
                    description="input",
                    variable_type=VariableType.INPUT,
                    causal_granularity="daily",
                    base_dtype="continuous",
                    aggregation="mean",
                ),
                Dimension(
                    name="Y",
                    description="outcome",
                    variable_type=VariableType.OUTCOME,
                    causal_granularity="daily",
                    base_dtype="continuous",
                    aggregation="mean",
                ),
            ],
            edges=[CausalEdge(cause="X", effect="Y", description="X causes Y")],
        )
        points = _count_rule_points(structure)
        # Dimension points + edge points (cause exists, effect exists, effect endogenous, same timescale)
        assert points >= 12 + 4

    def test_cross_timescale_edge_bonus(self):
        """Cross-timescale edge gets bonus points."""
        structure = self._make_structure(
            dims=[
                Dimension(
                    name="hourly_stress",
                    description="hourly",
                    variable_type=VariableType.INPUT,
                    causal_granularity="hourly",
                    base_dtype="continuous",
                    aggregation="mean",
                ),
                Dimension(
                    name="daily_mood",
                    description="daily",
                    variable_type=VariableType.OUTCOME,
                    causal_granularity="daily",
                    base_dtype="continuous",
                    aggregation="mean",
                ),
            ],
            edges=[CausalEdge(cause="hourly_stress", effect="daily_mood", description="Stress affects mood", aggregation="mean")],
        )
        points = _count_rule_points(structure)
        # Cross-timescale gives +2 instead of +1
        assert points >= 12 + 5  # dims + edge with bonus


class TestNormalizedScoring:
    """Tests for normalized scoring function."""

    def test_invalid_returns_zero(self):
        """Invalid structure returns 0."""
        pred = MockPrediction(structure="invalid")
        assert score_structure_proposal_normalized(None, pred) == 0.0

    def test_valid_returns_between_zero_and_one(self):
        """Valid structure returns score in [0, 1]."""
        valid = {
            "dimensions": [
                {
                    "name": "X",
                    "description": "input",
                    "variable_type": "input",
                    "causal_granularity": "daily",
                    "base_dtype": "continuous",
                    "aggregation": "mean",
                },
                {
                    "name": "Y",
                    "description": "outcome",
                    "variable_type": "outcome",
                    "causal_granularity": "daily",
                    "base_dtype": "continuous",
                    "aggregation": "mean",
                },
            ],
            "edges": [{"cause": "X", "effect": "Y", "description": "X causes Y"}],
        }
        pred = MockPrediction(structure=json.dumps(valid))
        score = score_structure_proposal_normalized(None, pred)
        assert 0 < score <= 1.0


class TestExogenousEffectViolation:
    """Test that exogenous variables as effects return 0."""

    def test_exogenous_as_effect_returns_zero(self):
        """Input (exogenous) as edge effect should fail validation."""
        invalid = {
            "dimensions": [
                {
                    "name": "mood",
                    "description": "outcome",
                    "variable_type": "outcome",
                    "causal_granularity": "daily",
                    "base_dtype": "continuous",
                    "aggregation": "mean",
                },
                {
                    "name": "weather",
                    "description": "input",
                    "variable_type": "input",
                    "causal_granularity": "daily",
                    "base_dtype": "continuous",
                    "aggregation": "mean",
                },
            ],
            "edges": [{"cause": "mood", "effect": "weather", "description": "Invalid"}],  # Invalid: exogenous effect
        }
        pred = MockPrediction(structure=json.dumps(invalid))
        assert score_structure_proposal(None, pred) == 0.0


class TestAggregationRuleViolation:
    """Test aggregation rule violations return 0."""

    def test_finer_to_coarser_without_aggregation_returns_zero(self):
        """Finer->coarser edge without aggregation should fail."""
        invalid = {
            "dimensions": [
                {
                    "name": "hourly_stress",
                    "description": "hourly",
                    "variable_type": "input",
                    "causal_granularity": "hourly",
                    "base_dtype": "continuous",
                    "aggregation": "mean",
                },
                {
                    "name": "daily_mood",
                    "description": "daily",
                    "variable_type": "outcome",
                    "causal_granularity": "daily",
                    "base_dtype": "continuous",
                    "aggregation": "mean",
                },
            ],
            "edges": [
                {"cause": "hourly_stress", "effect": "daily_mood", "description": "Missing aggregation"}
            ],
        }
        pred = MockPrediction(structure=json.dumps(invalid))
        assert score_structure_proposal(None, pred) == 0.0

    def test_coarser_to_finer_with_aggregation_returns_zero(self):
        """Coarser->finer edge with aggregation should fail."""
        invalid = {
            "dimensions": [
                {
                    "name": "weekly_stress",
                    "description": "weekly",
                    "variable_type": "input",
                    "causal_granularity": "weekly",
                    "base_dtype": "continuous",
                    "aggregation": "mean",
                },
                {
                    "name": "daily_mood",
                    "description": "daily",
                    "variable_type": "outcome",
                    "causal_granularity": "daily",
                    "base_dtype": "continuous",
                    "aggregation": "mean",
                },
            ],
            "edges": [
                {
                    "cause": "weekly_stress",
                    "effect": "daily_mood",
                    "description": "Invalid aggregation",
                    "aggregation": "mean",  # Not allowed
                }
            ],
        }
        pred = MockPrediction(structure=json.dumps(invalid))
        assert score_structure_proposal(None, pred) == 0.0
