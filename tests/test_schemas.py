"""Tests for DSEM schema validation."""

import pytest

from causal_agent.orchestrator.schemas import (
    CausalEdge,
    Dimension,
    DSEMStructure,
    GRANULARITY_HOURS,
)


class TestDimension:
    """Tests for Dimension validation."""

    def test_valid_endogenous_time_varying(self):
        """Endogenous time-varying dimension is valid."""
        dim = Dimension(
            name="mood",
            description="Daily mood rating",
            time_granularity="daily",
            dtype="continuous",
            role="endogenous",
        )
        assert dim.name == "mood"
        assert dim.role == "endogenous"

    def test_valid_exogenous_time_varying(self):
        """Exogenous time-varying dimension is valid."""
        dim = Dimension(
            name="weather",
            description="Daily temperature",
            time_granularity="daily",
            dtype="continuous",
            role="exogenous",
        )
        assert dim.role == "exogenous"

    def test_valid_exogenous_time_invariant(self):
        """Exogenous time-invariant dimension is valid."""
        dim = Dimension(
            name="age",
            description="Participant age",
            time_granularity=None,
            dtype="continuous",
            role="exogenous",
        )
        assert dim.time_granularity is None

    def test_valid_latent_random_effect(self):
        """Latent random effect (exogenous + time-invariant) is valid."""
        dim = Dimension(
            name="person_intercept",
            description="Person-specific baseline",
            time_granularity=None,
            dtype="continuous",
            role="exogenous",
            is_latent=True,
        )
        assert dim.is_latent is True

    def test_invalid_latent_must_be_exogenous(self):
        """Latent dimension must be exogenous."""
        with pytest.raises(ValueError, match="is_latent=True requires role='exogenous'"):
            Dimension(
                name="bad",
                description="Invalid",
                time_granularity=None,
                dtype="continuous",
                role="endogenous",
                is_latent=True,
            )

    def test_invalid_latent_must_be_time_invariant(self):
        """Latent dimension must be time-invariant."""
        with pytest.raises(ValueError, match="is_latent=True requires time_granularity=None"):
            Dimension(
                name="bad",
                description="Invalid",
                time_granularity="daily",
                dtype="continuous",
                role="exogenous",
                is_latent=True,
            )

    def test_invalid_endogenous_must_be_time_varying(self):
        """Endogenous dimension must be time-varying."""
        with pytest.raises(ValueError, match="Endogenous variables must be time-varying"):
            Dimension(
                name="bad",
                description="Invalid",
                time_granularity=None,
                dtype="continuous",
                role="endogenous",
            )


class TestCausalEdge:
    """Tests for CausalEdge."""

    def test_contemporaneous_edge(self):
        """Contemporaneous edge (lag=0) is valid."""
        edge = CausalEdge(cause="stress", effect="mood", lag=0)
        assert edge.lag == 0

    def test_lagged_edge(self):
        """Lagged edge is valid."""
        edge = CausalEdge(cause="sleep", effect="mood", lag=24)
        assert edge.lag == 24

    def test_edge_with_aggregation(self):
        """Edge with aggregation is valid."""
        edge = CausalEdge(cause="hourly_stress", effect="daily_mood", lag=24, aggregation="mean")
        assert edge.aggregation == "mean"


class TestDSEMStructure:
    """Tests for DSEMStructure validation."""

    def _make_dims(self, *specs):
        """Helper to create dimensions from (name, granularity, role) tuples."""
        return [
            Dimension(
                name=name,
                description=f"{name} description",
                time_granularity=gran,
                dtype="continuous",
                role=role,
            )
            for name, gran, role in specs
        ]

    def test_valid_simple_structure(self):
        """Simple valid structure passes validation."""
        structure = DSEMStructure(
            dimensions=self._make_dims(
                ("stress", "daily", "exogenous"),
                ("mood", "daily", "endogenous"),
            ),
            edges=[CausalEdge(cause="stress", effect="mood", lag=0)],
        )
        assert len(structure.dimensions) == 2
        assert len(structure.edges) == 1

    def test_valid_lagged_edge(self):
        """Same-timescale lagged edge with 1 unit lag is valid."""
        structure = DSEMStructure(
            dimensions=self._make_dims(
                ("sleep", "daily", "endogenous"),
                ("mood", "daily", "endogenous"),
            ),
            edges=[CausalEdge(cause="sleep", effect="mood", lag=24)],  # 1 day = 24h
        )
        assert structure.edges[0].lag == 24

    def test_invalid_edge_cause_not_in_dimensions(self):
        """Edge cause must exist in dimensions."""
        with pytest.raises(ValueError, match="Edge cause 'unknown' not in dimensions"):
            DSEMStructure(
                dimensions=self._make_dims(("mood", "daily", "endogenous")),
                edges=[CausalEdge(cause="unknown", effect="mood", lag=0)],
            )

    def test_invalid_edge_effect_not_in_dimensions(self):
        """Edge effect must exist in dimensions."""
        with pytest.raises(ValueError, match="Edge effect 'unknown' not in dimensions"):
            DSEMStructure(
                dimensions=self._make_dims(("stress", "daily", "exogenous")),
                edges=[CausalEdge(cause="stress", effect="unknown", lag=0)],
            )

    def test_invalid_exogenous_cannot_be_effect(self):
        """Exogenous variable cannot be an effect."""
        with pytest.raises(ValueError, match="Exogenous variable 'weather' cannot be an effect"):
            DSEMStructure(
                dimensions=self._make_dims(
                    ("mood", "daily", "endogenous"),
                    ("weather", "daily", "exogenous"),
                ),
                edges=[CausalEdge(cause="mood", effect="weather", lag=0)],
            )

    def test_invalid_contemporaneous_cross_timescale(self):
        """Contemporaneous edge requires same timescale."""
        with pytest.raises(ValueError, match="Contemporaneous edge.*requires same timescale"):
            DSEMStructure(
                dimensions=self._make_dims(
                    ("hourly_stress", "hourly", "exogenous"),
                    ("daily_mood", "daily", "endogenous"),
                ),
                edges=[CausalEdge(cause="hourly_stress", effect="daily_mood", lag=0)],
            )

    def test_invalid_same_scale_wrong_lag(self):
        """Same-timescale edge must have lag=0 or lag=1 unit."""
        with pytest.raises(ValueError, match="Same-timescale edge must have lag=0 or lag=24h"):
            DSEMStructure(
                dimensions=self._make_dims(
                    ("sleep", "daily", "endogenous"),
                    ("mood", "daily", "endogenous"),
                ),
                edges=[CausalEdge(cause="sleep", effect="mood", lag=48)],  # 2 days, not allowed
            )

    def test_invalid_cross_scale_wrong_lag(self):
        """Cross-timescale edge must have lag = coarser granularity."""
        with pytest.raises(ValueError, match="Cross-timescale edge must have lag=168h"):
            DSEMStructure(
                dimensions=self._make_dims(
                    ("weekly_stress", "weekly", "exogenous"),
                    ("daily_mood", "daily", "endogenous"),
                ),
                edges=[CausalEdge(cause="weekly_stress", effect="daily_mood", lag=24)],  # should be 168
            )

    def test_valid_cross_scale_coarser_to_finer(self):
        """Coarser cause -> finer effect with correct lag is valid."""
        structure = DSEMStructure(
            dimensions=self._make_dims(
                ("weekly_stress", "weekly", "exogenous"),
                ("daily_mood", "daily", "endogenous"),
            ),
            edges=[CausalEdge(cause="weekly_stress", effect="daily_mood", lag=168)],
        )
        assert structure.edges[0].lag == 168

    def test_invalid_finer_to_coarser_needs_aggregation(self):
        """Finer cause -> coarser effect requires aggregation."""
        with pytest.raises(ValueError, match="Aggregation required for finer->coarser edge"):
            DSEMStructure(
                dimensions=self._make_dims(
                    ("hourly_activity", "hourly", "exogenous"),
                    ("daily_mood", "daily", "endogenous"),
                ),
                edges=[CausalEdge(cause="hourly_activity", effect="daily_mood", lag=24)],
            )

    def test_valid_finer_to_coarser_with_aggregation(self):
        """Finer cause -> coarser effect with aggregation is valid."""
        structure = DSEMStructure(
            dimensions=self._make_dims(
                ("hourly_activity", "hourly", "exogenous"),
                ("daily_mood", "daily", "endogenous"),
            ),
            edges=[CausalEdge(cause="hourly_activity", effect="daily_mood", lag=24, aggregation="mean")],
        )
        assert structure.edges[0].aggregation == "mean"

    def test_invalid_coarser_to_finer_no_aggregation_allowed(self):
        """Coarser cause -> finer effect must not have aggregation."""
        with pytest.raises(ValueError, match="Aggregation not allowed for coarser->finer edge"):
            DSEMStructure(
                dimensions=self._make_dims(
                    ("weekly_stress", "weekly", "exogenous"),
                    ("daily_mood", "daily", "endogenous"),
                ),
                edges=[CausalEdge(cause="weekly_stress", effect="daily_mood", lag=168, aggregation="mean")],
            )

    def test_to_networkx(self):
        """Structure converts to NetworkX graph."""
        structure = DSEMStructure(
            dimensions=self._make_dims(
                ("stress", "daily", "exogenous"),
                ("mood", "daily", "endogenous"),
            ),
            edges=[CausalEdge(cause="stress", effect="mood", lag=0)],
        )
        G = structure.to_networkx()
        assert "stress" in G.nodes
        assert "mood" in G.nodes
        assert ("stress", "mood") in G.edges


class TestGranularityHours:
    """Tests for granularity constants."""

    def test_granularity_values(self):
        """Granularity hours are correct."""
        assert GRANULARITY_HOURS["hourly"] == 1
        assert GRANULARITY_HOURS["daily"] == 24
        assert GRANULARITY_HOURS["weekly"] == 168
        assert GRANULARITY_HOURS["monthly"] == 720
        assert GRANULARITY_HOURS["yearly"] == 8760
