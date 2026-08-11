"""
Tests for Module 07: Natural Language Explainability & Engineering Insights Engine.

Covers:
  - ExplainabilityEngine initialization
  - Local template generation under DRY, WET, FLOODED, and DRYING conditions
  - KeyFactors creation and impacts
  - Risk assessment text logic
  - Recommended action generation
  - Schema integrity of EngineeringInsight output
"""

from __future__ import annotations

import numpy as np
import pytest

from app.core.condition_classifier import (
    ConditionAssessment,
    ConditionClassifier,
    SurfaceMetrics,
    TrackCondition,
    TyreCompound,
    TyreRecommendation,
)
from app.core.explainability import EngineeringInsight, ExplainabilityEngine, KeyFactor
from app.core.perception import PerceptionResult
from app.core.temporal_reasoner import (
    TemporalAnalysis,
    TyreWindowAlert,
    WetnessMomentum,
    WetnessTrend,
)


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------

def _make_perception() -> PerceptionResult:
    return PerceptionResult(
        frame_index=1,
        timestamp="2025-07-15T14:30:00Z",
        image_hash="test_hash_001",
        segmentation_mask=np.zeros((64, 64), dtype=np.uint8),
        class_proportions={"dry_surface": 0.90, "puddle": 0.10},
        clip_scores={"wet_severe": 0.20, "dry_evolved": 0.80},
        dinov2_cls_token=np.zeros(768, dtype=np.float32),
        dinov2_patch_embeddings=np.zeros((256, 768), dtype=np.float32),
        attention_map=np.zeros((16, 16), dtype=np.float32),
        confidence_agreement=0.85,
        processing_time_ms=1.5,
    )


def _make_condition(
    cond: TrackCondition = TrackCondition.DRY,
    wetness: float = 0.0,
    compound: TyreCompound = TyreCompound.SLICK_SOFT,
    pit_open: bool = False,
) -> ConditionAssessment:
    metrics = SurfaceMetrics(
        wetness_index=wetness,
        puddle_coverage_pct=wetness * 10.0,
        wet_surface_pct=wetness * 5.0,
        damp_surface_pct=0.0,
        dry_surface_pct=max(0.0, 100.0 - wetness * 15.0),
        rubber_ratio=0.10,
        clip_wet_confidence=wetness * 0.5,
        clip_dry_confidence=max(0.0, 1.0 - wetness),
        dominant_condition="dry_evolved",
    )
    rec = TyreRecommendation(
        compound=compound,
        confidence=0.85,
        lap_delta_seconds=0.0,
        reasoning="Test reasoning string.",
        pit_window_open=pit_open,
    )
    return ConditionAssessment(
        frame_index=1,
        timestamp="2025-07-15T14:30:00Z",
        image_hash="test_hash_001",
        track_condition=cond,
        metrics=metrics,
        tyre_recommendation=rec,
        condition_changed=False,
        previous_condition=None,
        processing_time_ms=1.0,
        raw_clip_scores={},
    )


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------

class TestExplainabilityEngine:
    """Tests for ExplainabilityEngine."""

    def test_engine_init(self) -> None:
        engine = ExplainabilityEngine()
        assert engine.registry is None

    def test_explain_dry_condition(self) -> None:
        engine = ExplainabilityEngine()
        perc = _make_perception()
        cond = _make_condition(TrackCondition.DRY, wetness=0.0, compound=TyreCompound.SLICK_SOFT)

        insight = engine.explain(perc, cond)
        assert isinstance(insight, EngineeringInsight)
        assert "DRY" in insight.headline
        assert "SLICK_SOFT" in insight.headline
        assert "LOCAL_TEMPLATE_ENGINE" == insight.generation_provider
        assert len(insight.key_factors) >= 4
        assert insight.processing_time_ms >= 0

    def test_explain_flooded_condition(self) -> None:
        engine = ExplainabilityEngine()
        perc = _make_perception()
        cond = _make_condition(TrackCondition.FLOODED, wetness=0.8, compound=TyreCompound.WET, pit_open=True)

        insight = engine.explain(perc, cond)
        assert "FLOODED" in insight.headline
        assert "CRITICAL" in insight.risk_assessment
        assert "Box box box" in insight.recommended_action

    def test_explain_with_temporal_analysis(self) -> None:
        engine = ExplainabilityEngine()
        perc = _make_perception()
        cond = _make_condition(TrackCondition.WET, wetness=0.4, compound=TyreCompound.WET, pit_open=True)

        temp = TemporalAnalysis(
            frame_index=1,
            window_size_actual=10,
            trend=WetnessTrend.DECREASING,
            volatility=WetnessMomentum(
                slope_per_frame=-0.01,
                r_squared=0.90,
                projected_wetness_in_5=0.10,
                projected_wetness_in_15=0.0,
            ),  # type: ignore
            momentum=WetnessMomentum(
                slope_per_frame=-0.01,
                r_squared=0.90,
                projected_wetness_in_5=0.10,
                projected_wetness_in_15=0.0,
            ),
            condition_stability_frames=5,
            dominant_condition_in_window=TrackCondition.WET,
            wetness_mean=0.45,
            wetness_std=0.05,
            tyre_window_alert=TyreWindowAlert(
                alert_active=True,
                from_compound=TyreCompound.WET,
                to_compound=TyreCompound.SLICK_SOFT,
                estimated_frames_remaining=3,
                message="Crossover window approaching",
            ),
            condition_change_count=0,
            processing_time_ms=0.5,
        )

        insight = engine.explain(perc, cond, temp)
        assert "[PIT WINDOW ALERT]" in insight.headline
        assert "Crossover window approaching" in insight.detailed_summary
        assert any(kf.category == "TEMPORAL" for kf in insight.key_factors)
