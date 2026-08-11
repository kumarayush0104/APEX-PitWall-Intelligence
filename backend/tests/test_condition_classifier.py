"""
Tests for Module 05: Surface Metrics & Condition Classifier.

All tests are pure Python — no real model inference, no GPU needed.
Covers:
  - SurfaceMetrics computation
  - Threshold-based condition classification
  - Tyre compound recommendation logic
  - Condition transition detection (WET → DRYING)
  - Full ConditionAssessment output schema
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
from app.core.perception import PerceptionResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_perception(
    *,
    puddle: float = 0.0,
    wet_surface: float = 0.0,
    damp_surface: float = 0.0,
    dry_surface: float = 1.0,
    rubber_line: float = 0.0,
    clip_wet_severe: float = 0.0,
    clip_dry_evolved: float = 1.0,
    frame_index: int = 0,
) -> PerceptionResult:
    """Build a synthetic PerceptionResult with the given surface proportions."""
    return PerceptionResult(
        frame_index=frame_index,
        timestamp="2025-07-15T14:30:00Z",
        image_hash="abc123",
        segmentation_mask=np.zeros((64, 64), dtype=np.uint8),
        class_proportions={
            "puddle":       puddle,
            "wet_surface":  wet_surface,
            "damp_surface": damp_surface,
            "dry_surface":  dry_surface,
            "rubber_line":  rubber_line,
        },
        clip_scores={
            "wet_severe":     clip_wet_severe,
            "wet_moderate":   0.0,
            "sudden_shower":  0.0,
            "transitional":   0.0,
            "drying":         0.0,
            "dry_evolved":    clip_dry_evolved,
            "dry_green":      0.0,
            "marbles_offline":0.0,
        },
        dinov2_cls_token=np.zeros(768, dtype=np.float32),
        dinov2_patch_embeddings=np.zeros((256, 768), dtype=np.float32),
        attention_map=np.zeros((16, 16), dtype=np.float32),
        confidence_agreement=0.75,
        processing_time_ms=1.0,
    )


# ---------------------------------------------------------------------------
# Module-level constants tests
# ---------------------------------------------------------------------------

class TestTrackConditionEnum:
    """Ensure the enum has the correct members."""

    def test_all_members_present(self) -> None:
        expected = {"DRY", "DAMP", "WET", "FLOODED", "DRYING", "UNSAFE"}
        actual = {m.value for m in TrackCondition}
        assert expected == actual

    def test_string_values_match_names(self) -> None:
        for member in TrackCondition:
            assert member.value == member.name


class TestTyreCompoundEnum:
    """Ensure the compound enum is complete."""

    def test_all_compounds_present(self) -> None:
        names = {m.name for m in TyreCompound}
        assert "SLICK_SOFT" in names
        assert "INTERMEDIATE" in names
        assert "WET" in names
        assert "STAY_OUT" in names


# ---------------------------------------------------------------------------
# SurfaceMetrics computation
# ---------------------------------------------------------------------------

class TestSurfaceMetrics:
    """Unit tests for ConditionClassifier._compute_metrics."""

    def test_dry_track_has_low_wetness(self) -> None:
        clf = ConditionClassifier()
        perc = _make_perception(dry_surface=0.98, rubber_line=0.02)
        m = clf._compute_metrics(perc)
        assert m.wetness_index < 0.05
        assert m.puddle_coverage_pct == pytest.approx(0.0)
        assert m.dry_surface_pct > 90.0

    def test_flooded_track_has_high_wetness(self) -> None:
        clf = ConditionClassifier()
        perc = _make_perception(
            puddle=0.30, wet_surface=0.20, dry_surface=0.50, clip_wet_severe=0.80
        )
        m = clf._compute_metrics(perc)
        assert m.wetness_index >= 0.85  # capped at 1.0
        assert m.puddle_coverage_pct == pytest.approx(30.0)
        assert m.clip_wet_confidence > 0.50

    def test_rubber_ratio_captured(self) -> None:
        clf = ConditionClassifier()
        perc = _make_perception(dry_surface=0.80, rubber_line=0.20)
        m = clf._compute_metrics(perc)
        assert m.rubber_ratio == pytest.approx(0.20)

    def test_dominant_condition_is_string(self) -> None:
        clf = ConditionClassifier()
        perc = _make_perception()
        m = clf._compute_metrics(perc)
        assert isinstance(m.dominant_condition, str)
        assert len(m.dominant_condition) > 0

    def test_metrics_values_bounded(self) -> None:
        clf = ConditionClassifier()
        perc = _make_perception(puddle=1.0, clip_wet_severe=1.0)
        m = clf._compute_metrics(perc)
        assert 0.0 <= m.wetness_index <= 1.0
        assert 0.0 <= m.clip_wet_confidence <= 1.0
        assert 0.0 <= m.clip_dry_confidence <= 1.0


# ---------------------------------------------------------------------------
# Condition Classification
# ---------------------------------------------------------------------------

class TestConditionClassification:
    """Test the threshold decision rules in _classify_condition."""

    def test_fully_dry_classified_dry(self) -> None:
        clf = ConditionClassifier()
        perc = _make_perception(dry_surface=1.0)
        result = clf.assess(perc)
        assert result.track_condition == TrackCondition.DRY

    def test_puddle_30pct_classified_flooded(self) -> None:
        clf = ConditionClassifier()
        perc = _make_perception(puddle=0.30, wet_surface=0.20, dry_surface=0.50)
        result = clf.assess(perc)
        assert result.track_condition == TrackCondition.FLOODED

    def test_wet_surface_classified_wet(self) -> None:
        clf = ConditionClassifier()
        perc = _make_perception(puddle=0.05, wet_surface=0.10, dry_surface=0.85)
        result = clf.assess(perc)
        assert result.track_condition == TrackCondition.WET

    def test_damp_surface_classified_damp(self) -> None:
        clf = ConditionClassifier()
        perc = _make_perception(damp_surface=0.08, dry_surface=0.92)
        result = clf.assess(perc)
        assert result.track_condition == TrackCondition.DAMP

    def test_clip_wet_confidence_overrides_segformer(self) -> None:
        """High CLIP wet score should push classification to WET even with clean seg."""
        clf = ConditionClassifier()
        perc = _make_perception(dry_surface=0.99, clip_wet_severe=0.65)
        result = clf.assess(perc)
        assert result.track_condition in (TrackCondition.WET, TrackCondition.DAMP)


# ---------------------------------------------------------------------------
# Condition Transitions
# ---------------------------------------------------------------------------

class TestConditionTransitions:
    """Validate transition logic — DAMP after WET becomes DRYING."""

    def test_first_frame_has_no_previous(self) -> None:
        clf = ConditionClassifier()
        perc = _make_perception(dry_surface=1.0)
        result = clf.assess(perc)
        assert result.previous_condition is None

    def test_second_frame_has_previous(self) -> None:
        clf = ConditionClassifier()
        clf.assess(_make_perception(dry_surface=1.0))
        second = clf.assess(_make_perception(dry_surface=1.0, frame_index=1))
        assert second.previous_condition == TrackCondition.DRY

    def test_wet_then_damp_becomes_drying(self) -> None:
        clf = ConditionClassifier()
        # First frame: wet
        clf.assess(_make_perception(puddle=0.07, wet_surface=0.08, dry_surface=0.85))
        # Second frame: damp — should be DRYING
        result = clf.assess(
            _make_perception(damp_surface=0.06, dry_surface=0.94, frame_index=1)
        )
        assert result.track_condition == TrackCondition.DRYING

    def test_condition_changed_flag_is_true_on_change(self) -> None:
        clf = ConditionClassifier()
        clf.assess(_make_perception(dry_surface=1.0))
        second = clf.assess(
            _make_perception(puddle=0.07, wet_surface=0.08, dry_surface=0.85, frame_index=1)
        )
        assert second.condition_changed is True

    def test_condition_changed_flag_is_false_when_same(self) -> None:
        clf = ConditionClassifier()
        clf.assess(_make_perception(dry_surface=1.0))
        second = clf.assess(_make_perception(dry_surface=1.0, frame_index=1))
        assert second.condition_changed is False


# ---------------------------------------------------------------------------
# Tyre Recommendations
# ---------------------------------------------------------------------------

class TestTyreRecommendations:
    """Validate compound logic and reasoning text."""

    def test_flooded_gets_wet_tyre(self) -> None:
        clf = ConditionClassifier()
        perc = _make_perception(puddle=0.30, wet_surface=0.20, dry_surface=0.50)
        result = clf.assess(perc)
        assert result.tyre_recommendation.compound == TyreCompound.WET

    def test_wet_gets_wet_tyre(self) -> None:
        clf = ConditionClassifier()
        perc = _make_perception(puddle=0.05, wet_surface=0.10, dry_surface=0.85)
        result = clf.assess(perc)
        assert result.tyre_recommendation.compound == TyreCompound.WET

    def test_damp_gets_intermediate_or_slick(self) -> None:
        clf = ConditionClassifier()
        perc = _make_perception(damp_surface=0.08, dry_surface=0.92)
        result = clf.assess(perc)
        assert result.tyre_recommendation.compound in (
            TyreCompound.INTERMEDIATE, TyreCompound.SLICK_MEDIUM
        )

    def test_dry_heavily_rubbered_gets_soft(self) -> None:
        clf = ConditionClassifier()
        perc = _make_perception(dry_surface=0.80, rubber_line=0.20)
        result = clf.assess(perc)
        assert result.tyre_recommendation.compound == TyreCompound.SLICK_SOFT

    def test_dry_green_track_gets_hard(self) -> None:
        clf = ConditionClassifier()
        perc = _make_perception(dry_surface=1.0, rubber_line=0.0)
        result = clf.assess(perc)
        assert result.tyre_recommendation.compound == TyreCompound.SLICK_HARD

    def test_flooded_pit_window_open(self) -> None:
        clf = ConditionClassifier()
        perc = _make_perception(puddle=0.30, wet_surface=0.20, dry_surface=0.50)
        result = clf.assess(perc)
        assert result.tyre_recommendation.pit_window_open is True

    def test_dry_pit_window_closed(self) -> None:
        clf = ConditionClassifier()
        perc = _make_perception(dry_surface=1.0)
        result = clf.assess(perc)
        assert result.tyre_recommendation.pit_window_open is False

    def test_confidence_is_bounded(self) -> None:
        clf = ConditionClassifier()
        for dry in [1.0, 0.9, 0.5, 0.1]:
            perc = _make_perception(
                dry_surface=dry,
                puddle=max(0.0, (1.0 - dry) * 0.5),
                wet_surface=max(0.0, (1.0 - dry) * 0.5),
            )
            result = clf.assess(perc)
            c = result.tyre_recommendation.confidence
            assert 0.0 <= c <= 1.0, f"Confidence out of range: {c}"

    def test_reasoning_is_nonempty_string(self) -> None:
        clf = ConditionClassifier()
        for d in [1.0, 0.85, 0.5, 0.0]:
            perc = _make_perception(dry_surface=d)
            result = clf.assess(perc)
            assert isinstance(result.tyre_recommendation.reasoning, str)
            assert len(result.tyre_recommendation.reasoning) > 20


# ---------------------------------------------------------------------------
# ConditionAssessment schema
# ---------------------------------------------------------------------------

class TestConditionAssessmentSchema:
    """Verify the full output dataclass structure."""

    def test_assessment_has_all_fields(self) -> None:
        clf = ConditionClassifier()
        perc = _make_perception()
        result = clf.assess(perc)
        assert isinstance(result, ConditionAssessment)
        assert result.frame_index == 0
        assert result.image_hash == "abc123"
        assert isinstance(result.metrics, SurfaceMetrics)
        assert isinstance(result.tyre_recommendation, TyreRecommendation)
        assert isinstance(result.processing_time_ms, float)
        assert result.processing_time_ms > 0

    def test_raw_clip_scores_preserved(self) -> None:
        clf = ConditionClassifier()
        perc = _make_perception(clip_wet_severe=0.42)
        result = clf.assess(perc)
        assert result.raw_clip_scores.get("wet_severe") == pytest.approx(0.42)

    def test_timestamp_preserved(self) -> None:
        clf = ConditionClassifier()
        perc = _make_perception()
        result = clf.assess(perc)
        assert result.timestamp == "2025-07-15T14:30:00Z"
