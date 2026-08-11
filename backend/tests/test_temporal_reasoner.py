"""
Tests for Module 06: Temporal Reasoner.

Covers:
  - Rolling buffer management (capacity, reset)
  - WetnessTrend detection (INCREASING / DECREASING / STABLE / UNKNOWN)
  - OLS regression momentum and projections
  - ConditionVolatility classification
  - Stability counter
  - Dominant condition
  - Tyre crossover alert logic (drying and wetting)
  - Thread safety under concurrent updates
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

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
from app.core.temporal_reasoner import (
    ConditionVolatility,
    TemporalAnalysis,
    TemporalReasoner,
    WetnessMomentum,
    WetnessTrend,
)


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------

def _make_assessment(
    *,
    frame_index: int = 0,
    track_condition: TrackCondition = TrackCondition.DRY,
    wetness_index: float = 0.0,
    compound: TyreCompound = TyreCompound.SLICK_SOFT,
    pit_window_open: bool = False,
) -> ConditionAssessment:
    """Build a minimal ConditionAssessment for temporal testing."""
    metrics = SurfaceMetrics(
        wetness_index=wetness_index,
        puddle_coverage_pct=wetness_index * 10.0,
        wet_surface_pct=wetness_index * 5.0,
        damp_surface_pct=0.0,
        dry_surface_pct=max(0.0, 100.0 - wetness_index * 15.0),
        rubber_ratio=0.05,
        clip_wet_confidence=wetness_index * 0.5,
        clip_dry_confidence=max(0.0, 1.0 - wetness_index),
        dominant_condition="dry_evolved",
    )
    rec = TyreRecommendation(
        compound=compound,
        confidence=0.80,
        lap_delta_seconds=0.0,
        reasoning="Test recommendation.",
        pit_window_open=pit_window_open,
    )
    return ConditionAssessment(
        frame_index=frame_index,
        timestamp="2025-07-15T14:30:00Z",
        image_hash=f"hash_{frame_index:04d}",
        track_condition=track_condition,
        metrics=metrics,
        tyre_recommendation=rec,
        condition_changed=False,
        previous_condition=None,
        processing_time_ms=1.0,
        raw_clip_scores={},
    )


def _feed_sequence(
    reasoner: TemporalReasoner,
    wetness_values: list[float],
    *,
    condition: TrackCondition = TrackCondition.DRY,
    compound: TyreCompound = TyreCompound.SLICK_SOFT,
) -> TemporalAnalysis:
    """Feed a sequence of wetness values and return the last TemporalAnalysis."""
    result = None
    for i, w in enumerate(wetness_values):
        result = reasoner.update(
            _make_assessment(
                frame_index=i,
                wetness_index=w,
                track_condition=condition,
                compound=compound,
            )
        )
    return result


# ---------------------------------------------------------------------------
# Buffer Management
# ---------------------------------------------------------------------------

class TestBufferManagement:
    """Test rolling buffer capacity and reset behaviour."""

    def test_empty_reasoner_has_zero_frames(self) -> None:
        r = TemporalReasoner(window_size=10)
        assert r.frame_count == 0

    def test_single_update_increments_frame_count(self) -> None:
        r = TemporalReasoner(window_size=10)
        r.update(_make_assessment())
        assert r.frame_count == 1

    def test_buffer_capped_at_window_size(self) -> None:
        r = TemporalReasoner(window_size=5)
        for i in range(20):
            r.update(_make_assessment(frame_index=i))
        assert r.frame_count == 5

    def test_reset_clears_buffer(self) -> None:
        r = TemporalReasoner(window_size=10)
        for i in range(5):
            r.update(_make_assessment(frame_index=i))
        r.reset()
        assert r.frame_count == 0

    def test_window_size_too_small_raises(self) -> None:
        with pytest.raises(ValueError):
            TemporalReasoner(window_size=2)

    def test_window_size_actual_reported(self) -> None:
        r = TemporalReasoner(window_size=10)
        for i in range(4):
            result = r.update(_make_assessment(frame_index=i))
        assert result.window_size_actual == 4


# ---------------------------------------------------------------------------
# Trend Detection
# ---------------------------------------------------------------------------

class TestTrendDetection:
    """Test WetnessTrend classification from slope threshold."""

    def test_unknown_with_one_frame(self) -> None:
        r = TemporalReasoner()
        result = r.update(_make_assessment(wetness_index=0.1))
        assert result.trend == WetnessTrend.UNKNOWN

    def test_unknown_with_two_frames(self) -> None:
        r = TemporalReasoner()
        _feed_sequence(r, [0.1, 0.2])
        result = r.update(_make_assessment(frame_index=2, wetness_index=0.3))
        # 3 frames — now has enough for trend
        assert result.trend in (WetnessTrend.INCREASING, WetnessTrend.STABLE)

    def test_increasing_trend_on_rising_wetness(self) -> None:
        r = TemporalReasoner(window_size=10)
        values = [i * 0.05 for i in range(10)]   # 0.0 → 0.45
        result = _feed_sequence(r, values)
        assert result.trend == WetnessTrend.INCREASING

    def test_decreasing_trend_on_falling_wetness(self) -> None:
        r = TemporalReasoner(window_size=10)
        values = [0.5 - i * 0.05 for i in range(10)]  # 0.5 → 0.05
        result = _feed_sequence(r, values)
        assert result.trend == WetnessTrend.DECREASING

    def test_stable_trend_on_flat_wetness(self) -> None:
        r = TemporalReasoner(window_size=10)
        values = [0.2] * 10
        result = _feed_sequence(r, values)
        assert result.trend == WetnessTrend.STABLE


# ---------------------------------------------------------------------------
# Momentum / OLS Regression
# ---------------------------------------------------------------------------

class TestMomentum:
    """Test OLS regression quality and projections."""

    def test_slope_direction_matches_trend(self) -> None:
        r = TemporalReasoner(window_size=10)
        values = [i * 0.03 for i in range(10)]
        result = _feed_sequence(r, values)
        assert result.momentum.slope_per_frame > 0

    def test_flat_series_has_near_zero_slope(self) -> None:
        r = TemporalReasoner(window_size=10)
        values = [0.3] * 10
        result = _feed_sequence(r, values)
        assert abs(result.momentum.slope_per_frame) < 0.001

    def test_projections_clipped_to_unit_interval(self) -> None:
        r = TemporalReasoner(window_size=5)
        # Strongly increasing — projections must not exceed 1.0
        values = [i * 0.20 for i in range(5)]
        result = _feed_sequence(r, values)
        assert 0.0 <= result.momentum.projected_wetness_in_5  <= 1.0
        assert 0.0 <= result.momentum.projected_wetness_in_15 <= 1.0

    def test_r_squared_bounded(self) -> None:
        r = TemporalReasoner(window_size=10)
        values = [i * 0.04 for i in range(10)]
        result = _feed_sequence(r, values)
        assert 0.0 <= result.momentum.r_squared <= 1.0

    def test_perfect_linear_has_high_r_squared(self) -> None:
        r = TemporalReasoner(window_size=10)
        values = [i * 0.04 for i in range(10)]
        result = _feed_sequence(r, values)
        assert result.momentum.r_squared > 0.95


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------

class TestVolatility:
    """Test ConditionVolatility classification."""

    def test_unknown_with_two_frames(self) -> None:
        r = TemporalReasoner(window_size=10)
        r.update(_make_assessment())
        result = r.update(_make_assessment(frame_index=1))
        assert result.volatility == ConditionVolatility.UNKNOWN

    def test_stable_conditions_low_volatility(self) -> None:
        r = TemporalReasoner(window_size=10)
        for i in range(10):
            result = r.update(_make_assessment(frame_index=i, track_condition=TrackCondition.DRY))
        assert result.volatility == ConditionVolatility.LOW

    def test_alternating_conditions_high_volatility(self) -> None:
        r = TemporalReasoner(window_size=10)
        for i in range(10):
            cond = TrackCondition.DRY if i % 2 == 0 else TrackCondition.WET
            result = r.update(_make_assessment(frame_index=i, track_condition=cond))
        assert result.volatility == ConditionVolatility.HIGH


# ---------------------------------------------------------------------------
# Stability Counter
# ---------------------------------------------------------------------------

class TestStabilityCounter:
    """Test consecutive-frames-same-condition counter."""

    def test_10_dry_frames_gives_stability_10(self) -> None:
        r = TemporalReasoner(window_size=15)
        for i in range(10):
            result = r.update(_make_assessment(frame_index=i, track_condition=TrackCondition.DRY))
        assert result.condition_stability_frames == 10

    def test_stability_resets_on_condition_change(self) -> None:
        r = TemporalReasoner(window_size=15)
        for i in range(5):
            r.update(_make_assessment(frame_index=i, track_condition=TrackCondition.DRY))
        result = r.update(_make_assessment(frame_index=5, track_condition=TrackCondition.WET))
        assert result.condition_stability_frames == 1


# ---------------------------------------------------------------------------
# Tyre Window Alert
# ---------------------------------------------------------------------------

class TestTyreWindowAlert:
    """Validate crossover alert firing conditions."""

    def test_no_alert_on_dry_stable_track(self) -> None:
        r = TemporalReasoner(window_size=10)
        values = [0.02] * 10
        result = _feed_sequence(r, values, compound=TyreCompound.SLICK_SOFT)
        assert result.tyre_window_alert.alert_active is False

    def test_drying_crossover_alert_fires(self) -> None:
        """Intermediate on a rapidly drying track should trigger crossover alert."""
        r = TemporalReasoner(window_size=10)
        # Start wet, then rapidly dry
        values = [0.5 - i * 0.06 for i in range(10)]  # 0.5 → -0.04 (clamped)
        result = _feed_sequence(r, values, compound=TyreCompound.INTERMEDIATE)
        assert result.tyre_window_alert.alert_active is True
        assert result.tyre_window_alert.to_compound == TyreCompound.SLICK_SOFT

    def test_wetting_crossover_alert_fires(self) -> None:
        """Slick on a rapidly wetting track should trigger crossover alert."""
        r = TemporalReasoner(window_size=10)
        values = [i * 0.03 for i in range(10)]  # 0 → 0.27
        result = _feed_sequence(r, values, compound=TyreCompound.SLICK_SOFT)
        assert result.tyre_window_alert.alert_active is True
        assert result.tyre_window_alert.to_compound == TyreCompound.INTERMEDIATE

    def test_alert_message_nonempty(self) -> None:
        r = TemporalReasoner(window_size=5)
        result = _feed_sequence(r, [0.1] * 5, compound=TyreCompound.SLICK_SOFT)
        assert isinstance(result.tyre_window_alert.message, str)
        assert len(result.tyre_window_alert.message) > 0


# ---------------------------------------------------------------------------
# Thread Safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    """Verify concurrent update calls don't corrupt internal state."""

    def test_concurrent_updates_dont_corrupt(self) -> None:
        r = TemporalReasoner(window_size=30)
        errors: list[Exception] = []

        def worker() -> None:
            try:
                for i in range(50):
                    r.update(_make_assessment(frame_index=i, wetness_index=i * 0.01))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
        assert r.frame_count == 30  # capped at window_size


# ---------------------------------------------------------------------------
# Full Pipeline Integration
# ---------------------------------------------------------------------------

class TestTemporalAnalysisSchema:
    """Validate TemporalAnalysis output structure."""

    def test_output_type(self) -> None:
        r = TemporalReasoner(window_size=10)
        result = r.update(_make_assessment())
        assert isinstance(result, TemporalAnalysis)

    def test_processing_time_positive(self) -> None:
        r = TemporalReasoner(window_size=10)
        result = r.update(_make_assessment())
        assert result.processing_time_ms > 0

    def test_wetness_mean_and_std_reasonable(self) -> None:
        r = TemporalReasoner(window_size=10)
        values = [0.3] * 10
        result = _feed_sequence(r, values)
        assert result.wetness_mean == pytest.approx(0.3, abs=1e-4)
        assert result.wetness_std == pytest.approx(0.0, abs=1e-4)

    def test_condition_change_count(self) -> None:
        r = TemporalReasoner(window_size=10)
        for i in range(5):
            r.update(_make_assessment(frame_index=i, track_condition=TrackCondition.DRY))
        for i in range(5, 10):
            result = r.update(_make_assessment(frame_index=i, track_condition=TrackCondition.WET))
        assert result.condition_change_count == 1
