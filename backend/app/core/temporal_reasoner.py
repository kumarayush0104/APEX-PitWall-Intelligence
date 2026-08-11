"""
APEX Temporal Reasoner — Module 06.

Accumulates a rolling window of ConditionAssessment frames and produces:
  1. TemporalTrend — is the track getting wetter, drying, or stable?
  2. ConditionStability — how long has the current condition persisted?
  3. WetnessMomentum — rate of change in wetness index (Δ/frame)
  4. TyreWindowAlert — when the tyre crossover window is approaching

Design:
  - Pure Python + NumPy. No model inference.
  - Stateful ring-buffer of configurable depth (default 30 frames ≈ 30 s at 1 fps).
  - Thread-safe via threading.Lock.
  - CPU budget: <1 ms per frame.

Usage:
    reasoner = TemporalReasoner(window_size=30)
    trend = reasoner.update(assessment)
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque

import numpy as np
from loguru import logger

from app.core.condition_classifier import (
    ConditionAssessment,
    TrackCondition,
    TyreCompound,
)


# ---------------------------------------------------------------------------
# Trend & Stability Enumerations
# ---------------------------------------------------------------------------

class WetnessTrend(str, Enum):
    """Direction of wetness change over the recent window."""
    INCREASING  = "INCREASING"   # Getting wetter
    DECREASING  = "DECREASING"   # Drying out
    STABLE      = "STABLE"       # No meaningful change
    UNKNOWN     = "UNKNOWN"      # Insufficient data (< 3 frames)


class ConditionVolatility(str, Enum):
    """How rapidly conditions are changing."""
    HIGH    = "HIGH"    # Multiple condition changes in window
    MEDIUM  = "MEDIUM"  # Some changes
    LOW     = "LOW"     # Stable for most of window
    UNKNOWN = "UNKNOWN" # Insufficient data


# ---------------------------------------------------------------------------
# Output Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class WetnessMomentum:
    """Linear regression over recent wetness_index values."""
    slope_per_frame: float          # Δ wetness per frame (negative = drying)
    r_squared: float                # Goodness of fit [0..1]
    projected_wetness_in_5: float   # Projected wetness index in 5 frames
    projected_wetness_in_15: float  # Projected wetness index in 15 frames


@dataclass
class TyreWindowAlert:
    """Alert when a tyre compound crossover is imminent."""
    alert_active: bool
    from_compound: TyreCompound | None
    to_compound: TyreCompound | None
    estimated_frames_remaining: int | None   # Estimated frames until optimal crossover
    message: str


@dataclass
class TemporalAnalysis:
    """Complete temporal analysis output for one frame update."""
    frame_index: int
    window_size_actual: int                     # How many frames are in the current window
    trend: WetnessTrend
    volatility: ConditionVolatility
    momentum: WetnessMomentum
    condition_stability_frames: int             # Consecutive frames with same condition
    dominant_condition_in_window: TrackCondition
    wetness_mean: float                         # Mean wetness over window
    wetness_std: float                          # Std deviation
    tyre_window_alert: TyreWindowAlert
    condition_change_count: int                 # Number of condition changes in window
    processing_time_ms: float


# ---------------------------------------------------------------------------
# Temporal Reasoner
# ---------------------------------------------------------------------------

class TemporalReasoner:
    """
    Stateful multi-frame trend analyser.

    The reasoner maintains a rolling ring-buffer of ConditionAssessment objects.
    On each .update() call it computes wetness momentum via OLS regression,
    detects trends, measures volatility, and fires tyre crossover alerts.
    """

    # Thresholds
    TREND_SLOPE_THRESHOLD = 0.005   # |Δ/frame| considered meaningful
    CROSSOVER_ALERT_FRAMES = 5      # Alert when projected crossover is within N frames

    def __init__(self, window_size: int = 30) -> None:
        """
        Args:
            window_size: Number of frames to retain in the rolling window.
                         At 1 fps this is 30 seconds of history.
        """
        if window_size < 3:
            raise ValueError("window_size must be >= 3 for trend analysis.")
        self._window_size = window_size
        self._buffer: Deque[ConditionAssessment] = deque(maxlen=window_size)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, assessment: ConditionAssessment) -> TemporalAnalysis:
        """
        Ingest a new ConditionAssessment and return the latest TemporalAnalysis.

        Thread-safe.
        """
        import time  # noqa: PLC0415
        t0 = time.perf_counter()

        with self._lock:
            self._buffer.append(assessment)
            frames = list(self._buffer)

        n = len(frames)
        wetness_series = np.array([f.metrics.wetness_index for f in frames], dtype=np.float64)
        conditions     = [f.track_condition for f in frames]
        compounds      = [f.tyre_recommendation.compound for f in frames]

        # Core computations
        trend        = self._compute_trend(wetness_series)
        volatility   = self._compute_volatility(conditions)
        momentum     = self._compute_momentum(wetness_series)
        stability    = self._compute_stability(conditions)
        dominant     = self._dominant_condition(conditions)
        change_count = sum(
            1 for i in range(1, n) if conditions[i] != conditions[i - 1]
        )
        alert = self._tyre_window_alert(compounds, momentum)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        logger.debug(
            "Temporal: n={} trend={} slope={:.4f} stability={} alert={}",
            n, trend.value, momentum.slope_per_frame, stability, alert.alert_active,
        )

        return TemporalAnalysis(
            frame_index=assessment.frame_index,
            window_size_actual=n,
            trend=trend,
            volatility=volatility,
            momentum=momentum,
            condition_stability_frames=stability,
            dominant_condition_in_window=dominant,
            wetness_mean=round(float(wetness_series.mean()), 4),
            wetness_std=round(float(wetness_series.std()), 4),
            tyre_window_alert=alert,
            condition_change_count=change_count,
            processing_time_ms=elapsed_ms,
        )

    def reset(self) -> None:
        """Clear the rolling buffer (e.g., after session restart)."""
        with self._lock:
            self._buffer.clear()

    @property
    def frame_count(self) -> int:
        """Number of frames currently in the buffer."""
        with self._lock:
            return len(self._buffer)

    # ------------------------------------------------------------------
    # Private Computation Helpers
    # ------------------------------------------------------------------

    def _compute_trend(self, wetness: np.ndarray) -> WetnessTrend:
        if len(wetness) < 3:
            return WetnessTrend.UNKNOWN
        momentum = self._compute_momentum(wetness)
        slope = momentum.slope_per_frame
        if slope > self.TREND_SLOPE_THRESHOLD:
            return WetnessTrend.INCREASING
        if slope < -self.TREND_SLOPE_THRESHOLD:
            return WetnessTrend.DECREASING
        return WetnessTrend.STABLE

    def _compute_volatility(self, conditions: list[TrackCondition]) -> ConditionVolatility:
        if len(conditions) < 3:
            return ConditionVolatility.UNKNOWN
        changes = sum(
            1 for i in range(1, len(conditions))
            if conditions[i] != conditions[i - 1]
        )
        ratio = changes / max(1, len(conditions) - 1)
        if ratio > 0.30:
            return ConditionVolatility.HIGH
        if ratio > 0.10:
            return ConditionVolatility.MEDIUM
        return ConditionVolatility.LOW

    def _compute_momentum(self, wetness: np.ndarray) -> WetnessMomentum:
        n = len(wetness)
        if n < 2:
            return WetnessMomentum(
                slope_per_frame=0.0,
                r_squared=0.0,
                projected_wetness_in_5=float(wetness[-1]) if n else 0.0,
                projected_wetness_in_15=float(wetness[-1]) if n else 0.0,
            )
        x = np.arange(n, dtype=np.float64)
        # OLS: y = slope * x + intercept
        slope, intercept = np.polyfit(x, wetness, 1)
        # R²
        y_pred = slope * x + intercept
        ss_res = float(np.sum((wetness - y_pred) ** 2))
        ss_tot = float(np.sum((wetness - wetness.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

        # Projections from the last observed frame
        last_x = n - 1
        proj_5  = float(np.clip(slope * (last_x + 5)  + intercept, 0.0, 1.0))
        proj_15 = float(np.clip(slope * (last_x + 15) + intercept, 0.0, 1.0))

        return WetnessMomentum(
            slope_per_frame=round(float(slope), 6),
            r_squared=round(max(0.0, r2), 4),
            projected_wetness_in_5=round(proj_5, 4),
            projected_wetness_in_15=round(proj_15, 4),
        )

    def _compute_stability(self, conditions: list[TrackCondition]) -> int:
        """Count consecutive frames at the tail sharing the same condition."""
        if not conditions:
            return 0
        last = conditions[-1]
        count = 0
        for c in reversed(conditions):
            if c == last:
                count += 1
            else:
                break
        return count

    def _dominant_condition(self, conditions: list[TrackCondition]) -> TrackCondition:
        """Return the most frequently occurring condition in the window."""
        if not conditions:
            return TrackCondition.DRY
        from collections import Counter  # noqa: PLC0415
        return Counter(conditions).most_common(1)[0][0]

    def _tyre_window_alert(
        self,
        compounds: list[TyreCompound],
        momentum: WetnessMomentum,
    ) -> TyreWindowAlert:
        """
        Fire an alert when current recommendation differs from 5-frame projection.

        Logic:
          - If the current compound is WET/INTERMEDIATE and track is drying
            (negative slope + projected wetness low enough for slick), alert.
          - If the current compound is SLICK and track is wetting up, alert.
        """
        if not compounds:
            return TyreWindowAlert(False, None, None, None, "")

        current = compounds[-1]
        slope   = momentum.slope_per_frame
        proj5   = momentum.projected_wetness_in_5

        # Drying crossover: intermediate → slick
        if (
            current in (TyreCompound.WET, TyreCompound.INTERMEDIATE)
            and slope < -self.TREND_SLOPE_THRESHOLD
            and proj5 < 0.05
        ):
            frames_est = max(1, int(abs(0.05 / slope))) if slope != 0.0 else None
            return TyreWindowAlert(
                alert_active=True,
                from_compound=current,
                to_compound=TyreCompound.SLICK_SOFT,
                estimated_frames_remaining=frames_est,
                message=(
                    f"Crossover window approaching: {current.value} → SLICK. "
                    f"Projected wetness in 5 frames: {proj5:.3f}. "
                    f"Consider pitting in ~{frames_est} lap(s)."
                ),
            )

        # Wetting crossover: slick → wet
        if (
            current in (TyreCompound.SLICK_SOFT, TyreCompound.SLICK_MEDIUM, TyreCompound.SLICK_HARD)
            and slope > self.TREND_SLOPE_THRESHOLD
            and proj5 > 0.12
        ):
            frames_est = max(1, int(abs((proj5 - 0.12) / slope))) if slope != 0.0 else None
            return TyreWindowAlert(
                alert_active=True,
                from_compound=current,
                to_compound=TyreCompound.INTERMEDIATE,
                estimated_frames_remaining=frames_est,
                message=(
                    f"Deteriorating conditions: {current.value} → INTERMEDIATE. "
                    f"Projected wetness in 5 frames: {proj5:.3f}. "
                    "Intermediate or full wet required soon."
                ),
            )

        return TyreWindowAlert(
            alert_active=False,
            from_compound=current,
            to_compound=None,
            estimated_frames_remaining=None,
            message="No crossover alert. Current compound optimal for projected conditions.",
        )
