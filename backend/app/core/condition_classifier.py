"""
APEX Surface Metrics & Condition Classifier — Module 05.

Consumes PerceptionResult outputs and produces:
  1. TrackCondition enum (DRY, DAMP, WET, FLOODED, DRYING, UNSAFE)
  2. SurfaceMetrics (quantified wetness, coverage percentages, rubber ratio)
  3. TyreRecommendation (compound, lap delta, confidence, pit window)
  4. ConditionTransition (was dry, now wet → sudden_shower flag)

This module is pure Python / NumPy — no model inference. It is the
"race engineer reasoning layer" that converts raw pixel analysis into
actionable strategy data.

CPU budget: ~1 ms per frame. Negligible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from loguru import logger

from app.core.perception import CONDITION_PROMPTS, PerceptionResult, SurfaceClass


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class TrackCondition(str, Enum):
    """Overall track condition as assessed by the APEX system."""
    DRY      = "DRY"       # No moisture. Slick territory.
    DAMP     = "DAMP"      # Moisture traces. Intermediate recommended.
    WET      = "WET"       # Active wetness. Full wet required.
    FLOODED  = "FLOODED"   # Standing water. Red flag risk zone.
    DRYING   = "DRYING"    # Was wet, now drying. Intermediate window.
    UNSAFE   = "UNSAFE"    # Unacceptable conditions. Race suspension risk.


class TyreCompound(str, Enum):
    """F1 tyre compound recommendation."""
    SLICK_SOFT   = "SLICK_SOFT"
    SLICK_MEDIUM = "SLICK_MEDIUM"
    SLICK_HARD   = "SLICK_HARD"
    INTERMEDIATE = "INTERMEDIATE"
    WET          = "WET"
    STAY_OUT     = "STAY_OUT"     # No pit recommended this lap


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class SurfaceMetrics:
    """Quantified surface state derived from PerceptionResult."""
    wetness_index: float           # 0.0 (bone dry) to 1.0 (flooded) weighted score
    puddle_coverage_pct: float     # % of track area classified as PUDDLE
    wet_surface_pct: float         # % of track area classified as WET_SURFACE
    damp_surface_pct: float        # % of track area classified as DAMP_SURFACE
    dry_surface_pct: float         # % of track area classified as DRY_SURFACE
    rubber_ratio: float            # Rubber line coverage (proxy for track evolution)
    clip_wet_confidence: float     # CLIP aggregate wet probability
    clip_dry_confidence: float     # CLIP aggregate dry probability
    dominant_condition: str        # Most probable CLIP prompt key


@dataclass
class TyreRecommendation:
    """Tyre strategy recommendation produced by the classifier."""
    compound: TyreCompound
    confidence: float              # 0.0..1.0
    lap_delta_seconds: float       # Estimated lap time penalty vs dry slick (positive = slower)
    reasoning: str                 # Human-readable engineering rationale
    pit_window_open: bool          # True if conditions justify a pit stop now
    alternative_compound: TyreCompound | None = None


@dataclass
class ConditionAssessment:
    """
    Complete condition output consumed by the API response layer.
    """
    frame_index: int
    timestamp: str
    image_hash: str
    track_condition: TrackCondition
    metrics: SurfaceMetrics
    tyre_recommendation: TyreRecommendation
    condition_changed: bool                         # True if condition differs from previous
    previous_condition: TrackCondition | None       # None on first frame
    processing_time_ms: float
    raw_clip_scores: dict[str, float]


# ---------------------------------------------------------------------------
# Condition Classifier Engine
# ---------------------------------------------------------------------------

class ConditionClassifier:
    """
    Converts PerceptionResult → ConditionAssessment using engineering rules.

    Strategy logic:
      - Primary signal: SegFormer surface pixel proportions (spatial ground truth)
      - Cross-check: CLIP zero-shot scores (semantic validation)
      - Agreement weighting: high agreement → higher confidence
      - Temporal context: previous condition used to detect transitions

    Thresholds derived from motorsport engineering heuristics:
      - Flooded: puddle > 20%
      - Wet: (puddle + wet) > 12% OR CLIP wet_severe > 35%
      - Damp: (puddle + wet + damp) > 5% OR CLIP transitional/drying > 30%
      - Dry: everything else
      - Drying: previous was WET/FLOODED and now is DAMP
    """

    def __init__(self) -> None:
        self._previous_condition: TrackCondition | None = None

    def assess(
        self,
        perception: PerceptionResult,
    ) -> ConditionAssessment:
        """Produce a full ConditionAssessment from a PerceptionResult."""
        import time  # noqa: PLC0415
        t0 = time.perf_counter()

        # 1. Compute surface metrics
        metrics = self._compute_metrics(perception)

        # 2. Classify track condition
        condition = self._classify_condition(metrics)

        # 3. Detect condition transition
        condition_changed = (condition != self._previous_condition)
        prev = self._previous_condition

        # Upgrade DAMP to DRYING if we were previously WET/FLOODED
        if (
            condition == TrackCondition.DAMP
            and prev in (TrackCondition.WET, TrackCondition.FLOODED)
        ):
            condition = TrackCondition.DRYING

        # 4. Generate tyre recommendation
        recommendation = self._recommend_tyre(condition, metrics, prev)

        if condition_changed:
            logger.info(
                "Track condition changed: {} → {} | wetness={:.3f}",
                prev.value if prev else "UNKNOWN",
                condition.value,
                metrics.wetness_index,
            )

        self._previous_condition = condition
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return ConditionAssessment(
            frame_index=perception.frame_index,
            timestamp=perception.timestamp,
            image_hash=perception.image_hash,
            track_condition=condition,
            metrics=metrics,
            tyre_recommendation=recommendation,
            condition_changed=condition_changed,
            previous_condition=prev,
            processing_time_ms=elapsed_ms,
            raw_clip_scores=perception.clip_scores,
        )

    def _compute_metrics(self, perception: PerceptionResult) -> SurfaceMetrics:
        """Extract quantified surface state from PerceptionResult."""
        props = perception.class_proportions
        clips = perception.clip_scores

        puddle_pct  = props.get("puddle", 0.0)
        wet_pct     = props.get("wet_surface", 0.0)
        damp_pct    = props.get("damp_surface", 0.0)
        dry_pct     = props.get("dry_surface", 0.0)
        rubber_pct  = props.get("rubber_line", 0.0)

        # Weighted wetness index: puddle is 3x more critical than damp
        wetness_index = min(1.0,
            puddle_pct * 3.0 +
            wet_pct    * 2.0 +
            damp_pct   * 0.8
        )

        # CLIP wet confidence: sum of wet-oriented prompt probabilities
        clip_wet = (
            clips.get("wet_severe",   0.0) * 1.00 +
            clips.get("wet_moderate", 0.0) * 0.85 +
            clips.get("sudden_shower",0.0) * 0.90 +
            clips.get("transitional", 0.0) * 0.40
        )
        clip_wet = min(1.0, clip_wet)

        # CLIP dry confidence
        clip_dry = (
            clips.get("dry_evolved",    0.0) * 1.00 +
            clips.get("dry_green",      0.0) * 1.00 +
            clips.get("marbles_offline",0.0) * 0.50 +
            clips.get("drying",         0.0) * 0.40
        )
        clip_dry = min(1.0, clip_dry)

        # Dominant CLIP prompt
        dominant = max(clips, key=clips.get) if clips else "dry_evolved"

        return SurfaceMetrics(
            wetness_index=round(wetness_index, 4),
            puddle_coverage_pct=round(puddle_pct * 100, 2),
            wet_surface_pct=round(wet_pct * 100, 2),
            damp_surface_pct=round(damp_pct * 100, 2),
            dry_surface_pct=round(dry_pct * 100, 2),
            rubber_ratio=round(rubber_pct, 4),
            clip_wet_confidence=round(clip_wet, 4),
            clip_dry_confidence=round(clip_dry, 4),
            dominant_condition=dominant,
        )

    def _classify_condition(self, m: SurfaceMetrics) -> TrackCondition:
        """Apply engineering threshold rules to classify track condition."""
        # Safety check — extreme puddle coverage
        if m.puddle_coverage_pct > 25.0 or m.wetness_index > 0.85:
            return TrackCondition.FLOODED

        # Full wet
        if (
            m.puddle_coverage_pct + m.wet_surface_pct > 12.0
            or m.clip_wet_confidence > 0.55
            or m.wetness_index > 0.40
        ):
            return TrackCondition.WET

        # Damp
        if (
            m.puddle_coverage_pct + m.wet_surface_pct + m.damp_surface_pct > 5.0
            or m.clip_wet_confidence > 0.25
            or m.wetness_index > 0.08
        ):
            return TrackCondition.DAMP

        return TrackCondition.DRY

    def _recommend_tyre(
        self,
        condition: TrackCondition,
        metrics: SurfaceMetrics,
        previous: TrackCondition | None,
    ) -> TyreRecommendation:
        """Map condition → tyre compound + strategy reasoning."""

        if condition == TrackCondition.FLOODED:
            return TyreRecommendation(
                compound=TyreCompound.WET,
                confidence=0.97,
                lap_delta_seconds=+8.0,
                reasoning=(
                    "Standing water detected across more than 25% of track surface. "
                    "Full wet tyres mandatory. Safety car deployment probable. "
                    "Pit immediately if on intermediates or slicks."
                ),
                pit_window_open=True,
                alternative_compound=None,
            )

        if condition == TrackCondition.WET:
            # If on a drying trajectory, intermediates may be viable
            alt = TyreCompound.INTERMEDIATE if metrics.clip_wet_confidence < 0.70 else None
            return TyreRecommendation(
                compound=TyreCompound.WET,
                confidence=round(metrics.clip_wet_confidence * 0.8 + 0.1, 3),
                lap_delta_seconds=+5.5,
                reasoning=(
                    f"Active wet surface detected. Wetness index {metrics.wetness_index:.2f}. "
                    f"Puddle coverage {metrics.puddle_coverage_pct:.1f}%. "
                    "Full wet tyres recommended. "
                    + ("Intermediate viable if rain eases within 2 laps." if alt else "")
                ),
                pit_window_open=True,
                alternative_compound=alt,
            )

        if condition == TrackCondition.DRYING:
            return TyreRecommendation(
                compound=TyreCompound.INTERMEDIATE,
                confidence=0.78,
                lap_delta_seconds=+1.5,
                reasoning=(
                    f"Track drying from previous wet. Wetness index {metrics.wetness_index:.2f}. "
                    f"Damp coverage {metrics.damp_surface_pct:.1f}%. "
                    "Intermediate is the optimal crossover compound. "
                    "Monitor for slick crossover window in 3-5 laps."
                ),
                pit_window_open=False,
                alternative_compound=TyreCompound.SLICK_SOFT,
            )

        if condition == TrackCondition.DAMP:
            # Decide intermediate vs slick based on wetness magnitude
            if metrics.wetness_index > 0.15:
                return TyreRecommendation(
                    compound=TyreCompound.INTERMEDIATE,
                    confidence=0.72,
                    lap_delta_seconds=+1.8,
                    reasoning=(
                        f"Damp surface with wetness index {metrics.wetness_index:.2f}. "
                        "Intermediate provides safety margin. "
                        "Risk of aquaplaning on slicks through standing patches."
                    ),
                    pit_window_open=True,
                    alternative_compound=TyreCompound.SLICK_MEDIUM,
                )
            else:
                return TyreRecommendation(
                    compound=TyreCompound.SLICK_MEDIUM,
                    confidence=0.62,
                    lap_delta_seconds=+0.3,
                    reasoning=(
                        f"Marginal dampness (index {metrics.wetness_index:.2f}). "
                        "Slick medium is viable on rubbered racing line. "
                        "Off-line conditions remain slippery. "
                        f"Rubber ratio {metrics.rubber_ratio:.2f} — "
                        + ("good track evolution." if metrics.rubber_ratio > 0.1 else "track still green, caution advised.")
                    ),
                    pit_window_open=False,
                    alternative_compound=TyreCompound.INTERMEDIATE,
                )

        # Dry
        rubber = metrics.rubber_ratio
        if rubber > 0.15:
            return TyreRecommendation(
                compound=TyreCompound.SLICK_SOFT,
                confidence=0.91,
                lap_delta_seconds=0.0,
                reasoning=(
                    f"Fully dry track. Rubber ratio {rubber:.2f} — heavily rubbered in. "
                    "Slick soft is optimal for maximum grip and lap time."
                ),
                pit_window_open=False,
                alternative_compound=TyreCompound.SLICK_MEDIUM,
            )
        elif rubber > 0.05:
            return TyreRecommendation(
                compound=TyreCompound.SLICK_MEDIUM,
                confidence=0.85,
                lap_delta_seconds=+0.15,
                reasoning=(
                    f"Dry track with moderate rubber deposit (ratio {rubber:.2f}). "
                    "Medium slick balances pace and durability."
                ),
                pit_window_open=False,
                alternative_compound=TyreCompound.SLICK_SOFT,
            )
        else:
            return TyreRecommendation(
                compound=TyreCompound.SLICK_HARD,
                confidence=0.78,
                lap_delta_seconds=+0.35,
                reasoning=(
                    f"Dry but low rubber ratio ({rubber:.2f}) — track still green. "
                    "Hard slick provides stability until track rubbers in. "
                    "Expect significant lap time improvement over next 5-10 laps."
                ),
                pit_window_open=False,
                alternative_compound=TyreCompound.SLICK_MEDIUM,
            )
