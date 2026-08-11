"""
Pydantic schemas for the APEX / PitWall Intelligence API.

Organisation:
  - Enumerations (shared across modules)
  - Base response models
  - Health / system schemas
  - Core pipeline schemas (surface metrics → physics → temporal → confidence → recommendation)
  - Visualisation schemas
  - Explanation schemas
  - Composite result (APEXResult)
  - Session management schemas
  - Upload / demo schemas

Conventions:
  - All API responses inherit BaseResponse.
  - Every field has a description (auto-populates OpenAPI docs).
  - Timestamps are always UTC ISO-8601 strings.
  - No Optional[X] without an explicit default — use X | None = None.
  - Ranges are enforced with ge= / le= constraints where applicable.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow_iso() -> str:
    """Return current UTC time as ISO-8601 string (timezone-aware)."""
    return datetime.now(timezone.utc).isoformat()


# ===========================================================================
# Enumerations
# ===========================================================================

class ConditionState(str, Enum):
    """
    Physics-grounded track condition state machine.

    States are ordered from most to least wet.
    The TRANSITIONAL state is the highest-value window for tyre decisions.
    """
    WET_SEVERE    = "WET_SEVERE"     # μ ≈ 0.25–0.35  │ Full wets mandatory
    WET_MODERATE  = "WET_MODERATE"   # μ ≈ 0.35–0.50  │ Full wets / inters viable
    TRANSITIONAL  = "TRANSITIONAL"   # μ ≈ 0.50–0.70  │ CRITICAL decision window
    DRYING        = "DRYING"         # μ ≈ 0.60–0.80  │ Inters optimal, slick imminent
    DRY_GREEN     = "DRY_GREEN"      # μ ≈ 0.75–0.90  │ Slicks on, still rubbering in
    DRY_EVOLVED   = "DRY_EVOLVED"    # μ ≈ 0.90–1.05  │ Full grip, normal strategy
    UNKNOWN       = "UNKNOWN"        # Insufficient data to classify


class RecommendationPriority(str, Enum):
    """
    Strategy recommendation urgency levels for the race engineer.
    """
    IMMEDIATE        = "IMMEDIATE"         # Act now — optimal window open
    HIGH             = "HIGH"             # Pit in next 1–2 laps
    MONITOR          = "MONITOR"          # Approaching window — watch closely
    HOLD             = "HOLD"             # No action required
    ABORT            = "ABORT"            # Conditions worsening — abandon plan
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"  # Not enough frames yet


class TrendDirection(str, Enum):
    """Direction of grip coefficient change over temporal window."""
    IMPROVING  = "IMPROVING"    # dμ/dt > threshold
    WORSENING  = "WORSENING"    # dμ/dt < -threshold
    STABLE     = "STABLE"       # |dμ/dt| ≤ threshold
    UNKNOWN    = "UNKNOWN"      # Fewer than MIN_FRAMES_FOR_TREND frames


class ModelProvider(str, Enum):
    """Where a model is being served."""
    LOCAL       = "local"       # Loaded in process memory
    HF_API      = "hf_api"      # Served by HuggingFace Inference API
    NOT_LOADED  = "not_loaded"  # Lazy — not yet initialised
    DISABLED    = "disabled"    # Explicitly disabled by config


class TyreCompound(str, Enum):
    """F1 tyre compounds relevant to wet/dry transitions."""
    FULL_WET      = "FULL_WET"       # Blue — standing water, heavy rain
    INTERMEDIATE  = "INTERMEDIATE"   # Green — damp / mixed / drying
    SOFT          = "SOFT"           # Red — dry performance
    MEDIUM        = "MEDIUM"         # Yellow — dry balanced
    HARD          = "HARD"           # White — dry endurance


# ===========================================================================
# Base response
# ===========================================================================

class BaseResponse(BaseModel):
    """Root of all API response models."""
    success: bool = Field(default=True, description="False on error.")
    timestamp: str = Field(
        default_factory=_utcnow_iso,
        description="UTC ISO-8601 timestamp of the response.",
    )
    version: str = Field(default="1.0.0", description="API version.")


class ErrorResponse(BaseResponse):
    """Standardised error response returned for all 4xx / 5xx errors."""
    success: bool = False
    error: str = Field(..., description="Human-readable error message.")
    error_code: str = Field(..., description="Machine-readable error code (SCREAMING_SNAKE_CASE).")
    detail: str | None = Field(
        default=None,
        description="Additional debug detail (only present when DEBUG=true).",
    )


# ===========================================================================
# Health / System schemas
# ===========================================================================

class ModelStatus(BaseModel):
    """Runtime status of a single AI model."""
    loaded: bool = Field(..., description="True if model is currently in memory.")
    provider: ModelProvider = Field(..., description="How this model is being served.")
    model_id: str = Field(..., description="HuggingFace model repository identifier.")
    memory_mb: float | None = Field(
        default=None, description="Approximate RAM/VRAM used by this model in MB."
    )
    load_time_seconds: float | None = Field(
        default=None, description="Wall-clock seconds taken to load the model."
    )
    last_inference_ms: float | None = Field(
        default=None, description="Latency of the most recent inference call in ms."
    )
    error: str | None = Field(
        default=None, description="Error message if model failed to load."
    )


class SystemInfo(BaseModel):
    """Hardware and runtime environment information."""
    python_version: str = Field(..., description="Python interpreter version string.")
    torch_version: str | None = Field(
        default=None, description="PyTorch version, or null if not installed."
    )
    platform: str = Field(..., description="Operating system name.")
    platform_release: str = Field(..., description="OS version string.")
    cpu_count: int = Field(..., description="Number of logical CPU cores.")
    resolved_device: str = Field(
        ..., description="Actual compute device: 'cuda' or 'cpu'."
    )
    cuda_available: bool = Field(..., description="True if CUDA is available.")
    cuda_device_name: str | None = Field(
        default=None, description="NVIDIA GPU model name, if available."
    )
    cuda_device_memory_gb: float | None = Field(
        default=None, description="Total GPU VRAM in GB, if available."
    )
    ram_total_gb: float | None = Field(
        default=None, description="Total system RAM in GB."
    )
    ram_available_gb: float | None = Field(
        default=None, description="Currently available system RAM in GB."
    )


class ServiceStatus(BaseModel):
    """Runtime status of a backend service."""
    status: str = Field(..., description="'ready' | 'disabled' | 'degraded' | 'error'.")
    detail: str | None = Field(default=None, description="Optional detail message.")


class HealthResponse(BaseResponse):
    """
    Comprehensive system health response.

    Returned by GET /api/v1/health. Used by the frontend status panel,
    monitoring systems, and manual debugging.
    """
    status: str = Field(
        ..., description="Overall system status: 'healthy' | 'degraded' | 'unhealthy'."
    )
    system: SystemInfo
    models: dict[str, ModelStatus] = Field(
        ..., description="Per-model status keyed by model role (dinov2, segformer, clip, vlm)."
    )
    services: dict[str, ServiceStatus] = Field(
        ..., description="Per-service status keyed by service name."
    )
    config: dict[str, Any] = Field(
        ..., description="Non-sensitive configuration summary."
    )


# ===========================================================================
# Surface metrics (Module 05 output)
# ===========================================================================

class SurfaceMetrics(BaseModel):
    """
    Quantitative visual metrics derived from segmentation and image analysis.

    All values are normalised to [0.0, 1.0] unless otherwise noted.
    These feed directly into the physics layer's grip coefficient formula.
    """
    wet_pixel_ratio: float = Field(
        ..., ge=0.0, le=1.0,
        description=(
            "Fraction of visible track pixels classified as wet or puddle. "
            "0.0 = fully dry, 1.0 = completely flooded."
        ),
    )
    specular_intensity: float = Field(
        ..., ge=0.0, le=1.0,
        description=(
            "Normalised mean luminance of specular highlights on the track surface. "
            "High values indicate reflective water films."
        ),
    )
    rubber_line_coverage: float = Field(
        ..., ge=0.0, le=1.0,
        description=(
            "Fraction of detected track area showing dark rubber polymer deposit "
            "(the 'racing line'). Higher = more grip available on the dry line."
        ),
    )
    puddle_count: int = Field(
        ..., ge=0,
        description="Number of distinct standing-water puddle regions detected.",
    )
    puddle_area_fraction: float = Field(
        ..., ge=0.0, le=1.0,
        description="Total puddle area as fraction of total visible track area.",
    )
    reflectance_score: float = Field(
        ..., ge=0.0, le=1.0,
        description=(
            "Composite surface reflectance score (HSV-based). "
            "High = wet/shiny, Low = dry/matte."
        ),
    )
    racing_line_dry_pct: float = Field(
        ..., ge=0.0, le=1.0,
        description=(
            "Percentage of the expected racing line corridor that is visibly dry. "
            "THIS is the primary crossover indicator — engineers watch this number."
        ),
    )
    off_line_wetness: float = Field(
        ..., ge=0.0, le=1.0,
        description=(
            "Wetness level of the off-line (non-racing-line) area. "
            "Critical for assessing risk of overtaking or tyre temperature on out-lap."
        ),
    )


# ===========================================================================
# Physics estimate (Module 06 output)
# ===========================================================================

class PhysicsEstimate(BaseModel):
    """
    Physics-grounded grip coefficient estimate with confidence interval.

    μ̂ (mu_hat) is computed from surface metrics using a weighted formula:
        μ̂ = 0.35×(1-wet_pixel_ratio) + 0.25×(1-specular) + 0.25×rubber + 0.15×drying_bonus
    Then scaled to the real engineering range [0.25, 1.05].
    """
    mu_hat: float = Field(
        ...,
        description=(
            "Estimated grip coefficient (μ̂). "
            "Scale: 0.25 = completely soaked, 1.05 = fully rubbered-in dry. "
            "Wet asphalt typically 0.3–0.4; rubber-in line 0.9–1.0."
        ),
    )
    mu_lower: float = Field(
        ..., description="Lower bound of the μ̂ confidence interval."
    )
    mu_upper: float = Field(
        ..., description="Upper bound of the μ̂ confidence interval."
    )
    condition_state: ConditionState = Field(
        ..., description="Discrete condition state from the physics state machine."
    )
    risk_level: str = Field(
        ..., description="Current risk classification: LOW | MEDIUM | HIGH | CRITICAL."
    )
    recommended_compound: TyreCompound = Field(
        ..., description="Tyre compound that the physics estimate suggests is optimal right now."
    )


# ===========================================================================
# Temporal analysis (Module 07 output)
# ===========================================================================

class TemporalAnalysis(BaseModel):
    """
    Temporal reasoning over the N-frame sliding window.

    Tracks how track conditions are EVOLVING — not just what they are now.
    The drying_rate and frames_to_crossover are the key forecasting outputs.
    """
    smoothed_mu: float = Field(
        ...,
        description=(
            "Exponentially-weighted moving average of μ̂ over the temporal window. "
            "More stable than the raw per-frame estimate."
        ),
    )
    trend_direction: TrendDirection
    drying_rate_per_frame: float = Field(
        ...,
        description=(
            "Rate of grip change: Δμ per analysed frame (not calendar second). "
            "Positive = conditions improving (drying). "
            "Negative = conditions worsening (shower incoming or spray)."
        ),
    )
    frames_to_crossover: int | None = Field(
        default=None,
        description=(
            "Estimated number of future frames until grip crosses the tyre-change threshold. "
            "None if conditions are worsening or stable below threshold."
        ),
    )
    minutes_to_crossover: float | None = Field(
        default=None,
        description="Frames-to-crossover converted to calendar minutes (using VIDEO_ANALYSIS_FPS).",
    )
    condition_stability: float = Field(
        ..., ge=0.0, le=1.0,
        description=(
            "Temporal consistency score. "
            "1.0 = conditions very stable (low variance in window). "
            "0.0 = high variance / rapidly changing conditions."
        ),
    )
    anomaly_flag: bool = Field(
        ...,
        description=(
            "True if the current frame's μ̂ is a statistical outlier relative to "
            "the sliding window (possible single-frame noise or sudden event)."
        ),
    )
    frames_in_window: int = Field(
        ..., ge=0,
        description="Number of frames currently in the temporal window (ramps up from 0).",
    )


# ===========================================================================
# Confidence calibration (Module 08 output)
# ===========================================================================

class ConfidenceBundle(BaseModel):
    """
    Calibrated confidence scores for the current APEX prediction.

    Three independent sources (DINOv2, SegFormer, CLIP) are combined.
    Disagreement between sources → lower confidence → weaker recommendation.
    This prevents overconfident recommendations on ambiguous inputs.
    """
    overall_confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Combined confidence score across all sources and temporal consistency.",
    )
    source_agreement: float = Field(
        ..., ge=0.0, le=1.0,
        description=(
            "Agreement score between DINOv2, SegFormer, and CLIP predictions. "
            "1.0 = unanimous agreement. 0.0 = all three disagree."
        ),
    )
    temporal_consistency: float = Field(
        ..., ge=0.0, le=1.0,
        description="How consistent the current prediction is with recent frame history.",
    )
    uncertainty_level: str = Field(
        ..., description="Bucketed uncertainty: LOW | MEDIUM | HIGH."
    )
    uncertainty_sources: list[str] = Field(
        default_factory=list,
        description=(
            "List of factors contributing to uncertainty "
            "(e.g., 'model_disagreement', 'anomalous_frame', 'low_visibility')."
        ),
    )


# ===========================================================================
# Strategy recommendation (Module 09 output)
# ===========================================================================

class StrategyRecommendation(BaseModel):
    """
    Complete tyre strategy recommendation for the race engineer.

    Contains everything needed for a 3-second decision in a pit wall scenario:
    priority → action → evidence → risk → alternative → forecast.
    """
    priority: RecommendationPriority
    primary_action: str = Field(
        ...,
        description=(
            "The primary recommended action. "
            "e.g., 'Pit for intermediate tyres in next 2 laps.'"
        ),
    )
    alternative_action: str | None = Field(
        default=None,
        description=(
            "Alternative strategy if primary carries too much risk. "
            "e.g., 'If grip improves 10% in next lap, consider soft slick gamble.'"
        ),
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Recommendation confidence — directly from ConfidenceBundle.",
    )
    evidence_list: list[str] = Field(
        ...,
        description=(
            "3–5 concise bullet points of visual evidence. "
            "e.g., 'Racing line is 72% dry.', 'Standing water at Turn 1 braking zone.'"
        ),
    )
    risk_assessment: str = Field(
        ..., description="One-sentence risk statement for the engineer."
    )
    forecast_text: str | None = Field(
        default=None,
        description=(
            "Near-future condition forecast. "
            "e.g., 'At current rate, slick crossover in ~3.5 minutes.'"
        ),
    )
    crossover_score: float = Field(
        ..., ge=0.0, le=1.0,
        description=(
            "Proximity to the tyre crossover point. "
            "0.0 = far from crossover, 1.0 = at or past crossover."
        ),
    )
    pit_window_risk: float = Field(
        ..., ge=0.0, le=1.0,
        description=(
            "Risk of pitting now. Combines uncertainty with condition severity. "
            "High risk = wait for more evidence."
        ),
    )
    tyre_delta_estimate: str | None = Field(
        default=None,
        description=(
            "Estimated per-lap time delta vs. optimal compound. "
            "e.g., '+1.8 s/lap' (current tyre is costing 1.8 seconds per lap vs. optimal)."
        ),
    )
    optimal_compound: TyreCompound = Field(
        ..., description="Tyre compound this recommendation targets."
    )


# ===========================================================================
# Visualisation bundle (Module 04 output)
# ===========================================================================

class VisualizationBundle(BaseModel):
    """Base64-encoded visualisation images for the frontend canvas."""
    overlay_b64: str = Field(
        ...,
        description=(
            "Original track frame with segmentation overlay blended in. "
            "Base64-encoded PNG."
        ),
    )
    attention_heatmap_b64: str = Field(
        ...,
        description=(
            "DINOv2 self-attention heatmap highlighting regions the model attends to. "
            "Base64-encoded PNG."
        ),
    )
    segmentation_b64: str = Field(
        ...,
        description="Raw segmentation mask colourised by surface class. Base64-encoded PNG.",
    )
    class_legend: dict[str, str] = Field(
        ..., description="Surface class name → hex colour string mapping for the UI legend.",
    )
    original_width: int = Field(..., description="Original image width in pixels.")
    original_height: int = Field(..., description="Original image height in pixels.")


# ===========================================================================
# Explanation bundle (Module 10 output)
# ===========================================================================

class ExplanationBundle(BaseModel):
    """
    Natural language or structured explanation of the recommendation.

    Generated by Qwen2-VL via HF Inference API when available.
    Falls back to a deterministic structured template when VLM is unavailable.
    The fallback is indistinguishable in structure — only the fluency differs.
    """
    engineer_summary: str = Field(
        ...,
        description=(
            "80–120 word engineer-facing explanation. "
            "References specific visual evidence and links it to the recommendation."
        ),
    )
    key_observations: list[str] = Field(
        ...,
        description="3 concise visual observations extracted from the current frame.",
    )
    confidence_statement: str = Field(
        ..., description="One sentence describing prediction confidence and its basis."
    )
    risk_statement: str = Field(
        ..., description="One sentence summarising the primary risk if action is wrong."
    )
    generated_by: str = Field(
        ...,
        description="'vlm_api' | 'vlm_local' | 'structured_fallback'. Shown in UI for transparency.",
    )


# ===========================================================================
# Full APEX pipeline result
# ===========================================================================

class APEXResult(BaseModel):
    """
    Complete output of the APEX pipeline for a single analysed frame.

    This is the canonical response object for image analysis.
    Every field is always populated — no Optional fields at the result level.
    Pipeline failures produce structured fallbacks, not missing fields.
    """
    session_id: str = Field(..., description="UUID of the analysis session.")
    frame_index: int = Field(..., ge=0, description="Zero-based index of this frame within the session.")
    timestamp: str = Field(default_factory=_utcnow_iso)

    # Core pipeline outputs
    surface_metrics: SurfaceMetrics
    physics_estimate: PhysicsEstimate
    temporal_analysis: TemporalAnalysis
    confidence: ConfidenceBundle
    recommendation: StrategyRecommendation
    visualization: VisualizationBundle
    explanation: ExplanationBundle

    # Performance metadata
    pipeline_timing_ms: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Latency of each pipeline stage in milliseconds. "
            "Keys: 'perception', 'surface', 'physics', 'temporal', "
            "'confidence', 'recommendation', 'explanation', 'visualization'."
        ),
    )
    total_latency_ms: float = Field(
        ..., description="Total end-to-end pipeline latency in milliseconds."
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal pipeline warnings (e.g., VLM fallback used).",
    )


# ===========================================================================
# Session management
# ===========================================================================

class SessionInfo(BaseModel):
    """Metadata about an analysis session."""
    session_id: str
    created_at: str
    last_updated: str
    frame_count: int = Field(..., ge=0)
    current_state: ConditionState | None = Field(
        default=None, description="Most recent condition state, if any frames processed."
    )
    current_mu: float | None = Field(
        default=None, description="Most recent grip estimate μ̂, if any."
    )
    current_recommendation: RecommendationPriority | None = Field(
        default=None, description="Most recent recommendation priority."
    )
    demo_scenario: str | None = Field(
        default=None, description="Demo scenario ID if this session was started from Demo Mode."
    )


class SessionTimelineResponse(BaseResponse):
    """Full ordered history of a session — used by the Session Timeline view."""
    session: SessionInfo
    frames: list[APEXResult] = Field(
        ..., description="All frames in chronological order."
    )
    summary_stats: dict[str, Any] = Field(
        default_factory=dict,
        description="Aggregate stats: min/max/avg μ̂, peak recommendation, total drying time.",
    )


# ===========================================================================
# Upload / Demo schemas
# ===========================================================================

class AnalysisResponse(BaseResponse):
    """Response to POST /api/v1/analyze/image."""
    session_id: str
    result: APEXResult


class VideoUploadResponse(BaseResponse):
    """Response to POST /api/v1/analyze/video — processing is async."""
    session_id: str
    message: str = Field(
        default="Video analysis started. Connect to WebSocket /ws/{session_id} for live updates.",
    )
    total_frames_estimated: int | None = Field(
        default=None, description="Estimated number of frames that will be processed."
    )
    processing_fps: float | None = Field(
        default=None, description="Configured analysis frame rate."
    )


class DemoClipMetadata(BaseModel):
    """Metadata for a pre-loaded demo race scenario."""
    clip_id: str = Field(..., description="Unique identifier for this scenario.")
    name: str = Field(..., description="Human-readable scenario name.")
    description: str = Field(..., description="Scenario description shown in the UI.")
    condition_scenario: str = Field(
        ...,
        description=(
            "Primary scenario type: 'wet_severe', 'wet_to_drying', "
            "'drying_to_dry', 'dry_evolved', 'sudden_shower'."
        ),
    )
    frame_count: int = Field(..., ge=1, description="Number of pre-processed frames.")
    thumbnail_b64: str | None = Field(
        default=None, description="Thumbnail image (base64 PNG) for the demo gallery."
    )
    historical_context: str | None = Field(
        default=None,
        description="Optional motorsport context (e.g., 'Inspired by 2021 Belgian GP').",
    )


class DemoListResponse(BaseResponse):
    """List of available demo scenarios."""
    clips: list[DemoClipMetadata]
    demo_mode_enabled: bool


# ===========================================================================
# WebSocket message schemas
# ===========================================================================

class WebSocketFrameMessage(BaseModel):
    """Message sent FROM backend TO frontend over WebSocket."""
    type: str = Field(..., description="Message type: 'frame_result' | 'progress' | 'error' | 'complete'.")
    session_id: str
    payload: dict[str, Any] = Field(default_factory=dict)


class WebSocketProgressMessage(BaseModel):
    """Progress update during video processing."""
    session_id: str
    frames_processed: int
    frames_total: int
    progress_pct: float = Field(..., ge=0.0, le=100.0)
    current_state: ConditionState | None = None
    current_mu: float | None = None
