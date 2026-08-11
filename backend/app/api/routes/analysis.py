"""
APEX Analysis Router — Module 08.

FastAPI router providing:
  - POST /api/v1/analyze : Single-frame track condition analysis
  - GET  /api/v1/history : Historical analysis queries
  - WS   /api/v1/stream  : WebSocket live frame stream ingestion
"""

from __future__ import annotations

import base64
import json
import time
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from loguru import logger

from app.api.dependencies import require_model_registry
from app.core.pipeline import PipelineResult, UnifiedPipeline
from app.models.registry import ModelRegistry

router = APIRouter(prefix="/api/v1", tags=["analysis"])

# In-memory history buffer (holds up to 100 recent analysis results for telemetry UI)
HISTORY_BUFFER: list[dict[str, Any]] = []
PIPELINE_INSTANCE: UnifiedPipeline | None = None


def get_pipeline(registry: ModelRegistry = Depends(require_model_registry)) -> UnifiedPipeline:
    """Dependency provider for UnifiedPipeline singleton."""
    global PIPELINE_INSTANCE
    if PIPELINE_INSTANCE is None or PIPELINE_INSTANCE.registry is not registry:
        PIPELINE_INSTANCE = UnifiedPipeline(registry=registry)
    return PIPELINE_INSTANCE


@router.post("/analyze")
async def analyze_frame(
    file: UploadFile | None = File(None),
    image_base64: str | None = Form(None),
    frame_index: int = Form(0),
    generate_visualizations: bool = Form(True),
    pipeline: UnifiedPipeline = Depends(get_pipeline),
) -> dict[str, Any]:
    """
    Analyze a track frame via image file upload or base64 data.
    """
    if file is None and not image_base64:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must supply either 'file' upload or 'image_base64' string.",
        )

    try:
        if file is not None:
            source = await file.read()
        else:
            source = image_base64

        result: PipelineResult = pipeline.process_frame(
            source_image=source,
            frame_index=frame_index,
            generate_visualizations=generate_visualizations,
        )

        response_payload = {
            "frame_index": result.frame_index,
            "timestamp": result.timestamp,
            "image_hash": result.image_hash,
            "track_condition": result.condition.track_condition.value,
            "metrics": {
                "wetness_index": result.condition.metrics.wetness_index,
                "puddle_coverage_pct": result.condition.metrics.puddle_coverage_pct,
                "wet_surface_pct": result.condition.metrics.wet_surface_pct,
                "damp_surface_pct": result.condition.metrics.damp_surface_pct,
                "dry_surface_pct": result.condition.metrics.dry_surface_pct,
                "rubber_ratio": result.condition.metrics.rubber_ratio,
                "clip_wet_confidence": result.condition.metrics.clip_wet_confidence,
                "clip_dry_confidence": result.condition.metrics.clip_dry_confidence,
                "dominant_condition": result.condition.metrics.dominant_condition,
            },
            "tyre_recommendation": {
                "compound": result.condition.tyre_recommendation.compound.value,
                "confidence": result.condition.tyre_recommendation.confidence,
                "lap_delta_seconds": result.condition.tyre_recommendation.lap_delta_seconds,
                "reasoning": result.condition.tyre_recommendation.reasoning,
                "pit_window_open": result.condition.tyre_recommendation.pit_window_open,
                "alternative_compound": (
                    result.condition.tyre_recommendation.alternative_compound.value
                    if result.condition.tyre_recommendation.alternative_compound
                    else None
                ),
            },
            "temporal_analysis": {
                "trend": result.temporal.trend.value,
                "volatility": result.temporal.volatility.value,
                "momentum_slope": result.temporal.momentum.slope_per_frame,
                "projected_wetness_in_5": result.temporal.momentum.projected_wetness_in_5,
                "stability_frames": result.temporal.condition_stability_frames,
                "tyre_window_alert": {
                    "alert_active": result.temporal.tyre_window_alert.alert_active,
                    "message": result.temporal.tyre_window_alert.message,
                    "from_compound": (
                        result.temporal.tyre_window_alert.from_compound.value
                        if result.temporal.tyre_window_alert.from_compound
                        else None
                    ),
                    "to_compound": (
                        result.temporal.tyre_window_alert.to_compound.value
                        if result.temporal.tyre_window_alert.to_compound
                        else None
                    ),
                },
            },
            "explainability": {
                "headline": result.explainability.headline,
                "detailed_summary": result.explainability.detailed_summary,
                "risk_assessment": result.explainability.risk_assessment,
                "recommended_action": result.explainability.recommended_action,
                "key_factors": [
                    {
                        "category": kf.category,
                        "factor": kf.factor,
                        "impact": kf.impact,
                        "description": kf.description,
                    }
                    for kf in result.explainability.key_factors
                ],
            },
            "visualization": {
                "overlay": result.visualization.overlay_b64,
                "segmentation_mask": result.visualization.segmentation_b64,
                "attention_heatmap": result.visualization.attention_heatmap_b64,
                "class_legend": result.visualization.class_legend,
                "dimensions": {"width": result.visualization.original_width, "height": result.visualization.original_height},
            },
            "sector_risk": {
                "sector_1": {
                    "risk_level": "CRITICAL" if result.condition.metrics.wetness_index > 0.7 else ("HIGH" if result.condition.metrics.wetness_index > 0.4 else ("MEDIUM" if result.condition.metrics.wetness_index > 0.2 else "LOW")),
                    "grip_mu": round(max(0.15, 1.0 - result.condition.metrics.wetness_index * 0.7), 2),
                    "water_depth_mm": round(result.condition.metrics.puddle_coverage_pct * 0.25, 1),
                },
                "sector_2": {
                    "risk_level": "CRITICAL" if result.condition.metrics.wetness_index > 0.6 else ("HIGH" if result.condition.metrics.wetness_index > 0.35 else ("MEDIUM" if result.condition.metrics.wetness_index > 0.15 else "LOW")),
                    "grip_mu": round(max(0.12, 1.0 - result.condition.metrics.wetness_index * 0.8), 2),
                    "water_depth_mm": round(result.condition.metrics.puddle_coverage_pct * 0.28, 1),
                },
                "sector_3": {
                    "risk_level": "CRITICAL" if result.condition.metrics.wetness_index > 0.8 else ("HIGH" if result.condition.metrics.wetness_index > 0.5 else ("MEDIUM" if result.condition.metrics.wetness_index > 0.25 else "LOW")),
                    "grip_mu": round(max(0.18, 1.0 - result.condition.metrics.wetness_index * 0.65), 2),
                    "water_depth_mm": round(result.condition.metrics.puddle_coverage_pct * 0.20, 1),
                },
            },
            "processing_time_ms": result.total_processing_time_ms,
        }

        # Keep history buffer for /history endpoint
        HISTORY_BUFFER.append(response_payload)
        if len(HISTORY_BUFFER) > 100:
            HISTORY_BUFFER.pop(0)

        return response_payload

    except Exception as exc:
        logger.exception("Error processing frame analysis")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Frame analysis failed: {str(exc)}",
        )


@router.get("/history")
async def get_history(limit: int = 20) -> dict[str, Any]:
    """Retrieve recent frame analysis history."""
    items = HISTORY_BUFFER[-limit:]
    return {
        "count": len(items),
        "history": items,
    }


@router.websocket("/stream")
async def stream_frames(
    websocket: WebSocket,
    registry: ModelRegistry = Depends(require_model_registry),
) -> None:
    """
    WebSocket endpoint for real-time video frame streaming.
    Receives base64 image strings, runs pipeline, and yields JSON responses.
    """
    await websocket.accept()
    logger.info("WebSocket telemetry stream connected")

    pipeline = get_pipeline(registry)
    frame_idx = 0

    try:
        while True:
            data = await websocket.receive_text()
            if not data:
                continue

            # Safely handle JSON payloads: {"image": "base64..."} from frontend api.js
            try:
                payload = json.loads(data)
                image_source = payload.get("image", data) if isinstance(payload, dict) else data
            except json.JSONDecodeError:
                image_source = data

            result = pipeline.process_frame(
                source_image=image_source,
                frame_index=frame_idx,
                generate_visualizations=True,
            )
            frame_idx += 1

            await websocket.send_json({
                "frame_index": result.frame_index,
                "track_condition": result.condition.track_condition.value,
                "wetness_index": result.condition.metrics.wetness_index,
                "tyre_recommendation": result.condition.tyre_recommendation.compound.value,
                "headline": result.explainability.headline,
                "segmentation_overlay": result.visualization.overlay_b64,
                "processing_time_ms": result.total_processing_time_ms,
            })
    except WebSocketDisconnect:
        logger.info("WebSocket telemetry stream disconnected")
    except Exception as exc:
        logger.error("WebSocket stream error: {}", exc)
        await websocket.close()
