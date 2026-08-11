"""
APEX Unified Intelligence Pipeline Engine — Module 08.

Orchestrates all 7 modules into a single thread-safe pipeline execution context:
  Module 01: Model Registry & Lazy Loader
  Module 02: Image Utilities & Safety Layer
  Module 03: Perception Layer (DINOv2, SegFormer, CLIP)
  Module 04: Visualization Engine (Masks, Heatmaps, Bundles)
  Module 05: Surface Metrics & Condition Classifier
  Module 06: Temporal Reasoner (Rolling Window & Crossover Alerts)
  Module 07: Explainability Engine (Engineering Insights & Rationale)

Guarantees strict sequential model execution (DINOv2 -> SegFormer -> CLIP)
under the CPU-first 8GB RAM budget constraint.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from app.core.condition_classifier import ConditionAssessment, ConditionClassifier
from app.core.explainability import EngineeringInsight, ExplainabilityEngine
from app.core.perception import SURFACE_CLASS_COLORS, PerceptionEngine, PerceptionResult
from app.core.temporal_reasoner import TemporalAnalysis, TemporalReasoner
from app.models.registry import ModelRegistry
from app.utils.image_utils import PreprocessedFrame, load_image, preprocess_frame
from app.utils.visualization import VisualizationBundle, create_visualization_bundle


# ---------------------------------------------------------------------------
# Pipeline Output Container
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Complete multi-stage APEX analysis result for a single input frame."""
    frame_index: int
    timestamp: str
    image_hash: str
    perception: PerceptionResult
    condition: ConditionAssessment
    temporal: TemporalAnalysis
    explainability: EngineeringInsight
    visualization: VisualizationBundle
    total_processing_time_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Unified Pipeline Orchestrator
# ---------------------------------------------------------------------------

class UnifiedPipeline:
    """
    Main entry point for APEX track intelligence analysis.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        temporal_window_size: int = 30,
    ) -> None:
        self.registry = registry
        self.perception_engine = PerceptionEngine(registry=registry)
        self.condition_classifier = ConditionClassifier()
        self.temporal_reasoner = TemporalReasoner(window_size=temporal_window_size)
        self.explainability_engine = ExplainabilityEngine(registry=registry)

    def process_frame(
        self,
        source_image: Any,
        frame_index: int = 0,
        generate_visualizations: bool = True,
    ) -> PipelineResult:
        """
        Execute the full APEX pipeline on a raw image source.

        Args:
            source_image: PIL Image, raw bytes, base64 string, filepath, or NumPy array.
            frame_index: Sequential frame number.
            generate_visualizations: Whether to render base64 visual overlays.

        Returns:
            PipelineResult containing all stage outputs.
        """
        t0 = time.perf_counter()

        # 1. Image Preprocessing & Safety Downscaling (Module 02)
        frame: PreprocessedFrame = preprocess_frame(
            input_data=source_image,
            frame_index=frame_index,
            registry=self.registry,
        )

        # 2. Perception Layer Analysis (Module 03)
        perception: PerceptionResult = self.perception_engine.analyze(frame)

        # 3. Surface Metrics & Condition Classification (Module 05)
        condition: ConditionAssessment = self.condition_classifier.assess(perception)

        # 4. Temporal Trend Reasoning (Module 06)
        temporal: TemporalAnalysis = self.temporal_reasoner.update(condition)

        # 5. Explainability & Insights Generation (Module 07)
        explainability: EngineeringInsight = self.explainability_engine.explain(
            perception=perception,
            condition=condition,
            temporal=temporal,
        )

        # 6. Visualization Overlay Generation (Module 04)
        if generate_visualizations:
            visualization: VisualizationBundle = create_visualization_bundle(
                frame=frame,
                perception_result=perception,
            )
        else:
            visualization = VisualizationBundle(
                overlay_b64="",
                segmentation_b64="",
                attention_heatmap_b64="",
                class_legend=SURFACE_CLASS_COLORS,
                original_width=frame.width,
                original_height=frame.height,
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Pipeline frame #{} processed in {:.1f} ms | Condition: {} | Wetness: {:.3f}",
            frame_index, elapsed_ms, condition.track_condition.value, condition.metrics.wetness_index,
        )

        return PipelineResult(
            frame_index=frame_index,
            timestamp=frame.timestamp,
            image_hash=frame.image_hash,
            perception=perception,
            condition=condition,
            temporal=temporal,
            explainability=explainability,
            visualization=visualization,
            total_processing_time_ms=elapsed_ms,
        )

    def reset_temporal_state(self) -> None:
        """Reset temporal history ring buffer."""
        self.temporal_reasoner.reset()
