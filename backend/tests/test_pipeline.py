"""
Tests for Module 08: Unified Intelligence Pipeline Engine.

Covers:
  - UnifiedPipeline initialization
  - End-to-end frame processing with synthetic image and mocked models
  - Output schema validation (PipelineResult, perception, condition, temporal, explainability, visualization)
  - Reset temporal state
"""

from __future__ import annotations

from unittest.mock import MagicMock
import numpy as np
import pytest
import torch
from PIL import Image

from app.core.pipeline import PipelineResult, UnifiedPipeline
from app.models.registry import ModelRegistry


@pytest.fixture
def mock_registry() -> MagicMock:
    reg = MagicMock(spec=ModelRegistry)
    reg.device = "cpu"
    reg.torch_dtype = torch.float32

    # Mock DINOv2
    dino_model = MagicMock()
    dino_outputs = MagicMock()
    dino_outputs.last_hidden_state = torch.randn(1, 197, 768)
    dino_outputs.attentions = [torch.rand(1, 12, 197, 197)]
    dino_model.return_value = dino_outputs
    dino_proc = MagicMock()
    dino_proc.return_value = {"pixel_values": torch.randn(1, 3, 224, 224)}
    reg.get_dinov2.return_value = (dino_model, dino_proc)

    # Mock SegFormer
    seg_model = MagicMock()
    seg_outputs = MagicMock()
    seg_outputs.logits = torch.randn(1, 19, 16, 16)
    seg_model.return_value = seg_outputs
    seg_proc = MagicMock()
    seg_proc.return_value = {"pixel_values": torch.randn(1, 3, 64, 64)}
    reg.get_segformer.return_value = (seg_model, seg_proc)

    # Mock CLIP
    clip_model = MagicMock()
    clip_outputs = MagicMock()
    clip_outputs.logits_per_image = torch.tensor([[5.0, 2.0, 1.0, 0.5, 0.1, 0.0, 0.2, 0.1]])
    clip_model.return_value = clip_outputs
    clip_proc = MagicMock()
    clip_proc.return_value = {"pixel_values": torch.randn(1, 3, 224, 224), "input_ids": torch.randint(0, 100, (8, 10))}
    reg.get_clip.return_value = (clip_model, clip_proc)

    # Mock status
    status = MagicMock()
    status.models = {}
    reg.get_status.return_value = status

    return reg


class TestUnifiedPipeline:
    """Test UnifiedPipeline execution and orchestrations."""

    def test_pipeline_init(self, mock_registry: MagicMock) -> None:
        pipeline = UnifiedPipeline(registry=mock_registry)
        assert pipeline.registry is mock_registry

    def test_process_synthetic_frame(self, mock_registry: MagicMock) -> None:
        pipeline = UnifiedPipeline(registry=mock_registry)

        img = Image.new("RGB", (128, 128), color="darkgray")
        result = pipeline.process_frame(img, frame_index=0)

        assert isinstance(result, PipelineResult)
        assert result.frame_index == 0
        assert result.perception is not None
        assert result.condition is not None
        assert result.temporal is not None
        assert result.explainability is not None
        assert result.visualization is not None
        assert result.total_processing_time_ms > 0
        assert len(result.visualization.overlay_b64) > 0

    def test_reset_temporal_state(self, mock_registry: MagicMock) -> None:
        pipeline = UnifiedPipeline(registry=mock_registry)

        img = Image.new("RGB", (64, 64), color="blue")
        pipeline.process_frame(img, frame_index=0)
        assert pipeline.temporal_reasoner.frame_count == 1

        pipeline.reset_temporal_state()
        assert pipeline.temporal_reasoner.frame_count == 0
