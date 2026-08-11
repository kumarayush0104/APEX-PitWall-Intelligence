"""
Module 03 tests — Perception Layer (Core Vision).

Tests for SurfaceClass, PerceptionResult, SegFormer class remapping,
CLIP prompt confidence agreement calculation, and PerceptionEngine mocking.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

# Ensure backend root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.perception import (
    CONDITION_PROMPTS,
    SURFACE_CLASS_COLORS,
    SURFACE_CLASS_NAMES,
    PerceptionEngine,
    PerceptionResult,
    SurfaceClass,
)
from app.utils.image_utils import PreprocessedFrame


# ---------------------------------------------------------------------------
# Tests for Constants & Enums
# ---------------------------------------------------------------------------

class TestPerceptionConstants:
    """Verify SurfaceClass enum and prompt constants."""

    def test_surface_class_enum_values(self) -> None:
        assert SurfaceClass.BACKGROUND == 0
        assert SurfaceClass.DRY_SURFACE == 1
        assert SurfaceClass.DAMP_SURFACE == 2
        assert SurfaceClass.WET_SURFACE == 3
        assert SurfaceClass.PUDDLE == 4
        assert SurfaceClass.RUBBER_LINE == 5
        assert SurfaceClass.OCCLUDER == 6

    def test_surface_class_names_complete(self) -> None:
        for sc in SurfaceClass:
            assert sc in SURFACE_CLASS_NAMES
            assert isinstance(SURFACE_CLASS_NAMES[sc], str)

    def test_surface_class_colors_complete(self) -> None:
        for sc in SurfaceClass:
            name = SURFACE_CLASS_NAMES[sc]
            assert name in SURFACE_CLASS_COLORS
            assert SURFACE_CLASS_COLORS[name].startswith("#")

    def test_condition_prompts_count(self) -> None:
        assert len(CONDITION_PROMPTS) == 8
        for key in ("wet_severe", "wet_moderate", "transitional", "drying", "dry_green", "dry_evolved", "sudden_shower", "marbles_offline"):
            assert key in CONDITION_PROMPTS


# ---------------------------------------------------------------------------
# Tests for PerceptionEngine Helpers
# ---------------------------------------------------------------------------

class TestPerceptionEngineHelpers:
    """Test confidence agreement calculation & segmentation remapping."""

    @pytest.fixture
    def engine(self) -> PerceptionEngine:
        mock_registry = MagicMock()
        return PerceptionEngine(registry=mock_registry)

    def test_confidence_agreement_identical(self, engine: PerceptionEngine) -> None:
        # High wetness in both SegFormer and CLIP -> High agreement
        class_props = {"damp_surface": 0.2, "wet_surface": 0.5, "puddle": 0.2}
        clip_scores = {"wet_severe": 0.6, "wet_moderate": 0.3}
        agreement = engine._compute_confidence_agreement(class_props, clip_scores)
        assert 0.7 <= agreement <= 1.0

    def test_confidence_agreement_disagreement(self, engine: PerceptionEngine) -> None:
        # High wetness in SegFormer, but CLIP says 100% dry_evolved -> Low agreement
        class_props = {"wet_surface": 0.9, "puddle": 0.1}
        clip_scores = {"dry_evolved": 1.0, "wet_severe": 0.0}
        agreement = engine._compute_confidence_agreement(class_props, clip_scores)
        assert agreement < 0.5

    def test_remap_segmentation_road_pixels(self, engine: PerceptionEngine) -> None:
        # Create 10x10 Cityscapes mask with road (class 0)
        cityscapes_mask = np.zeros((10, 10), dtype=np.uint8)
        # Add car (class 13) at corner
        cityscapes_mask[0, 0] = 13

        # Create dummy PIL image
        img_arr = np.full((10, 10, 3), 128, dtype=np.uint8)  # Mid-gray -> Dry Surface
        pil_img = Image.fromarray(img_arr)

        apex_mask = engine._remap_segmentation(cityscapes_mask, pil_img)

        assert apex_mask.shape == (10, 10)
        assert apex_mask[0, 0] == SurfaceClass.OCCLUDER
        assert apex_mask[5, 5] in (SurfaceClass.DRY_SURFACE, SurfaceClass.DAMP_SURFACE)


# ---------------------------------------------------------------------------
# End-to-End PerceptionEngine Test (Mocked Models)
# ---------------------------------------------------------------------------

class TestPerceptionEngineAnalyze:
    """Test analyze pipeline execution with mocked ModelRegistry outputs."""

    def test_perception_analyze_full_pipeline(self) -> None:
        import torch

        mock_registry = MagicMock()
        mock_registry.device = "cpu"

        # Mock DINOv2
        mock_dinov2_model = MagicMock()
        mock_dinov2_outputs = MagicMock()
        mock_dinov2_outputs.last_hidden_state = torch.randn(1, 197, 768)
        mock_dinov2_outputs.attentions = [torch.rand(1, 12, 197, 197)]
        mock_dinov2_model.return_value = mock_dinov2_outputs

        mock_dinov2_proc = MagicMock()
        mock_dinov2_proc.return_value = {"pixel_values": torch.randn(1, 3, 224, 224)}
        mock_registry.get_dinov2.return_value = (mock_dinov2_model, mock_dinov2_proc)

        # Mock SegFormer
        mock_seg_model = MagicMock()
        mock_seg_outputs = MagicMock()
        mock_seg_outputs.logits = torch.randn(1, 19, 16, 16)
        mock_seg_model.return_value = mock_seg_outputs

        mock_seg_proc = MagicMock()
        mock_seg_proc.return_value = {"pixel_values": torch.randn(1, 3, 64, 64)}
        mock_registry.get_segformer.return_value = (mock_seg_model, mock_seg_proc)

        # Mock CLIP
        mock_clip_model = MagicMock()
        mock_clip_outputs = MagicMock()
        mock_clip_outputs.logits_per_image = torch.tensor([[5.0, 2.0, 1.0, 0.5, 0.1, 0.0, 0.2, 0.1]])
        mock_clip_model.return_value = mock_clip_outputs

        mock_clip_proc = MagicMock()
        mock_clip_proc.return_value = {"pixel_values": torch.randn(1, 3, 224, 224), "input_ids": torch.randint(0, 100, (8, 10))}
        mock_registry.get_clip.return_value = (mock_clip_model, mock_clip_proc)

        # Create input frame
        pil_img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        frame = PreprocessedFrame(
            pil_image=pil_img,
            original_size=(64, 64),
            resized_size=(64, 64),
            aspect_ratio=1.0,
            image_hash="test_hash_123",
            frame_index=1,
        )

        engine = PerceptionEngine(registry=mock_registry)
        result = engine.analyze(frame)

        assert isinstance(result, PerceptionResult)
        assert result.frame_index == 1
        assert result.image_hash == "test_hash_123"
        assert result.segmentation_mask.shape == (64, 64)
        assert result.dinov2_cls_token.shape == (768,)
        assert result.dinov2_patch_embeddings.shape == (196, 768)
        assert result.attention_map.shape == (14, 14)
        assert len(result.clip_scores) == 8
        assert 0.0 <= result.confidence_agreement <= 1.0
        assert result.processing_time_ms > 0
