"""
APEX Perception Layer — Module 03.

Primary computer vision intelligence module:
  1. DINOv2 (ViT-B/14): Extracts 768-d CLS token, patch embeddings, and
     self-attention maps for XAI visual heatmaps.
  2. SegFormer (B2-Cityscapes): Generates semantic segmentation, remapped to 7
     track surface categories (BACKGROUND, DRY_SURFACE, DAMP_SURFACE, WET_SURFACE,
     PUDDLE, RUBBER_LINE, OCCLUDER) with luminance/HSV surface refinements.
  3. CLIP (ViT-L/14): Computes zero-shot probabilities across 8 motorsport condition prompts.
  4. Cross-Validation: Calculates confidence agreement between SegFormer surface
     analysis and CLIP prompt probabilities.

Hardware optimized: CPU-first, memory-aware, uses torch.no_grad().
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger
from PIL import Image

if TYPE_CHECKING:
    import torch
    from app.models.registry import ModelRegistry
    from app.utils.image_utils import PreprocessedFrame


# ---------------------------------------------------------------------------
# Surface Class Definitions & Prompts
# ---------------------------------------------------------------------------

class SurfaceClass(IntEnum):
    BACKGROUND   = 0  # Sky, vegetation, buildings, barriers
    DRY_SURFACE  = 1  # Matte asphalt, dry racing surface
    DAMP_SURFACE = 2  # Darkened asphalt, slight sheen, moisture
    WET_SURFACE  = 3  # Water film, high specular reflection
    PUDDLE       = 4  # Standing water, strong specular highlights
    RUBBER_LINE  = 5  # Dark rubber deposit corridor (racing line)
    OCCLUDER     = 6  # Cars, marshals, kerbs, objects to ignore


SURFACE_CLASS_NAMES: dict[int, str] = {
    SurfaceClass.BACKGROUND:   "background",
    SurfaceClass.DRY_SURFACE:  "dry_surface",
    SurfaceClass.DAMP_SURFACE: "damp_surface",
    SurfaceClass.WET_SURFACE:  "wet_surface",
    SurfaceClass.PUDDLE:       "puddle",
    SurfaceClass.RUBBER_LINE:  "rubber_line",
    SurfaceClass.OCCLUDER:     "occluder",
}

# Hex colors for visualization overlay (legend mapping)
SURFACE_CLASS_COLORS: dict[str, str] = {
    "background":   "#1A1D24",
    "dry_surface":  "#00C853",
    "damp_surface": "#FF9800",
    "wet_surface":  "#2196F3",
    "puddle":       "#00D4FF",
    "rubber_line":  "#7E57C2",
    "occluder":     "#E91E63",
}

# 8 Motorsport Condition Prompts for CLIP zero-shot classification
CONDITION_PROMPTS: dict[str, str] = {
    "wet_severe":     "a Formula 1 race track completely covered in standing water with strong reflections",
    "wet_moderate":   "a wet racing circuit with puddles and a shiny reflective surface",
    "transitional":   "a drying race track with patches of wet and dry asphalt",
    "drying":         "a race track that is mostly dry with some damp areas remaining",
    "dry_green":      "a clean dry race track with fresh asphalt and no rubber deposits",
    "dry_evolved":    "a dark rubbered-in racing line on a dry race track",
    "sudden_shower":   "heavy rain falling on a motorsport track with spray",
    "marbles_offline": "rubber marbles and debris accumulating off the racing line",
}


# ---------------------------------------------------------------------------
# Perception Result Dataclass
# ---------------------------------------------------------------------------

@dataclass
class PerceptionResult:
    """
    Complete output container for Stage 1 Perception analysis.
    """
    frame_index: int
    timestamp: str
    image_hash: str
    segmentation_mask: np.ndarray             # (H, W) uint8 array [0..6]
    class_proportions: dict[str, float]        # Name -> fraction of track area [0.0..1.0]
    clip_scores: dict[str, float]              # Prompt key -> probability [0.0..1.0]
    dinov2_cls_token: np.ndarray               # (768,) float32 embedding
    dinov2_patch_embeddings: np.ndarray        # (N, 768) float32 embeddings
    attention_map: np.ndarray                  # (Grid_H, Grid_W) normalized attention [0.0..1.0]
    confidence_agreement: float                # Agreement between SegFormer & CLIP [0.0..1.0]
    processing_time_ms: float                  # Stage latency in milliseconds
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core Perception Pipeline Engine
# ---------------------------------------------------------------------------

class PerceptionEngine:
    """
    Perception intelligence stage executing DINOv2, SegFormer, and CLIP.
    """

    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry

    def analyze(self, frame: PreprocessedFrame) -> PerceptionResult:
        """
        Run the complete multi-model perception analysis on a preprocessed frame.

        Args:
            frame: PreprocessedFrame from Module 02.

        Returns:
            PerceptionResult container populated with features, masks, and scores.
        """
        t0 = time.perf_counter()
        pil_img = frame.pil_image

        # 1. Run DINOv2 Feature & Attention Map Extraction
        dinov2_cls, dinov2_patches, attention_map = self._run_dinov2(frame, pil_img)

        # 2. Run SegFormer Semantic Segmentation & Surface Refinement
        seg_mask, class_props = self._run_segformer(frame, pil_img)

        # 3. Run CLIP Zero-Shot Condition Classification
        clip_scores = self._run_clip(frame, pil_img)

        # 4. Compute Cross-Validation Agreement
        agreement = self._compute_confidence_agreement(class_props, clip_scores)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.debug("Perception stage completed in {:.1f} ms | agreement={:.2f}", elapsed_ms, agreement)

        return PerceptionResult(
            frame_index=frame.frame_index,
            timestamp=frame.timestamp,
            image_hash=frame.image_hash,
            segmentation_mask=seg_mask,
            class_proportions=class_props,
            clip_scores=clip_scores,
            dinov2_cls_token=dinov2_cls,
            dinov2_patch_embeddings=dinov2_patches,
            attention_map=attention_map,
            confidence_agreement=agreement,
            processing_time_ms=elapsed_ms,
        )

    # =========================================================================
    # DINOv2 Stage
    # =========================================================================

    def _run_dinov2(
        self,
        frame: PreprocessedFrame,
        pil_img: Image.Image,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run DINOv2 forward pass with attention extraction.
        """
        import torch  # noqa: PLC0415

        t_start = time.perf_counter()
        model, processor = self.registry.get_dinov2()
        device = self.registry.device

        # Get or prepare inputs
        if frame.dinov2_tensor is not None:
            pixel_values = frame.dinov2_tensor
        else:
            inputs = processor(images=pil_img, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, output_attentions=True)

        # Record latency
        self.registry.record_inference_time(
            from_role_name("dinov2"), (time.perf_counter() - t_start) * 1000.0
        )

        # Extract CLS token (768,) and patch embeddings (196, 768)
        last_hidden_state = outputs.last_hidden_state[0]  # (1 + N_patches, 768)
        cls_token = last_hidden_state[0].cpu().numpy().astype(np.float32)
        patch_embeddings = last_hidden_state[1:].cpu().numpy().astype(np.float32)

        # Extract attention map from last layer
        # outputs.attentions is tuple of (1, num_heads, seq_len, seq_len)
        if outputs.attentions is not None and len(outputs.attentions) > 0:
            last_attn = outputs.attentions[-1][0]  # (num_heads, seq_len, seq_len)
            # Mean attention from CLS token (index 0) to patches (indices 1..)
            cls_attn = last_attn[:, 0, 1:].mean(dim=0).cpu().numpy()  # (N_patches,)
            grid_size = int(np.sqrt(len(cls_attn)))
            if grid_size * grid_size == len(cls_attn):
                attn_map = cls_attn.reshape(grid_size, grid_size)
            else:
                attn_map = cls_attn.reshape(14, 14)  # Default for ViT-B/14
        else:
            attn_map = np.ones((14, 14), dtype=np.float32) / 196.0

        # Normalize attention map to [0.0, 1.0]
        min_v, max_v = attn_map.min(), attn_map.max()
        if max_v > min_v:
            attn_map = (attn_map - min_v) / (max_v - min_v)
        else:
            attn_map = np.zeros_like(attn_map)

        return cls_token, patch_embeddings, attn_map

    # =========================================================================
    # SegFormer Stage
    # =========================================================================

    def _run_segformer(
        self,
        frame: PreprocessedFrame,
        pil_img: Image.Image,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """
        Run SegFormer segmentation and remap to 7 APEX surface classes.
        """
        import torch  # noqa: PLC0415
        import torch.nn.functional as F  # noqa: PLC0415

        t_start = time.perf_counter()
        model, processor = self.registry.get_segformer()
        device = self.registry.device

        # Get or prepare inputs
        if frame.segformer_inputs is not None:
            inputs = frame.segformer_inputs
        else:
            raw_inputs = processor(images=pil_img, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in raw_inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits  # (1, 19, H/4, W/4)

            # Resize logits to match PIL image size
            orig_w, orig_h = pil_img.size
            upsampled_logits = F.interpolate(
                logits,
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False,
            )
            raw_mask = torch.argmax(upsampled_logits[0], dim=0).cpu().numpy().astype(np.uint8)

        self.registry.record_inference_time(
            from_role_name("segformer"), (time.perf_counter() - t_start) * 1000.0
        )

        # Apply Cityscapes 19-class to 7 APEX surface classes remapping
        apex_mask = self._remap_segmentation(raw_mask, pil_img)

        # Compute surface proportions over visible track area
        track_pixels = (apex_mask != SurfaceClass.BACKGROUND) & (apex_mask != SurfaceClass.OCCLUDER)
        total_track = int(np.sum(track_pixels))

        class_props: dict[str, float] = {}
        for sc in SurfaceClass:
            name = SURFACE_CLASS_NAMES[sc]
            if total_track > 0:
                count = int(np.sum(apex_mask == sc))
                class_props[name] = float(count / total_track) if sc not in (SurfaceClass.BACKGROUND, SurfaceClass.OCCLUDER) else float(count / apex_mask.size)
            else:
                class_props[name] = 0.0

        return apex_mask, class_props

    def _remap_segmentation(self, cityscapes_mask: np.ndarray, pil_img: Image.Image) -> np.ndarray:
        """
        Remap 19-class Cityscapes mask + HSV surface analysis to 7 APEX surface classes.
        """
        h, w = cityscapes_mask.shape
        apex_mask = np.full((h, w), SurfaceClass.BACKGROUND, dtype=np.uint8)

        # Cityscapes mapping:
        # 0 (road), 1 (sidewalk) -> Track area candidate
        # 11..18 (vehicles, riders, persons) -> OCCLUDER
        is_road = (cityscapes_mask == 0) | (cityscapes_mask == 1)
        is_occluder = (cityscapes_mask >= 11) & (cityscapes_mask <= 18)

        apex_mask[is_occluder] = SurfaceClass.OCCLUDER

        if not np.any(is_road):
            # Fallback if SegFormer misses road: assume bottom 60% center is track
            h_start = int(h * 0.4)
            w_start, w_end = int(w * 0.2), int(w * 0.8)
            is_road[h_start:, w_start:w_end] = True

        # Perform HSV surface refinement on track area
        img_np = np.array(pil_img)  # RGB uint8
        # Convert RGB to HSV
        from PIL import ImageStat  # noqa: PLC0415
        
        # Calculate luminance (V) and saturation (S) efficiently
        r, g, b = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]
        gray = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)
        
        max_c = np.maximum(np.maximum(r, g), b)
        min_c = np.minimum(np.minimum(r, g), b)
        sat = np.zeros_like(gray)
        non_zero_max = max_c > 0
        sat[non_zero_max] = ((max_c[non_zero_max] - min_c[non_zero_max]) * 255 // max_c[non_zero_max]).astype(np.uint8)

        # Apply thresholds on road pixels:
        road_indices = np.where(is_road)

        road_gray = gray[road_indices]
        road_sat = sat[road_indices]

        # Classification logic per pixel on track:
        # 1. Puddle: High specular highlight (gray > 215) + low saturation (sat < 40)
        is_puddle = (road_gray > 215) & (road_sat < 40)
        
        # 2. Wet Surface: Moderately high luminance reflection (gray > 165) + low saturation (sat < 50)
        is_wet = (road_gray > 165) & (road_gray <= 215) & (road_sat < 50)
        
        # 3. Rubber Line: Dark rubber deposit (gray < 55)
        is_rubber = (road_gray < 55)
        
        # 4. Damp Surface: Darkened asphalt (55 <= gray < 115)
        is_damp = (road_gray >= 55) & (road_gray < 115)
        
        # 5. Dry Surface: Normal matte asphalt (115 <= gray <= 165)
        is_dry = (road_gray >= 115) & (road_gray <= 165)

        # Assign classes
        track_classes = np.full(len(road_indices[0]), SurfaceClass.DRY_SURFACE, dtype=np.uint8)
        track_classes[is_damp] = SurfaceClass.DAMP_SURFACE
        track_classes[is_rubber] = SurfaceClass.RUBBER_LINE
        track_classes[is_wet] = SurfaceClass.WET_SURFACE
        track_classes[is_puddle] = SurfaceClass.PUDDLE

        apex_mask[road_indices] = track_classes

        return apex_mask

    # =========================================================================
    # CLIP Stage
    # =========================================================================

    def _run_clip(
        self,
        frame: PreprocessedFrame,
        pil_img: Image.Image,
    ) -> dict[str, float]:
        """
        Run CLIP zero-shot classification across 8 condition prompts.
        """
        import torch  # noqa: PLC0415
        import torch.nn.functional as F  # noqa: PLC0415

        t_start = time.perf_counter()
        model, processor = self.registry.get_clip()
        device = self.registry.device

        prompts_list = list(CONDITION_PROMPTS.values())
        prompt_keys = list(CONDITION_PROMPTS.keys())

        # Prepare inputs
        inputs = processor(
            text=prompts_list,
            images=pil_img,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image  # (1, 8)
            probs = F.softmax(logits_per_image, dim=-1)[0].cpu().numpy()

        self.registry.record_inference_time(
            from_role_name("clip"), (time.perf_counter() - t_start) * 1000.0
        )

        return {key: float(probs[i]) for i, key in enumerate(prompt_keys)}

    # =========================================================================
    # Cross-Validation Agreement
    # =========================================================================

    def _compute_confidence_agreement(
        self,
        class_props: dict[str, float],
        clip_scores: dict[str, float],
    ) -> float:
        """
        Compute agreement score [0.0, 1.0] between SegFormer wetness & CLIP prompt scores.
        """
        # SegFormer wetness indicator: damp + wet + puddle ratio on track
        seg_wet = class_props.get("damp_surface", 0.0) * 0.5 + \
                  class_props.get("wet_surface", 0.0) * 0.85 + \
                  class_props.get("puddle", 0.0) * 1.0

        # CLIP wetness indicator: sum of wet-related prompt probabilities
        clip_wet = clip_scores.get("wet_severe", 0.0) * 1.0 + \
                   clip_scores.get("wet_moderate", 0.0) * 0.85 + \
                   clip_scores.get("sudden_shower", 0.0) * 0.90 + \
                   clip_scores.get("transitional", 0.0) * 0.50 + \
                   clip_scores.get("drying", 0.0) * 0.30

        # Agreement is inverse of absolute difference
        diff = abs(seg_wet - clip_wet)
        agreement = max(0.0, min(1.0, 1.0 - diff))
        return float(agreement)


def from_role_name(name: str):
    from app.models.registry import ModelRole  # noqa: PLC0415
    return ModelRole(name)
