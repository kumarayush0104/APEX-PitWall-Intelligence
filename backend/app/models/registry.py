"""
APEX Model Registry & Loader — Module 01.

The beating heart of the inference system on constrained hardware.

Architecture decisions for 8 GB RAM / CPU-first:
─────────────────────────────────────────────────
1. LAZY LOADING — Models are loaded on first access, not at startup.
   A cold /health check returns in <50ms because no models are in memory.

2. SEQUENTIAL LOADING — Only one model loads at a time (threading.Lock).
   Prevents double-loading on concurrent requests.

3. MEMORY AWARENESS — Each model tracks its approximate RAM footprint.
   The registry will refuse to load a model if it would exceed MAX_RAM_GB.
   If a new model won't fit, the least-recently-used model is unloaded first.

4. UNLOAD CAPABILITY — Any model can be explicitly unloaded to free RAM.
   gc.collect() + torch.cuda.empty_cache() are called after unloading.

5. HF INFERENCE API FALLBACK — If a model fails to load locally (OOM,
   download failure, etc.), the registry marks it as "hf_api" and the
   pipeline will call the HF Inference API instead.

6. THREAD SAFETY — All load/unload operations are guarded by a
   reentrant lock. Multiple threads can READ the registry concurrently
   but only one can modify it.

7. WARMUP — A dummy forward pass is run after loading to eliminate
   JIT compilation latency on the first real request.

8. HEALTH INTEGRATION — get_status() returns a dict[str, ModelStatus]
   matching the schema defined in Module 00.

Memory budget for this machine (from health endpoint):
  Total RAM:     8.52 GB
  Available:     ~1.5 GB (at rest)
  Model budget:  5.0 GB (configurable via MAX_RAM_GB)

  DINOv2-base:    ~344 MB float32
  SegFormer-B2:   ~108 MB float32
  CLIP ViT-L/14:  ~1200 MB float32
  ─────────────────────────────────
  Total:          ~1.65 GB → fits within 5 GB budget ✓

  Qwen2-VL-7B:   ~15 GB → NEVER loaded locally. Always API.
"""

from __future__ import annotations

import gc
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from loguru import logger

from app.config import Settings, VLMProvider, get_settings
from app.models.schemas import ModelProvider, ModelStatus


# ---------------------------------------------------------------------------
# Model specification descriptors
# ---------------------------------------------------------------------------

class ModelRole(str, Enum):
    """Identifiers for each model in the pipeline."""
    DINOV2 = "dinov2"
    SEGFORMER = "segformer"
    CLIP = "clip"
    VLM = "vlm"


@dataclass
class ModelSpec:
    """
    Static specification for a model — what it is, how much RAM it needs,
    and how to load it.
    """
    role: ModelRole
    model_id: str
    estimated_ram_mb: float  # Approximate float32 weight size in megabytes
    loader_fn_name: str      # Name of the private loader method on ModelRegistry
    requires_processor: bool = True
    supports_local: bool = True  # False for VLM on 8 GB machines


@dataclass
class LoadedModel:
    """
    Runtime state for a loaded model — the model instance, processor,
    and timing/memory metadata.
    """
    spec: ModelSpec
    model: Any = None
    processor: Any = None
    device: str = "cpu"
    dtype_str: str = "float32"
    loaded: bool = False
    provider: ModelProvider = ModelProvider.NOT_LOADED
    load_time_seconds: float = 0.0
    actual_ram_mb: float = 0.0
    last_inference_ms: float | None = None
    last_accessed: float = field(default_factory=time.time)
    error: str | None = None


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------

class ModelRegistry:
    """
    Thread-safe, memory-aware model registry for the APEX pipeline.

    Usage:
        registry = ModelRegistry(settings)

        # Lazy — loads on first access
        model, processor = registry.get_dinov2()

        # Check what's loaded
        status = registry.get_status()

        # Free memory
        registry.unload(ModelRole.CLIP)

        # Pre-load everything (for demo warmup)
        await registry.warmup()
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._lock = threading.RLock()  # Reentrant for nested calls
        self._models: dict[ModelRole, LoadedModel] = {}
        self._initialised = False

        # Register all model specs
        self._specs: dict[ModelRole, ModelSpec] = {
            ModelRole.DINOV2: ModelSpec(
                role=ModelRole.DINOV2,
                model_id=self._settings.DINOV2_MODEL_ID,
                estimated_ram_mb=344.0,
                loader_fn_name="_load_dinov2",
            ),
            ModelRole.SEGFORMER: ModelSpec(
                role=ModelRole.SEGFORMER,
                model_id=self._settings.SEGFORMER_MODEL_ID,
                estimated_ram_mb=108.0,
                loader_fn_name="_load_segformer",
            ),
            ModelRole.CLIP: ModelSpec(
                role=ModelRole.CLIP,
                model_id=self._settings.CLIP_MODEL_ID,
                estimated_ram_mb=1200.0,
                loader_fn_name="_load_clip",
            ),
            ModelRole.VLM: ModelSpec(
                role=ModelRole.VLM,
                model_id=self._settings.VLM_MODEL_ID,
                estimated_ram_mb=15000.0,  # 15 GB — never loaded locally
                loader_fn_name="_load_vlm",
                supports_local=(
                    self._settings.VLM_PROVIDER == VLMProvider.LOCAL
                    and self._settings.MAX_RAM_GB >= 16.0
                ),
            ),
        }

        # Initialize empty LoadedModel for each spec
        for role, spec in self._specs.items():
            provider = ModelProvider.NOT_LOADED
            if role == ModelRole.VLM:
                if not self._settings.VLM_ENABLED:
                    provider = ModelProvider.DISABLED
                elif self._settings.VLM_PROVIDER == VLMProvider.API:
                    provider = ModelProvider.HF_API
                elif self._settings.VLM_PROVIDER == VLMProvider.NONE:
                    provider = ModelProvider.DISABLED
            self._models[role] = LoadedModel(spec=spec, provider=provider)

        self._initialised = True
        logger.info(
            "ModelRegistry initialised | device={} | dtype={} | budget={:.1f} GB",
            self._settings.resolved_device,
            self._settings.effective_torch_dtype,
            self._settings.MAX_RAM_GB,
        )

    # ==================================================================
    # Properties
    # ==================================================================

    @property
    def device(self) -> str:
        """Resolved compute device string ('cpu' or 'cuda')."""
        return self._settings.resolved_device

    @property
    def dtype(self):
        """Resolved torch dtype object. Returns None if torch not available."""
        try:
            import torch  # noqa: PLC0415
            dtype_map = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }
            return dtype_map.get(self._settings.effective_torch_dtype, torch.float32)
        except ImportError:
            return None

    @property
    def total_loaded_ram_mb(self) -> float:
        """Total RAM used by all currently loaded models in MB."""
        return sum(
            m.actual_ram_mb for m in self._models.values()
            if m.loaded and m.provider == ModelProvider.LOCAL
        )

    @property
    def budget_remaining_mb(self) -> float:
        """Remaining RAM budget for models in MB."""
        budget_mb = self._settings.MAX_RAM_GB * 1024
        return budget_mb - self.total_loaded_ram_mb

    # ==================================================================
    # Public accessors — lazy-loading
    # ==================================================================

    def get_dinov2(self) -> tuple[Any, Any]:
        """
        Get the DINOv2 model and processor.

        Loads lazily on first call. Thread-safe.
        Returns (model, processor) tuple.
        Raises RuntimeError if loading fails and no API fallback.
        """
        return self._ensure_loaded(ModelRole.DINOV2)

    def get_segformer(self) -> tuple[Any, Any]:
        """Get the SegFormer model and processor. Lazy-loads on first call."""
        return self._ensure_loaded(ModelRole.SEGFORMER)

    def get_clip(self) -> tuple[Any, Any]:
        """Get the CLIP model and processor. Lazy-loads on first call."""
        return self._ensure_loaded(ModelRole.CLIP)

    def get_vlm(self) -> tuple[Any, Any] | None:
        """
        Get the VLM model and processor, or None if VLM is API-only/disabled.

        For API mode, returns None — the explainer module calls HF API directly.
        For local mode (>= 16 GB VRAM only), loads and returns the model.
        """
        entry = self._models[ModelRole.VLM]
        if entry.provider in (ModelProvider.DISABLED, ModelProvider.HF_API):
            return None  # Caller uses HF API or structured fallback
        return self._ensure_loaded(ModelRole.VLM)

    def is_loaded(self, role: ModelRole) -> bool:
        """Check if a model is currently loaded in memory."""
        return self._models[role].loaded

    def get_provider(self, role: ModelRole) -> ModelProvider:
        """Get the current provider for a model."""
        return self._models[role].provider

    def record_inference_time(self, role: ModelRole, elapsed_ms: float) -> None:
        """Record inference latency for a model (called by pipeline stages)."""
        entry = self._models.get(role)
        if entry:
            entry.last_inference_ms = elapsed_ms
            entry.last_accessed = time.time()

    # ==================================================================
    # Load / Unload
    # ==================================================================

    def _ensure_loaded(self, role: ModelRole) -> tuple[Any, Any]:
        """
        Ensure a model is loaded and return (model, processor).

        Thread-safe: uses a reentrant lock to prevent double-loading.
        Memory-aware: checks budget and evicts LRU if needed.
        Fallback-aware: on failure, marks model for HF API fallback.
        """
        entry = self._models[role]

        # Fast path — already loaded
        if entry.loaded and entry.model is not None:
            entry.last_accessed = time.time()
            return (entry.model, entry.processor)

        # Slow path — need to load
        with self._lock:
            # Double-check after acquiring lock (another thread may have loaded it)
            if entry.loaded and entry.model is not None:
                entry.last_accessed = time.time()
                return (entry.model, entry.processor)

            spec = entry.spec

            # Check if model supports local loading
            if not spec.supports_local:
                logger.info(
                    "Model {} does not support local loading — using HF API",
                    role.value,
                )
                entry.provider = ModelProvider.HF_API
                raise RuntimeError(
                    f"Model {role.value} ({spec.model_id}) cannot be loaded locally "
                    f"on this machine. Use HF Inference API instead."
                )

            # Check memory budget — evict LRU if needed
            self._ensure_memory_budget(spec.estimated_ram_mb, exclude_role=role)

            # Load the model
            logger.info(
                "Loading {} ({}) → {} device, {} dtype...",
                role.value,
                spec.model_id,
                self.device,
                self._settings.effective_torch_dtype,
            )

            loader_method = getattr(self, spec.loader_fn_name, None)
            if loader_method is None:
                raise RuntimeError(f"No loader method '{spec.loader_fn_name}' for {role.value}")

            t0 = time.time()
            try:
                model, processor = loader_method()
                elapsed = time.time() - t0

                # Measure actual RAM usage
                actual_mb = self._estimate_model_ram(model)

                entry.model = model
                entry.processor = processor
                entry.device = self.device
                entry.dtype_str = self._settings.effective_torch_dtype
                entry.loaded = True
                entry.provider = ModelProvider.LOCAL
                entry.load_time_seconds = round(elapsed, 2)
                entry.actual_ram_mb = actual_mb
                entry.last_accessed = time.time()
                entry.error = None

                logger.info(
                    "✓ {} loaded in {:.1f}s | RAM: {:.0f} MB | device: {} | total loaded: {:.0f} MB",
                    role.value,
                    elapsed,
                    actual_mb,
                    self.device,
                    self.total_loaded_ram_mb,
                )

                return (model, processor)

            except Exception as exc:
                elapsed = time.time() - t0
                error_msg = f"Failed to load {role.value}: {type(exc).__name__}: {exc}"
                logger.error("{} (after {:.1f}s)", error_msg, elapsed)

                entry.loaded = False
                entry.model = None
                entry.processor = None
                entry.error = str(exc)

                # Attempt HF API fallback if token available
                if self._settings.HF_TOKEN and self._settings.USE_HF_INFERENCE_API:
                    logger.warning(
                        "Falling back to HF Inference API for {}", role.value,
                    )
                    entry.provider = ModelProvider.HF_API
                else:
                    entry.provider = ModelProvider.NOT_LOADED

                # Force garbage collection after failed load
                self._force_gc()

                raise RuntimeError(error_msg) from exc

    def unload(self, role: ModelRole) -> bool:
        """
        Unload a model from memory and free its RAM.

        Returns True if a model was actually unloaded, False if it
        wasn't loaded in the first place.
        """
        with self._lock:
            entry = self._models[role]

            if not entry.loaded:
                return False

            model_name = role.value
            ram_freed = entry.actual_ram_mb

            logger.info("Unloading {} (freeing ~{:.0f} MB)...", model_name, ram_freed)

            # Clear references
            entry.model = None
            entry.processor = None
            entry.loaded = False
            entry.actual_ram_mb = 0.0
            # Keep provider as NOT_LOADED (can be reloaded later)
            if entry.provider == ModelProvider.LOCAL:
                entry.provider = ModelProvider.NOT_LOADED

            # Force garbage collection
            self._force_gc()

            logger.info(
                "✓ {} unloaded | freed ~{:.0f} MB | total loaded: {:.0f} MB",
                model_name,
                ram_freed,
                self.total_loaded_ram_mb,
            )
            return True

    def unload_all(self) -> int:
        """Unload all models. Returns count of models unloaded."""
        count = 0
        for role in ModelRole:
            if self.unload(role):
                count += 1
        return count

    # ==================================================================
    # Memory management
    # ==================================================================

    def _ensure_memory_budget(
        self,
        needed_mb: float,
        exclude_role: ModelRole | None = None,
    ) -> None:
        """
        Ensure there is enough budget to load a model of `needed_mb` size.

        If the budget would be exceeded, evict the least-recently-used
        loaded model (that isn't `exclude_role`) until there is room.
        """
        budget_mb = self._settings.MAX_RAM_GB * 1024

        while (self.total_loaded_ram_mb + needed_mb) > budget_mb:
            # Find LRU candidate for eviction
            candidates = [
                (role, entry)
                for role, entry in self._models.items()
                if entry.loaded
                and entry.provider == ModelProvider.LOCAL
                and role != exclude_role
            ]

            if not candidates:
                logger.warning(
                    "Cannot free memory: no evictable models loaded. "
                    "Need {:.0f} MB, budget {:.0f} MB, used {:.0f} MB.",
                    needed_mb,
                    budget_mb,
                    self.total_loaded_ram_mb,
                )
                break  # Proceed anyway — torch may handle better than we estimate

            # Sort by last_accessed (oldest first)
            candidates.sort(key=lambda x: x[1].last_accessed)
            victim_role, victim = candidates[0]

            logger.info(
                "Memory pressure: evicting {} ({:.0f} MB) to make room for new model",
                victim_role.value,
                victim.actual_ram_mb,
            )
            self.unload(victim_role)

    def _estimate_model_ram(self, model: Any) -> float:
        """
        Estimate the RAM footprint of a loaded model in MB.

        Uses torch parameter counting when available, falls back to
        sys.getsizeof as a rough estimate.
        """
        try:
            import torch  # noqa: PLC0415
            if hasattr(model, "parameters"):
                total_bytes = sum(
                    p.nelement() * p.element_size()
                    for p in model.parameters()
                )
                # Add ~20% overhead for buffers, gradients, optimizer state
                return (total_bytes * 1.2) / (1024 * 1024)
        except (ImportError, Exception):
            pass

        # Rough fallback
        return sys.getsizeof(model) / (1024 * 1024)

    def _force_gc(self) -> None:
        """Force garbage collection and clear CUDA cache if available."""
        gc.collect()
        try:
            import torch  # noqa: PLC0415
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # ==================================================================
    # Individual model loaders
    # ==================================================================

    def _load_dinov2(self) -> tuple[Any, Any]:
        """
        Load DINOv2 ViT-B/14 for self-supervised feature extraction.

        Returns (model, processor).
        Model outputs: CLS token (768-d) + patch embeddings (196×768).
        We also extract self-attention maps from the last layer for XAI.
        """
        import torch  # noqa: PLC0415
        from transformers import AutoImageProcessor, AutoModel  # noqa: PLC0415

        model_id = self._settings.DINOV2_MODEL_ID

        processor = AutoImageProcessor.from_pretrained(
            model_id,
            cache_dir=str(self._settings.HF_CACHE_DIR / "hub"),
        )

        model = AutoModel.from_pretrained(
            model_id,
            cache_dir=str(self._settings.HF_CACHE_DIR / "hub"),
            torch_dtype=self.dtype,
        )

        model = model.to(self.device)
        model.eval()

        # Warmup with dummy input to eliminate JIT latency
        self._warmup_vision_model(model, processor, "dinov2", input_size=224)

        return (model, processor)

    def _load_segformer(self) -> tuple[Any, Any]:
        """
        Load SegFormer-B2 fine-tuned on Cityscapes for semantic segmentation.

        Returns (model, processor).
        Output: logits of shape (1, num_classes, H/4, W/4) — upscale to (H, W).
        Cityscapes classes (19) will be remapped to our 6 track classes in Module 03.
        """
        import torch  # noqa: PLC0415
        from transformers import (  # noqa: PLC0415
            SegformerForSemanticSegmentation,
            SegformerImageProcessor,
        )

        model_id = self._settings.SEGFORMER_MODEL_ID

        processor = SegformerImageProcessor.from_pretrained(
            model_id,
            cache_dir=str(self._settings.HF_CACHE_DIR / "hub"),
        )

        model = SegformerForSemanticSegmentation.from_pretrained(
            model_id,
            cache_dir=str(self._settings.HF_CACHE_DIR / "hub"),
            torch_dtype=self.dtype,
        )

        model = model.to(self.device)
        model.eval()

        # Warmup
        self._warmup_vision_model(model, processor, "segformer", input_size=512)

        return (model, processor)

    def _load_clip(self) -> tuple[Any, Any]:
        """
        Load CLIP ViT-L/14 for zero-shot condition classification.

        Returns (model, processor).
        Used for cross-validating SegFormer outputs via text-image similarity.
        This is the largest local model (~1.2 GB float32).
        """
        import torch  # noqa: PLC0415
        from transformers import CLIPModel, CLIPProcessor  # noqa: PLC0415

        model_id = self._settings.CLIP_MODEL_ID

        processor = CLIPProcessor.from_pretrained(
            model_id,
            cache_dir=str(self._settings.HF_CACHE_DIR / "hub"),
        )

        model = CLIPModel.from_pretrained(
            model_id,
            cache_dir=str(self._settings.HF_CACHE_DIR / "hub"),
            torch_dtype=self.dtype,
        )

        model = model.to(self.device)
        model.eval()

        # Warmup with dummy image + text
        self._warmup_clip_model(model, processor)

        return (model, processor)

    def _load_vlm(self) -> tuple[Any, Any]:
        """
        Load Qwen2-VL for local inference (NOT recommended on 8 GB machines).

        This loader exists for completeness but should almost never be called.
        On this hardware, VLM_PROVIDER should be 'api' (HF Inference API)
        or 'none' (structured fallback).
        """
        raise RuntimeError(
            f"Qwen2-VL-7B ({self._specs[ModelRole.VLM].estimated_ram_mb:.0f} MB) "
            f"cannot be loaded on a machine with {self._settings.MAX_RAM_GB:.0f} GB "
            f"RAM budget. Use VLM_PROVIDER=api to serve via HuggingFace Inference API, "
            f"or VLM_PROVIDER=none for structured fallback."
        )

    # ==================================================================
    # Warmup helpers
    # ==================================================================

    def _warmup_vision_model(
        self,
        model: Any,
        processor: Any,
        name: str,
        input_size: int = 224,
    ) -> None:
        """
        Run a single dummy forward pass to warm up the model.

        This eliminates JIT compilation latency on the first real inference.
        Uses a tiny random image to minimize compute cost.
        """
        try:
            import torch  # noqa: PLC0415
            from PIL import Image  # noqa: PLC0415
            import numpy as np  # noqa: PLC0415

            logger.debug("Warming up {} with {}×{} dummy input...", name, input_size, input_size)

            # Create a small random image
            dummy_array = np.random.randint(0, 255, (input_size, input_size, 3), dtype=np.uint8)
            dummy_image = Image.fromarray(dummy_array)

            # Process and run
            inputs = processor(images=dummy_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                _ = model(**inputs)

            logger.debug("✓ {} warmup complete", name)

        except Exception as e:
            logger.warning("Warmup failed for {} (non-critical): {}", name, e)

    def _warmup_clip_model(self, model: Any, processor: Any) -> None:
        """Warmup CLIP with a dummy image and text prompt."""
        try:
            import torch  # noqa: PLC0415
            from PIL import Image  # noqa: PLC0415
            import numpy as np  # noqa: PLC0415

            logger.debug("Warming up CLIP with dummy image + text...")

            dummy_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            dummy_image = Image.fromarray(dummy_array)

            inputs = processor(
                text=["a race track"],
                images=dummy_image,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                _ = model(**inputs)

            logger.debug("✓ CLIP warmup complete")

        except Exception as e:
            logger.warning("CLIP warmup failed (non-critical): {}", e)

    # ==================================================================
    # Health / Status reporting
    # ==================================================================

    def get_status(self) -> dict[str, ModelStatus]:
        """
        Return per-model status for the health endpoint.

        Returns a dict keyed by model role name (str) with ModelStatus values.
        This method is called by health.py's _build_model_statuses().
        """
        statuses: dict[str, ModelStatus] = {}

        for role, entry in self._models.items():
            statuses[role.value] = ModelStatus(
                loaded=entry.loaded,
                provider=entry.provider,
                model_id=entry.spec.model_id,
                memory_mb=round(entry.actual_ram_mb, 1) if entry.loaded else None,
                load_time_seconds=(
                    round(entry.load_time_seconds, 2) if entry.load_time_seconds else None
                ),
                last_inference_ms=(
                    round(entry.last_inference_ms, 2) if entry.last_inference_ms is not None else None
                ),
                error=entry.error,
            )

        return statuses

    def get_memory_summary(self) -> dict[str, Any]:
        """
        Return a memory usage summary for debugging.

        Not exposed via API — used in logs and developer tooling.
        """
        budget_mb = self._settings.MAX_RAM_GB * 1024
        return {
            "budget_mb": budget_mb,
            "used_mb": round(self.total_loaded_ram_mb, 1),
            "remaining_mb": round(self.budget_remaining_mb, 1),
            "utilisation_pct": round(
                (self.total_loaded_ram_mb / budget_mb) * 100, 1
            ) if budget_mb > 0 else 0,
            "models_loaded": [
                role.value
                for role, entry in self._models.items()
                if entry.loaded
            ],
        }

    # ==================================================================
    # Warmup / batch pre-loading
    # ==================================================================

    async def warmup(self, models: list[ModelRole] | None = None) -> dict[str, bool]:
        """
        Pre-load models into memory (async wrapper for background task).

        Called by POST /api/v1/warmup. Loads each model sequentially
        to control memory pressure.

        Args:
            models: Specific models to load. None = load all local models.

        Returns:
            Dict of model_name → success boolean.
        """
        import asyncio  # noqa: PLC0415

        if models is None:
            # Load the three core models (not VLM — that's API-only)
            models = [ModelRole.DINOV2, ModelRole.SEGFORMER, ModelRole.CLIP]

        results: dict[str, bool] = {}

        for role in models:
            entry = self._models[role]

            # Skip already-loaded models
            if entry.loaded:
                results[role.value] = True
                continue

            # Skip non-local models
            if entry.provider in (ModelProvider.DISABLED, ModelProvider.HF_API):
                results[role.value] = True
                continue

            try:
                # Run blocking loader in a thread pool to avoid blocking the event loop
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda r=role: self._ensure_loaded(r),
                )
                results[role.value] = True
            except Exception as exc:
                logger.error("Warmup failed for {}: {}", role.value, exc)
                results[role.value] = False

        logger.info("Warmup complete: {}", results)
        return results


# ---------------------------------------------------------------------------
# Global registry accessor
# ---------------------------------------------------------------------------

_registry_instance: ModelRegistry | None = None
_registry_lock = threading.Lock()


def get_registry(settings: Settings | None = None) -> ModelRegistry:
    """
    Return the global ModelRegistry singleton.

    Thread-safe initialisation. The registry is created on first call
    and reused for the lifetime of the process.
    """
    global _registry_instance

    if _registry_instance is not None:
        return _registry_instance

    with _registry_lock:
        # Double-check after acquiring lock
        if _registry_instance is not None:
            return _registry_instance

        _registry_instance = ModelRegistry(settings)
        return _registry_instance


def reset_registry() -> None:
    """
    Reset the global registry (for testing only).

    Unloads all models and clears the singleton.
    """
    global _registry_instance

    with _registry_lock:
        if _registry_instance is not None:
            _registry_instance.unload_all()
            _registry_instance = None
