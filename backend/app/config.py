"""
Application configuration — Pydantic Settings v2.

Design principles:
- CPU-first defaults: every default is safe for an 8 GB RAM machine.
- No torch imported at config load time (optional dependency, lazy check).
- Hardware-aware computed fields for device, dtype, and memory budget.
- HF Inference API is the default for the VLM — never self-host 7B on 8 GB RAM.
- Every setting can be overridden via environment variable or .env file.
"""

from __future__ import annotations

import os
import platform
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, computed_field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Path anchors — resolved relative to *this* file, not the working directory
# ---------------------------------------------------------------------------
_APP_ROOT = Path(__file__).resolve().parent          # backend/app/
_BACKEND_ROOT = _APP_ROOT.parent                      # backend/
_PROJECT_ROOT = _BACKEND_ROOT.parent                  # project root (F1/)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class DeviceChoice(str, Enum):
    AUTO = "auto"
    CUDA = "cuda"
    CPU = "cpu"


class VLMProvider(str, Enum):
    """Where to run the Vision-Language Model."""
    LOCAL = "local"   # Self-hosted — requires significant VRAM (not viable on 8 GB)
    API = "api"       # HuggingFace Inference API — recommended default
    NONE = "none"     # Disabled — use structured fallback only


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """
    Central configuration for APEX / PitWall Intelligence.

    Environment variable names match field names (case-insensitive).
    All values have safe defaults for CPU-only / 8 GB RAM development.
    """

    model_config = SettingsConfigDict(
        # Look for .env next to main.py (i.e., in backend/)
        env_file=str(_BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",          # silently ignore unknown env vars
        validate_default=True,
    )

    # ------------------------------------------------------------------
    # Application identity
    # ------------------------------------------------------------------
    APP_NAME: str = "PitWall Intelligence"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = (
        "AI-powered race track condition analysis and tyre strategy engine. "
        "Powered by APEX — Adaptive Perception & Evolution eXplainer."
    )
    DEBUG: bool = Field(default=False, description="Enable debug mode (verbose logging, stack traces).")
    HOST: str = Field(default="0.0.0.0", description="Uvicorn bind host.")
    PORT: int = Field(default=8000, description="Uvicorn bind port.")

    # ------------------------------------------------------------------
    # CORS — permissive by default for local development
    # ------------------------------------------------------------------
    CORS_ORIGINS: list[str] = Field(
        default=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
            "http://localhost:5500",   # VS Code Live Server default
            "http://127.0.0.1:5500",
            "null",                    # file:// origin (opening index.html directly)
        ],
        description="Allowed CORS origins. In production restrict this.",
    )
    CORS_ALLOW_ALL: bool = Field(
        default=True,
        description="If true, allow all origins (*). Set false in production.",
    )

    # ------------------------------------------------------------------
    # Compute / Hardware
    # ------------------------------------------------------------------
    DEVICE: DeviceChoice = Field(
        default=DeviceChoice.AUTO,
        description=(
            "Compute device. 'auto' checks for CUDA and falls back to CPU. "
            "On an 8 GB RAM laptop without a capable GPU, this will resolve to 'cpu'."
        ),
    )
    TORCH_DTYPE: Literal["float32", "float16", "bfloat16"] = Field(
        default="float32",
        description=(
            "Model weight dtype. float16 halves VRAM on GPU. "
            "ALWAYS overridden to float32 on CPU (float16 is not efficient on CPU)."
        ),
    )
    MAX_RAM_GB: float = Field(
        default=5.0,
        description=(
            "RAM budget (GB) for model weights. "
            "Default 5.0 GB leaves ~3 GB headroom on an 8 GB machine "
            "for OS, browser, and Python overhead."
        ),
    )

    # ------------------------------------------------------------------
    # HuggingFace Hub
    # ------------------------------------------------------------------
    HF_TOKEN: str | None = Field(
        default=None,
        description=(
            "HuggingFace API token. Required for VLM via HF Inference API. "
            "Get one at https://huggingface.co/settings/tokens"
        ),
    )
    HF_CACHE_DIR: Path = Field(
        default=_BACKEND_ROOT / ".hf_cache",
        description="Local directory for HuggingFace model and dataset cache.",
    )
    USE_HF_INFERENCE_API: bool = Field(
        default=False,
        description=(
            "Force all models to use HF Inference API instead of local inference. "
            "Useful for very constrained hardware. Requires HF_TOKEN."
        ),
    )

    # ------------------------------------------------------------------
    # AI Model identifiers (HuggingFace Hub)
    # ------------------------------------------------------------------
    DINOV2_MODEL_ID: str = Field(
        default="facebook/dinov2-base",
        description=(
            "DINOv2 ViT-B/14. ~86M parameters, ~344 MB float32. "
            "Chosen over larger variants to fit within 8 GB RAM budget. "
            "Self-supervised semantic features — no fine-tuning needed."
        ),
    )
    SEGFORMER_MODEL_ID: str = Field(
        default="nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
        description=(
            "SegFormer-B2 fine-tuned on Cityscapes. ~27M params, ~108 MB. "
            "Best speed/accuracy tradeoff for road-scene segmentation. "
            "B5 would be more accurate but too slow on CPU."
        ),
    )
    CLIP_MODEL_ID: str = Field(
        default="openai/clip-vit-large-patch14",
        description=(
            "CLIP ViT-L/14. ~307M params, ~1.2 GB float32. "
            "The largest model in our local stack. "
            "Provides zero-shot cross-validation of segmentation results."
        ),
    )
    VLM_MODEL_ID: str = Field(
        default="Qwen/Qwen2-VL-7B-Instruct",
        description=(
            "Qwen2-VL 7B for natural language explanations. "
            "~15 GB float32 — NEVER loaded locally on 8 GB machines. "
            "Always served via HF Inference API or falls back to structured template."
        ),
    )

    # ------------------------------------------------------------------
    # VLM Configuration
    # ------------------------------------------------------------------
    VLM_ENABLED: bool = Field(
        default=True,
        description=(
            "Enable VLM natural language explanations. "
            "If true but VLM is unavailable, falls back to structured template. "
            "The system works perfectly either way."
        ),
    )
    VLM_PROVIDER: VLMProvider = Field(
        default=VLMProvider.API,
        description=(
            "'api' = HF Inference API (default, requires HF_TOKEN). "
            "'local' = self-hosted (requires GPU with ≥16 GB VRAM — not this machine). "
            "'none' = always use structured fallback."
        ),
    )
    VLM_API_TIMEOUT_SECONDS: float = Field(
        default=25.0,
        description=(
            "Timeout for VLM API calls in seconds. "
            "If exceeded, structured fallback is used immediately."
        ),
    )
    VLM_CACHE_ENABLED: bool = Field(
        default=True,
        description="Cache VLM explanations by state fingerprint to minimize API calls.",
    )
    VLM_CACHE_SIZE: int = Field(
        default=64,
        description="Maximum number of cached VLM explanations (LRU eviction).",
    )

    # ------------------------------------------------------------------
    # Weather Service (optional — system works fully offline without it)
    # ------------------------------------------------------------------
    WEATHER_ENABLED: bool = Field(
        default=False,
        description=(
            "Enable OpenWeatherMap integration for weather-visual fusion. "
            "System functions completely without this. Enable only if you have an API key."
        ),
    )
    WEATHER_API_KEY: str | None = Field(
        default=None,
        description="OpenWeatherMap API key. Required if WEATHER_ENABLED=true.",
    )
    WEATHER_CACHE_TTL_SECONDS: int = Field(
        default=300,
        description="Weather data cache TTL (5 minutes — weather changes slowly).",
    )
    WEATHER_API_BASE_URL: str = "https://api.openweathermap.org/data/3.0"

    # ------------------------------------------------------------------
    # Session Store (in-memory, sufficient for hackathon)
    # ------------------------------------------------------------------
    SESSION_TTL_SECONDS: int = Field(
        default=7200,
        description="How long to retain session data in memory (2 hours).",
    )
    SESSION_MAX_FRAMES: int = Field(
        default=200,
        description="Maximum frames per session history before oldest are discarded.",
    )
    SESSION_MAX_CONCURRENT: int = Field(
        default=10,
        description="Maximum number of concurrent active sessions.",
    )

    # ------------------------------------------------------------------
    # File Uploads
    # ------------------------------------------------------------------
    MAX_UPLOAD_SIZE_MB: int = Field(
        default=500,
        description="Maximum upload file size in megabytes.",
    )
    ALLOWED_IMAGE_TYPES: list[str] = Field(
        default=["image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"],
    )
    ALLOWED_VIDEO_TYPES: list[str] = Field(
        default=[
            "video/mp4",
            "video/quicktime",
            "video/avi",
            "video/x-msvideo",
            "video/x-matroska",
        ],
    )

    # ------------------------------------------------------------------
    # Inference Pipeline
    # ------------------------------------------------------------------
    VIDEO_ANALYSIS_FPS: float = Field(
        default=1.0,
        description=(
            "Frames per second to extract for video analysis. "
            "1.0 fps is sufficient for track condition changes (slow phenomenon). "
            "Lower values = faster processing on modest hardware."
        ),
    )
    VIDEO_MAX_FRAMES: int = Field(
        default=120,
        description=(
            "Maximum frames to process per video. "
            "120 frames at 1 fps = 2-minute clip. "
            "Reduced from 300 to accommodate 8 GB RAM constraint."
        ),
    )
    TEMPORAL_WINDOW_SIZE: int = Field(
        default=8,
        description="Sliding window depth for temporal reasoning (number of frames).",
    )
    MIN_FRAMES_FOR_TREND: int = Field(
        default=3,
        description="Minimum frames in window before computing grip trend / forecast.",
    )

    # ------------------------------------------------------------------
    # Demo Mode
    # ------------------------------------------------------------------
    DEMO_MODE_ENABLED: bool = Field(
        default=True,
        description="Enable pre-loaded race scenarios for judge demonstration.",
    )
    DEMO_CLIPS_DIR: Path = Field(
        default=_BACKEND_ROOT / "demo_clips",
        description="Directory containing pre-processed demo scenario frame sequences.",
    )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    LOG_LEVEL: Literal["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
    )
    LOG_TO_FILE: bool = Field(
        default=True,
        description="Write logs to rotating file in addition to stderr.",
    )
    LOG_DIR: Path = Field(
        default=_BACKEND_ROOT / "logs",
    )
    LOG_ROTATION: str = "10 MB"
    LOG_RETENTION: str = "7 days"

    # ==================================================================
    # Computed fields (derived from base settings — no env vars)
    # ==================================================================

    @computed_field  # type: ignore[misc]
    @property
    def resolved_device(self) -> str:
        """
        Actual compute device in use ('cuda' or 'cpu').

        Checks torch.cuda availability when DEVICE='auto'.
        Torch is imported lazily here — if not installed, defaults to 'cpu'.
        Does NOT raise if torch is missing.
        """
        if self.DEVICE == DeviceChoice.CUDA:
            return "cuda"
        if self.DEVICE == DeviceChoice.CPU:
            return "cpu"
        # AUTO: try to detect CUDA
        try:
            import torch  # noqa: PLC0415
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    @computed_field  # type: ignore[misc]
    @property
    def is_gpu_available(self) -> bool:
        """True if CUDA is resolved and available."""
        return self.resolved_device == "cuda"

    @computed_field  # type: ignore[misc]
    @property
    def effective_torch_dtype(self) -> str:
        """
        Effective dtype for model weights.

        float16 and bfloat16 are only efficient on GPU.
        On CPU they are slower than float32 and have no benefit.
        """
        if not self.is_gpu_available and self.TORCH_DTYPE != "float32":
            return "float32"
        return self.TORCH_DTYPE

    @computed_field  # type: ignore[misc]
    @property
    def max_upload_bytes(self) -> int:
        """Max upload size in bytes (for FastAPI size validation)."""
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    @computed_field  # type: ignore[misc]
    @property
    def vlm_is_functional(self) -> bool:
        """
        True if VLM explanations will actually work at runtime.

        Requires: VLM_ENABLED=true AND either:
          - VLM_PROVIDER='api' + HF_TOKEN is set
          - VLM_PROVIDER='local' (not recommended on this hardware)
        """
        if not self.VLM_ENABLED:
            return False
        if self.VLM_PROVIDER == VLMProvider.NONE:
            return False
        if self.VLM_PROVIDER == VLMProvider.API and not self.HF_TOKEN:
            return False  # Will fall back to structured template
        return True

    # ==================================================================
    # Validators
    # ==================================================================

    @field_validator("HF_CACHE_DIR", "LOG_DIR", "DEMO_CLIPS_DIR", mode="before")
    @classmethod
    def coerce_to_path(cls, v: Any) -> Path:
        return Path(v)

    @model_validator(mode="after")
    def _warn_vlm_api_without_token(self) -> "Settings":
        """Emit a warning when VLM API is requested but no token is set."""
        if (
            self.VLM_ENABLED
            and self.VLM_PROVIDER == VLMProvider.API
            and not self.HF_TOKEN
        ):
            import warnings  # noqa: PLC0415
            warnings.warn(
                "VLM_PROVIDER='api' requires HF_TOKEN. "
                "VLM explanations will automatically use the structured fallback. "
                "Set HF_TOKEN in your .env file to enable LLM-generated explanations.",
                UserWarning,
                stacklevel=2,
            )
        return self

    @model_validator(mode="after")
    def _disable_weather_without_key(self) -> "Settings":
        """Automatically disable weather if API key is missing."""
        if self.WEATHER_ENABLED and not self.WEATHER_API_KEY:
            import warnings  # noqa: PLC0415
            warnings.warn(
                "WEATHER_ENABLED=true but WEATHER_API_KEY is not set. "
                "Weather integration will be disabled. "
                "System operates perfectly without weather data.",
                UserWarning,
                stacklevel=2,
            )
            object.__setattr__(self, "WEATHER_ENABLED", False)
        return self

    @model_validator(mode="after")
    def _warn_vlm_local_on_low_ram(self) -> "Settings":
        """Warn if user tries to self-host the 7B VLM on a low-RAM machine."""
        if self.VLM_PROVIDER == VLMProvider.LOCAL and self.MAX_RAM_GB <= 8.0:
            import warnings  # noqa: PLC0415
            warnings.warn(
                f"VLM_PROVIDER='local' with MAX_RAM_GB={self.MAX_RAM_GB} is not viable. "
                "Qwen2-VL-7B requires ~15 GB RAM. "
                "Switching VLM to 'api' provider. "
                "Update VLM_PROVIDER in your .env to suppress this warning.",
                UserWarning,
                stacklevel=2,
            )
            object.__setattr__(self, "VLM_PROVIDER", VLMProvider.API)
        return self

    # ==================================================================
    # Setup helpers
    # ==================================================================

    def setup_environment(self) -> None:
        """
        Apply settings that must exist as OS environment variables.
        Call exactly once at application startup (done in lifespan).
        """
        # HuggingFace cache directories
        hf_hub_cache = str(self.HF_CACHE_DIR / "hub")
        os.environ["HF_HOME"] = str(self.HF_CACHE_DIR)
        os.environ["HF_HUB_CACHE"] = hf_hub_cache
        os.environ["TRANSFORMERS_CACHE"] = hf_hub_cache
        os.environ["HF_DATASETS_CACHE"] = str(self.HF_CACHE_DIR / "datasets")

        if self.HF_TOKEN:
            os.environ["HF_TOKEN"] = self.HF_TOKEN
            os.environ["HUGGING_FACE_HUB_TOKEN"] = self.HF_TOKEN

        # Disable tokenizer parallelism — prevents deadlocks with multiple workers
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        # Quieter transformers output in non-debug mode
        if not self.DEBUG:
            os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

        # Create required directories
        for directory in (self.HF_CACHE_DIR, self.LOG_DIR, self.DEMO_CLIPS_DIR):
            directory.mkdir(parents=True, exist_ok=True)

    def get_system_info(self) -> dict[str, Any]:
        """
        Collect hardware and runtime information.

        Safe to call without torch or any optional dependency.
        Used by the health endpoint and startup logging.
        """
        import sys  # noqa: PLC0415

        info: dict[str, Any] = {
            "python_version": sys.version.split()[0],
            "platform": platform.system(),
            "platform_release": platform.release(),
            "cpu_count": os.cpu_count() or 1,
            "device_config": self.DEVICE.value,
            "resolved_device": self.resolved_device,
            "effective_dtype": self.effective_torch_dtype,
            "cuda_available": False,
            "cuda_device_name": None,
            "cuda_device_memory_gb": None,
            "torch_version": None,
            "ram_total_gb": None,
            "ram_available_gb": None,
        }

        # Torch / CUDA (optional)
        try:
            import torch  # noqa: PLC0415
            info["torch_version"] = torch.__version__
            info["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                info["cuda_device_name"] = props.name
                info["cuda_device_memory_gb"] = round(props.total_memory / 1e9, 2)
        except ImportError:
            pass

        # RAM (optional — psutil)
        try:
            import psutil  # noqa: PLC0415
            mem = psutil.virtual_memory()
            info["ram_total_gb"] = round(mem.total / 1e9, 2)
            info["ram_available_gb"] = round(mem.available / 1e9, 2)
        except ImportError:
            pass

        return info

    def summary(self) -> dict[str, Any]:
        """
        Return a non-sensitive configuration summary for logging and health checks.
        Never includes tokens or API keys.
        """
        return {
            "app_version": self.APP_VERSION,
            "debug": self.DEBUG,
            "device": self.resolved_device,
            "dtype": self.effective_torch_dtype,
            "max_ram_gb": self.MAX_RAM_GB,
            "vlm_enabled": self.VLM_ENABLED,
            "vlm_provider": self.VLM_PROVIDER.value,
            "vlm_functional": self.vlm_is_functional,
            "weather_enabled": self.WEATHER_ENABLED,
            "demo_mode": self.DEMO_MODE_ENABLED,
            "temporal_window": self.TEMPORAL_WINDOW_SIZE,
            "max_upload_mb": self.MAX_UPLOAD_SIZE_MB,
            "hf_token_set": bool(self.HF_TOKEN),
        }


# ---------------------------------------------------------------------------
# Global settings accessor
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the cached Settings singleton.

    Use this function everywhere — never instantiate Settings directly in routes.
    The lru_cache ensures only one Settings instance exists for the process lifetime.
    """
    settings = Settings()
    settings.setup_environment()
    return settings
