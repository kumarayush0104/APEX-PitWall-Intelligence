#!/usr/bin/env python3
"""
APEX Model Pre-Download Script.

Downloads all required HuggingFace models to the local cache BEFORE
running the application or demo. This eliminates cold-start download
latency during live demonstrations.

Run this script once on a fast/reliable internet connection.
After download, the application works fully offline (except VLM via API).

Usage:
    cd backend
    python scripts/download_models.py

    # Download only specific models:
    python scripts/download_models.py --models dinov2 segformer
    python scripts/download_models.py --skip-clip          # Skip 890 MB CLIP

Estimated download sizes:
    DINOv2-base:         ~340 MB
    SegFormer-B2:        ~110 MB
    CLIP ViT-L/14:       ~890 MB
    Total:               ~1.34 GB

The VLM (Qwen2-VL-7B) is NOT downloaded locally — it is served via
the HuggingFace Inference API (~15 GB is not feasible for 8 GB RAM machines).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add the backend root to sys.path so we can import app modules
_BACKEND_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_BACKEND_ROOT))

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
DIM    = "\033[2m"


def _log(icon: str, message: str) -> None:
    print(f"  {icon}  {message}")


def _hr(label: str) -> None:
    print(f"\n{BOLD}{CYAN}{'─' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  {label}{RESET}")
    print(f"{CYAN}{'─' * 60}{RESET}")


def _load_settings():
    """Load settings, applying .env file and environment setup."""
    try:
        from app.config import get_settings  # noqa: PLC0415
        return get_settings()
    except Exception as e:
        print(f"\n{RED}Could not load settings: {e}{RESET}")
        print("Make sure you are running from the backend/ directory.")
        sys.exit(1)


def _download_with_progress(model_id: str, model_type: str, size_estimate: str) -> bool:
    """
    Download a model to the local HuggingFace cache.

    Returns True on success, False on failure.
    """
    _hr(f"Downloading {model_type}")
    _log(CYAN, f"Model:  {model_id}")
    _log(CYAN, f"Size:   {size_estimate} (approximate)")

    t0 = time.time()
    try:
        if model_type == "dinov2":
            from transformers import AutoImageProcessor, AutoModel  # noqa: PLC0415
            _log(YELLOW, "Downloading processor...")
            AutoImageProcessor.from_pretrained(model_id)
            _log(YELLOW, "Downloading model weights...")
            AutoModel.from_pretrained(model_id)

        elif model_type == "segformer":
            from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor  # noqa: PLC0415
            _log(YELLOW, "Downloading processor...")
            SegformerImageProcessor.from_pretrained(model_id)
            _log(YELLOW, "Downloading model weights...")
            SegformerForSemanticSegmentation.from_pretrained(model_id)

        elif model_type == "clip":
            from transformers import CLIPModel, CLIPProcessor  # noqa: PLC0415
            _log(YELLOW, "Downloading processor...")
            CLIPProcessor.from_pretrained(model_id)
            _log(YELLOW, "Downloading model weights...")
            CLIPModel.from_pretrained(model_id)

        elapsed = time.time() - t0
        _log(f"{GREEN}✓", f"Downloaded in {elapsed:.1f}s")
        return True

    except Exception as e:
        elapsed = time.time() - t0
        _log(f"{RED}✗", f"FAILED after {elapsed:.1f}s: {e}")
        print(f"\n  {YELLOW}Troubleshooting:{RESET}")
        print(f"  - Check your internet connection")
        print(f"  - If 401 error: set HF_TOKEN in backend/.env")
        print(f"  - If disk full: check available space")
        print(f"  - Retry: the download will resume from where it stopped")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pre-download APEX AI models to local cache.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["dinov2", "segformer", "clip"],
        default=["dinov2", "segformer", "clip"],
        help="Which models to download (default: all).",
    )
    parser.add_argument(
        "--skip-clip",
        action="store_true",
        help="Skip CLIP download (saves 890 MB — reduces zero-shot cross-validation quality).",
    )
    args = parser.parse_args()

    print(f"\n{BOLD}{'═' * 60}{RESET}")
    print(f"{BOLD}  APEX — Model Pre-Download{RESET}")
    print(f"{BOLD}{'═' * 60}{RESET}")

    settings = _load_settings()
    _log(CYAN, f"Cache directory: {settings.HF_CACHE_DIR}")
    _log(CYAN, f"HF Token:        {'set ✓' if settings.HF_TOKEN else 'NOT SET (public models only)'}")

    models_to_download = args.models
    if args.skip_clip and "clip" in models_to_download:
        models_to_download = [m for m in models_to_download if m != "clip"]
        _log(YELLOW, "Skipping CLIP download (--skip-clip flag)")

    model_config = {
        "dinov2": (settings.DINOV2_MODEL_ID,   "~340 MB"),
        "segformer": (settings.SEGFORMER_MODEL_ID, "~110 MB"),
        "clip":    (settings.CLIP_MODEL_ID,    "~890 MB"),
    }

    # Estimate total download
    size_map = {"dinov2": 340, "segformer": 110, "clip": 890}
    total_mb = sum(size_map[m] for m in models_to_download)
    _log(CYAN, f"Total download:  ~{total_mb} MB ({len(models_to_download)} models)")

    print(f"\n  {YELLOW}Note: The VLM (Qwen2-VL-7B, ~15 GB) is NOT downloaded locally.")
    print(f"  It will be served via the HuggingFace Inference API (requires HF_TOKEN).{RESET}")

    print(f"\n  Starting downloads...\n")
    total_t0 = time.time()
    results = {}

    for model_key in models_to_download:
        model_id, size_hint = model_config[model_key]
        success = _download_with_progress(model_id, model_key, size_hint)
        results[model_key] = success

    # Summary
    total_elapsed = time.time() - total_t0
    print(f"\n{BOLD}{'═' * 60}{RESET}")
    print(f"{BOLD}  Download Summary{RESET}")
    print(f"{'─' * 60}")
    for model_key, success in results.items():
        icon = f"{GREEN}✓" if success else f"{RED}✗"
        status = "Success" if success else "FAILED"
        print(f"  {icon}  {model_key:<15} {status}{RESET}")
    print(f"{'─' * 60}")
    print(f"  Total time: {total_elapsed:.1f}s")

    failed = [k for k, v in results.items() if not v]
    if failed:
        print(f"\n{RED}{BOLD}  {len(failed)} model(s) failed to download.{RESET}")
        print(f"  {YELLOW}Failed models will be downloaded on first use (slower).{RESET}")
        return 1
    else:
        print(f"\n{GREEN}{BOLD}  All models downloaded successfully!{RESET}")
        print(f"  {DIM}Models are cached at: {settings.HF_CACHE_DIR}{RESET}")
        print(f"\n  {CYAN}Next: python scripts/verify_setup.py{RESET}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
