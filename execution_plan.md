# APEX Engineering Execution Plan
### PitWall Intelligence — Module-Level Implementation Roadmap

> **Classification:** Principal Engineering Execution Document  
> **Status:** Approved Architecture → Implementation Roadmap  
> **Frozen Spec:** `implementation_plan.md` v1.0  
> **Author:** Technical Lead / Principal Architect

---

## Part 0 — Pre-Implementation Decisions (Resolve Before Day 1)

These decisions block module construction and **must be locked before writing a single line of code**.

| # | Decision | Options | Recommendation | Blocker Risk |
|---|---|---|---|---|
| D1 | **VLM Hosting** | A) Self-host Qwen2-VL-7B (GPU, 16 GB VRAM) B) HF Inference API (token, ~3s latency) C) Qwen2-VL-2B self-hosted (CPU viable, lower quality) | **B for demo, A if GPU available** | 🔴 HIGH — affects explainer module design |
| D2 | **Demo Mode** | A) Pre-loaded clips B) Upload-only | **A — mandatory for judge experience** | 🔴 HIGH — affects frontend state management |
| D3 | **Weather API** | A) OpenWeatherMap free tier B) Skip, visual-only | **A — adds ~30 min work, huge demo value** | 🟡 MEDIUM |
| D4 | **Database** | A) SQLite (simple) B) Postgres (production) | **A — SQLite is sufficient, zero setup** | 🟢 LOW |
| D5 | **Frontend Framework** | A) Vanilla HTML/CSS/JS B) React/Vite | **A — zero build toolchain friction, faster iteration** | 🟡 MEDIUM |
| D6 | **Session Persistence** | A) In-memory dict B) Redis C) SQLite | **A for demo, add B flag for future** | 🟢 LOW |
| D7 | **Model Precision** | A) float32 B) float16 (GPU) C) int8 quant (CPU) | **B if GPU, C if CPU-only** | 🟡 MEDIUM — affects inference speed |
| D8 | **SegFormer variant** | A) B2-cityscapes B) B5-cityscapes (more accurate, heavier) | **A — better speed/accuracy tradeoff for demo** | 🟢 LOW |

---

## Part 1 — Complete Module Specifications

---

### MODULE 00 — Project Scaffold & Configuration

| Attribute | Detail |
|---|---|
| **Purpose** | Establish folder structure, environment, config, logging, and shared utilities |
| **Responsibilities** | Create all directories; write `.env`; configure logging; set up Pydantic settings; write `requirements.txt`; write `docker-compose.yml` skeleton; configure CORS |
| **Inputs** | D1–D8 decisions; Python 3.11 environment |
| **Outputs** | Runnable FastAPI skeleton; importable config; working health endpoint |
| **Dependencies** | None — this is the root module |
| **Libraries** | `fastapi`, `uvicorn`, `pydantic-settings`, `python-dotenv`, `loguru` |
| **APIs Used** | None |
| **Difficulty** | 🟢 Easy |
| **Time Estimate** | 30 minutes |
| **Testing Strategy** | `GET /health` returns `200 OK` with model status `{loaded: false}` |
| **Edge Cases** | Missing `.env` file → Pydantic raises on startup; handle with defaults |
| **Possible Failures** | Port conflicts; Windows path separators in file configs |
| **Optimizations** | Use `lifespan` context manager (FastAPI 0.95+) for model loading — cleaner than `@app.on_event` |

**Folder Structure to Create:**
```
pitwall/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── routes/
│   │   │   │   ├── analysis.py
│   │   │   │   ├── session.py
│   │   │   │   └── websocket.py
│   │   │   └── dependencies.py
│   │   ├── core/
│   │   │   ├── pipeline.py
│   │   │   ├── perception.py
│   │   │   ├── temporal.py
│   │   │   ├── physics.py
│   │   │   ├── recommender.py
│   │   │   └── explainer.py
│   │   ├── models/
│   │   │   ├── registry.py
│   │   │   └── schemas.py
│   │   ├── services/
│   │   │   ├── session_store.py
│   │   │   ├── weather.py
│   │   │   └── video_processor.py
│   │   └── utils/
│   │       ├── image_utils.py
│   │       ├── visualization.py
│   │       └── calibration.py
│   ├── tests/
│   ├── demo_clips/          # Pre-loaded F1 rain footage frames
│   ├── .env.example
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── index.html
│   ├── src/
│   │   ├── main.js
│   │   ├── styles/
│   │   ├── components/
│   │   ├── services/
│   │   └── utils/
│   └── assets/
├── docker-compose.yml
└── README.md
```

---

### MODULE 01 — Model Registry & Loader

| Attribute | Detail |
|---|---|
| **Purpose** | Load, cache, and expose all HuggingFace models at startup; manage model lifecycle; prevent redundant reloads |
| **Responsibilities** | Download & cache DINOv2, SegFormer, CLIP at startup; expose model instances via singleton registry; handle GPU/CPU device selection; report load status to health endpoint; manage memory footprint |
| **Inputs** | `.env` config: `HF_CACHE_DIR`, `DEVICE` (`cuda`/`cpu`), `SEGFORMER_MODEL_ID`, `DINOV2_MODEL_ID`, `CLIP_MODEL_ID` |
| **Outputs** | `ModelRegistry` singleton with `.dinov2`, `.segformer`, `.clip`, `.processor_*` attributes; loaded and warm |
| **Dependencies** | MODULE 00 (config, logging) |
| **Libraries** | `transformers`, `torch`, `torchvision`, `Pillow`, `accelerate` |
| **APIs Used** | HuggingFace Hub (`from_pretrained`) |
| **Difficulty** | 🟡 Medium |
| **Time Estimate** | 1–1.5 hours |
| **Testing Strategy** | Unit test: `registry.dinov2` is not None; `registry.device` matches environment; run a dummy inference to confirm no shape errors |
| **Edge Cases** | CUDA OOM on load → gracefully fall back to CPU with quantization; HF Hub unavailable → check local cache; first load = slow download (mitigate with pre-download script) |
| **Possible Failures** | `transformers` version incompatibility with Qwen2-VL; CUDA driver version mismatch; model cache corruption |
| **Optimizations** | Warm up each model with a 1×1 dummy input after loading (eliminates first-request JIT latency); use `torch.compile` if PyTorch 2.x is available; load SegFormer with `half()` on GPU |

**Critical Note:** Write a `scripts/download_models.py` that pre-downloads all models before demo. This eliminates cold-start download during the live demo — a catastrophic failure mode.

---

### MODULE 02 — Image Preprocessing Utilities

| Attribute | Detail |
|---|---|
| **Purpose** | Normalize all incoming media (images, video frames) into a consistent format consumed by all downstream models |
| **Responsibilities** | Accept: file upload bytes, base64 string, URL, or numpy array; output: PIL Image + model-specific preprocessed tensors for DINOv2, SegFormer, CLIP; handle aspect ratio, resizing, EXIF rotation; validate image integrity |
| **Inputs** | Raw bytes / base64 / file path; target model name |
| **Outputs** | `PreprocessedFrame` dataclass: `{pil_image, dinov2_tensor, segformer_inputs, clip_inputs, original_size, timestamp}` |
| **Dependencies** | MODULE 00, MODULE 01 |
| **Libraries** | `Pillow`, `numpy`, `torch`, `cv2` (OpenCV), `transformers` (processors) |
| **APIs Used** | None |
| **Difficulty** | 🟢 Easy |
| **Time Estimate** | 45 minutes |
| **Testing Strategy** | Feed 5 test images (wet, dry, dark, bright, corrupted); assert output tensor shapes match model expectations |
| **Edge Cases** | HEIC/HEIF images from iOS; very large images (>8MP) causing OOM; grayscale images; corrupt files; zero-byte uploads; portrait-mode videos |
| **Possible Failures** | OpenCV not installed correctly on Windows; PIL EXIF stripping changes orientation |
| **Optimizations** | Cache processed tensors keyed by image hash to avoid reprocessing same frame; use async file I/O for upload handling |

---

### MODULE 03 — Perception Layer (Core Vision)

| Attribute | Detail |
|---|---|
| **Purpose** | The primary visual intelligence stage. Extracts semantic features via DINOv2, generates pixel-level surface segmentation via SegFormer, and cross-validates via CLIP zero-shot scoring |
| **Responsibilities** | Run DINOv2 forward pass → extract `[CLS]` token + patch embeddings; run SegFormer → generate semantic segmentation mask with 6 classes; run CLIP → compute cosine similarity to 8 condition text prompts; aggregate into `PerceptionResult`; extract DINOv2 attention maps for XAI |
| **Inputs** | `PreprocessedFrame` from MODULE 02 |
| **Outputs** | `PerceptionResult` dataclass: `{segmentation_mask (H×W), class_probs (6,), clip_scores (8,), patch_embeddings (196, 768), attention_maps (12, 196, 196), confidence_agreement, raw_logits}` |
| **Dependencies** | MODULE 01, MODULE 02 |
| **Libraries** | `torch`, `transformers`, `numpy`, `scipy` |
| **APIs Used** | None (local inference) |
| **Difficulty** | 🔴 Hard |
| **Time Estimate** | 3–4 hours |
| **Testing Strategy** | Visual regression test: run on 10 labeled test images; assert top predicted class matches label; assert segmentation mask has correct shape; CLIP scores for "wet" > 0.3 on wet images; attention maps are non-uniform (not flat) |
| **Edge Cases** | Night-time images (minimal reflectance signal); in-tunnel sections; camera lens water drops (frame-level artifacts vs. track wetness); sun glare occluding the track; smoke/spray obscuring the surface |
| **Possible Failures** | SegFormer class mapping mismatch (Cityscapes classes ≠ our 6 classes — needs remapping); CLIP context window overflow on long prompts; DINOv2 attention extraction hooks breaking across library versions |
| **Optimizations** | Run DINOv2 and CLIP in parallel (`torch.no_grad()` blocks); cache patch embeddings when same frame is analyzed twice; use SegFormer-B2 instead of B5 to halve inference time with minimal accuracy loss |

**SegFormer Class Remapping Strategy:**
Cityscapes has 19 classes. We remap to our 6 target classes:
```
road + sidewalk        → DRY_SURFACE candidate
sky + vegetation       → BACKGROUND
person + vehicle       → OCCLUDER (mask out)
Custom logic on luminance/saturation → WET / DAMP / PUDDLE / RUBBER_LINE
```
Additional wetness detection is done via luminance + saturation thresholds layered ON TOP of SegFormer's road mask. This is a critical design insight.

**CLIP Prompt Engineering (The 8 Prompts):**
```python
CONDITION_PROMPTS = {
    "wet_severe":     "a Formula 1 race track completely covered in standing water with strong reflections",
    "wet_moderate":   "a wet racing circuit with puddles and a shiny reflective surface",
    "transitional":   "a drying race track with patches of wet and dry asphalt",
    "drying":         "a race track that is mostly dry with some damp areas remaining",
    "dry_green":      "a clean dry race track with fresh asphalt and no rubber deposits",
    "dry_evolved":    "a dry Formula 1 circuit with a dark rubbered-in racing line",
    "marbles":        "racing rubber debris and marbles accumulated at the edge of an F1 track",
    "puddle":         "standing water puddles on a race track surface",
}
```

---

### MODULE 04 — Visualization Engine

| Attribute | Detail |
|---|---|
| **Purpose** | Convert model outputs (segmentation masks, attention maps) into visual overlays consumable by the frontend; generate all heatmap imagery |
| **Responsibilities** | Colorize segmentation mask with class color scheme; blend overlay onto original image (alpha compositing); generate attention heatmap from DINOv2 attention weights; resize all outputs to consistent display resolution; encode all outputs as base64 PNG for JSON transport |
| **Inputs** | `PerceptionResult` + original `PIL Image` |
| **Outputs** | `VisualizationBundle`: `{overlay_b64, attention_heatmap_b64, segmentation_b64, class_legend}` |
| **Dependencies** | MODULE 02, MODULE 03 |
| **Libraries** | `Pillow`, `numpy`, `matplotlib` (colormaps only), `cv2`, `base64` |
| **APIs Used** | None |
| **Difficulty** | 🟡 Medium |
| **Time Estimate** | 1.5 hours |
| **Testing Strategy** | Visual inspection of outputs; assert base64 strings decode to valid PNG; assert overlay image dimensions match original |
| **Edge Cases** | Very dark images where overlay colors are invisible; high-resolution originals causing slow encoding; attention maps all near-uniform (low information) |
| **Possible Failures** | matplotlib backend issues on headless server (use `Agg` backend); color bleed at class boundaries |
| **Optimizations** | Resize images to max 1024px before overlay generation; use OpenCV for faster blending than PIL; cache attention heatmap computation per frame hash |

**Color Scheme (matches frontend):**
```python
CLASS_COLORS = {
    "puddle":       (0,   100, 255, 180),   # Blue
    "wet":          (0,   160, 255, 140),   # Light blue  
    "damp":         (255, 200,   0, 120),   # Amber
    "dry":          (0,   220, 100,  60),   # Green (subtle)
    "rubber_line":  ( 40,  40,  40, 180),   # Dark (nearly black)
    "marble":       (255, 140,   0, 160),   # Orange
}
```

---

### MODULE 05 — Surface Feature Extractor

| Attribute | Detail |
|---|---|
| **Purpose** | Extract quantitative surface metrics from perception outputs that feed the physics layer; bridge between raw model outputs and engineering units |
| **Responsibilities** | Compute `wet_pixel_ratio` from segmentation mask; compute `specular_intensity` from luminance analysis; estimate `rubber_line_coverage` from dark band detection; compute `puddle_count` and `puddle_area_fraction`; compute `reflectance_score` from HSV analysis; return named metrics dict |
| **Inputs** | `PerceptionResult` + original `PIL Image` |
| **Outputs** | `SurfaceMetrics`: `{wet_pixel_ratio, specular_intensity, rubber_line_coverage, puddle_count, puddle_area_fraction, reflectance_score, off_line_wetness, racing_line_wetness}` |
| **Dependencies** | MODULE 03 |
| **Libraries** | `numpy`, `cv2`, `scipy.ndimage` |
| **APIs Used** | None |
| **Difficulty** | 🟡 Medium |
| **Time Estimate** | 2 hours |
| **Testing Strategy** | Feed synthetic test images with known pixel distributions; assert metric values are in expected ranges; compare wet image vs. dry image metrics (wet should have >0.3 wet_pixel_ratio, >0.5 specular_intensity) |
| **Edge Cases** | Tunnel segments (low luminance suppresses reflectance signal); track markings (white lines) misidentified as dry patches; camera overexposure washing out wetness signal |
| **Possible Failures** | Segmentation errors propagate to metrics — mitigate by weighting CLIP score agreement |
| **Optimizations** | All metric computations are numpy vectorized — no loops; pre-compute HSV conversion once and reuse |

**Rubber Line Detection Algorithm:**
```
1. Extract road pixels from SegFormer mask
2. Convert to HSV
3. Find pixels with: V < 60, S < 30 (dark, desaturated = rubber)
4. Erode noise → connected components
5. Largest elongated component = racing line candidate
6. Coverage = racing_line_pixels / total_road_pixels
```

---

### MODULE 06 — Physics & State Machine Layer

| Attribute | Detail |
|---|---|
| **Purpose** | Convert raw visual metrics into physics-grounded engineering values; maintain and transition the condition state machine; estimate grip coefficient μ̂ |
| **Responsibilities** | Compute μ̂ from `SurfaceMetrics` using weighted formula; determine current `ConditionState` enum value; enforce valid state transitions (e.g., cannot jump from WET_SEVERE to DRY_EVOLVED in one frame); compute `drying_rate` (Δμ per frame); output `PhysicsEstimate` |
| **Inputs** | `SurfaceMetrics` (current frame) + `TemporalState` (previous frames from MODULE 07) |
| **Outputs** | `PhysicsEstimate`: `{mu_hat, mu_lower_bound, mu_upper_bound, condition_state, drying_rate, estimated_crossover_frames, risk_level}` |
| **Dependencies** | MODULE 05 (SurfaceMetrics), MODULE 07 (temporal context) |
| **Libraries** | `numpy`, `dataclasses` |
| **APIs Used** | None |
| **Difficulty** | 🟡 Medium |
| **Time Estimate** | 2 hours |
| **Testing Strategy** | Unit tests for each state transition rule; test μ̂ formula with extreme inputs (all wet = 0.25, all dry = 1.0); test state machine rejects invalid transitions |
| **Edge Cases** | Rapid condition oscillation (intermittent rain) — apply hysteresis; single anomalous frame degrading state — require 3 consecutive frames to confirm state change |
| **Possible Failures** | μ̂ formula weights need tuning — define them as config constants for easy adjustment |
| **Optimizations** | Hysteresis prevents state thrashing; Bayesian update with weather prior if weather API is enabled |

**Grip Formula (from design spec):**
```
μ̂ = 0.35 × (1 - wet_pixel_ratio)
   + 0.25 × (1 - specular_intensity)
   + 0.25 × rubber_line_coverage
   + 0.15 × temporal_drying_rate_bonus
→ Scaled to [0.25, 1.05] range
```

**State Transition Rules:**
```
Hysteresis window: 3 consecutive frames required for state change
Forbidden instant jumps: WET_SEVERE → DRY_GREEN (must pass through intermediates)
Emergency override: If puddle_count > 5 AND wet_pixel_ratio > 0.6 → force WET_SEVERE
```

---

### MODULE 07 — Temporal Reasoning Layer

| Attribute | Detail |
|---|---|
| **Purpose** | Maintain a sliding window of track state history; detect trends; smooth noisy predictions; compute grip recovery rate; forecast near-term conditions |
| **Responsibilities** | Maintain `deque` of last N=8 `PhysicsEstimate` frames per session; compute exponentially weighted moving average (EWMA) of μ̂; detect trend direction (improving/worsening/stable); compute `drying_rate` (linear regression slope over N frames); flag anomalies (outlier frames); estimate frames-to-crossover; flag sudden condition reversals |
| **Inputs** | `PhysicsEstimate` (current) + session_id (to look up history) |
| **Outputs** | `TemporalAnalysis`: `{smoothed_mu, trend_direction, drying_rate_per_frame, frames_to_crossover, condition_stability, anomaly_flag, history_vector}` |
| **Dependencies** | MODULE 06 |
| **Libraries** | `numpy`, `collections.deque`, `scipy.stats` (linear regression) |
| **APIs Used** | None |
| **Difficulty** | 🟡 Medium |
| **Time Estimate** | 2 hours |
| **Testing Strategy** | Feed 8-frame synthetic sequences (steady wet, rapid drying, sudden shower); assert trend detection matches expected direction; assert anomaly flag triggers on outlier frame |
| **Edge Cases** | Cold start (fewer than 3 frames in history) — return wide confidence interval; session timeout (old frames in buffer) — apply time-based expiry |
| **Possible Failures** | Linear regression on 3 points is noisy — use minimum 5 frames for regression, wider bounds otherwise |
| **Optimizations** | EWMA is O(1) per frame — lightweight; deque O(1) append/pop; entire temporal layer adds <1ms to inference |

**Frames-to-Crossover Formula:**
```
If drying_rate > 0 (improving):
    mu_needed = compound_threshold - smoothed_mu  (e.g., 0.75 for slick threshold)
    frames_to_crossover = mu_needed / drying_rate
    minutes_to_crossover = frames_to_crossover × frame_interval_seconds / 60
```

---

### MODULE 08 — Confidence Calibration

| Attribute | Detail |
|---|---|
| **Purpose** | Produce calibrated, trustworthy confidence scores; detect and flag high-uncertainty situations; prevent over-confident recommendations on ambiguous inputs |
| **Responsibilities** | Compute agreement score between DINOv2 classification, SegFormer, and CLIP outputs; combine into overall `confidence` score; apply uncertainty multiplier based on temporal consistency; return `ConfidenceBundle` with component-level breakdown |
| **Inputs** | `PerceptionResult` (clip_scores, class agreement) + `TemporalAnalysis` (stability) |
| **Outputs** | `ConfidenceBundle`: `{overall_confidence, source_agreement, temporal_consistency, uncertainty_level, uncertainty_sources_list}` |
| **Dependencies** | MODULE 03, MODULE 07 |
| **Libraries** | `numpy`, `scipy.stats` |
| **APIs Used** | None |
| **Difficulty** | 🟢 Easy |
| **Time Estimate** | 1 hour |
| **Testing Strategy** | Test: unanimous model agreement → >0.85 confidence; test: all models disagree → <0.45 confidence; test: anomaly flag → confidence capped at 0.60 |
| **Edge Cases** | All three models agree on wrong answer (systematic failure) — no mitigation, but low temporal consistency should catch it over frames |
| **Possible Failures** | Confidence score not properly bounded [0,1] — add explicit clip |
| **Optimizations** | Simple weighted average — trivially fast |

**Confidence Formula:**
```
source_agreement = 1 - std([dinov2_label, segformer_label, clip_top1])  # normalized
temporal_consistency = 1 - anomaly_flag × 0.4
condition_entropy = -sum(p × log(p)) for segformer class probs (normalized)

overall_confidence = 0.5 × source_agreement
                   + 0.3 × temporal_consistency
                   + 0.2 × (1 - condition_entropy)
```

---

### MODULE 09 — Recommendation Engine

| Attribute | Detail |
|---|---|
| **Purpose** | The strategic brain. Translates physics estimates and temporal analysis into actionable, risk-adjusted tyre strategy recommendations that a race engineer can act on immediately |
| **Responsibilities** | Compute `CrossoverScore`; compute `PitWindowRisk`; determine `RecommendationPriority` (IMMEDIATE/HIGH/MONITOR/HOLD/ABORT); generate structured recommendation with primary action, alternative action, evidence list, risk statement, and forecast; ensure recommendations are consistent across frames (no flip-flopping) |
| **Inputs** | `PhysicsEstimate` + `TemporalAnalysis` + `ConfidenceBundle` |
| **Outputs** | `StrategyRecommendation`: `{priority, primary_action, alternative_action, confidence, evidence_list, risk_assessment, forecast_text, crossover_score, pit_window_risk, tyre_delta_estimate}` |
| **Dependencies** | MODULE 06, MODULE 07, MODULE 08 |
| **Libraries** | `dataclasses`, `enum` |
| **APIs Used** | None |
| **Difficulty** | 🟡 Medium |
| **Time Estimate** | 2 hours |
| **Testing Strategy** | Scenario-based testing: wet_severe → HOLD; transitional + high drying rate → HIGH; dry_evolved → HOLD (already on slicks); compound 8 test scenarios; assert no IMMEDIATE recommendation when confidence < 0.60 |
| **Edge Cases** | Conditions improving but rain forecast incoming → should flag ABORT risk; confidence too low to recommend → emit "INSUFFICIENT DATA — MONITOR MANUALLY" |
| **Possible Failures** | Recommendation priority oscillating between HIGH and MONITOR across frames — apply priority hysteresis (require 2 consecutive frames at same priority) |
| **Optimizations** | Add `tyre_delta_estimate` field: approximate time loss per lap on current vs. optimal compound — gives engineer a concrete number to act on |

**Priority Mapping:**
```
IMMEDIATE   → CrossoverScore > 0.85 AND confidence > 0.75 AND risk < 0.30
HIGH        → CrossoverScore > 0.70 AND confidence > 0.60 AND risk < 0.50
MONITOR     → CrossoverScore 0.50–0.70 OR high drying rate but not yet at threshold
HOLD        → CrossoverScore < 0.50 AND stable/worsening conditions
ABORT       → drying_rate < -0.01 (conditions worsening) OR sudden shower flag
```

---

### MODULE 10 — VLM Explainability Layer

| Attribute | Detail |
|---|---|
| **Purpose** | Generate natural-language, engineer-facing explanations for every recommendation; make the AI trustworthy and interpretable |
| **Responsibilities** | Construct a structured prompt for Qwen2-VL combining: frame description, segmentation statistics, current state, recommendation; call VLM via HF Inference API or local inference; parse and validate response; cache explanation per state fingerprint to avoid redundant calls |
| **Inputs** | `StrategyRecommendation` + `SurfaceMetrics` + `TemporalAnalysis` + overlay image (base64) |
| **Outputs** | `ExplanationBundle`: `{engineer_summary (str, 80–120 words), key_observations (list of 3), confidence_statement (str), risk_statement (str)}` |
| **Dependencies** | MODULE 04, MODULE 09 |
| **Libraries** | `transformers` (Qwen2-VL), `huggingface_hub` (InferenceClient), `asyncio` |
| **APIs Used** | HF Inference API (`Qwen/Qwen2-VL-7B-Instruct`) OR local inference |
| **Difficulty** | 🔴 Hard |
| **Time Estimate** | 3 hours |
| **Testing Strategy** | Assert output length 50–200 words; assert no hallucinated compound names; assert explanation mentions at least one metric from input; test fallback when API is unavailable |
| **Edge Cases** | API rate limit (HF free tier: ~100 req/day) — implement explanation caching by state fingerprint; VLM timeout → use structured fallback template; VLM outputs nonsensical text → validate with regex before serving |
| **Possible Failures** | HF Inference API cold start (model not loaded → 30–60s wait) — warm up on system start; rate limiting during live demo; Qwen2-VL refusing motorsport queries due to safety filters |
| **Optimizations** | **Cache by state fingerprint**: if condition_state + recommendation_priority + confidence_bucket are same as previous call, return cached explanation (eliminates 80% of VLM calls); Run VLM asynchronously — don't block the main response; provide results in two phases: fast (metrics, recommendation) then slow (explanation streams in) |

**Prompt Template:**
```
System: You are an AI assistant for a Formula 1 race engineer. Be precise, professional, and concise.

User: Analyze this race track camera frame.

Current Conditions:
- Track State: {condition_state}  
- Estimated Grip (μ): {mu_hat:.2f} (scale: 0.25=soaked, 1.05=full rubber)
- Wet Coverage: {wet_pixel_ratio:.0%} of visible track surface
- Racing Line Dry: {rubber_line_coverage:.0%}  
- Drying Rate: {drying_rate_display}
- Trend: {trend_direction}
- Confidence: {overall_confidence:.0%}

AI Recommendation: {priority} — {primary_action}

Write a 2–3 sentence technical briefing for the race engineer explaining WHY this recommendation is being made, what visual evidence supports it, and what risk to watch.
```

---

### MODULE 11 — APEX Pipeline Orchestrator

| Attribute | Detail |
|---|---|
| **Purpose** | The master coordinator. Chains Modules 02–10 in the correct order; handles async execution; returns a single unified `APEXResult` to the API layer |
| **Responsibilities** | Accept a `PreprocessedFrame`; execute pipeline stages; handle partial failures gracefully (if VLM fails, return without explanation but flag it); emit timing metrics for each stage; return complete `APEXResult` |
| **Inputs** | `PreprocessedFrame` + `session_id` |
| **Outputs** | `APEXResult`: `{perception, surface_metrics, physics_estimate, temporal_analysis, confidence, recommendation, visualization, explanation, pipeline_timing_ms}` |
| **Dependencies** | All core modules (02–10) |
| **Libraries** | `asyncio`, `time`, `loguru` |
| **APIs Used** | None (orchestrates internal modules) |
| **Difficulty** | 🟡 Medium |
| **Time Estimate** | 1.5 hours |
| **Testing Strategy** | Integration test: pass a real wet track image; assert all fields in `APEXResult` are populated; assert total pipeline time < 5s on CPU, < 2s on GPU |
| **Edge Cases** | Module N fails → log error, mark field as None, continue pipeline; all modules fail → return error response with diagnostics |
| **Possible Failures** | Async deadlock if VLM call blocks event loop — always run VLM in `run_in_executor`; memory leak if preprocessed tensors not freed |
| **Optimizations** | **Parallelise independent stages**: DINOv2 and CLIP can run simultaneously (no dependency between them); visualization can be started immediately after segmentation while physics runs; VLM runs last and asynchronously |

**Execution Graph (within pipeline):**
```
PreprocessedFrame
    ├──→ [PARALLEL] DINOv2 Feature Extract
    ├──→ [PARALLEL] SegFormer Segmentation
    └──→ [PARALLEL] CLIP Zero-Shot Score
              ↓ (all complete)
         PerceptionResult
              ├──→ Surface Feature Extraction
              └──→ Visualization Generation (async, non-blocking)
                        ↓
                   SurfaceMetrics
                        ↓
                   Physics Estimate ←── Temporal History
                        ↓
                   Temporal Analysis Update
                        ↓
                   Confidence Calibration
                        ↓
                   Strategy Recommendation
                        ↓
                   VLM Explanation (async, returns when ready)
                        ↓
                   APEXResult (assembled)
```

---

### MODULE 12 — Session Store

| Attribute | Detail |
|---|---|
| **Purpose** | Maintain per-session state: temporal frame history, current recommendation, timeline of all `APEXResult` frames for the session |
| **Responsibilities** | Create/retrieve/delete sessions; store ordered list of `APEXResult` per session; enforce max session history (100 frames); provide timeline query endpoint; handle session expiry (TTL: 2 hours) |
| **Inputs** | `session_id` (UUID) + `APEXResult` |
| **Outputs** | Session retrieval, timeline list, current state |
| **Dependencies** | MODULE 11 |
| **Libraries** | `uuid`, `datetime`, `collections.deque`, `asyncio.Lock` |
| **APIs Used** | None |
| **Difficulty** | 🟢 Easy |
| **Time Estimate** | 45 minutes |
| **Testing Strategy** | Assert session creation/retrieval; assert timeline is ordered; assert old frames expire; concurrent write test with asyncio |
| **Edge Cases** | Concurrent writes to same session from WebSocket stream — use `asyncio.Lock` per session; session ID collision (UUID4 — negligible risk) |
| **Possible Failures** | Memory growth if sessions never expire — implement background cleanup task |
| **Optimizations** | For hackathon: in-memory dict is sufficient; Redis can be added later with same interface |

---

### MODULE 13 — Video Processing Service

| Attribute | Detail |
|---|---|
| **Purpose** | Accept uploaded video files; extract frames at configurable FPS; feed frames sequentially through the APEX pipeline; manage async processing with progress updates |
| **Responsibilities** | Accept video file upload (MP4, MOV, AVI); extract frames at configurable rate (default: 1 fps for analysis, 4 fps for fast demo); run each frame through MODULE 11; update session store; emit progress via WebSocket; return session_id immediately (async processing) |
| **Inputs** | Video file bytes + `session_id` + `target_fps` |
| **Outputs** | Session timeline populated with frame-by-frame results; WebSocket progress events |
| **Dependencies** | MODULE 11, MODULE 12 |
| **Libraries** | `cv2` (OpenCV), `asyncio`, `concurrent.futures` |
| **APIs Used** | None |
| **Difficulty** | 🟡 Medium |
| **Time Estimate** | 2 hours |
| **Testing Strategy** | Upload a 10-second test clip; assert correct number of frames extracted; assert session timeline has >0 results; assert progress events are emitted |
| **Edge Cases** | Corrupt video file; video with no audio track (shouldn't matter but OpenCV may warn); very long videos (>10 min) — cap at 300 frames; codec issues on Windows (H.265 may need ffmpeg) |
| **Possible Failures** | OpenCV codec issues on Windows — recommend ffmpeg as fallback; large video causing memory spike — stream frames without buffering entire video |
| **Optimizations** | Process frames in a background `ThreadPoolExecutor` to avoid blocking the event loop; limit max concurrent video processing jobs to 2 |

---

### MODULE 14 — FastAPI Route Layer

| Attribute | Detail |
|---|---|
| **Purpose** | Expose all APEX capabilities as clean, documented REST + WebSocket endpoints |
| **Responsibilities** | `POST /api/v1/analyze/image` → sync image analysis; `POST /api/v1/analyze/video` → async video upload, returns session_id; `GET /api/v1/session/{id}` → current session state; `GET /api/v1/session/{id}/timeline` → full frame history; `WS /ws/live` → real-time frame streaming; `GET /api/v1/demo/{clip_name}` → trigger pre-loaded demo clip; `GET /api/v1/health` → system health + model status |
| **Inputs** | HTTP requests, WebSocket frames, multipart file uploads |
| **Outputs** | JSON responses + WebSocket messages conforming to `APIResponse` schema |
| **Dependencies** | MODULE 11, MODULE 12, MODULE 13 |
| **Libraries** | `fastapi`, `python-multipart`, `websockets`, `pydantic` |
| **APIs Used** | None (this IS the API) |
| **Difficulty** | 🟡 Medium |
| **Time Estimate** | 2 hours |
| **Testing Strategy** | Pytest with `httpx.AsyncClient`; test all endpoints; test WebSocket with `websockets` test client; test file size validation; test 404/422 error responses |
| **Edge Cases** | Large file upload timeout; WebSocket disconnect mid-stream; concurrent requests overwhelming inference |
| **Possible Failures** | CORS misconfiguration blocking frontend; WebSocket handshake failures; file upload size limit (default FastAPI: 10MB — increase to 500MB for video) |
| **Optimizations** | Add `X-Process-Time` response header for debugging; use `StreamingResponse` for large result payloads; enable gzip compression via middleware |

---

### MODULE 15 — Weather Fusion Service (Optional Enhancement)

| Attribute | Detail |
|---|---|
| **Purpose** | Fetch ambient weather data for a given circuit; use as a Bayesian prior to strengthen or weaken visual predictions |
| **Responsibilities** | Accept circuit name → lookup GPS coordinates; call OpenWeatherMap API; parse precipitation probability, humidity, wind speed; return `WeatherContext`; cache results for 5 minutes (weather doesn't change per-second) |
| **Inputs** | Circuit name (string) or lat/lng |
| **Outputs** | `WeatherContext`: `{precipitation_prob, humidity, temp_celsius, wind_speed, condition_description, weather_icon}` |
| **Dependencies** | MODULE 00 (config for API key) |
| **Libraries** | `httpx` (async HTTP), `cachetools` |
| **APIs Used** | OpenWeatherMap One Call API 3.0 (free tier: 1000 calls/day) |
| **Difficulty** | 🟢 Easy |
| **Time Estimate** | 1 hour |
| **Testing Strategy** | Mock API response; assert correct parsing; assert cache prevents duplicate calls |
| **Edge Cases** | API key missing → disable weather fusion gracefully; unknown circuit name → return None, system continues without weather |
| **Possible Failures** | Rate limiting; network unavailability during demo — always have offline fallback |
| **Optimizations** | Cache at circuit level for 5 minutes; weather data updates the μ̂ confidence interval width (high precipitation prob → wider uncertainty bounds) |

---

### MODULE 16 — Frontend Design System

| Attribute | Detail |
|---|---|
| **Purpose** | Establish all CSS custom properties, typography, color palette, spacing system, and reusable component styles that the entire frontend is built upon |
| **Responsibilities** | Define CSS variables (colors, spacing, typography, shadows, transitions); import Google Fonts (Inter + JetBrains Mono); create base reset; define utility classes; create component base styles (cards, badges, gauges, buttons, panels) |
| **Inputs** | Design specification from architecture document |
| **Outputs** | `base.css`, `components.css`, `animations.css` fully built |
| **Dependencies** | None |
| **Libraries** | Vanilla CSS, Google Fonts CDN |
| **APIs Used** | None |
| **Difficulty** | 🟡 Medium |
| **Time Estimate** | 2 hours |
| **Testing Strategy** | Visual review in browser; test in Chrome + Firefox; verify fonts load; test dark theme consistency |
| **Edge Cases** | Google Fonts CDN offline → system font fallback in CSS; narrow viewport breaking three-column layout |
| **Possible Failures** | CSS custom property browser compatibility (IE — not a concern for hackathon); specificity conflicts |
| **Optimizations** | Use CSS Grid for main layout (simpler than flexbox chains); preload fonts with `<link rel="preload">`; minimize repaints by using `transform` and `opacity` for animations |

**Design Token Reference:**
```css
:root {
  /* Colors */
  --bg-primary:    #0A0B0D;
  --bg-secondary:  #111318;
  --bg-card:       #161820;
  --bg-elevated:   #1C1F28;
  --accent-red:    #FF1E00;
  --accent-red-dim: #CC1800;
  --data-green:    #00C853;
  --data-amber:    #FF9800;
  --data-blue:     #2196F3;
  --data-purple:   #9C27B0;
  --text-primary:  #F0F2F5;
  --text-secondary: #8A8F9E;
  --text-dim:      #4A4F60;
  --border:        #232630;
  --border-bright: #333848;
  
  /* Typography */
  --font-ui:    'Inter', system-ui, sans-serif;
  --font-data:  'JetBrains Mono', 'Fira Code', monospace;
  
  /* Spacing */
  --space-xs: 4px; --space-sm: 8px; --space-md: 16px;
  --space-lg: 24px; --space-xl: 32px; --space-2xl: 48px;
  
  /* Timing */
  --transition-fast: 150ms ease;
  --transition-normal: 300ms ease;
  --transition-slow: 500ms ease;
}
```

---

### MODULE 17 — Frontend Command Center Layout

| Attribute | Detail |
|---|---|
| **Purpose** | The master shell of the application; three-panel layout with header, providing the visual scaffolding all components plug into |
| **Responsibilities** | Three-column CSS Grid layout (left panel 320px, center flexible, right panel 340px); fixed header with system status indicators; collapsible panels for mobile; navigation between "Live Analysis" and "Session Timeline" views |
| **Inputs** | CSS design system from MODULE 16 |
| **Outputs** | `index.html` structural skeleton; `CommandCenter.js` layout controller |
| **Dependencies** | MODULE 16 |
| **Libraries** | Vanilla CSS Grid, Vanilla JS |
| **APIs Used** | None |
| **Difficulty** | 🟡 Medium |
| **Time Estimate** | 1.5 hours |
| **Testing Strategy** | Visual review at 1440px, 1920px; verify panel overflow is handled; verify header is always visible |
| **Edge Cases** | Very small screens → show single-column stacked layout; browser zoom >150% → check overflow |
| **Possible Failures** | CSS Grid gap causing unexpected scrollbars; fixed header overlapping content |
| **Optimizations** | Use `dvh` units (dynamic viewport height) for full-height panels instead of `100vh` |

---

### MODULE 18 — Track Analysis Canvas Component

| Attribute | Detail |
|---|---|
| **Purpose** | The visual centrepiece. Displays the original track frame with segmentation overlay; handles heatmap toggle; shows attention map; handles zoom/pan |
| **Responsibilities** | Display uploaded/demo image; render segmentation overlay at configurable opacity; toggle between: original / segmentation / attention map / blended views; show class legend; animate overlay fade-in on update; display confidence badge over image |
| **Inputs** | `VisualizationBundle` (base64 images from API) |
| **Outputs** | Interactive canvas component in center panel |
| **Dependencies** | MODULE 16, MODULE 17 |
| **Libraries** | HTML5 Canvas, Vanilla JS |
| **APIs Used** | None (renders data from API response) |
| **Difficulty** | 🟡 Medium |
| **Time Estimate** | 2 hours |
| **Testing Strategy** | Load all three visualization types; test toggle buttons; test that overlay updates smoothly when new frame arrives |
| **Edge Cases** | Slow network → show loading skeleton; broken image URL → show placeholder; canvas fallback if WebGL unavailable |
| **Possible Failures** | CORS issues loading base64 images onto canvas (shouldn't apply to base64); canvas taint from cross-origin images |
| **Optimizations** | CSS `mix-blend-mode: screen` for overlay blending (GPU-accelerated); transition opacity on canvas rather than re-drawing; requestAnimationFrame for smooth updates |

---

### MODULE 19 — Grip Gauge Component

| Attribute | Detail |
|---|---|
| **Purpose** | Real-time display of the estimated grip coefficient μ̂; the most impactful single number on screen |
| **Responsibilities** | Render animated circular arc gauge from 0.25 to 1.05; color-code by condition state (red → amber → green); show numeric value with one decimal; show uncertainty band (lower–upper bounds); animate smoothly on value change; show condition state label below gauge |
| **Inputs** | `{mu_hat, mu_lower, mu_upper, condition_state}` |
| **Outputs** | SVG-based animated gauge |
| **Dependencies** | MODULE 16 |
| **Libraries** | Vanilla JS + SVG (no D3 needed for a single gauge) |
| **APIs Used** | None |
| **Difficulty** | 🟡 Medium |
| **Time Estimate** | 1.5 hours |
| **Testing Strategy** | Render at min/max/mid values; test color transitions; test animation at update rate |
| **Edge Cases** | Value outside [0.25, 1.05] range → clamp and log warning; NaN value → show "---" |
| **Possible Failures** | SVG arc math errors for edge angles (0° and 360°); animation jitter if update rate is very high |
| **Optimizations** | Use CSS custom property animation (`@property`) for smooth arc transitions without JS requestAnimationFrame loops |

---

### MODULE 20 — Grip Timeline Chart Component

| Attribute | Detail |
|---|---|
| **Purpose** | Display the full temporal evolution of μ̂ across all analyzed frames in the session; the key temporal intelligence visualization |
| **Responsibilities** | Render line chart of μ̂ over time (frames/timestamp); show uncertainty band as area fill; mark condition state transitions with vertical annotations; mark IMMEDIATE recommendations with alert markers; animate new data points appending; show crossover threshold line for current target compound |
| **Inputs** | Session timeline array of `{timestamp, mu_hat, mu_lower, mu_upper, condition_state, recommendation_priority}` |
| **Outputs** | D3.js interactive chart |
| **Dependencies** | MODULE 16 |
| **Libraries** | **D3.js v7** (CDN) |
| **APIs Used** | `GET /api/v1/session/{id}/timeline` |
| **Difficulty** | 🔴 Hard |
| **Time Estimate** | 3 hours |
| **Testing Strategy** | Render with mock 20-frame dataset; test tooltip behavior; test zoom/pan; test that new frames animate in smoothly |
| **Edge Cases** | 0 frames → show "Start analysis to see timeline" empty state; 1 frame → single point, no line; >100 frames → scroll/zoom required |
| **Possible Failures** | D3 v7 import via CDN may conflict with ES module scope; performance with >200 data points without canvas rendering |
| **Optimizations** | Use D3 `join()` pattern for efficient enter/update/exit; limit visible range to last 50 frames with scroll; canvas rendering if >100 points |

---

### MODULE 21 — Recommendation Panel Component

| Attribute | Detail |
|---|---|
| **Purpose** | The primary decision-support output. Displays the strategy recommendation in a way that a race engineer can read and act on in under 3 seconds |
| **Responsibilities** | Display priority badge (IMMEDIATE/HIGH/MONITOR/HOLD/ABORT) with color coding; display primary action text; display confidence meter; display 3-bullet evidence list; display alternative action; display risk statement; display crossover forecast ("~3.5 min"); animate slide-in on priority change; pulse animation on IMMEDIATE priority |
| **Inputs** | `StrategyRecommendation` object |
| **Outputs** | Dynamic HTML recommendation card |
| **Dependencies** | MODULE 16 |
| **Libraries** | Vanilla JS |
| **APIs Used** | None (renders data from API response) |
| **Difficulty** | 🟢 Easy |
| **Time Estimate** | 1.5 hours |
| **Testing Strategy** | Render all 5 priority states; verify color coding; verify animations; verify text truncation for long strings |
| **Edge Cases** | Missing optional fields → graceful degradation; priority change animation must not block screen readers |
| **Possible Failures** | CSS animation causing layout reflow (use `transform` only); IMMEDIATE pulse animation being too distracting |
| **Optimizations** | Priority badge uses CSS custom property for color — one property change triggers full visual update; text animates with `@keyframes` character-by-character typing effect for VLM output |

**Priority Color Mapping:**
```
IMMEDIATE → --accent-red    → pulsing border animation
HIGH      → --data-amber    → static highlight
MONITOR   → --data-blue     → calm indicator
HOLD      → --text-secondary → muted
ABORT     → #FF0040         → flashing red-pink
```

---

### MODULE 22 — Evidence & Explainability Panel

| Attribute | Detail |
|---|---|
| **Purpose** | Display the AI's reasoning in a way that builds engineer trust — showing WHAT was seen and WHY the recommendation was made |
| **Responsibilities** | Display VLM-generated explanation text (with typewriter animation for streaming effect); display DINOv2 attention map thumbnail; display source agreement bar (how much DINOv2 / SegFormer / CLIP agreed); display surface metrics table (wet%, reflectance, rubber line coverage); display confidence breakdown |
| **Inputs** | `ExplanationBundle` + `VisualizationBundle` + `ConfidenceBundle` + `SurfaceMetrics` |
| **Outputs** | Evidence panel component |
| **Dependencies** | MODULE 16 |
| **Libraries** | Vanilla JS |
| **APIs Used** | None |
| **Difficulty** | 🟡 Medium |
| **Time Estimate** | 1.5 hours |
| **Testing Strategy** | Test with and without VLM explanation (fallback state); test typewriter animation timing; verify attention map renders |
| **Edge Cases** | VLM explanation unavailable → show structured template output; very long explanation → clamp with expand toggle |
| **Possible Failures** | Typewriter animation conflicting with live updates; attention map not loading if backend slow |
| **Optimizations** | Typewriter animation runs at character-level with requestAnimationFrame; pre-render attention map as background while user reads metrics |

---

### MODULE 23 — Media Upload & Demo Mode

| Attribute | Detail |
|---|---|
| **Purpose** | The entry point for all media input; handles drag-and-drop upload AND demo mode activation; must look effortless |
| **Responsibilities** | Drag-and-drop zone for image/video upload; click-to-browse file picker; demo clip gallery (5 pre-loaded F1 rain scenarios with thumbnail previews); WebSocket live stream mode (accepts webcam or RTSP); show upload progress for video files; validate file types and sizes before upload |
| **Inputs** | User file interaction; demo clip selection |
| **Outputs** | Media sent to API; session_id received; pipeline begins |
| **Dependencies** | MODULE 16, `api.js` service |
| **Libraries** | Vanilla JS (File API, drag events) |
| **APIs Used** | `POST /api/v1/analyze/image`, `POST /api/v1/analyze/video`, `GET /api/v1/demo/{clip}` |
| **Difficulty** | 🟡 Medium |
| **Time Estimate** | 2 hours |
| **Testing Strategy** | Test drag-and-drop; test file picker; test demo clip triggers API correctly; test file type validation rejects non-image/video |
| **Edge Cases** | User drops multiple files → only process first; mobile touch doesn't support drag → show tap-to-browse instead; demo clips not loaded on server → show error gracefully |
| **Possible Failures** | Large video file causing browser memory issues before upload completes — stream upload using chunked FormData |
| **Optimizations** | Show image preview immediately from FileReader before upload completes (feels instant); demo clips should load with `0`-click — just click thumbnail to instantly trigger analysis |

**Demo Clips to Pre-prepare:**
```
demo_01: 2021 Belgian GP — Spa standing water, full wet conditions
demo_02: 2016 Monaco GP — mixed drying conditions
demo_03: 2019 German GP — sudden shower, strategy chaos
demo_04: Generic dry track — high rubber evolution
demo_05: Synthetic transition sequence — wet → drying → dry
```
*(Use frames from free broadcast archives / synthetically generate with stable diffusion)*

---

### MODULE 24 — WebSocket Real-Time Service

| Attribute | Detail |
|---|---|
| **Purpose** | Enable real-time frame streaming from browser to backend; power the "Live Camera" mode |
| **Responsibilities** | Manage WebSocket connection lifecycle (connect, reconnect, ping/pong); send frames from webcam capture at configurable FPS (default 0.5 fps to avoid overloading backend); receive `APEXResult` messages and dispatch to UI components; handle connection errors gracefully; show connection status indicator |
| **Inputs** | WebSocket connection to `WS /ws/live`; video frames from MediaDevices API |
| **Outputs** | Stream of `APEXResult` updates dispatched to components |
| **Dependencies** | MODULE 14 (backend WS endpoint), all frontend components |
| **Libraries** | Browser native WebSocket API; MediaDevices API |
| **APIs Used** | `WS /ws/live` |
| **Difficulty** | 🟡 Medium |
| **Time Estimate** | 2 hours |
| **Testing Strategy** | Test connection establishment; test graceful reconnect after disconnect; test frame sending rate limiter; test dispatch to components |
| **Edge Cases** | Browser camera permission denied → fallback to upload mode; WebSocket blocked by corporate firewall → fallback to polling; server restart mid-session → auto-reconnect |
| **Possible Failures** | Message queue backlog if backend is slower than frame rate — implement client-side rate limiter and drop frames if ACK not received |
| **Optimizations** | Send frames as JPEG with quality 0.7 (not PNG) to reduce payload size; only send new frame if previous result was received |

---

### MODULE 25 — Session Timeline View

| Attribute | Detail |
|---|---|
| **Purpose** | Post-session review of the complete analysis — shows every frame analyzed, the full grip evolution timeline, and key decision moments |
| **Responsibilities** | Fetch full session timeline; render scrollable frame gallery; highlight frames with IMMEDIATE/HIGH recommendations; show aggregate stats (min/max/avg μ̂, total drying time, key events); allow frame-click to view detailed analysis for that moment; export as PDF report (optional) |
| **Inputs** | `GET /api/v1/session/{id}/timeline` |
| **Outputs** | Interactive timeline view |
| **Dependencies** | MODULE 20 (timeline chart), MODULE 16 |
| **Libraries** | Vanilla JS, D3.js |
| **APIs Used** | `GET /api/v1/session/{id}/timeline` |
| **Difficulty** | 🟡 Medium |
| **Time Estimate** | 2 hours |
| **Testing Strategy** | Load with mock 30-frame session; test frame click navigation; test stats calculation |
| **Edge Cases** | Session with 0 frames; session expired → show "Session data unavailable" |
| **Possible Failures** | Loading 100 frames of thumbnails simultaneously → virtualize the list |
| **Optimizations** | Lazy-load frame thumbnails using IntersectionObserver |

---

## Part 2 — Complete Dependency Graph

```
MODULE 00 (Scaffold)
    │
    ├──→ MODULE 01 (Model Registry)
    │         │
    │         ├──→ MODULE 02 (Image Preprocessing)
    │         │         │
    │         │         └──→ MODULE 03 (Perception Layer)
    │         │                   │
    │         │                   ├──→ MODULE 04 (Visualization Engine)
    │         │                   │
    │         │                   └──→ MODULE 05 (Surface Feature Extractor)
    │         │                             │
    │         │                             └──→ MODULE 06 (Physics & State Machine)
    │         │                                       │
    │         │                                       ├──→ MODULE 07 (Temporal Reasoning)
    │         │                                       │         │
    │         │                                       │         └──→ MODULE 08 (Confidence Calibration)
    │         │                                       │                   │
    │         │                                       │                   └──→ MODULE 09 (Recommendation Engine)
    │         │                                       │                             │
    │         │                                       │                             └──→ MODULE 10 (VLM Explainer)
    │         │                                       │                                       │
    │         │                                       │                                       └──→ MODULE 11 (APEX Orchestrator)
    │         │                                       │                                                 │
    │         │                                       └──────────────────────────────────────────────── ┘
    │         │
    │         └──→ MODULE 15 (Weather Service) ──→ MODULE 06 (optional fusion)
    │
    ├──→ MODULE 12 (Session Store)
    │         │
    │         └──→ MODULE 13 (Video Processor) ──→ MODULE 11
    │
    └──→ MODULE 14 (API Routes) ──→ MODULE 11, 12, 13
              │
              └──→ MODULE 24 (WebSocket Service) [backend side]


FRONTEND (independent of backend implementation order):
MODULE 16 (Design System)
    │
    ├──→ MODULE 17 (Layout)
    │         │
    │         ├──→ MODULE 18 (Canvas)
    │         ├──→ MODULE 19 (Grip Gauge)
    │         ├──→ MODULE 20 (Timeline Chart)
    │         ├──→ MODULE 21 (Recommendation Panel)
    │         ├──→ MODULE 22 (Evidence Panel)
    │         ├──→ MODULE 23 (Upload / Demo)
    │         ├──→ MODULE 24 (WebSocket Client)
    │         └──→ MODULE 25 (Session Timeline)
```

**Critical Path (longest chain that must complete before demo is possible):**
```
M00 → M01 → M02 → M03 → M05 → M06 → M07 → M08 → M09 → M11 → M14
  ↑
  └── M16 → M17 → M21 → M23 → [API Integration]
```

---

## Part 3 — Milestone Plan

### MILESTONE 1: Infrastructure & Inference Core ✅
**Goal:** Models are loaded and producing inference results from the CLI.  
**Deliverable:** `python scripts/test_inference.py wet_track.jpg` outputs a complete `APEXResult` JSON to stdout.  
**Modules:** M00, M01, M02, M03, M04, M05  
**Estimated Time:** 6–8 hours  
**Go/No-Go Test:** Wet track image → segmentation mask has >40% wet pixels; CLIP score for "wet track" > 0.6

---

### MILESTONE 2: Physics Intelligence ✅
**Goal:** Raw vision outputs converted to engineering units; state machine and temporal reasoning operational.  
**Deliverable:** Feed a 10-frame synthetic sequence; `μ̂` curve shows expected evolution; state machine transitions correctly.  
**Modules:** M06, M07, M08, M09  
**Estimated Time:** 5–6 hours  
**Go/No-Go Test:** Wet → drying sequence produces MONITOR → HIGH priority progression

---

### MILESTONE 3: Full Backend API ✅
**Goal:** All API endpoints operational; image and video upload works; WebSocket streams data.  
**Deliverable:** Postman/curl tests pass for all endpoints.  
**Modules:** M10, M11, M12, M13, M14, M15  
**Estimated Time:** 5–7 hours  
**Go/No-Go Test:** `curl -X POST /api/v1/analyze/image -F file=@wet_track.jpg` returns full `APEXResult`

---

### MILESTONE 4: Frontend Shell & Core Components ✅
**Goal:** The UI looks stunning and the key panels render with mock data.  
**Deliverable:** Frontend loads in browser; all panels render with hardcoded demo data.  
**Modules:** M16, M17, M18, M19, M20, M21, M22  
**Estimated Time:** 6–8 hours  
**Go/No-Go Test:** Show design to someone unfamiliar — they say "this looks professional"

---

### MILESTONE 5: Full Integration ✅
**Goal:** Frontend connected to live backend; upload → analysis → display pipeline works end-to-end.  
**Deliverable:** Upload a wet track image → all panels update with real AI results.  
**Modules:** M23, M24, M25 + API integration in all frontend components  
**Estimated Time:** 4–5 hours  
**Go/No-Go Test:** Demo mode clip → full analysis → recommendation displayed in <5 seconds

---

### MILESTONE 6: Polish, Demo Mode & Deployment ✅
**Goal:** Demo-ready system; Docker works; demo clips preloaded; no rough edges.  
**Deliverable:** `docker-compose up` starts everything; 5 demo clips work; zero console errors.  
**Modules:** Demo clip preparation + Docker + README  
**Estimated Time:** 3–4 hours  
**Go/No-Go Test:** Run full demo flow 3 times without touching keyboard except for clicks; zero failures

---

## Part 4 — Blocker Analysis (Pre-Coding)

### 🔴 CRITICAL BLOCKERS (Must resolve before starting)

| # | Blocker | Resolution | Time to Resolve |
|---|---|---|---|
| B1 | **VLM hosting decision** (D1) | Decide: HF Inference API token OR local GPU check | 10 minutes |
| B2 | **HuggingFace token** for gated models | Create HF account + token + store in `.env` | 15 minutes |
| B3 | **GPU availability check** | Run `torch.cuda.is_available()` — drives D1, D7 decisions | 5 minutes |
| B4 | **Model download pre-run** | Run `scripts/download_models.py` before demo day | 30–60 minutes (network) |
| B5 | **Python environment** | Create `venv`, install base deps, confirm `import torch` works | 20 minutes |
| B6 | **OpenCV on Windows** | `pip install opencv-python-headless` (avoid GUI conflicts) | 5 minutes |
| B7 | **Demo footage sourcing** | Find / generate 5 demo clip frame sequences for demo mode | 1–2 hours |

### 🟡 MEDIUM BLOCKERS (Resolve within first 2 hours)

| # | Blocker | Resolution |
|---|---|---|
| B8 | SegFormer class mapping to our 6 classes | Write and validate `SEGFORMER_CLASS_REMAP` dict before M03 |
| B9 | CLIP prompt tuning | Validate 8 prompts against 10 test images before M03 goes to production |
| B10 | CORS configuration | Test frontend-to-backend communication before M14 integration |
| B11 | Windows file path separators | Use `pathlib.Path` throughout backend (no hardcoded `/`) |

### 🟢 LOW BLOCKERS (Can resolve during implementation)

| # | Blocker | Resolution |
|---|---|---|
| B12 | D3.js version conflicts | Pin to D3 v7 CDN URL with SRI hash |
| B13 | WebSocket behind reverse proxy | Add `proxy_read_timeout 3600` to Nginx config |
| B14 | SQLite concurrent writes | Use WAL mode: `PRAGMA journal_mode=WAL` |

---

## Part 5 — Hackathon-Winning Improvements (High ROI Only)

These are additions that significantly increase the probability of winning without adding excessive complexity:

### ✅ IMP-1: Racing Line Drying Indicator (HIGH IMPACT, LOW EFFORT)
**What:** A dedicated metric showing `% of racing line that is dry` as a standalone prominent display, separate from overall wet%.  
**Why:** This is exactly what engineers care about — the racing line is what determines crossover. A team showing they understand this distinction will immediately impress motorsport judges.  
**Effort:** 2 hours (sub-metric of M05)

### ✅ IMP-2: Compound Recommendation with Tyre Delta (HIGH IMPACT, MEDIUM EFFORT)
**What:** Not just "pit for intermediates" but "estimated +1.8s/lap delta on current compound vs. intermediates."  
**Why:** This is the actual calculation race engineers perform. Showing a time delta makes the system immediately actionable.  
**Effort:** 2 hours (extend M09)

### ✅ IMP-3: Typewriter Animation for VLM Output (HIGH IMPACT, LOW EFFORT)
**What:** Stream the VLM explanation character by character as it arrives (or simulate streaming for cached results).  
**Why:** Creates a "live AI reasoning" effect that is dramatically more impressive than text appearing instantly. Judges will watch the AI "think" in real time.  
**Effort:** 45 minutes (M22)

### ✅ IMP-4: Pre-loaded Demo Clips with Named Scenarios (HIGH IMPACT, LOW EFFORT)
**What:** 5 clickable demo scenarios each with a name ("2021 Belgian GP — Race Start", "Monaco Dry After Rain", etc.).  
**Why:** Eliminates upload friction for judges; each scenario demonstrates a different system capability; judges can explore multiple conditions without re-uploading.  
**Effort:** 1 hour setup + demo data preparation

### ✅ IMP-5: Sector-Level Risk Map (MEDIUM IMPACT, MEDIUM EFFORT)
**What:** Divide the track image into a 3×3 grid of sectors; show per-sector grip estimate.  
**Why:** "Sector 2 still showing standing water" is immediately actionable; it's how F1 engineers actually think about track conditions.  
**Effort:** 3 hours (extend M05 + M18)

### ⚠️ IMP-6: Live Weather Radar Overlay (LOW IMPACT, HIGH EFFORT)
**What:** Integrate a weather radar tile over a circuit map.  
**Why:** Cool visually but adds significant complexity and API integration risk.  
**Verdict:** Skip. Too much risk for marginal demo value.

---

## Part 6 — Day-by-Day Implementation Order

### DAY 1 — Foundation & AI Core (Target: 8–10 hours)

```
MORNING (0–3h):
  ☐ Lock all D1–D8 decisions
  ☐ Resolve blockers B1–B7
  ☐ MODULE 00: Project scaffold, config, logging, FastAPI skeleton
  ☐ MODULE 01: Model registry — load DINOv2, SegFormer, CLIP
  ☐ Run `scripts/download_models.py` (let run in background)

AFTERNOON (3–7h):
  ☐ MODULE 02: Image preprocessing utilities
  ☐ MODULE 03: Perception layer — DINOv2 + SegFormer + CLIP
  ☐ Validate SegFormer class remapping on 5 test images
  ☐ Tune CLIP prompts on 10 test images

EVENING (7–10h):
  ☐ MODULE 04: Visualization engine (overlays, heatmaps, base64)
  ☐ MODULE 05: Surface feature extraction (wet%, specular, rubber line)
  ☐ MILESTONE 1 GO/NO-GO TEST
```

---

### DAY 2 — Intelligence & Physics Layer (Target: 8–10 hours)

```
MORNING (0–3h):
  ☐ MODULE 06: Physics & state machine (μ̂ formula, state transitions)
  ☐ MODULE 07: Temporal reasoning (sliding window, EWMA, trend, forecast)

AFTERNOON (3–6h):
  ☐ MODULE 08: Confidence calibration (multi-source agreement)
  ☐ MODULE 09: Recommendation engine (CrossoverScore, priorities, evidence)
  ☐ MODULE 10: VLM explainability (prompt, API call, cache, fallback)
  ☐ MILESTONE 2 GO/NO-GO TEST

EVENING (6–10h):
  ☐ MODULE 11: APEX Pipeline Orchestrator
  ☐ MODULE 12: Session Store
  ☐ MODULE 15: Weather Service (quick win, 1 hour)
  ☐ Unit tests for M06–M09 (critical path)
```

---

### DAY 3 — API Layer & Frontend Foundation (Target: 8–10 hours)

```
MORNING (0–3h):
  ☐ MODULE 13: Video processor
  ☐ MODULE 14: FastAPI routes (all endpoints)
  ☐ MILESTONE 3 GO/NO-GO TEST (Postman / curl verification)

AFTERNOON (3–7h):
  ☐ MODULE 16: Frontend design system (CSS tokens, typography, colors)
  ☐ MODULE 17: Command Center layout (three-panel shell)
  ☐ MODULE 19: Grip Gauge (SVG animated, highest visual impact component)
  ☐ MODULE 21: Recommendation Panel (static render with mock data)

EVENING (7–10h):
  ☐ MODULE 18: Track Analysis Canvas
  ☐ MODULE 22: Evidence & Explainability Panel
  ☐ MILESTONE 4 GO/NO-GO TEST (visual review with mock data)
```

---

### DAY 4 — Integration, Timeline & Polish (Target: 8–10 hours)

```
MORNING (0–3h):
  ☐ MODULE 20: Grip Timeline Chart (D3.js)
  ☐ MODULE 23: Upload Zone + Demo Mode
  ☐ MODULE 24: WebSocket (client + server integration)

AFTERNOON (3–6h):
  ☐ MODULE 25: Session Timeline view
  ☐ Full frontend ↔ backend integration
  ☐ IMP-1: Racing line drying indicator
  ☐ IMP-3: Typewriter animation for VLM
  ☐ MILESTONE 5 GO/NO-GO TEST

EVENING (6–10h):
  ☐ IMP-2: Tyre delta estimate in recommendation
  ☐ IMP-4: Named demo scenarios (all 5 clips loaded and tested)
  ☐ Bug fixing pass
  ☐ Performance profiling (target: <3s image analysis, <5s video frame)
```

---

### DAY 5 — Hardening, Demo Prep & Deployment (Target: 6–8 hours)

```
MORNING (0–3h):
  ☐ Docker: Dockerfile for backend; nginx for frontend; docker-compose.yml
  ☐ README.md (architecture diagram, setup instructions, API docs)
  ☐ MILESTONE 6 GO/NO-GO TEST

AFTERNOON (3–5h):
  ☐ Demo run-through × 3 (no keyboard, clicks only)
  ☐ Fix any rough edges found during run-through
  ☐ Performance optimization pass (quantization, caching)
  ☐ Error handling audit (what happens if backend is slow?)

EVENING (5–7h):
  ☐ Final polish: animations, transitions, empty states
  ☐ Cross-browser testing (Chrome, Firefox, Edge)
  ☐ Deployment verification (docker-compose up from clean state)
  ☐ Prepare demo script (30s hook → 3-min full walkthrough)
```

---

## Part 7 — Total Time Estimate

| Category | Modules | Time |
|---|---|---|
| Scaffold & Config | M00 | 0.5h |
| Model Infrastructure | M01, M02 | 2.5h |
| AI Perception Core | M03, M04, M05 | 7.5h |
| Physics & Temporal | M06, M07, M08 | 5h |
| Recommendations & XAI | M09, M10, M11 | 6.5h |
| Backend Services | M12, M13, M14, M15 | 5.75h |
| Frontend Design | M16, M17 | 3.5h |
| Frontend Components | M18–M25 | 12.5h |
| Integration & Testing | Cross-module | 3h |
| Demo Prep & Deployment | Docker + README + demo clips | 4h |
| **TOTAL** | | **~50 hours** |

*Assumes 1 developer. 2 developers splitting backend/frontend: ~28 hours wall-clock time.*

---

## Part 8 — Risk Register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| VLM API rate-limited during demo | Medium | HIGH | Cache all VLM calls; run demo clips pre-cached |
| SegFormer produces poor results on race track images | Medium | HIGH | CLIP zero-shot provides backup classification; degrade gracefully |
| GPU VRAM OOM with multiple models | Low | HIGH | CPU fallback; model quantization; load one at a time |
| Demo video codec incompatible | Medium | Medium | Pre-validate all demo clips; provide frame sequences as fallback |
| WebSocket instability | Low | Medium | HTTP polling fallback for image mode; WS only for live stream |
| First demo clip takes 60s to process | Medium | HIGH | Pre-process all demo clips and cache results; demo shows cached results |
| D3.js chart performance at 100+ frames | Low | Low | Canvas rendering fallback; virtualize visible range |
| VLM generates unprofessional output | Low | Medium | Validate output; regex sanitize; fallback to structured template |

---

*Execution plan finalized. Architecture frozen. Ready to implement.*  
*Next step: Resolve Pre-Implementation Decisions D1–D8, then begin DAY 1 execution.*
