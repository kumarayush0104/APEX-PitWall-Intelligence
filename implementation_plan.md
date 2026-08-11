# PitWall Intelligence — Technical Design Document
### AI-Powered Track Condition Analysis & Tyre Strategy Engine

> **Classification:** Principal Engineering Design Document  
> **Status:** Awaiting Review & Approval Before Implementation  
> **Version:** 1.0 — Pre-Implementation

---

## Executive Summary

**PitWall Intelligence** is a production-quality, AI-powered system that ingests images or video frames of a race track and produces a continuously evolving, temporally-aware, explainable model of track conditions. It does not merely classify images. It reasons about *grip physics*, *track evolution dynamics*, *tyre crossover windows*, and *risk-adjusted strategy recommendations* — exactly as a Formula 1 race engineer does.

This document presents a rigorous comparison of multiple architectures, a justified final design choice, and a complete implementation roadmap.

---

## 1. Domain Research Synthesis

### 1.1 The Physics of Track Conditions

Real F1 engineers do not think in buckets of "wet / damp / dry." They think in terms of:

| Concept | Engineering Meaning |
|---|---|
| **Grip coefficient (μ)** | The ratio of friction force to normal force. Wet asphalt: ~0.3μ. Rubbered-in dry: ~1.0μ. |
| **Track Evolution Rate** | How quickly rubber is being deposited per lap. Highest during the first 10–20 laps of a dry session. |
| **Crossover Point** | The exact lap-time delta (typically 10–12%) at which switching tyres produces a net advantage. |
| **Green Track** | Post-rain, freshly washed track with no rubber. Maximum danger. Grip is unpredictable. |
| **Rubbered-In Line** | Dark, stained band of asphalt with polymer molecular adhesion. Most grip when dry. Most dangerous when wet (polished rubber acts like ice). |
| **Tyre Delta** | The per-lap time cost of being on the wrong tyre compound. Intermediates on a dry track lose ~1.5–2.0 s/lap. Slicks on standing water = race-ending accident. |

### 1.2 Visual Signatures of Track Conditions

This is what our AI system must learn to see:

**Fully Wet:**
- High-luminance specular reflections across the full track surface
- Visible standing water / puddles (dark, mirror-like regions)
- Spray rooster-tails visible from onboard cameras
- No visible dark rubber racing line (it has been washed away)
- Drain flows visible on track edges

**Drying / Transitional (Most Critical State):**
- Patchy surface: high-contrast alternating wet/dry zones
- The rubbered racing line appears *darker* and *less reflective* as it dries first
- The off-line areas may remain wet/specular
- Strong Shadows / Sun patches create spatial variation
- This state is the highest-value, highest-risk decision window

**Damp:**
- Low overall reflectivity, track appears uniformly darker
- No standing water visible
- Racing line visible as darkest band
- Track at ~30–50% recovery

**Fully Dry / Rubbered-In:**
- Distinct dark racing line visible (high rubber deposit)
- Off-line areas lighter / more abrasive-looking
- No reflections / specular highlights
- High visual contrast between racing line and the rest

**"Marbles" Zone:**
- Scattered high-brightness particulate matter off the racing line
- Textured appearance at edges
- Indicates rubber rolloff — could improve or worsen conditions for specific manoeuvres

### 1.3 What Real F1 Teams Monitor

Real teams use:
- **Pirelli Wet Weather Indicator** (lap-time vs. compound telemetry)
- **ATM2 Weather RADAR integration** (rain fronts, precipitation probability, cloud speed)
- **Track sensors** (embedded temperature / humidity sensors at specific marshal posts)
- **Driver voice feedback** (subjective but irreplaceable — especially for "it feels like it's coming")
- **onboard camera analysis** (spray visibility, aquaplaning moments)
- **Fan/vent system** for heated tyre blanket management during pit window

**PitWall Intelligence will be the AI layer that automates the visual + temporal reasoning component**, which is currently done manually by engineers watching monitors in real time.

---

## 2. Problem Decomposition

The system must answer five engineering questions simultaneously:

```
Q1: WHAT ARE CURRENT CONDITIONS?        → Computer Vision Classification + Segmentation
Q2: HOW ARE CONDITIONS EVOLVING?        → Temporal Analysis, Trend Modelling
Q3: WHERE EXACTLY IS THE TRACK WET?    → Spatial Segmentation + Heatmap
Q4: WHAT SHOULD THE ENGINEER DO?        → Recommendation Engine
Q5: WHY? (EVIDENCE?)                    → Explainable AI + Visual Attention
```

---

## 3. Architecture Options — Comparative Analysis

### Option A: Single-Model Classifier (❌ Rejected — Obvious)

**Design:** ResNet/EfficientNet fine-tuned on 4 classes (dry/damp/wet/drying).  
**Why 180 teams will build this:** It's the first approach everyone thinks of.  
**Weaknesses:**
- No temporal reasoning — each frame is independent
- No spatial understanding — doesn't know WHERE it's wet
- No confidence calibration
- No explainability
- No recommendation engine
- Stateless — can't track evolution

**Verdict: ❌ Disqualified. This is exactly what we must not build.**

---

### Option B: Segmentation + Rule-Based Recommender (⚠️ Partial)

**Design:** SegFormer for pixel-level segmentation → hand-crafted rule-based recommender.  
**Improvements over A:** Spatial understanding, visual heatmap.  
**Weaknesses:**
- Rule-based recommender is brittle and overfit to expected scenarios
- Still stateless / no temporal memory
- Rules cannot capture nuance of "drying quickly" vs. "drying slowly"

**Verdict: ⚠️ Better but still insufficient.**

---

### Option C: Multi-Stage AI Pipeline with Temporal Memory (✅ Our Target)

**Design:** A layered system where:
1. A **perception layer** extracts visual features and segments the track surface
2. A **temporal reasoning layer** maintains a sliding-window history of track states
3. A **physics-informed fusion layer** combines visual evidence with track evolution physics
4. A **Bayesian recommendation engine** produces calibrated, confidence-weighted strategy recommendations
5. A **VLM explainability layer** generates natural-language justifications

**Strengths:**
- Understands evolution, not just state
- Explainable at every layer
- Confidence-calibrated
- Can forecast future conditions
- Production-grade architecture

**Verdict: ✅ This is our architecture.**

---

### Option D: End-to-End Video Transformer (❌ Infeasible for hackathon)

**Design:** A single VideoMAE or Video-LLaMA model trained end-to-end on labeled F1 video sequences.  
**Weaknesses:**
- Requires enormous labeled motorsport video dataset (doesn't exist publicly)
- Training time infeasible for a hackathon
- Black box — very hard to explain

**Verdict: ❌ Architecturally interesting but not buildable here.**

---

## 4. Chosen Architecture: APEX (Adaptive Perception & Evolution eXplainer)

> **Name: APEX — Adaptive Perception & Evolution eXplainer**  
> A multi-stage AI pipeline purpose-built to reason about track evolution.

### 4.1 System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        APEX SYSTEM ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   INPUT LAYER                                                           │
│   ┌──────────────┐    ┌──────────────┐    ┌────────────────────────┐   │
│   │  Image/Frame │    │  Video Clip  │    │  Weather API (optional) │  │
│   └──────┬───────┘    └──────┬───────┘    └───────────┬────────────┘   │
│          │                   │                         │                │
│          └──────────┬────────┘                         │                │
│                     ▼                                  │                │
│   STAGE 1: PERCEPTION LAYER                           │                │
│   ┌────────────────────────────────────────┐          │                │
│   │  DINOv2 Feature Extractor              │          │                │
│   │  (Self-supervised ViT-B/14 backbone)   │          │                │
│   │  → Rich semantic patch embeddings      │          │                │
│   ├────────────────────────────────────────┤          │                │
│   │  SegFormer-B2 (Fine-tuned)            │          │                │
│   │  → Pixel-level condition segmentation │          │                │
│   │  → Classes: wet / damp / dry /        │          │                │
│   │    rubber_line / puddle / marble      │          │                │
│   ├────────────────────────────────────────┤          │                │
│   │  CLIP Zero-Shot Scorer                │          │                │
│   │  → Cross-validates segmentation       │          │                │
│   │  → Provides alternative evidence      │          │                │
│   └──────────────────┬─────────────────────┘          │                │
│                      │                                 │                │
│   STAGE 2: TEMPORAL REASONING LAYER                   │                │
│   ┌──────────────────▼──────────────────────────────┐ │                │
│   │  Sliding Window State Buffer (N=8 frames)       │ │                │
│   │  → Per-frame condition vectors                  │ │                │
│   │  → Grip recovery rate (dμ/dt estimate)          │ │                │
│   │  → Evolution trend classifier                   │ │                │
│   │  → LSTM / Temporal Attention over state history │ │                │
│   └──────────────────┬──────────────────────────────┘ │                │
│                      │                                 │                │
│   STAGE 3: FUSION & PHYSICS LAYER                     │                │
│   ┌──────────────────▼──────────────────────────────┐ │                │
│   │  Physics-Informed State Estimator               ◄─┘                │
│   │  → Estimates μ (grip coefficient)               │                  │
│   │  → Estimates drying rate from visual delta      │                  │
│   │  → Bayesian update with weather priors          │                  │
│   │  → Conformal Prediction for uncertainty bounds  │                  │
│   └──────────────────┬──────────────────────────────┘                  │
│                      │                                                  │
│   STAGE 4: RECOMMENDATION ENGINE                                        │
│   ┌──────────────────▼──────────────────────────────┐                  │
│   │  Strategy Reasoner                              │                  │
│   │  → Tyre crossover window estimation             │                  │
│   │  → Risk-adjusted pit window scoring             │                  │
│   │  → Primary recommendation + alternative         │                  │
│   │  → Confidence level + evidence list             │                  │
│   └──────────────────┬──────────────────────────────┘                  │
│                      │                                                  │
│   STAGE 5: EXPLAINABILITY LAYER                                         │
│   ┌──────────────────▼──────────────────────────────┐                  │
│   │  VLM Narrator (InternVL-2 or Qwen2-VL)         │                  │
│   │  → Takes: segmentation heatmap + state summary  │                  │
│   │  → Generates: Engineer-style natural language   │                  │
│   │    explanation of recommendation + evidence     │                  │
│   ├──────────────────────────────────────────────── ┤                  │
│   │  Attention Maps (DINOv2 Self-Attention)         │                  │
│   │  → Visual heatmap of what AI is attending to   │                  │
│   └──────────────────────────────────────────────── ┘                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 AI Model Stack (Hugging Face-Centric)

| Stage | Model | HuggingFace ID | Justification |
|---|---|---|---|
| Feature Extraction | DINOv2 ViT-B/14 | `facebook/dinov2-base` | Self-supervised, powerful semantic patches, no labels needed. Best zero-shot scene descriptor available. |
| Semantic Segmentation | SegFormer-B2 | `nvidia/segformer-b2-finetuned-cityscapes-1024-1024` | Efficient transformer, strong on road scenes, fine-tunable with limited data |
| CLIP Cross-Validation | CLIP ViT-L/14 | `openai/clip-vit-large-patch14` | Zero-shot condition scoring, generates text-similarity confidence for "wet track", "standing water", etc. |
| Temporal Reasoning | Custom LSTM over SegFormer outputs | Trained in-session | Lightweight temporal model trained on synthetic evolution sequences |
| VLM Explanation | Qwen2-VL-7B-Instruct | `Qwen/Qwen2-VL-7B-Instruct` | State-of-the-art open-source VLM. Excellent at structured scene analysis and report generation. |
| Attention XAI | DINOv2 self-attention | (extracted from backbone) | Transformer attention heads naturally map to semantically meaningful regions without GradCAM |

### 4.3 Why This Model Combination Wins

- **DINOv2** is trained via self-distillation on 142M images — it has an exceptional understanding of surface textures, reflectance, and scene geometry without any labels.
- **SegFormer** is purpose-built for road scene understanding (trained on Cityscapes, which includes rain/wet conditions).
- **CLIP** allows zero-shot verification: we can score any frame against prompts like "track with standing water", "dry asphalt racing line", "wet slippery corner" and get a calibrated probability vector — no training needed.
- **Qwen2-VL** enables generating real engineer-language explanations — not pre-written templates, but genuine reasoning from visual evidence.
- The **combination** means we have redundant evidence: if SegFormer says "wet" and CLIP says "dry", we flag high uncertainty — this is critical for engineering trust.

---

## 5. Track Condition State Machine

Rather than free-form classification, APEX uses a structured physics-aware state machine:

```
States:
  WET_SEVERE        → μ ≈ 0.25–0.35  |  Full wets mandatory
  WET_MODERATE      → μ ≈ 0.35–0.50  |  Full wets preferred / inters viable  
  TRANSITIONAL      → μ ≈ 0.50–0.70  |  CRITICAL WINDOW — highest decision value
  DRYING            → μ ≈ 0.60–0.80  |  Intermediates optimal, slick gamble imminent
  DRY_GREEN         → μ ≈ 0.75–0.90  |  Slicks on, track still rubbering in
  DRY_EVOLVED       → μ ≈ 0.90–1.05  |  Full grip, normal strategy

Transitions are governed by:
  - Wet pixel ratio (from segmentation)
  - Reflectance score (from DINOv2 features)
  - Historical trend (from temporal layer)
  - Time since last rain event (temporal prior)
```

**The TRANSITIONAL state is the crown jewel of the system.** This is where the most pit stop decisions are made, and where AI assistance is most valuable.

---

## 6. Grip Coefficient Estimation

Rather than reporting a meaningless "wetness percentage", APEX estimates the **relative grip coefficient (μ̂)** using a composite model:

```
μ̂ = w₁ × (1 - wet_pixel_ratio) 
   + w₂ × (1 - specular_intensity) 
   + w₃ × rubber_line_coverage 
   + w₄ × temporal_drying_rate_bonus
   
where:
  wet_pixel_ratio       = fraction of track pixels classified as wet/puddle
  specular_intensity    = mean luminance of specular highlights detected
  rubber_line_coverage  = fraction of expected racing line that is visibly dark/dry
  temporal_drying_rate_bonus = rate of positive μ change over last N frames

Normalized to [0, 1] and mapped to μ scale [0.25, 1.05]
```

This is a novel contribution — converting visual features directly into engineering units that a race engineer understands.

---

## 7. Temporal Evolution Engine

The sliding window maintains:

```python
TemporalState = {
    "timestamp": float,
    "grip_estimate": float,          # μ̂
    "condition_state": str,          # state machine label
    "wet_pixel_ratio": float,        # [0, 1]
    "confidence": float,             # model confidence
    "drying_rate": float,            # Δμ/frame — positive = improving
    "racing_line_dry_pct": float,    # % of racing line that is dry
    "puddle_count": int,             # detected puddles
    "segments": np.ndarray,          # segmentation mask
}
```

The LSTM-over-states model learns to:
1. Smooth noisy predictions (reject outliers)
2. Estimate how many frames until crossover
3. Distinguish "steady dry" from "drying rapidly" from "getting wetter"
4. Flag potential condition reversals (sudden shower detection)

---

## 8. Recommendation Engine

The recommendation engine does NOT use if-else logic. It uses a **scoring function** with confidence propagation:

```
CrossoverScore = f(μ̂, drying_rate, target_compound_μ_threshold)

PitWindowRisk = severity(current_state) × (1 - confidence)

Recommendation Priority:
  IMMEDIATE   → CrossoverScore > 0.85 AND PitWindowRisk < 0.30
  HIGH        → CrossoverScore > 0.70 AND PitWindowRisk < 0.50
  MONITOR     → CrossoverScore 0.50–0.70 OR drying_rate > 0.02/frame
  HOLD        → CrossoverScore < 0.50 AND condition_state == STABLE
  ABORT       → drying_rate < 0 AND condition_state worsening
```

Each recommendation includes:
- **Primary Action** (e.g., "Pit for intermediates in next 2 laps")
- **Confidence** (e.g., "78% confidence")
- **Key Evidence** (e.g., "Racing line is 65% dry, drying at 3% per frame")
- **Risk Assessment** (e.g., "Off-line sectors still showing standing water")
- **Alternative** (e.g., "If grip improves 15% more, consider soft slick gamble")
- **Forecast** (e.g., "Expected crossover in ~3.5 minutes at current rate")

---

## 9. Explainability Architecture

Every prediction is paired with three forms of evidence:

### 9.1 Visual Evidence (Segmentation Overlay)
A color-coded heatmap overlaid on the original frame:
- 🔵 Blue: Standing water / puddle
- 🟡 Yellow: Damp / transitional
- 🟢 Green: Dry surface
- ⚫ Black: Rubber-in racing line
- 🟠 Orange: Marble accumulation

### 9.2 Attention Evidence (DINOv2 Self-Attention)
DINOv2 attention heads highlight which regions the model finds most semantically salient — this naturally focuses on reflective surfaces, water bodies, and surface texture changes without any additional training.

### 9.3 Natural Language Evidence (VLM)
Qwen2-VL receives: the original frame + segmentation overlay + current state vector, and is prompted to generate an engineer-style summary:

> *"The track camera shows a drying surface with approximately 40% wet coverage remaining. The traditional racing line through Turn 3 and Turn 7 appears dry and rubbered-in. Significant standing water remains in the braking zones approaching Turns 1 and 4. At the current drying rate of 3.2% per minute, slick tyres are not yet viable. Intermediate tyres would perform optimally in these conditions. Recommend monitoring for pit window in 2–3 laps."*

---

## 10. System Architecture (Full Stack)

### 10.1 Backend Architecture

```
backend/
├── app/
│   ├── main.py                     # FastAPI app, CORS, lifespan
│   ├── config.py                   # Pydantic settings, env vars
│   ├── api/
│   │   ├── routes/
│   │   │   ├── analysis.py         # POST /analyze, POST /analyze/video
│   │   │   ├── session.py          # GET /session/{id}, session management
│   │   │   └── websocket.py        # WS /ws/live — real-time streaming
│   │   └── dependencies.py         # DI: model registry, session store
│   ├── core/
│   │   ├── pipeline.py             # Main APEX orchestrator
│   │   ├── perception.py           # DINOv2 + SegFormer + CLIP stage
│   │   ├── temporal.py             # Sliding window + LSTM temporal
│   │   ├── physics.py              # Grip estimation, state machine
│   │   ├── recommender.py          # Strategy recommendation engine
│   │   └── explainer.py            # VLM + attention XAI layer
│   ├── models/
│   │   ├── registry.py             # Model loader / warm-up / cache
│   │   └── schemas.py              # Pydantic request/response models
│   ├── services/
│   │   ├── session_store.py        # In-memory session state (Redis-compatible)
│   │   ├── weather.py              # Optional weather API fusion
│   │   └── video_processor.py      # Frame extraction, preprocessing
│   └── utils/
│       ├── image_utils.py
│       ├── visualization.py        # Heatmap generation, overlay composition
│       └── calibration.py          # Conformal prediction intervals
├── tests/
├── Dockerfile
└── requirements.txt
```

### 10.2 Key API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/analyze/image` | Single image analysis → full APEX pipeline result |
| `POST` | `/api/v1/analyze/video` | Video upload → async processing, returns session_id |
| `GET` | `/api/v1/session/{id}` | Get current session state + history |
| `WS` | `/ws/live` | WebSocket for real-time frame streaming |
| `GET` | `/api/v1/session/{id}/timeline` | Full temporal evolution timeline |
| `GET` | `/api/v1/session/{id}/recommendation` | Current strategy recommendation |
| `GET` | `/api/v1/health` | Health + model status |

### 10.3 Frontend Architecture

```
frontend/
├── index.html                      # Entry point
├── src/
│   ├── main.js                     # App bootstrap
│   ├── styles/
│   │   ├── base.css                # Design tokens, typography
│   │   ├── components.css          # Component styles
│   │   └── animations.css          # Keyframes, transitions
│   ├── components/
│   │   ├── CommandCenter.js        # Main layout controller
│   │   ├── TrackRadar.js           # The segmentation overlay canvas
│   │   ├── GripTimeline.js         # Temporal evolution chart (D3.js)
│   │   ├── RecommendationPanel.js  # Strategy recommendation card
│   │   ├── EvidencePanel.js        # XAI evidence display
│   │   ├── ConditionGauge.js       # Grip coefficient circular gauge
│   │   ├── UploadZone.js           # Drag-and-drop media input
│   │   └── LiveStream.js           # WebSocket frame streaming
│   ├── services/
│   │   ├── api.js                  # API client
│   │   └── websocket.js            # WS connection manager
│   └── utils/
│       ├── canvas.js               # Canvas overlay utilities
│       └── format.js               # Display formatting
```

### 10.4 Frontend Design Language

The UI must feel like professional motorsport telemetry software. Key design decisions:

- **Color Palette:** Near-black `#0A0B0D` background, accent `#FF1E00` (F1 red), data-green `#00C853`, warning-amber `#FF9800`, muted slate `#1A1D24`
- **Typography:** `JetBrains Mono` for data values, `Inter` for UI text — both from Google Fonts
- **Layout:** Three-column command center. Left: media input + condition state. Center: track analysis canvas with segmentation overlay. Right: recommendations + timeline
- **Key animations:** Grip gauge pulse on state change, recommendation card slide-in, attention heatmap fade transition
- **NO decorative animations** — every animation must convey data meaning

### 10.5 Signature UI Screens

1. **Command Center (Main):** The primary interface showing live analysis
2. **Track Evolution Timeline:** D3.js chart showing μ over time with annotated events
3. **Evidence Panel:** Side-by-side: original frame, segmentation overlay, attention map
4. **Recommendation Card:** Priority banner, confidence indicator, evidence list, forecast
5. **Session Report:** Post-session summary with full timeline and key decision moments

---

## 11. Data Strategy

### 11.1 No Perfect Dataset Exists — Our Strategy

There is no public dataset of F1 track condition images labeled with wet/dry/damp at pixel level. This is actually our advantage — it means no team has a fine-tuned model that's significantly better than ours. Our strategy:

**Tier 1: Zero-shot foundation (no training needed)**
- DINOv2 features are powerful enough for good zero-shot classification
- CLIP zero-shot scoring requires no fine-tuning
- These work out-of-box on first demo

**Tier 2: Transfer from road condition datasets**
- `nvidia/segformer-b2-finetuned-cityscapes-1024-1024` — already understands road/water/road markings
- `StreetSurfaceVis` dataset (HuggingFace) for surface texture understanding
- Road puddle detection papers' approaches (AIWD16 dataset)

**Tier 3: Synthetic augmentation**
- Generate training examples by augmenting road images with rain simulation, puddle overlays, wet reflections using Albumentations
- Use stable diffusion in-painting to create "partially wet track" training images

**Tier 4: CLIP-guided pseudo-labeling**
- Use CLIP to pseudo-label unlabeled road/track images from YouTube/internet
- Fine-tune SegFormer on pseudo-labeled data

### 11.2 HuggingFace Resources Used

| Resource | HF ID | Usage |
|---|---|---|
| DINOv2 model | `facebook/dinov2-base` | Feature backbone |
| SegFormer | `nvidia/segformer-b2-finetuned-cityscapes-1024-1024` | Segmentation |
| CLIP | `openai/clip-vit-large-patch14` | Zero-shot scoring |
| Qwen2-VL | `Qwen/Qwen2-VL-7B-Instruct` | VLM explanation (via Inference API) |
| StreetSurfaceVis | `your-username/streetsurfacevis` | Optional fine-tuning data |

---

## 12. Confidence & Uncertainty

Every output includes a **calibrated confidence interval** using:

1. **Multi-source agreement:** If DINOv2 classification, SegFormer segmentation, and CLIP scoring all agree → high confidence. If they disagree → flag uncertainty, reduce recommendation strength.

2. **Temporal consistency:** If current frame disagrees with trend of last 5 frames → flag as potential outlier or rapid change event.

3. **Entropy of segmentation distribution:** If the segmentation output has very low cross-class probability (clear), confidence is high. If it's diffuse → flag uncertainty.

4. **Risk-weighted output:** High uncertainty + high-stakes recommendation (e.g., "pit NOW for slicks") → system automatically elevates to "MONITOR" rather than "IMMEDIATE" to avoid false positives.

---

## 13. Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     DEPLOYMENT TOPOLOGY                          │
│                                                                   │
│   User Browser ────► Nginx/Caddy (reverse proxy)                │
│                              │                                   │
│                    ┌─────────┴──────────┐                        │
│                    │                    │                        │
│              Static Files         FastAPI Backend                │
│              (Frontend)           (Python 3.11)                  │
│                                        │                         │
│                            ┌───────────┼──────────────┐          │
│                            │           │              │           │
│                       Redis Cache   SQLite/        HF Models    │
│                       (sessions,    Postgres       (loaded in   │
│                        pub/sub)     (timeline)      memory)     │
│                                                                   │
│   Optional: HF Inference API for Qwen2-VL (if GPU unavailable)  │
└─────────────────────────────────────────────────────────────────┘
```

**Docker-first deployment:**
- `docker-compose.yml` with: `backend`, `frontend` (nginx), `redis` services
- `.env` configuration for HF tokens, model sizes, etc.
- GPU passthrough support for NVIDIA models

---

## 14. Innovation Highlights (What Judges Will Remember)

1. **Grip Coefficient Estimation (μ̂):** Converting visual features into real engineering units. No team will do this.

2. **TRANSITIONAL State Intelligence:** Special handling of the drying phase — this is where F1 races are won and lost, and our system specifically optimizes for this decision window.

3. **Multi-model Cross-validation with Uncertainty Propagation:** Three independent AI models (DINOv2, SegFormer, CLIP) must agree before a recommendation is made with high confidence.

4. **Physics-Informed Fusion:** The system uses actual grip coefficient physics, not arbitrary "wetness scores."

5. **VLM-Generated Engineer Explanations:** Qwen2-VL generates real-language explanations that sound like they came from a race engineer — not template strings.

6. **Temporal Grip Recovery Rate (dμ/dt):** The system reports HOW FAST conditions are improving, not just WHERE they are now — enabling forward-looking strategy.

7. **Crossover Window Forecasting:** "Switch to slicks in approximately 3.5 minutes" — a forecast, not just a current-state label.

---

## 15. Potential Weaknesses & Mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| No motorsport-specific training data | High | Zero-shot CLIP + DINOv2 work well out-of-box; synthetic augmentation fills gaps |
| Qwen2-VL inference too slow | Medium | Cache explanations; use HF Inference API; fallback to template + structured output |
| SegFormer not trained on race tracks | Medium | Fine-tune with synthetic data + pseudo-labels; CLIP provides cross-validation |
| Temporal LSTM overfitting | Low | Simple LSTM over engineered features, not raw pixels; robust to small data |
| WebSocket connection instability | Low | Reconnection logic + HTTP polling fallback |
| GPU not available at demo | Medium | CPU inference mode; smaller model variants; HF Inference API for VLM |

---

## 16. Open Questions for Review

> [!IMPORTANT]
> **Q1: VLM Approach** — Should we use Qwen2-VL-7B self-hosted (requires GPU/16GB RAM) or call the HuggingFace Inference API (requires HF token, adds latency)? The self-hosted version gives better demo performance but requires hardware.

> [!IMPORTANT]
> **Q2: Demo Mode** — Should we build a "Demo Mode" with pre-processed F1 race clips (e.g., 2021 Belgian GP wet race, 2016 Monaco wet, etc.) so judges can see the full temporal evolution without uploading media? This would be much more impressive than a cold-start demo.

> [!NOTE]
> **Q3: Weather API Integration** — Should we integrate a real weather API (OpenWeatherMap free tier) to automatically pull ambient conditions based on user-provided circuit name? This adds realism but minor complexity.

> [!NOTE]
> **Q4: Mobile Responsiveness** — The telemetry-style layout is inherently desktop-oriented (three columns). Should we build a simplified mobile view, or focus entirely on desktop excellence?

---

## 17. Implementation Phases

### Phase 1: Foundation (Backend Core)
- Project structure setup
- Model loading and inference pipeline (DINOv2 + SegFormer + CLIP)
- Basic FastAPI endpoints
- Heatmap generation

### Phase 2: Intelligence Layer
- Grip coefficient estimation
- State machine implementation
- Temporal sliding window
- Recommendation engine

### Phase 3: Explainability
- DINOv2 attention visualization
- VLM integration (Qwen2-VL or HF API)
- Structured evidence generation

### Phase 4: Frontend
- Design system and tokens
- Command Center layout
- Track analysis canvas (segmentation overlay)
- Grip timeline chart (D3.js)
- Recommendation panel
- Evidence panel

### Phase 5: Integration & Polish
- WebSocket real-time streaming
- Video processing pipeline
- Demo mode with pre-loaded clips
- Docker deployment
- README + API docs

---

## 18. Success Criteria

The demo must make judges think:
1. ✅ "This team deeply understands Formula 1 strategy"
2. ✅ "This AI explains itself — I can trust it"
3. ✅ "This interface looks like real racing software"
4. ✅ "They thought of things we didn't expect"
5. ✅ "I want to see this deployed at a real race"

The first 30 seconds of the demo should show:
→ Upload or select a wet/drying race footage clip  
→ The segmentation heatmap appears immediately over the frame  
→ The grip gauge shows 0.42μ with a "DRYING — TRANSITIONAL" state  
→ The recommendation card reads "PIT WINDOW: 2–3 LAPS | HIGH CONFIDENCE 81%"  
→ The VLM explanation text types out: *"Standing water detected in braking zones..."*  
→ The timeline shows the grip recovery curve rising over 8 frames  

---

*Document prepared by: APEX Design Team*  
*Ready for engineering approval.*
