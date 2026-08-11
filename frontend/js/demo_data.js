/**
 * APEX PitWall Intelligence — Demo Data & Scenario Engine
 * Pre-computed high-fidelity Formula 1 race scenarios for instant judge demonstration.
 */

window.APEX_DEMO_SCENARIOS = {
  belgian_gp: {
    id: "belgian_gp",
    name: "Belgian GP — Spa (Flooded)",
    circuit: "Circuit de Spa-Francorchamps",
    location: "Stavelot, Belgium",
    flag: "🇧🇪",
    weather: "HEAVY RAIN / STANDING WATER",
    track_temp: "14.2 °C",
    air_temp: "12.5 °C",
    frame_index: 42,
    timestamp: new Date().toISOString(),
    image_hash: "spa_wet_f42_9a8b7c",
    track_condition: "FLOODED",
    metrics: {
      wetness_index: 0.884,
      puddle_coverage_pct: 34.2,
      wet_surface_pct: 48.6,
      damp_surface_pct: 12.1,
      dry_surface_pct: 5.1,
      rubber_ratio: 0.08,
      clip_wet_confidence: 0.94,
      clip_dry_confidence: 0.06,
      dominant_condition: "FLOODED"
    },
    tyre_recommendation: {
      compound: "WET",
      confidence: 0.96,
      lap_delta_seconds: 0.0,
      reasoning: "Severe standing water at Eau Rouge braking zone (>30% puddle coverage). Full Wet compound required to prevent aquaplaning.",
      pit_window_open: true,
      alternative_compound: "INTERMEDIATE"
    },
    temporal_analysis: {
      trend: "WETTING",
      volatility: "HIGH",
      momentum_slope: 0.012,
      projected_wetness_in_5: 0.94,
      stability_frames: 18,
      tyre_window_alert: {
        alert_active: true,
        message: "CRITICAL: Heavy shower intensification in Sector 2. Box immediately for Full Wets.",
        from_compound: "INTERMEDIATE",
        to_compound: "WET"
      }
    },
    explainability: {
      headline: "AQUAPLANING RISK AT TURN 3 — FULL WET MANDATED",
      detailed_summary: "Perception engines detect 34.2% standing water coverage with deep pooling near Turn 3 (Raidillon entry). DINOv2 self-attention maps highlight heavy surface reflection and water displacement channels along the racing line. Tyre crossover score for WET compound is 0.96.",
      risk_assessment: "Extremely high risk of aquaplaning on Intermediate compound. Pitting now yields estimated +4.2s/lap advantage over dry/inter gambles.",
      recommended_action: "BOX THIS LAP for Full Wet tyres. Increase tire pressures by +1.5 PSI.",
      key_factors: [
        { category: "VISUAL", factor: "Standing Water", impact: "HIGH_RISK", description: "Deep puddles (>5mm depth) detected across 34% of track area." },
        { category: "DYNAMIC", factor: "Track Friction (μ)", impact: "CRITICAL", description: "Grip estimate μ = 0.28 (72% reduction vs dry baseline)." },
        { category: "TEMPORAL", factor: "Rain Intensity", impact: "HIGH_RISK", description: "Wetness index increasing at +0.012/frame." }
      ]
    },
    sector_risk: {
      sector1: { risk: "CRITICAL", wetness: 0.91, label: "Eau Rouge / Raidillon (Flooded)" },
      sector2: { risk: "HIGH", wetness: 0.84, label: "Les Combes to Stavelot (Wet)" },
      sector3: { risk: "HIGH", wetness: 0.89, label: "Blanchimont / Bus Stop (Flooded)" }
    },
    visualization: {
      class_legend: {
        "puddle": "#0064FF",
        "wet": "#00A0FF",
        "damp": "#FFC800",
        "dry": "#00DC64",
        "rubber": "#1E1E1E",
        "marbles": "#FF8C00"
      },
      dimensions: { width: 1280, height: 720 }
    }
  },

  monaco_gp: {
    id: "monaco_gp",
    name: "Monaco GP — Monte Carlo (Drying Line)",
    circuit: "Circuit de Monaco",
    location: "Monte Carlo, Monaco",
    flag: "🇲🇨",
    weather: "OVERCAST / TRACK DRYING",
    track_temp: "22.8 °C",
    air_temp: "19.4 °C",
    frame_index: 88,
    timestamp: new Date().toISOString(),
    image_hash: "monaco_drying_f88_3f2e1a",
    track_condition: "DRYING",
    metrics: {
      wetness_index: 0.385,
      puddle_coverage_pct: 1.2,
      wet_surface_pct: 18.4,
      damp_surface_pct: 42.8,
      dry_surface_pct: 37.6,
      rubber_ratio: 0.28,
      clip_wet_confidence: 0.32,
      clip_dry_confidence: 0.68,
      dominant_condition: "DAMP"
    },
    tyre_recommendation: {
      compound: "INTERMEDIATE",
      confidence: 0.84,
      lap_delta_seconds: 1.4,
      reasoning: "Dry racing line emerging through Casino Square and Tunnel exit. Intermediate compound optimal for next 2-3 laps before slick crossover.",
      pit_window_open: true,
      alternative_compound: "MEDIUM"
    },
    temporal_analysis: {
      trend: "DRYING",
      volatility: "MEDIUM",
      momentum_slope: -0.018,
      projected_wetness_in_5: 0.29,
      stability_frames: 24,
      tyre_window_alert: {
        alert_active: true,
        message: "CROSSOVER ALERT: Track drying rapidly. Slick tyre crossover anticipated in 3 laps (Est. Lap 24).",
        from_compound: "INTERMEDIATE",
        to_compound: "MEDIUM"
      }
    },
    explainability: {
      headline: "DRY RACING LINE EMERGING — SLICK CROSSOVER APPROACHING",
      detailed_summary: "SegFormer analysis indicates dry surface area increased to 37.6% along the ideal racing line. Damp patches remain off-line at Turn 1 (Sainte Devote) and Pool Chicane. Wetness trend slope is -0.018/frame, signaling impending slick window.",
      risk_assessment: "Intermediate tyre is currently 1.4s faster than Wets. Transition to Medium Slicks recommended within 3.5 minutes.",
      recommended_action: "Prepare pit crew with Medium Slicks. Monitor dampness at Casino Square apex.",
      key_factors: [
        { category: "VISUAL", factor: "Drying Line", impact: "FAVORABLE", description: "Clear dark dry groove visible along 78% of racing line." },
        { category: "DYNAMIC", factor: "Friction Recovery", impact: "FAVORABLE", description: "Grip μ recovered to 0.62 (Slick crossover threshold = 0.65)." },
        { category: "TEMPORAL", factor: "Drying Rate", impact: "FAVORABLE", description: "Track drying at 1.8% wetness reduction per minute." }
      ]
    },
    sector_risk: {
      sector1: { risk: "MEDIUM", wetness: 0.42, label: "Sainte Devote to Casino (Damp)" },
      sector2: { risk: "LOW", wetness: 0.28, label: "Mirabeau to Tunnel (Drying)" },
      sector3: { risk: "MEDIUM", wetness: 0.45, label: "Chicane to Rascasse (Damp)" }
    },
    visualization: {
      class_legend: {
        "puddle": "#0064FF",
        "wet": "#00A0FF",
        "damp": "#FFC800",
        "dry": "#00DC64",
        "rubber": "#1E1E1E",
        "marbles": "#FF8C00"
      },
      dimensions: { width: 1280, height: 720 }
    }
  },

  silverstone_gp: {
    id: "silverstone_gp",
    name: "Silverstone — Sudden Shower",
    circuit: "Silverstone Circuit",
    location: "Towcester, UK",
    flag: "🇬🇧",
    weather: "SUDDEN RAIN SHOWER",
    track_temp: "18.5 °C",
    air_temp: "15.8 °C",
    frame_index: 104,
    timestamp: new Date().toISOString(),
    image_hash: "silverstone_shower_f104_77c2",
    track_condition: "WET",
    metrics: {
      wetness_index: 0.652,
      puddle_coverage_pct: 12.8,
      wet_surface_pct: 54.2,
      damp_surface_pct: 22.0,
      dry_surface_pct: 11.0,
      rubber_ratio: 0.15,
      clip_wet_confidence: 0.88,
      clip_dry_confidence: 0.12,
      dominant_condition: "WET"
    },
    tyre_recommendation: {
      compound: "INTERMEDIATE",
      confidence: 0.92,
      lap_delta_seconds: 3.8,
      reasoning: "Sudden rain shower hit Sector 2 (Maggotts & Becketts). Slick tyres losing 3.8s/lap. Immediate pit call required for Intermediate compound.",
      pit_window_open: true,
      alternative_compound: "WET"
    },
    temporal_analysis: {
      trend: "WETTING",
      volatility: "HIGH",
      momentum_slope: 0.035,
      projected_wetness_in_5: 0.82,
      stability_frames: 6,
      tyre_window_alert: {
        alert_active: true,
        message: "URGENT: Rapid wetness spike in Sector 2 (+3.5%/frame). Pit for Intermediates NOW.",
        from_compound: "MEDIUM",
        to_compound: "INTERMEDIATE"
      }
    },
    explainability: {
      headline: "SUDDEN SHOWER IN SECTOR 2 — PIT IMMEDIATELY FOR INTERMEDIATES",
      detailed_summary: "Vision models detect rapid rain accumulation at Copse, Maggotts, and Becketts. Water film thickness rapidly exceeding slick operating window. Grip μ dropped from 0.88 to 0.44 in 3 frames.",
      risk_assessment: "High risk of spin/crash on slick tyres in high-speed sweeps. Immediate pit stop avoids estimated 12s lost on track.",
      recommended_action: "BOX THIS LAP for Intermediate tyres.",
      key_factors: [
        { category: "VISUAL", factor: "Surface Gloss / Rain", impact: "HIGH_RISK", description: "Specular reflections detected across Maggotts/Becketts complex." },
        { category: "DYNAMIC", factor: "Friction Drop", impact: "CRITICAL", description: "Grip μ reduced by 50% in 15 seconds." },
        { category: "TEMPORAL", factor: "Rain Front Arrival", impact: "HIGH_RISK", description: "Localized shower moving south across Hangar Straight." }
      ]
    },
    sector_risk: {
      sector1: { risk: "MEDIUM", wetness: 0.45, label: "Abbey to Wellington (Damp)" },
      sector2: { risk: "CRITICAL", wetness: 0.82, label: "Copse to Chapel (Heavy Wet)" },
      sector3: { risk: "HIGH", wetness: 0.68, label: "Stowe to Club (Wet)" }
    },
    visualization: {
      class_legend: {
        "puddle": "#0064FF",
        "wet": "#00A0FF",
        "damp": "#FFC800",
        "dry": "#00DC64",
        "rubber": "#1E1E1E",
        "marbles": "#FF8C00"
      },
      dimensions: { width: 1280, height: 720 }
    }
  },

  monza_gp: {
    id: "monza_gp",
    name: "Monza — Dry Track Evolution",
    circuit: "Autodromo Nazionale Monza",
    location: "Monza, Italy",
    flag: "🇮🇹",
    weather: "SUNNY / HIGH RUBBERING",
    track_temp: "38.5 °C",
    air_temp: "29.2 °C",
    frame_index: 150,
    timestamp: new Date().toISOString(),
    image_hash: "monza_dry_f150_00aa88",
    track_condition: "DRY",
    metrics: {
      wetness_index: 0.012,
      puddle_coverage_pct: 0.0,
      wet_surface_pct: 0.0,
      damp_surface_pct: 1.2,
      dry_surface_pct: 98.8,
      rubber_ratio: 0.42,
      clip_wet_confidence: 0.01,
      clip_dry_confidence: 0.99,
      dominant_condition: "DRY"
    },
    tyre_recommendation: {
      compound: "SOFT",
      confidence: 0.98,
      lap_delta_seconds: 0.0,
      reasoning: "Fully dry track with high rubber density (42% rubber ratio). Soft compound optimal for qualifying stint or short race stint.",
      pit_window_open: false,
      alternative_compound: "MEDIUM"
    },
    temporal_analysis: {
      trend: "STABLE",
      volatility: "LOW",
      momentum_slope: 0.000,
      projected_wetness_in_5: 0.01,
      stability_frames: 120,
      tyre_window_alert: {
        alert_active: false,
        message: "Conditions optimal. Rubber buildup providing maximum grip (μ = 0.92).",
        from_compound: null,
        to_compound: null
      }
    },
    explainability: {
      headline: "OPTIMAL DRY TRACK — MAXIMUM RUBBER BUILDOUT ON RACING LINE",
      detailed_summary: "SegFormer & DINOv2 confirm 98.8% dry surface with dense rubber deposit along Curva Grande and Parabolica entry. Grip μ is at maximum peak (0.92). Soft compound yields minimum lap time.",
      risk_assessment: "Zero wetness risk. Thermal degradation is primary stint limiter.",
      recommended_action: "MAINTAIN CURRENT STINT. Monitor rear tyre core temperature.",
      key_factors: [
        { category: "VISUAL", factor: "Rubber Deposit", impact: "FAVORABLE", description: "Dark rubber groove providing maximum mechanical grip." },
        { category: "DYNAMIC", factor: "Peak Grip", impact: "FAVORABLE", description: "Track friction μ = 0.92 (100% dry baseline)." },
        { category: "TEMPORAL", factor: "Condition Stability", impact: "FAVORABLE", description: "Zero variance over past 120 frames." }
      ]
    },
    sector_risk: {
      sector1: { risk: "SAFE", wetness: 0.01, label: "Prima Variante to Biassono (Dry)" },
      sector2: { risk: "SAFE", wetness: 0.01, label: "Lesmo 1 & 2 to Serraglio (Dry)" },
      sector3: { risk: "SAFE", wetness: 0.01, label: "Ascari to Parabolica (Dry)" }
    },
    visualization: {
      class_legend: {
        "puddle": "#0064FF",
        "wet": "#00A0FF",
        "damp": "#FFC800",
        "dry": "#00DC64",
        "rubber": "#1E1E1E",
        "marbles": "#FF8C00"
      },
      dimensions: { width: 1280, height: 720 }
    }
  }
};
