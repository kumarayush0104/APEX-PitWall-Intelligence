/**
 * APEX Component — Tyre Recommendation & Strategy Action Panel
 */

class RecommendationPanel {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
  }

  update(recommendation, metrics, temporal) {
    if (!this.container) return;
    if (!recommendation) {
      this.container.innerHTML = '<div style="color:var(--text-muted); padding:12px;">Awaiting strategy calculation...</div>';
      return;
    }

    const compound = recommendation.compound || "MEDIUM";
    const priority = recommendation.priority || (metrics?.wetness_index > 0.7 ? "CRITICAL" : metrics?.wetness_index > 0.3 ? "HIGH" : "MONITOR");
    const delta = recommendation.lap_delta_seconds ?? 0.0;
    const pitWindow = recommendation.pit_window_open ? "PIT WINDOW OPEN" : "STINT MAINTAINED";
    const pitWindowColor = recommendation.pit_window_open ? "var(--status-danger)" : "var(--status-safe)";

    this.container.innerHTML = `
      <div class="strategy-card">
        <div class="strategy-compound-header">
          <span class="compound-badge ${compound}">${compound}</span>
          <span class="priority-tag ${priority}">${priority}</span>
        </div>

        <div style="font-size:12px; font-weight:600; color:var(--text-primary); margin-top:4px;">
          ${recommendation.reasoning || "Optimal tyre compound selected based on vision friction estimates."}
        </div>

        <div class="strategy-metric-row">
          <div class="metric-stat-box">
            <div class="stat-label">LAP TIME DELTA</div>
            <div class="stat-value" style="color:${delta > 0 ? 'var(--status-danger)' : 'var(--status-safe)'};">
              ${delta > 0 ? '+' + delta.toFixed(1) + 's/lap' : '0.0s (OPTIMAL)'}
            </div>
          </div>

          <div class="metric-stat-box">
            <div class="stat-label">PIT STATUS</div>
            <div class="stat-value" style="color:${pitWindowColor}; font-size:11px;">
              ${pitWindow}
            </div>
          </div>
        </div>

        ${recommendation.alternative_compound ? `
          <div style="margin-top:6px; padding:6px 8px; background:var(--bg-secondary); border-radius:4px; border:1px dashed var(--border-strong); display:flex; justify-content:space-between; align-items:center; font-size:11px;">
            <span style="color:var(--text-secondary);">Gamble Alternative:</span>
            <span style="font-family:var(--font-data); font-weight:700; color:var(--accent-cyan);">${recommendation.alternative_compound}</span>
          </div>
        ` : ''}
      </div>
    `;
  }
}

window.RecommendationPanel = RecommendationPanel;
