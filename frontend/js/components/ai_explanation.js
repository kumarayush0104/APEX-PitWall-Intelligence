/**
 * APEX Component — Race Engineer AI Rationale & Visual Evidence Panel
 */

class AIExplanation {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
  }

  update(explainabilityData) {
    if (!this.container) return;
    if (!explainabilityData) {
      this.container.innerHTML = '<div style="color:var(--text-muted); padding:12px;">Awaiting AI reasoning generation...</div>';
      return;
    }

    const headline = explainabilityData.headline || "TRACK ANALYSIS COMPLETE";
    const summary = explainabilityData.detailed_summary || explainabilityData.headline || "";
    const risk = explainabilityData.risk_assessment || "";
    const action = explainabilityData.recommended_action || "";
    const keyFactors = explainabilityData.key_factors || [];

    this.container.innerHTML = `
      <div class="ai-summary-box">
        <div class="ai-header-row">
          <span>RACE ENGINEER RATIONALE</span>
          <span class="ai-badge">VLM / APEX V2</span>
        </div>

        <div style="font-family:var(--font-data); font-size:12px; font-weight:800; color:var(--text-primary); text-transform:uppercase;">
          ${headline}
        </div>

        <div style="color:var(--text-primary); font-size:11px; line-height:1.5;">
          ${summary}
        </div>

        ${risk ? `
          <div style="background:rgba(255, 30, 0, 0.1); border-left:3px solid var(--accent-f1); padding:6px 8px; border-radius:3px; font-size:11px; color:var(--text-primary);">
            <strong style="color:var(--accent-f1);">RISK ASSESSMENT:</strong> ${risk}
          </div>
        ` : ''}

        ${action ? `
          <div style="background:rgba(0, 230, 118, 0.1); border-left:3px solid var(--status-safe); padding:6px 8px; border-radius:3px; font-size:11px; color:var(--text-primary);">
            <strong style="color:var(--status-safe);">PIT WALL ACTION:</strong> ${action}
          </div>
        ` : ''}

        <div style="margin-top:4px; font-size:10px; font-weight:700; color:var(--text-secondary); text-transform:uppercase; letter-spacing:0.5px;">
          Visual Evidence Factors
        </div>

        <div class="evidence-list">
          ${keyFactors.length > 0 ? keyFactors.map(kf => `
            <div class="evidence-item">
              <div style="flex:1;">
                <div style="font-weight:700; color:var(--accent-cyan);">${kf.factor || kf.category}</div>
                <div style="color:var(--text-secondary); font-size:10px;">${kf.description || ''}</div>
              </div>
            </div>
          `).join('') : `
            <div class="evidence-item">
              <div style="font-size:10px; color:var(--text-secondary);">Perception engine cross-validated DINOv2 self-attention map & SegFormer surface classes.</div>
            </div>
          `}
        </div>
      </div>
    `;
  }
}

window.AIExplanation = AIExplanation;
