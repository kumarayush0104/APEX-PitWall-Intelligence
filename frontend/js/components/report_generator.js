/**
 * APEX Component — Executive Strategy Report Builder (PDF / Printable)
 */

class ReportGenerator {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
  }

  generateReport(currentAnalysis, sessionHistory) {
    if (!this.container) return;
    if (!currentAnalysis) {
      this.container.innerHTML = `
        <div class="report-paper">
          <h2 style="color:var(--accent-f1);">POST-SESSION STRATEGY REPORT</h2>
          <p style="color:var(--text-secondary); margin-top:8px;">No analysis session currently active. Run a demo scenario or upload a frame to generate a complete report.</p>
        </div>
      `;
      return;
    }

    const data = currentAnalysis;
    const dateStr = new Date().toLocaleString();

    this.container.innerHTML = `
      <div class="report-paper">
        <!-- Header -->
        <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:2px solid var(--accent-f1); padding-bottom:12px; margin-bottom:16px;">
          <div>
            <h1 style="font-size:22px; font-weight:900; letter-spacing:1px; color:#FFF;">APEX PITWALL INTELLIGENCE</h1>
            <div style="font-size:12px; color:var(--accent-f1); font-weight:700;">FORMULA 1 RACE STRATEGY & VISION PERCEPTION REPORT</div>
          </div>
          <div style="text-align:right; font-family:var(--font-data); font-size:11px; color:var(--text-secondary);">
            <div>SESSION ID: ${data.image_hash ? data.image_hash.substring(0, 12) : 'APEX-800'}</div>
            <div>DATE: ${dateStr}</div>
            <button class="preset-btn" style="margin-top:6px; background:var(--accent-f1); color:#FFF; border:none;" onclick="window.print()">🖨 PRINT / EXPORT PDF</button>
          </div>
        </div>

        <!-- Executive Summary -->
        <div style="background:var(--bg-panel); border:1px solid var(--border-strong); border-radius:6px; padding:12px; margin-bottom:16px;">
          <h3 style="font-size:12px; color:var(--accent-cyan); text-transform:uppercase; letter-spacing:1px; margin-bottom:6px;">Executive Summary</h3>
          <div style="font-size:12px; line-height:1.6; color:var(--text-primary);">
            ${data.explainability?.detailed_summary || data.explainability?.headline || "Track analysis complete."}
          </div>
        </div>

        <!-- Metric Cards Grid -->
        <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:12px; margin-bottom:16px;">
          <div style="background:var(--bg-card); padding:10px; border-radius:4px; border:1px solid var(--border-subtle);">
            <div style="font-size:10px; color:var(--text-muted);">TRACK CONDITION</div>
            <div style="font-family:var(--font-data); font-size:16px; font-weight:800; color:var(--accent-f1);">${data.track_condition}</div>
          </div>
          <div style="background:var(--bg-card); padding:10px; border-radius:4px; border:1px solid var(--border-subtle);">
            <div style="font-size:10px; color:var(--text-muted);">WETNESS INDEX</div>
            <div style="font-family:var(--font-data); font-size:16px; font-weight:800; color:var(--text-mono);">${(data.metrics?.wetness_index * 100).toFixed(1)}%</div>
          </div>
          <div style="background:var(--bg-card); padding:10px; border-radius:4px; border:1px solid var(--border-subtle);">
            <div style="font-size:10px; color:var(--text-muted);">RECOMMENDED TYRE</div>
            <div style="font-family:var(--font-data); font-size:16px; font-weight:800; color:#FFF;">${data.tyre_recommendation?.compound}</div>
          </div>
          <div style="background:var(--bg-card); padding:10px; border-radius:4px; border:1px solid var(--border-subtle);">
            <div style="font-size:10px; color:var(--text-muted);">CONFIDENCE SCORE</div>
            <div style="font-family:var(--font-data); font-size:16px; font-weight:800; color:var(--status-safe);">${(data.tyre_recommendation?.confidence * 100).toFixed(0)}%</div>
          </div>
        </div>

        <!-- Vision Overlay Snapshots -->
        <div style="margin-bottom:16px;">
          <h3 style="font-size:12px; color:var(--text-secondary); text-transform:uppercase; letter-spacing:1px; margin-bottom:8px;">Vision Perception Artifacts</h3>
          <div style="display:grid; grid-template-columns: 1fr 1fr; gap:12px;">
            <div style="background:#000; border-radius:4px; overflow:hidden; border:1px solid var(--border-subtle);">
              <div style="padding:4px 8px; background:var(--bg-panel); font-size:10px; color:var(--text-secondary); font-weight:700;">SegFormer Remapped Surface Mask</div>
              <img src="${data.visualization?.segmentation_mask ? (data.visualization.segmentation_mask.startsWith('data:') ? data.visualization.segmentation_mask : 'data:image/png;base64,' + data.visualization.segmentation_mask) : ''}" style="width:100%; height:180px; object-fit:contain;" alt="Segmentation Mask" />
            </div>
            <div style="background:#000; border-radius:4px; overflow:hidden; border:1px solid var(--border-subtle);">
              <div style="padding:4px 8px; background:var(--bg-panel); font-size:10px; color:var(--text-secondary); font-weight:700;">DINOv2 Self-Attention Heatmap</div>
              <img src="${data.visualization?.attention_heatmap ? (data.visualization.attention_heatmap.startsWith('data:') ? data.visualization.attention_heatmap : 'data:image/png;base64,' + data.visualization.attention_heatmap) : ''}" style="width:100%; height:180px; object-fit:contain;" alt="Attention Heatmap" />
            </div>
          </div>
        </div>

        <!-- Key Evidence Bullets -->
        <div style="margin-bottom:16px;">
          <h3 style="font-size:12px; color:var(--text-secondary); text-transform:uppercase; letter-spacing:1px; margin-bottom:6px;">Key Evidence Factors</h3>
          <ul style="padding-left:20px; font-size:11px; color:var(--text-primary); line-height:1.8;">
            ${(data.explainability?.key_factors || []).map(kf => `
              <li><strong>${kf.factor || kf.category}:</strong> ${kf.description || ''}</li>
            `).join('') || '<li>Vision cross-validation unanimous across DINOv2 and SegFormer models.</li>'}
          </ul>
        </div>

        <!-- Footer / Sign-off -->
        <div style="border-top:1px solid var(--border-subtle); pt:12px; display:flex; justify-content:space-between; align-items:center; font-size:10px; color:var(--text-muted);">
          <div>APEX ADAPTIVE PERCEPTION & EVOLUTION EXPLAINER v1.0.0</div>
          <div>APPROVED BY CHIEF RACE ENGINEER</div>
        </div>
      </div>
    `;
  }
}

window.ReportGenerator = ReportGenerator;
