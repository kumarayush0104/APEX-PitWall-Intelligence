/**
 * APEX Component — Session History & Multi-Frame Audit Log
 */

class SessionHistory {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.sessions = [];
  }

  setHistory(historyData) {
    this.sessions = historyData || [];
    this.render();
  }

  render() {
    if (!this.container) return;

    if (this.sessions.length === 0) {
      this.container.innerHTML = `
        <div style="text-align:center; padding:40px; color:var(--text-muted);">
          <div style="font-size:32px; margin-bottom:8px;">📊</div>
          <div>No session analysis history available yet.</div>
          <div style="font-size:11px; margin-top:4px;">Run a Demo Scenario or upload a frame to generate telemetry logs.</div>
        </div>
      `;
      return;
    }

    this.container.innerHTML = `
      <div style="width:100%; overflow-x:auto;">
        <table class="history-table">
          <thead>
            <tr>
              <th>Frame #</th>
              <th>Timestamp</th>
              <th>Condition</th>
              <th>Wetness %</th>
              <th>Grip (μ)</th>
              <th>Tyre Rec</th>
              <th>Pit Status</th>
              <th>Latency</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            ${this.sessions.map((s, idx) => `
              <tr>
                <td><strong>#${s.frame_index ?? idx}</strong></td>
                <td>${s.timestamp ? new Date(s.timestamp).toLocaleTimeString() : 'N/A'}</td>
                <td><span style="padding:2px 6px; border-radius:3px; font-weight:700; background:var(--bg-panel); color:${this.getConditionColor(s.track_condition)};">${s.track_condition || 'N/A'}</span></td>
                <td>${s.metrics?.wetness_index ? (s.metrics.wetness_index * 100).toFixed(1) + '%' : '0.0%'}</td>
                <td>${s.metrics?.wetness_index ? (1.0 - s.metrics.wetness_index * 0.7).toFixed(2) : '0.85'}</td>
                <td><span class="compound-badge ${s.tyre_recommendation?.compound || 'MEDIUM'}" style="font-size:10px; padding:2px 6px;">${s.tyre_recommendation?.compound || 'MEDIUM'}</span></td>
                <td style="color:${s.tyre_recommendation?.pit_window_open ? 'var(--status-danger)' : 'var(--status-safe)'}; font-weight:700;">${s.tyre_recommendation?.pit_window_open ? 'BOX NOW' : 'MAINTAIN'}</td>
                <td>${s.processing_time_ms ? s.processing_time_ms.toFixed(0) + ' ms' : '< 50 ms'}</td>
                <td><button class="preset-btn" onclick="window.apexApp.loadHistoryItem(${idx})">Inspect</button></td>
              </tr>
            `).join('')}
          </tbody>
        </table>
      </div>
    `;
  }

  getConditionColor(cond) {
    switch (cond) {
      case 'FLOODED': return 'var(--state-wet-severe)';
      case 'WET':     return 'var(--state-wet-moderate)';
      case 'DAMP':    return 'var(--state-transitional)';
      case 'DRYING':  return 'var(--state-drying)';
      case 'DRY':     return 'var(--state-dry-evolved)';
      default:        return 'var(--text-primary)';
    }
  }
}

window.SessionHistory = SessionHistory;
