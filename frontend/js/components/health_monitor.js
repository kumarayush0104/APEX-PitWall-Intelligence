/**
 * APEX Component — System Telemetry & Model Registry Health Monitor
 */

class HealthMonitor {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
  }

  update(healthData) {
    if (!this.container) return;
    if (!healthData) {
      this.container.innerHTML = '<div style="color:var(--text-muted); padding:20px; text-align:center;">Connecting to backend health monitor...</div>';
      return;
    }

    const sys = healthData.system || {};
    const models = healthData.models || {};
    const services = healthData.services || {};
    const ramPct = sys.ram_total_gb ? ((sys.ram_used_gb / sys.ram_total_gb) * 100).toFixed(1) : 45.0;

    this.container.innerHTML = `
      <div style="display:flex; flex-direction:column; gap:16px; width:100%; max-width:900px; margin:0 auto; padding:16px;">
        <!-- Header Telemetry Card -->
        <div style="background:var(--bg-panel); border:1px solid var(--border-strong); border-radius:6px; padding:16px; display:flex; justify-content:space-between; align-items:center;">
          <div>
            <h2 style="font-size:16px; font-weight:800; color:#FFF; letter-spacing:1px;">HARDWARE & MODEL REGISTRY TELEMETRY</h2>
            <div style="font-size:11px; color:var(--text-secondary); margin-top:2px;">CPU-First 8GB Constrained Hardware Optimization Engine</div>
          </div>
          <div style="display:flex; gap:12px;">
            <div style="background:var(--bg-card); padding:8px 14px; border-radius:4px; border:1px solid var(--border-subtle); text-align:center;">
              <div style="font-size:9px; color:var(--text-muted);">COMPUTE DEVICE</div>
              <div style="font-family:var(--font-data); font-weight:800; color:var(--accent-cyan); font-size:14px;">${(sys.resolved_device || 'CPU').toUpperCase()}</div>
            </div>
            <div style="background:var(--bg-card); padding:8px 14px; border-radius:4px; border:1px solid var(--border-subtle); text-align:center;">
              <div style="font-size:9px; color:var(--text-muted);">PRECISION</div>
              <div style="font-family:var(--font-data); font-weight:800; color:var(--status-safe); font-size:14px;">${sys.effective_dtype || 'FLOAT32'}</div>
            </div>
          </div>
        </div>

        <!-- RAM & Resource Gauge -->
        <div style="background:var(--bg-panel); border:1px solid var(--border-subtle); border-radius:6px; padding:16px;">
          <div style="display:flex; justify-content:space-between; font-size:12px; font-weight:700; margin-bottom:8px;">
            <span style="color:var(--text-secondary);">HOST RAM ALLOCATION:</span>
            <span style="font-family:var(--font-data); color:var(--text-mono);">${sys.ram_used_gb || 3.8} GB / ${sys.ram_total_gb || 8.0} GB (${ramPct}%)</span>
          </div>
          <div style="width:100%; height:12px; background:var(--bg-primary); border-radius:6px; overflow:hidden; border:1px solid var(--border-subtle);">
            <div style="width:${ramPct}%; height:100%; background:linear-gradient(90deg, var(--status-safe) 0%, var(--status-warning) 70%, var(--status-danger) 100%); border-radius:6px; transition:width 0.4s ease;"></div>
          </div>
          <div style="font-size:10px; color:var(--text-muted); margin-top:6px;">
            Model Budget Cap: 5.0 GB · Sequential Lazy Loading enforces strict single-model residency.
          </div>
        </div>

        <!-- Model Registry Status Table -->
        <div style="background:var(--bg-panel); border:1px solid var(--border-subtle); border-radius:6px; overflow:hidden;">
          <div style="padding:10px 14px; background:var(--bg-card); border-bottom:1px solid var(--border-subtle); font-weight:700; font-size:11px; color:var(--text-secondary); text-transform:uppercase; letter-spacing:1px;">
            Lazy Model Registry Status
          </div>
          <table class="history-table">
            <thead>
              <tr>
                <th>Model Role</th>
                <th>Model Identifier</th>
                <th>Status</th>
                <th>Estimated RAM</th>
                <th>Actual RAM</th>
                <th>Provider</th>
              </tr>
            </thead>
            <tbody>
              ${Object.entries(models).map(([name, m]) => `
                <tr>
                  <td><strong style="color:var(--accent-cyan);">${name.toUpperCase()}</strong></td>
                  <td style="font-size:10px; color:var(--text-secondary);">${m.model_id || 'N/A'}</td>
                  <td><span style="padding:2px 6px; border-radius:3px; font-weight:700; background:var(--bg-card); color:${m.loaded ? 'var(--status-safe)' : 'var(--text-muted)'};">${m.loaded ? 'LOADED IN RAM' : 'LAZY / STANDBY'}</span></td>
                  <td>${m.estimated_ram_mb || 0} MB</td>
                  <td>${m.actual_ram_mb || 0} MB</td>
                  <td>${m.provider || 'LOCAL'}</td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>
      </div>
    `;
  }
}

window.HealthMonitor = HealthMonitor;
