/**
 * APEX Component — Dynamic F1 Sector Risk Map
 */

class SectorMap {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.sectors = {
      sector1: { risk: "SAFE", wetness: 0.0, label: "Sector 1" },
      sector2: { risk: "SAFE", wetness: 0.0, label: "Sector 2" },
      sector3: { risk: "SAFE", wetness: 0.0, label: "Sector 3" }
    };
    this.render();
  }

  updateSectors(sectorData) {
    if (sectorData) {
      this.sectors = { ...this.sectors, ...sectorData };
    }
    this.render();
  }

  getRiskColor(risk) {
    switch (risk) {
      case "CRITICAL": return "#FF1744";
      case "HIGH":     return "#FF9100";
      case "MEDIUM":   return "#FFC800";
      case "LOW":      return "#00E676";
      default:         return "#00E676";
    }
  }

  render() {
    if (!this.container) return;

    const s1Color = this.getRiskColor(this.sectors.sector1.risk);
    const s2Color = this.getRiskColor(this.sectors.sector2.risk);
    const s3Color = this.getRiskColor(this.sectors.sector3.risk);

    this.container.innerHTML = `
      <div style="display:flex; flex-direction:column; gap:10px; width:100%;">
        <div style="position:relative; width:100%; height:140px; background:#0B0D12; border:1px solid var(--border-subtle); border-radius:6px; padding:8px; display:flex; align-items:center; justify-content:center;">
          <svg viewBox="0 0 300 120" style="width:100%; height:100%;">
            <defs>
              <filter id="glow-s1" x="-20%" y="-20%" width="140%" height="140%">
                <feGaussianBlur stdDeviation="3" result="blur" />
                <feComposite in="SourceGraphic" in2="blur" operator="over" />
              </filter>
            </defs>

            <!-- Circuit Layout Path split into 3 Sector segments -->
            <!-- Sector 1 -->
            <path d="M 40,80 C 40,30 80,20 120,20" fill="none" stroke="${s1Color}" stroke-width="6" stroke-linecap="round" filter="url(#glow-s1)"/>
            <!-- Sector 2 -->
            <path d="M 120,20 C 180,20 260,30 260,60 C 260,90 200,100 160,100" fill="none" stroke="${s2Color}" stroke-width="6" stroke-linecap="round"/>
            <!-- Sector 3 -->
            <path d="M 160,100 C 110,100 40,110 40,80" fill="none" stroke="${s3Color}" stroke-width="6" stroke-linecap="round"/>

            <!-- Sector Labels -->
            <text x="70" y="15" fill="${s1Color}" font-family="JetBrains Mono" font-size="9" font-weight="700">S1</text>
            <text x="210" y="15" fill="${s2Color}" font-family="JetBrains Mono" font-size="9" font-weight="700">S2</text>
            <text x="100" y="115" fill="${s3Color}" font-family="JetBrains Mono" font-size="9" font-weight="700">S3</text>
          </svg>
        </div>

        <!-- Sector Cards -->
        <div style="display:grid; grid-template-columns: 1fr 1fr 1fr; gap:6px;">
          ${this.renderSectorCard('S1', this.sectors.sector1, s1Color)}
          ${this.renderSectorCard('S2', this.sectors.sector2, s2Color)}
          ${this.renderSectorCard('S3', this.sectors.sector3, s3Color)}
        </div>
      </div>
    `;
  }

  renderSectorCard(id, data, color) {
    return `
      <div style="background:var(--bg-panel); border:1px solid var(--border-subtle); border-top:3px solid ${color}; border-radius:4px; padding:6px; font-family:var(--font-data);">
        <div style="display:flex; justify-size:space-between; align-items:center;">
          <span style="font-weight:800; font-size:11px; color:#FFF;">${id}</span>
          <span style="font-size:9px; font-weight:700; color:${color};">${data.risk}</span>
        </div>
        <div style="font-size:10px; color:var(--text-secondary); margin-top:2px;">
          Wet: ${(data.wetness * 100).toFixed(0)}%
        </div>
      </div>
    `;
  }
}

window.SectorMap = SectorMap;
