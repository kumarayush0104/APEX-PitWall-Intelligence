/**
 * APEX Component — Telemetry Timeline Chart (Canvas High FPS Engine)
 */

class TelemetryChart {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext('2d');
    this.history = []; // array of { frame, wetness, grip, timestamp }
    this.init();
  }

  init() {
    this.resize();
    window.addEventListener('resize', () => this.resize());
    this.draw();
  }

  resize() {
    if (!this.canvas || !this.canvas.parentElement) return;
    this.canvas.width = this.canvas.parentElement.clientWidth;
    this.canvas.height = this.canvas.parentElement.clientHeight || 180;
    this.draw();
  }

  addPoint(frameIndex, wetnessIndex, gripVal) {
    this.history.push({
      frame: frameIndex,
      wetness: Math.max(0, Math.min(1, wetnessIndex)),
      grip: Math.max(0, Math.min(1, gripVal)),
      timestamp: new Date().toLocaleTimeString()
    });
    if (this.history.length > 50) {
      this.history.shift();
    }
    this.draw();
  }

  setHistory(historyPoints) {
    this.history = historyPoints.map(p => ({
      frame: p.frame_index || p.frame || 0,
      wetness: p.metrics?.wetness_index ?? p.wetness ?? 0.5,
      grip: 1.0 - (p.metrics?.wetness_index ?? p.wetness ?? 0.5) * 0.7,
      timestamp: p.timestamp || ''
    }));
    this.draw();
  }

  draw() {
    if (!this.ctx) return;
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;
    const padding = { top: 20, right: 40, bottom: 25, left: 45 };

    ctx.clearRect(0, 0, w, h);

    // Background Grid
    ctx.strokeStyle = '#1E2330';
    ctx.lineWidth = 1;

    const graphW = w - padding.left - padding.right;
    const graphH = h - padding.top - padding.bottom;

    // Horizontal grid lines (0.0, 0.25, 0.5, 0.75, 1.0)
    for (let i = 0; i <= 4; i++) {
      const y = padding.top + (graphH * (i / 4));
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(w - padding.right, y);
      ctx.stroke();

      // Axis label
      const val = (1.0 - i / 4).toFixed(2);
      ctx.fillStyle = '#515A6B';
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.textAlign = 'right';
      ctx.fillText(val, padding.left - 8, y + 3);
    }

    // Threshold Reference Lines
    // Slick Crossover Line (μ = 0.65)
    const ySlick = padding.top + graphH * (1.0 - 0.65);
    ctx.strokeStyle = 'rgba(0, 230, 118, 0.35)';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(padding.left, ySlick);
    ctx.lineTo(w - padding.right, ySlick);
    ctx.stroke();
    ctx.fillStyle = 'rgba(0, 230, 118, 0.7)';
    ctx.fillText('SLICK THRESHOLD (0.65)', w - padding.right, ySlick - 4);

    // Inter/Wet Crossover Line (μ = 0.35)
    const yWet = padding.top + graphH * (1.0 - 0.35);
    ctx.strokeStyle = 'rgba(0, 176, 255, 0.35)';
    ctx.beginPath();
    ctx.moveTo(padding.left, yWet);
    ctx.lineTo(w - padding.right, yWet);
    ctx.stroke();
    ctx.fillStyle = 'rgba(0, 176, 255, 0.7)';
    ctx.fillText('WET THRESHOLD (0.35)', w - padding.right, yWet - 4);

    ctx.setLineDash([]); // Reset dash

    if (this.history.length < 2) {
      ctx.fillStyle = '#515A6B';
      ctx.font = '12px "Inter", sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Telemetry trend building... (Awaiting stream frames)', w / 2, h / 2);
      return;
    }

    // Draw Wetness Line (Blue)
    ctx.beginPath();
    this.history.forEach((pt, idx) => {
      const x = padding.left + (graphW * (idx / (this.history.length - 1)));
      const y = padding.top + (graphH * (1.0 - pt.wetness));
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = '#00B0FF';
    ctx.lineWidth = 2.5;
    ctx.shadowBlur = 8;
    ctx.shadowColor = 'rgba(0, 176, 255, 0.5)';
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Draw Grip μ Line (Green/Amber)
    ctx.beginPath();
    this.history.forEach((pt, idx) => {
      const x = padding.left + (graphW * (idx / (this.history.length - 1)));
      const y = padding.top + (graphH * (1.0 - pt.grip));
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = '#00E676';
    ctx.lineWidth = 2.5;
    ctx.shadowBlur = 8;
    ctx.shadowColor = 'rgba(0, 230, 118, 0.5)';
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Current Frame Indicator Dot
    const lastPt = this.history[this.history.length - 1];
    const lastX = padding.left + graphW;
    const lastY = padding.top + (graphH * (1.0 - lastPt.grip));

    ctx.beginPath();
    ctx.arc(lastX, lastY, 5, 0, Math.PI * 2);
    ctx.fillStyle = '#FF1E00';
    ctx.fill();
    ctx.strokeStyle = '#FFF';
    ctx.lineWidth = 2;
    ctx.stroke();
  }
}

window.TelemetryChart = TelemetryChart;
