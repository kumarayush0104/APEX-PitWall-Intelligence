/**
 * APEX Component — Circular Grip Gauge (Speedometer Telemetry Style)
 */

class GripGauge {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext('2d');
    this.currentGrip = 0.5;
    this.targetGrip = 0.5;
    this.animFrame = null;
    this.init();
  }

  init() {
    this.canvas.width = 240;
    this.canvas.height = 180;
    this.draw();
  }

  setGrip(val) {
    this.targetGrip = Math.max(0, Math.min(1, val));
    this.animate();
  }

  animate() {
    if (this.animFrame) cancelAnimationFrame(this.animFrame);
    const step = () => {
      const diff = this.targetGrip - this.currentGrip;
      if (Math.abs(diff) < 0.005) {
        this.currentGrip = this.targetGrip;
        this.draw();
      } else {
        this.currentGrip += diff * 0.12;
        this.draw();
        this.animFrame = requestAnimationFrame(step);
      }
    };
    this.animFrame = requestAnimationFrame(step);
  }

  draw() {
    if (!this.ctx) return;
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;
    const cx = w / 2;
    const cy = h - 25;
    const radius = 85;

    ctx.clearRect(0, 0, w, h);

    const startAngle = Math.PI * 0.85;
    const endAngle = Math.PI * 2.15;
    const totalAngle = endAngle - startAngle;

    // Track Background Arc
    ctx.beginPath();
    ctx.arc(cx, cy, radius, startAngle, endAngle);
    ctx.lineWidth = 14;
    ctx.strokeStyle = '#171B24';
    ctx.stroke();

    // Zone Arcs
    // Flooded (0.0 - 0.3)
    this.drawArcZone(ctx, cx, cy, radius, startAngle, startAngle + totalAngle * 0.3, '#0064FF');
    // Wet (0.3 - 0.55)
    this.drawArcZone(ctx, cx, cy, radius, startAngle + totalAngle * 0.3, startAngle + totalAngle * 0.55, '#00A0FF');
    // Damp (0.55 - 0.75)
    this.drawArcZone(ctx, cx, cy, radius, startAngle + totalAngle * 0.55, startAngle + totalAngle * 0.75, '#FFC800');
    // Dry (0.75 - 1.0)
    this.drawArcZone(ctx, cx, cy, radius, startAngle + totalAngle * 0.75, endAngle, '#00DC64');

    // Value Arc Glow
    const currentAngle = startAngle + totalAngle * this.currentGrip;
    ctx.beginPath();
    ctx.arc(cx, cy, radius, startAngle, currentAngle);
    ctx.lineWidth = 14;
    ctx.strokeStyle = this.getGripColor(this.currentGrip);
    ctx.shadowBlur = 12;
    ctx.shadowColor = this.getGripColor(this.currentGrip);
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Needle
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(currentAngle + Math.PI / 2);

    ctx.beginPath();
    ctx.moveTo(-4, 0);
    ctx.lineTo(0, -radius + 8);
    ctx.lineTo(4, 0);
    ctx.closePath();
    ctx.fillStyle = '#FFFFFF';
    ctx.fill();

    ctx.restore();

    // Pivot Center Cap
    ctx.beginPath();
    ctx.arc(cx, cy, 8, 0, Math.PI * 2);
    ctx.fillStyle = '#FF1E00';
    ctx.fill();

    // Value Text
    ctx.fillStyle = '#FFFFFF';
    ctx.font = 'bold 26px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.fillText((this.currentGrip).toFixed(2), cx, cy - 25);

    ctx.fillStyle = '#8C96A6';
    ctx.font = '700 9px "Inter", sans-serif';
    ctx.fillText('FRICTION COEFF (μ)', cx, cy - 10);
  }

  drawArcZone(ctx, cx, cy, radius, start, end, color) {
    ctx.beginPath();
    ctx.arc(cx, cy, radius, start, end);
    ctx.lineWidth = 14;
    ctx.strokeStyle = color;
    ctx.globalAlpha = 0.3;
    ctx.stroke();
    ctx.globalAlpha = 1.0;
  }

  getGripColor(grip) {
    if (grip < 0.3) return '#0064FF';
    if (grip < 0.55) return '#00A0FF';
    if (grip < 0.75) return '#FFC800';
    return '#00DC64';
  }
}

window.GripGauge = GripGauge;
