/**
 * APEX Component — Multi-Layer Vision Track Canvas
 */

class TrackCanvas {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.currentMode = 'overlay'; // 'original' | 'segmentation' | 'heatmap' | 'overlay'
    this.visualizationData = null;
    this.init();
  }

  init() {
    if (!this.container) return;
    this.render();
  }

  setMode(mode) {
    this.currentMode = mode;
    this.updateImage();
  }

  setVisualizationData(viz, frameImageBase64) {
    this.visualizationData = viz;
    this.frameImageBase64 = frameImageBase64;
    this.updateImage();
  }

  updateImage() {
    const imgEl = document.getElementById('apex-track-image');
    if (!imgEl) return;

    if (!this.visualizationData) {
      if (this.frameImageBase64) {
        imgEl.src = this.frameImageBase64.startsWith('data:') ? this.frameImageBase64 : `data:image/png;base64,${this.frameImageBase64}`;
      }
      return;
    }

    let src = '';
    switch (this.currentMode) {
      case 'original':
        src = this.frameImageBase64 ? (this.frameImageBase64.startsWith('data:') ? this.frameImageBase64 : `data:image/png;base64,${this.frameImageBase64}`) : '';
        break;
      case 'segmentation':
        src = this.visualizationData.segmentation_mask || this.visualizationData.overlay;
        break;
      case 'heatmap':
        src = this.visualizationData.attention_heatmap || this.visualizationData.overlay;
        break;
      case 'overlay':
      default:
        src = this.visualizationData.overlay || this.visualizationData.segmentation_mask;
        break;
    }

    if (src) {
      imgEl.src = src.startsWith('data:') ? src : `data:image/png;base64,${src}`;
    }
  }

  render() {
    if (!this.container) return;

    this.container.innerHTML = `
      <div class="visualizer-wrapper">
        <div class="visualizer-canvas-container">
          <img id="apex-track-image" class="visualizer-img" src="data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='800' height='450' viewBox='0 0 800 450'><rect width='800' height='450' fill='%230B0D12'/><text x='400' y='225' fill='%238C96A6' font-family='sans-serif' font-size='16' text-anchor='middle'>Awaiting Telemetry Feed / Image Upload</text></svg>" alt="APEX Track Visualization" />
        </div>

        <div class="layer-controls">
          <button class="layer-btn ${this.currentMode==='original'?'active':''}" onclick="window.apexTrackCanvas.setMode('original')">ORIGINAL</button>
          <button class="layer-btn ${this.currentMode==='overlay'?'active':''}" onclick="window.apexTrackCanvas.setMode('overlay')">OVERLAY</button>
          <button class="layer-btn ${this.currentMode==='segmentation'?'active':''}" onclick="window.apexTrackCanvas.setMode('segmentation')">SEGMENTATION</button>
          <button class="layer-btn ${this.currentMode==='heatmap'?'active':''}" onclick="window.apexTrackCanvas.setMode('heatmap')">DINOv2 HEATMAP</button>
        </div>
      </div>
    `;
  }
}

window.TrackCanvas = TrackCanvas;
