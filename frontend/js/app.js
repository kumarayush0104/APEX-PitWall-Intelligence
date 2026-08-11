/**
 * APEX PitWall Intelligence — Master Controller & Application Store
 */

class APEXApp {
  constructor() {
    this.currentView = 'command';
    this.currentScenario = 'belgian_gp';
    this.currentAnalysis = null;
    this.history = [];
    this.health = null;

    this.components = {};
    this.init();
  }

  async init() {
    console.log('[APEX App] Initializing PitWall Intelligence Command Center...');
    this.initComponents();
    this.bindEvents();
    this.startClock();

    // Check backend health & status
    await this.refreshHealth();

    // Load initial scenario (Belgian GP Spa) for immediate wow effect
    this.loadScenario('belgian_gp');
  }

  initComponents() {
    this.components.gripGauge = new window.GripGauge('grip-gauge-canvas');
    this.components.sectorMap = new window.SectorMap('sector-map-container');
    this.components.trackCanvas = new window.TrackCanvas('track-canvas-container');
    window.apexTrackCanvas = this.components.trackCanvas; // Global handle for mode switching
    this.components.telemetryChart = new window.TelemetryChart('telemetry-chart-canvas');
    this.components.recommendationPanel = new window.RecommendationPanel('recommendation-panel-container');
    this.components.aiExplanation = new window.AIExplanation('ai-explanation-container');
    this.components.sessionHistory = new window.SessionHistory('history-table-container');
    this.components.reportGenerator = new window.ReportGenerator('report-paper-container');
    this.components.healthMonitor = new window.HealthMonitor('health-monitor-container');
  }

  bindEvents() {
    // Navigation Tabs
    document.querySelectorAll('.nav-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const tab = e.currentTarget.dataset.tab;
        if (tab) this.switchView(tab);
      });
    });

    // Preset Scenario Buttons
    document.querySelectorAll('.preset-btn[data-scenario]').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const scenarioId = e.currentTarget.dataset.scenario;
        if (scenarioId) this.loadScenario(scenarioId);
      });
    });

    // File Upload Handler
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');

    if (dropzone && fileInput) {
      dropzone.addEventListener('click', () => fileInput.click());
      fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
          this.handleFileUpload(e.target.files[0]);
        }
      });

      dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
      });

      dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragover');
      });

      dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
          this.handleFileUpload(e.dataTransfer.files[0]);
        }
      });
    }
  }

  switchView(viewId) {
    this.currentView = viewId;

    // Update Nav Buttons
    document.querySelectorAll('.nav-btn').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.tab === viewId);
    });

    // Update Tab Containers
    document.querySelectorAll('.tab-view').forEach(view => {
      view.classList.toggle('active', view.id === `view-${viewId}`);
    });

    // Refresh specific view layout if needed
    if (viewId === 'command') {
      setTimeout(() => this.components.telemetryChart.resize(), 100);
    } else if (viewId === 'reports') {
      this.components.reportGenerator.generateReport(this.currentAnalysis, this.history);
    } else if (viewId === 'history') {
      this.components.sessionHistory.setHistory(this.history);
    } else if (viewId === 'health') {
      this.components.healthMonitor.update(this.health);
    }
  }

  loadScenario(scenarioId) {
    const scenario = window.APEX_DEMO_SCENARIOS[scenarioId];
    if (!scenario) return;

    this.currentScenario = scenarioId;

    // Update Active Preset Button
    document.querySelectorAll('.preset-btn[data-scenario]').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.scenario === scenarioId);
    });

    // Update Header Meta
    const circuitEl = document.getElementById('circuit-name-display');
    if (circuitEl) circuitEl.textContent = `${scenario.flag} ${scenario.circuit}`;

    // Apply Scenario Data to Components
    this.applyAnalysisResult(scenario);
  }

  async handleFileUpload(file) {
    console.log('[APEX App] Uploading frame file:', file.name);
    // Show spinner / loading status
    const statusPill = document.getElementById('backend-status-text');
    if (statusPill) statusPill.textContent = 'ANALYZING FRAME...';

    try {
      const result = await window.apexApi.analyzeFrame(file, Math.floor(Math.random() * 100));
      this.applyAnalysisResult(result);
      this.switchView('command');
    } catch (err) {
      alert(`Analysis failed: ${err.message}`);
    } finally {
      if (statusPill) statusPill.textContent = 'Backend Online';
    }
  }

  applyAnalysisResult(data) {
    this.currentAnalysis = data;
    this.history.unshift(data);
    if (this.history.length > 50) this.history.pop();

    // 1. Grip Gauge
    const wetness = data.metrics?.wetness_index ?? 0.5;
    const grip = 1.0 - (wetness * 0.7);
    this.components.gripGauge.setGrip(grip);

    // 2. Track Canvas Images
    if (data.visualization) {
      this.components.trackCanvas.setVisualizationData(data.visualization, data.visualization.overlay);
    }

    // 3. Sector Map
    if (data.sector_risk) {
      this.components.sectorMap.updateSectors(data.sector_risk);
    }

    // 4. Recommendation Card
    this.components.recommendationPanel.update(data.tyre_recommendation, data.metrics, data.temporal_analysis);

    // 5. AI Explanation & Evidence
    this.components.aiExplanation.update(data.explainability);

    // 6. Telemetry Chart
    this.components.telemetryChart.addPoint(data.frame_index || 0, wetness, grip);

    // 7. Update Header Telemetry Pills
    const wetnessPill = document.getElementById('header-wetness-val');
    if (wetnessPill) wetnessPill.textContent = `${(wetness * 100).toFixed(0)}% WET`;

    const conditionPill = document.getElementById('header-condition-val');
    if (conditionPill) conditionPill.textContent = data.track_condition || 'DRY';
  }

  loadHistoryItem(index) {
    const item = this.history[index];
    if (item) {
      this.applyAnalysisResult(item);
      this.switchView('command');
    }
  }

  async refreshHealth() {
    const h = await window.apexApi.checkHealth();
    if (h) {
      this.health = h;
      this.components.healthMonitor.update(h);
      const versionPill = document.getElementById('backend-version-display');
      if (versionPill) versionPill.textContent = `v${h.version || '1.0.0'} · ${(h.system?.resolved_device || 'CPU').toUpperCase()}`;
    }
  }

  startClock() {
    const clockEl = document.getElementById('session-clock-display');
    let seconds = 0;
    setInterval(() => {
      seconds++;
      const hrs = String(Math.floor(seconds / 3600)).padStart(2, '0');
      const mins = String(Math.floor((seconds % 3600) / 60)).padStart(2, '0');
      const secs = String(seconds % 60).padStart(2, '0');
      if (clockEl) clockEl.textContent = `${hrs}:${mins}:${secs}`;
    }, 1000);
  }
}

window.addEventListener('DOMContentLoaded', () => {
  window.apexApp = new APEXApp();
});
