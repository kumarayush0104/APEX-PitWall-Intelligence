/**
 * APEX PitWall Intelligence — API Communication & WebSocket Engine
 */

class APEXApi {
  constructor() {
    this.baseUrl = this.detectBaseUrl();
    this.wsUrl = this.detectWsUrl();
    this.ws = null;
    this.listeners = new Map();
  }

  detectBaseUrl() {
    const stored = localStorage.getItem('APEX_API_URL');
    if (stored) return stored;
    if (window.location.port === '8000') return '';
    return 'http://localhost:8000';
  }

  detectWsUrl() {
    const base = this.baseUrl || window.location.origin;
    const wsProto = base.startsWith('https') ? 'wss:' : 'ws:';
    const host = base.replace(/^https?:\/\//, '');
    return `${wsProto}//${host}/api/v1/stream`;
  }

  async checkHealth() {
    try {
      const resp = await fetch(`${this.baseUrl}/api/v1/health`, { signal: AbortSignal.timeout(5000) });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      return await resp.json();
    } catch (err) {
      console.warn('[APEX Api] Health check failed:', err.message);
      return null;
    }
  }

  async analyzeFrame(fileOrBase64, frameIndex = 0) {
    const formData = new FormData();
    if (fileOrBase64 instanceof File || fileOrBase64 instanceof Blob) {
      formData.append('file', fileOrBase64);
    } else if (typeof fileOrBase64 === 'string') {
      formData.append('image_base64', fileOrBase64);
    } else {
      throw new Error("Invalid input: must be File, Blob, or Base64 string.");
    }
    formData.append('frame_index', frameIndex);
    formData.append('generate_visualizations', 'true');

    const resp = await fetch(`${this.baseUrl}/api/v1/analyze`, {
      method: 'POST',
      body: formData,
    });

    if (!resp.ok) {
      const errJson = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(errJson.detail || `Analysis failed: ${resp.status}`);
    }

    return await resp.json();
  }

  async getHistory() {
    try {
      const resp = await fetch(`${this.baseUrl}/api/v1/history`);
      if (!resp.ok) return { history: [] };
      return await resp.json();
    } catch (err) {
      return { history: [] };
    }
  }

  connectWebSocket(onFrameCallback) {
    if (this.ws) {
      this.ws.close();
    }
    try {
      this.ws = new WebSocket(this.wsUrl);
      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (onFrameCallback) onFrameCallback(data);
        } catch (e) {
          console.error('[APEX WS] Parse error:', e);
        }
      };
      this.ws.onerror = (err) => console.warn('[APEX WS] Socket error:', err);
      this.ws.onclose = () => console.log('[APEX WS] Socket closed');
    } catch (err) {
      console.warn('[APEX WS] Connection failed:', err);
    }
  }

  sendWsFrame(base64Image) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ image: base64Image }));
    }
  }
}

window.apexApi = new APEXApi();
