/* ========================================================================
   SkyGuard API Client
   Communicates with FastAPI backend at /api/v1/*
   ======================================================================== */

const SkyGuardAPI = (function () {
    const BASE_URL = '/api/v1';

    /**
     * Core request wrapper with error handling.
     * @param {string} path - API path (relative to BASE_URL)
     * @param {object} options - fetch options
     */
    async function request(path, options = {}) {
        const url = `${BASE_URL}${path}`;
        const config = {
            headers: { 'Content-Type': 'application/json' },
            ...options,
        };

        if (config.body && typeof config.body === 'object') {
            config.body = JSON.stringify(config.body);
        }

        try {
            const response = await fetch(url, config);
            const text = await response.text();
            let data;
            try {
                data = text ? JSON.parse(text) : {};
            } catch {
                data = { raw: text };
            }

            if (!response.ok) {
                const errMsg = data.detail || data.message || data.error || `HTTP ${response.status}`;
                throw new Error(errMsg);
            }

            // APIResponse envelope: { success, message, data, error }
            if (data && typeof data === 'object' && 'success' in data) {
                if (!data.success) {
                    throw new Error(data.error || data.message || 'API request failed');
                }
                return data.data;
            }
            return data;
        } catch (err) {
            console.error(`[API] ${options.method || 'GET'} ${path} failed:`, err);
            throw err;
        }
    }

    /* --- Monitoring ------------------------------------------------------ */
    const monitor = {
        getStatus: () => request('/monitor/status'),
        getMetrics: () => request('/monitor/metrics'),
        getHealth: () => request('/monitor/health'),
    };

    /* --- Devices --------------------------------------------------------- */
    const devices = {
        list: (params = {}) => {
            const q = new URLSearchParams();
            if (params.type) q.set('type', params.type);
            if (params.status) q.set('status', params.status);
            const qs = q.toString();
            return request(`/devices/${qs ? '?' + qs : ''}`);
        },
        get: (id) => request(`/devices/${id}`),
        create: (device) => request('/devices/', { method: 'POST', body: device }),
        update: (id, device) => request(`/devices/${id}`, { method: 'PUT', body: device }),
        delete: (id) => request(`/devices/${id}`, { method: 'DELETE' }),
        connect: (id) => request(`/devices/${id}/connect`, { method: 'POST' }),
        disconnect: (id) => request(`/devices/${id}/disconnect`, { method: 'POST' }),
        listCameras: () => request('/devices/cameras/list'),
        listPTZ: () => request('/devices/ptz/list'),
    };

    /* --- Alerts ---------------------------------------------------------- */
    const alerts = {
        list: (params = {}) => {
            const q = new URLSearchParams();
            if (params.type) q.set('type', params.type);
            if (params.level) q.set('level', params.level);
            if (params.acknowledged !== undefined) q.set('acknowledged', params.acknowledged);
            if (params.device_id) q.set('device_id', params.device_id);
            if (params.limit) q.set('limit', params.limit);
            const qs = q.toString();
            return request(`/alerts/${qs ? '?' + qs : ''}`);
        },
        get: (id) => request(`/alerts/${id}`),
        create: (alert) => request('/alerts/', { method: 'POST', body: alert }),
        acknowledge: (id) => request(`/alerts/${id}/acknowledge`, { method: 'PUT' }),
        unacknowledge: (id) => request(`/alerts/${id}/unacknowledge`, { method: 'PUT' }),
        delete: (id) => request(`/alerts/${id}`, { method: 'DELETE' }),
        deleteAll: () => request('/alerts/', { method: 'DELETE' }),
        getCount: (params = {}) => {
            const q = new URLSearchParams();
            if (params.type) q.set('type', params.type);
            if (params.level) q.set('level', params.level);
            if (params.acknowledged !== undefined) q.set('acknowledged', params.acknowledged);
            const qs = q.toString();
            return request(`/alerts/count/total${qs ? '?' + qs : ''}`);
        },
        getUnacknowledged: () => request('/alerts/unacknowledged/list'),
    };

    /* --- Config ---------------------------------------------------------- */
    const config = {
        getAll: () => request('/config/'),
        getSystem: () => request('/config/system'),
        reload: () => request('/config/reload', { method: 'POST' }),
        // Camera configs
        listCameras: () => request('/config/camera'),
        getCamera: (id) => request(`/config/camera/${id}`),
        createCamera: (cfg) => request('/config/camera', { method: 'POST', body: cfg }),
        updateCamera: (id, cfg) => request(`/config/camera/${id}`, { method: 'PUT', body: cfg }),
        deleteCamera: (id) => request(`/config/camera/${id}`, { method: 'DELETE' }),
        // PTZ configs
        listPTZ: () => request('/config/ptz'),
        getPTZ: (id) => request(`/config/ptz/${id}`),
        createPTZ: (cfg) => request('/config/ptz', { method: 'POST', body: cfg }),
        updatePTZ: (id, cfg) => request(`/config/ptz/${id}`, { method: 'PUT', body: cfg }),
        deletePTZ: (id) => request(`/config/ptz/${id}`, { method: 'DELETE' }),
        // Zone configs
        listZones: () => request('/config/zones'),
        getZone: (id) => request(`/config/zones/${id}`),
        createZone: (cfg) => request('/config/zones', { method: 'POST', body: cfg }),
        updateZone: (id, cfg) => request(`/config/zones/${id}`, { method: 'PUT', body: cfg }),
        deleteZone: (id) => request(`/config/zones/${id}`, { method: 'DELETE' }),
    };

    /* --- WebSocket ------------------------------------------------------- */
    function connectWebSocket(channel, onMessage, onStatus) {
        const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
        const url = `${proto}//${location.host}/ws/${channel}`;
        let ws = null;
        let reconnectTimer = null;
        let closed = false;

        function connect() {
            ws = new WebSocket(url);
            ws.onopen = () => {
                if (onStatus) onStatus('connected');
            };
            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    if (onMessage) onMessage(data);
                } catch (e) {
                    console.error('[WS] Parse error:', e);
                }
            };
            ws.onerror = (err) => {
                console.error('[WS] Error:', err);
            };
            ws.onclose = () => {
                if (onStatus) onStatus('disconnected');
                if (!closed) {
                    reconnectTimer = setTimeout(connect, 3000);
                }
            };
        }

        connect();

        return {
            send: (data) => {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify(data));
                }
            },
            close: () => {
                closed = true;
                if (reconnectTimer) clearTimeout(reconnectTimer);
                if (ws) ws.close();
            },
        };
    }

    return {
        monitor,
        devices,
        alerts,
        config,
        connectWebSocket,
    };
})();

// Expose globally
window.SkyGuardAPI = SkyGuardAPI;
