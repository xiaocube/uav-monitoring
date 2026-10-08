/* ========================================================================
   SkyGuard Dashboard Application Logic
   Handles routing, rendering, and user interactions
   ======================================================================== */

(function () {
    'use strict';

    // --- State -----------------------------------------------------------
    const state = {
        currentPage: 'dashboard',
        refreshTimer: null,
        refreshInterval: 5000,
        wsAlert: null,
        wsStatus: null,
        metricsHistory: [],
    };

    // --- DOM Helpers -----------------------------------------------------
    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);
    const el = (tag, attrs = {}, children = []) => {
        const node = document.createElement(tag);
        Object.entries(attrs).forEach(([k, v]) => {
            if (k === 'class') node.className = v;
            else if (k === 'html') node.innerHTML = v;
            else if (k.startsWith('on') && typeof v === 'function') {
                node.addEventListener(k.slice(2).toLowerCase(), v);
            } else if (v !== null && v !== undefined) {
                node.setAttribute(k, v);
            }
        });
        (Array.isArray(children) ? children : [children]).forEach((c) => {
            if (c == null) return;
            if (typeof c === 'string') node.appendChild(document.createTextNode(c));
            else node.appendChild(c);
        });
        return node;
    };

    // --- Utility ---------------------------------------------------------
    function formatBytes(bytes) {
        if (!bytes) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
    }

    function formatDuration(seconds) {
        if (!seconds) return '--';
        const h = Math.floor(seconds / 3600);
        const m = Math.floor((seconds % 3600) / 60);
        const s = Math.floor(seconds % 60);
        if (h > 0) return `${h}h ${m}m ${s}s`;
        if (m > 0) return `${m}m ${s}s`;
        return `${s}s`;
    }

    function formatTime(iso) {
        if (!iso) return '--';
        try {
            const d = new Date(iso);
            return d.toLocaleTimeString('zh-CN', { hour12: false });
        } catch {
            return iso;
        }
    }

    function formatDateTime(iso) {
        if (!iso) return '--';
        try {
            const d = new Date(iso);
            return d.toLocaleString('zh-CN', { hour12: false });
        } catch {
            return iso;
        }
    }

    function timeAgo(iso) {
        if (!iso) return '从未';
        const diff = (Date.now() - new Date(iso).getTime()) / 1000;
        if (diff < 60) return `${Math.floor(diff)}秒前`;
        if (diff < 3600) return `${Math.floor(diff / 60)}分钟前`;
        if (diff < 86400) return `${Math.floor(diff / 3600)}小时前`;
        return `${Math.floor(diff / 86400)}天前`;
    }

    // --- Toast -----------------------------------------------------------
    function toast(message, type = 'info', duration = 3000) {
        const container = $('#toastContainer');
        const icons = {
            success: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>',
            error: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>',
            warning: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
            info: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>',
        };
        const t = el('div', { class: `toast toast-${type}` }, [
            el('span', { html: icons[type] || icons.info }),
            el('span', {}, message),
        ]);
        container.appendChild(t);
        setTimeout(() => {
            t.classList.add('removing');
            setTimeout(() => t.remove(), 300);
        }, duration);
    }

    // --- Modal -----------------------------------------------------------
    function showModal(title, bodyHtml, footerHtml = '') {
        const overlay = $('#modalOverlay');
        const modal = $('#modal');
        modal.innerHTML = `
            <div class="modal-header">
                <h2 class="modal-title">${title}</h2>
                <button class="modal-close" onclick="document.getElementById('modalOverlay').style.display='none'">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
                    </svg>
                </button>
            </div>
            <div class="modal-body">${bodyHtml}</div>
            ${footerHtml ? `<div class="modal-footer">${footerHtml}</div>` : ''}
        `;
        overlay.style.display = 'flex';
        overlay.onclick = (e) => {
            if (e.target === overlay) overlay.style.display = 'none';
        };
    }

    function closeModal() {
        $('#modalOverlay').style.display = 'none';
    }

    // --- Loading & Empty States -----------------------------------------
    function loadingState(text = '加载中...') {
        return `<div class="loading-container"><span class="spinner"></span>${text}</div>`;
    }

    function emptyState(title, desc) {
        return `
            <div class="empty-state">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                    <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/>
                </svg>
                <div class="empty-state-title">${title}</div>
                <div class="empty-state-desc">${desc}</div>
            </div>
        `;
    }

    // --- Status Badge Helpers -------------------------------------------
    function deviceStatusBadge(status) {
        const map = {
            online: 'badge-success',
            offline: 'badge-neutral',
            connecting: 'badge-warning',
            error: 'badge-danger',
        };
        const labels = { online: '在线', offline: '离线', connecting: '连接中', error: '错误' };
        return `<span class="badge ${map[status] || 'badge-neutral'}">${labels[status] || status}</span>`;
    }

    function alertLevelBadge(level) {
        const map = {
            critical: 'badge-critical',
            warning: 'badge-warning',
            info: 'badge-info',
        };
        const labels = { critical: '严重', warning: '警告', info: '信息' };
        return `<span class="badge ${map[level] || 'badge-neutral'}">${labels[level] || level}</span>`;
    }

    function alertTypeLabel(type) {
        const map = {
            drone_detected: '无人机检测',
            drone_entered_zone: '入侵禁飞区',
            drone_lost: '目标丢失',
            device_offline: '设备离线',
            device_error: '设备错误',
            system_error: '系统错误',
        };
        return map[type] || type;
    }

    function deviceTypeLabel(type) {
        const map = {
            camera: '摄像头',
            ptz: '云台',
            detector: '检测器',
            tracker: '跟踪器',
            stream: '视频流',
        };
        return map[type] || type;
    }

    // --- Icon Helpers ---------------------------------------------------
    const icons = {
        cpu: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/></svg>',
        memory: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 19v-3"/><path d="M10 19v-3"/><path d="M14 19v-3"/><path d="M18 19v-3"/><path d="M8 11V9"/><path d="M16 11V9"/><path d="M12 11V9"/><path d="M2 15h20"/><path d="M2 7a2 2 0 012-2h16a2 2 0 012 2v1.1a2 2 0 000 3.837V17a2 2 0 01-2 2H4a2 2 0 01-2-2v-5.1a2 2 0 000-3.837Z"/></svg>',
        gpu: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="6" width="20" height="12" rx="2"/><path d="M6 12h.01M10 12h.01M14 12h.01M18 12h.01"/></svg>',
        clock: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>',
        activity: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>',
        camera: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2"/></svg>',
        target: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>',
        alert: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
        info: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>',
        server: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="2" width="20" height="8" rx="2"/><rect x="2" y="14" width="20" height="8" rx="2"/><line x1="6" y1="6" x2="6.01" y2="6"/><line x1="6" y1="18" x2="6.01" y2="18"/></svg>',
        power: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18.36 6.64a9 9 0 11-12.73 0"/><line x1="12" y1="2" x2="12" y2="12"/></svg>',
        edit: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>',
        trash: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/></svg>',
        check: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>',
        plus: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>',
    };

    // ====================================================================
    // PAGE: DASHBOARD
    // ====================================================================
    async function renderDashboard() {
        const content = $('#pageContent');
        content.innerHTML = `
            <div class="stats-grid" id="statsGrid">${loadingState('加载状态...')}</div>
            <div class="grid-2" style="margin-bottom:16px">
                <div class="card" id="resourceCard">${loadingState()}</div>
                <div class="card" id="systemCard">${loadingState()}</div>
            </div>
            <div class="card">
                <div class="card-header">
                    <div>
                        <div class="card-title">系统活动</div>
                        <div class="card-subtitle">实时监控数据流</div>
                    </div>
                </div>
                <div class="chart-container" id="activityChart">
                    ${loadingState('加载图表...')}
                </div>
            </div>
        `;

        try {
            const [status, metrics] = await Promise.all([
                SkyGuardAPI.monitor.getStatus(),
                SkyGuardAPI.monitor.getMetrics(),
            ]);
            renderDashboardStats(status);
            renderResourceCard(metrics);
            renderSystemCard(status, metrics);
            renderActivityChart(status);

            // Record history
            state.metricsHistory.push({
                time: Date.now(),
                cpu: status.cpu_usage,
                memory: status.memory_usage,
                fps: status.avg_fps,
            });
            if (state.metricsHistory.length > 30) {
                state.metricsHistory.shift();
            }
        } catch (err) {
            content.innerHTML = `<div class="card"><div class="empty-state">
                <div class="empty-state-title">无法加载仪表板数据</div>
                <div class="empty-state-desc">${err.message}</div>
            </div></div>`;
        }
    }

    function renderDashboardStats(status) {
        const grid = $('#statsGrid');
        if (!grid) return;
        const uptimeClass = status.uptime > 86400 ? 'success' : '';
        const cpuClass = status.cpu_usage > 80 ? 'danger' : status.cpu_usage > 60 ? 'warning' : 'success';
        const memClass = status.memory_usage > 80 ? 'danger' : status.memory_usage > 60 ? 'warning' : 'success';
        const fpsClass = status.avg_fps > 25 ? 'success' : status.avg_fps > 15 ? 'warning' : 'danger';

        grid.innerHTML = `
            <div class="stat-card ${uptimeClass}">
                <div class="stat-label">${icons.clock} 运行时间</div>
                <div class="stat-value">${formatDuration(status.uptime)}</div>
            </div>
            <div class="stat-card ${cpuClass}">
                <div class="stat-label">${icons.cpu} CPU 使用率</div>
                <div class="stat-value">${status.cpu_usage.toFixed(1)}<span class="stat-unit">%</span></div>
                <div class="progress-bar"><div class="progress-fill ${cpuClass === 'success' ? 'success' : cpuClass}" style="width:${status.cpu_usage}%"></div></div>
            </div>
            <div class="stat-card ${memClass}">
                <div class="stat-label">${icons.memory} 内存使用率</div>
                <div class="stat-value">${status.memory_usage.toFixed(1)}<span class="stat-unit">%</span></div>
                <div class="progress-bar"><div class="progress-fill ${memClass === 'success' ? 'success' : memClass}" style="width:${status.memory_usage}%"></div></div>
            </div>
            <div class="stat-card">
                <div class="stat-label">${icons.gpu} GPU 使用率</div>
                <div class="stat-value">${status.gpu_usage !== null ? status.gpu_usage.toFixed(1) : '--'}<span class="stat-unit">%</span></div>
            </div>
            <div class="stat-card">
                <div class="stat-label">${icons.camera} 活动视频流</div>
                <div class="stat-value">${status.active_streams}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">${icons.target} 活动跟踪目标</div>
                <div class="stat-value">${status.active_tracks}</div>
            </div>
            <div class="stat-card ${fpsClass}">
                <div class="stat-label">${icons.activity} 平均 FPS</div>
                <div class="stat-value">${status.avg_fps.toFixed(1)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">${icons.server} 已加载模型</div>
                <div class="stat-value">${status.models_loaded}</div>
            </div>
        `;
    }

    function renderResourceCard(metrics) {
        const card = $('#resourceCard');
        if (!card) return;
        const mem = metrics.memory;
        const disk = metrics.disk;
        card.innerHTML = `
            <div class="card-header">
                <div class="card-title">资源监控</div>
            </div>
            <div style="display:flex;flex-direction:column;gap:16px">
                <div>
                    <div style="display:flex;justify-content:space-between;margin-bottom:6px">
                        <span style="font-size:13px;color:var(--text-secondary)">内存</span>
                        <span style="font-size:12px;font-family:var(--font-mono);color:var(--text-muted)">${formatBytes(mem.used)} / ${formatBytes(mem.total)}</span>
                    </div>
                    <div class="progress-bar"><div class="progress-fill ${mem.percent > 80 ? 'danger' : mem.percent > 60 ? 'warning' : 'success'}" style="width:${mem.percent}%"></div></div>
                </div>
                <div>
                    <div style="display:flex;justify-content:space-between;margin-bottom:6px">
                        <span style="font-size:13px;color:var(--text-secondary)">磁盘</span>
                        <span style="font-size:12px;font-family:var(--font-mono);color:var(--text-muted)">${formatBytes(disk.used)} / ${formatBytes(disk.total)}</span>
                    </div>
                    <div class="progress-bar"><div class="progress-fill ${disk.percent > 80 ? 'danger' : disk.percent > 60 ? 'warning' : 'success'}" style="width:${disk.percent}%"></div></div>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:13px">
                    <span style="color:var(--text-secondary)">CPU 核心数</span>
                    <span style="font-family:var(--font-mono);color:var(--text-primary)">${metrics.cpu.count}</span>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:13px">
                    <span style="color:var(--text-secondary)">网络连接数</span>
                    <span style="font-family:var(--font-mono);color:var(--text-primary)">${metrics.network.connections}</span>
                </div>
            </div>
        `;
    }

    function renderSystemCard(status, metrics) {
        const card = $('#systemCard');
        if (!card) return;
        card.innerHTML = `
            <div class="card-header">
                <div class="card-title">系统信息</div>
            </div>
            <div style="display:flex;flex-direction:column;gap:12px">
                <div style="display:flex;justify-content:space-between;font-size:13px">
                    <span style="color:var(--text-secondary)">系统状态</span>
                    <span class="badge badge-success">运行中</span>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:13px">
                    <span style="color:var(--text-secondary)">运行时间</span>
                    <span style="font-family:var(--font-mono);color:var(--text-primary)">${formatDuration(status.uptime)}</span>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:13px">
                    <span style="color:var(--text-secondary)">CPU 频率</span>
                    <span style="font-family:var(--font-mono);color:var(--text-primary)">${metrics.cpu.freq ? metrics.cpu.freq.toFixed(0) + ' MHz' : '--'}</span>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:13px">
                    <span style="color:var(--text-secondary)">可用内存</span>
                    <span style="font-family:var(--font-mono);color:var(--text-primary)">${formatBytes(metrics.memory.available)}</span>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:13px">
                    <span style="color:var(--text-secondary)">磁盘剩余</span>
                    <span style="font-family:var(--font-mono);color:var(--text-primary)">${formatBytes(metrics.disk.free)}</span>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:13px">
                    <span style="color:var(--text-secondary)">更新时间</span>
                    <span style="font-family:var(--font-mono);color:var(--text-muted)">${formatTime(metrics.timestamp)}</span>
                </div>
            </div>
        `;
    }

    function renderActivityChart(status) {
        const container = $('#activityChart');
        if (!container) return;
        const history = state.metricsHistory;
        if (history.length < 2) {
            container.innerHTML = `<div class="empty-state" style="padding:24px">
                <div class="empty-state-desc">正在收集数据，请稍候...</div>
            </div>`;
            return;
        }

        const w = 600;
        const h = 200;
        const padding = { top: 20, right: 20, bottom: 30, left: 40 };
        const plotW = w - padding.left - padding.right;
        const plotH = h - padding.top - padding.bottom;
        const maxVal = 100;

        const xScale = (i) => padding.left + (i / (history.length - 1)) * plotW;
        const yScale = (v) => padding.top + plotH - (v / maxVal) * plotH;

        const cpuPath = history.map((d, i) => `${i === 0 ? 'M' : 'L'} ${xScale(i)} ${yScale(d.cpu)}`).join(' ');
        const memPath = history.map((d, i) => `${i === 0 ? 'M' : 'L'} ${xScale(i)} ${yScale(d.memory)}`).join(' ');
        const fpsPath = history.map((d, i) => `${i === 0 ? 'M' : 'L'} ${xScale(i)} ${yScale(Math.min(d.fps * 2, 100))}`).join(' ');

        container.innerHTML = `
            <svg class="chart-svg" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">
                <defs>
                    <linearGradient id="chartGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stop-color="#00D9FF" stop-opacity="0.3"/>
                        <stop offset="100%" stop-color="#00D9FF" stop-opacity="0"/>
                    </linearGradient>
                </defs>
                <!-- Grid lines -->
                ${[0, 25, 50, 75, 100].map((v) => `
                    <line class="chart-axis" x1="${padding.left}" y1="${yScale(v)}" x2="${w - padding.right}" y2="${yScale(v)}"/>
                    <text class="chart-label" x="${padding.left - 8}" y="${yScale(v) + 3}" text-anchor="end">${v}</text>
                `).join('')}
                <!-- CPU line -->
                <path d="${cpuPath}" fill="none" stroke="#00D9FF" stroke-width="2" stroke-linecap="round"/>
                <!-- Memory line -->
                <path d="${memPath}" fill="none" stroke="#3FB950" stroke-width="2" stroke-linecap="round" stroke-dasharray="4 2"/>
                <!-- FPS line (scaled) -->
                <path d="${fpsPath}" fill="none" stroke="#D29922" stroke-width="1.5" stroke-linecap="round" stroke-dasharray="2 2"/>
                <!-- Legend -->
                <g transform="translate(${padding.left}, ${h - 8})">
                    <circle cx="0" cy="-3" r="4" fill="#00D9FF"/>
                    <text class="chart-label" x="8" y="0">CPU</text>
                    <circle cx="60" cy="-3" r="4" fill="#3FB950"/>
                    <text class="chart-label" x="68" y="0">内存</text>
                    <circle cx="120" cy="-3" r="4" fill="#D29922"/>
                    <text class="chart-label" x="128" y="0">FPS (×2)</text>
                </g>
            </svg>
        `;
    }

    // ====================================================================
    // PAGE: DEVICES
    // ====================================================================
    let deviceFilter = { type: '', status: '' };

    async function renderDevices() {
        const content = $('#pageContent');
        content.innerHTML = `
            <div class="filter-bar">
                <select class="form-select" id="deviceTypeFilter" style="width:auto;min-width:140px">
                    <option value="">全部类型</option>
                    <option value="camera">摄像头</option>
                    <option value="ptz">云台</option>
                    <option value="detector">检测器</option>
                    <option value="tracker">跟踪器</option>
                    <option value="stream">视频流</option>
                </select>
                <select class="form-select" id="deviceStatusFilter" style="width:auto;min-width:140px">
                    <option value="">全部状态</option>
                    <option value="online">在线</option>
                    <option value="offline">离线</option>
                    <option value="connecting">连接中</option>
                    <option value="error">错误</option>
                </select>
                <button class="btn btn-primary" id="addDeviceBtn">${icons.plus} 添加设备</button>
            </div>
            <div class="device-grid" id="deviceGrid">${loadingState('加载设备...')}</div>
        `;

        $('#deviceTypeFilter').value = deviceFilter.type;
        $('#deviceStatusFilter').value = deviceFilter.status;
        $('#deviceTypeFilter').addEventListener('change', (e) => {
            deviceFilter.type = e.target.value;
            loadDevices();
        });
        $('#deviceStatusFilter').addEventListener('change', (e) => {
            deviceFilter.status = e.target.value;
            loadDevices();
        });
        $('#addDeviceBtn').addEventListener('click', () => showDeviceForm());

        await loadDevices();
    }

    async function loadDevices() {
        const grid = $('#deviceGrid');
        if (!grid) return;
        grid.innerHTML = loadingState('加载设备...');

        try {
            const params = {};
            if (deviceFilter.type) params.type = deviceFilter.type;
            if (deviceFilter.status) params.status = deviceFilter.status;
            const devices = await SkyGuardAPI.devices.list(params);

            if (!devices || devices.length === 0) {
                grid.innerHTML = emptyState('暂无设备', '点击"添加设备"来注册新的监控设备');
                return;
            }

            grid.innerHTML = devices.map((d) => `
                <div class="device-card">
                    <div class="device-header">
                        <div>
                            <div class="device-name">${d.name}</div>
                            <div class="device-type">${deviceTypeLabel(d.type)}</div>
                        </div>
                        ${deviceStatusBadge(d.status)}
                    </div>
                    <div class="device-info">
                        <div class="device-info-row">
                            <span class="device-info-label">ID</span>
                            <span class="device-info-value">${d.id.substring(0, 8)}...</span>
                        </div>
                        <div class="device-info-row">
                            <span class="device-info-label">地址</span>
                            <span class="device-info-value">${d.host}:${d.port}</span>
                        </div>
                        <div class="device-info-row">
                            <span class="device-info-label">最后活动</span>
                            <span class="device-info-value">${timeAgo(d.last_seen)}</span>
                        </div>
                        ${d.error ? `<div class="device-info-row"><span class="device-info-label">错误</span><span class="device-info-value" style="color:var(--danger)">${d.error}</span></div>` : ''}
                    </div>
                    <div class="device-actions">
                        ${d.status === 'online'
                            ? `<button class="btn btn-secondary btn-sm" onclick="SkyGuardApp.disconnectDevice('${d.id}')">${icons.power} 断开</button>`
                            : `<button class="btn btn-primary btn-sm" onclick="SkyGuardApp.connectDevice('${d.id}')">${icons.power} 连接</button>`
                        }
                        <button class="btn btn-ghost btn-sm" onclick="SkyGuardApp.editDevice('${d.id}')">${icons.edit} 编辑</button>
                        <button class="btn btn-ghost btn-sm" onclick="SkyGuardApp.deleteDevice('${d.id}')" style="color:var(--danger)">${icons.trash}</button>
                    </div>
                </div>
            `).join('');
        } catch (err) {
            grid.innerHTML = `<div class="card">${emptyState('加载失败', err.message)}</div>`;
        }
    }

    function showDeviceForm(device = null) {
        const isEdit = !!device;
        const d = device || { id: '', name: '', type: 'camera', status: 'offline', host: '', port: 554, config: {} };
        const body = `
            <form id="deviceForm">
                <div class="form-group">
                    <label class="form-label">设备名称</label>
                    <input class="form-input" name="name" value="${d.name}" placeholder="例如: 前门摄像头" required>
                </div>
                <div class="form-row">
                    <div class="form-group">
                        <label class="form-label">设备类型</label>
                        <select class="form-select" name="type">
                            <option value="camera" ${d.type === 'camera' ? 'selected' : ''}>摄像头</option>
                            <option value="ptz" ${d.type === 'ptz' ? 'selected' : ''}>云台</option>
                            <option value="detector" ${d.type === 'detector' ? 'selected' : ''}>检测器</option>
                            <option value="tracker" ${d.type === 'tracker' ? 'selected' : ''}>跟踪器</option>
                            <option value="stream" ${d.type === 'stream' ? 'selected' : ''}>视频流</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label class="form-label">初始状态</label>
                        <select class="form-select" name="status">
                            <option value="offline" ${d.status === 'offline' ? 'selected' : ''}>离线</option>
                            <option value="online" ${d.status === 'online' ? 'selected' : ''}>在线</option>
                            <option value="error" ${d.status === 'error' ? 'selected' : ''}>错误</option>
                        </select>
                    </div>
                </div>
                <div class="form-row">
                    <div class="form-group">
                        <label class="form-label">主机地址</label>
                        <input class="form-input" name="host" value="${d.host}" placeholder="192.168.1.100" required>
                    </div>
                    <div class="form-group">
                        <label class="form-label">端口</label>
                        <input class="form-input" type="number" name="port" value="${d.port}" placeholder="554" required>
                    </div>
                </div>
                ${isEdit ? `<input type="hidden" name="id" value="${d.id}">` : ''}
            </form>
        `;
        const footer = `
            <button class="btn btn-secondary" onclick="document.getElementById('modalOverlay').style.display='none'">取消</button>
            <button class="btn btn-primary" onclick="SkyGuardApp.saveDevice(${isEdit})">${isEdit ? '更新' : '创建'}</button>
        `;
        showModal(isEdit ? '编辑设备' : '添加设备', body, footer);
    }

    async function saveDevice(isEdit) {
        const form = $('#deviceForm');
        const formData = new FormData(form);
        const data = {
            name: formData.get('name'),
            type: formData.get('type'),
            status: formData.get('status'),
            host: formData.get('host'),
            port: parseInt(formData.get('port')),
            config: {},
        };

        try {
            if (isEdit) {
                data.id = formData.get('id');
                await SkyGuardAPI.devices.update(data.id, data);
                toast('设备更新成功', 'success');
            } else {
                data.id = 'dev-' + Date.now().toString(36);
                await SkyGuardAPI.devices.create(data);
                toast('设备创建成功', 'success');
            }
            closeModal();
            loadDevices();
        } catch (err) {
            toast('保存失败: ' + err.message, 'error');
        }
    }

    async function editDevice(id) {
        try {
            const device = await SkyGuardAPI.devices.get(id);
            showDeviceForm(device);
        } catch (err) {
            toast('加载设备失败: ' + err.message, 'error');
        }
    }

    async function deleteDevice(id) {
        if (!confirm('确定要删除此设备吗？')) return;
        try {
            await SkyGuardAPI.devices.delete(id);
            toast('设备已删除', 'success');
            loadDevices();
        } catch (err) {
            toast('删除失败: ' + err.message, 'error');
        }
    }

    async function connectDevice(id) {
        try {
            await SkyGuardAPI.devices.connect(id);
            toast('设备已连接', 'success');
            loadDevices();
        } catch (err) {
            toast('连接失败: ' + err.message, 'error');
        }
    }

    async function disconnectDevice(id) {
        try {
            await SkyGuardAPI.devices.disconnect(id);
            toast('设备已断开', 'success');
            loadDevices();
        } catch (err) {
            toast('断开失败: ' + err.message, 'error');
        }
    }

    // ====================================================================
    // PAGE: ALERTS
    // ====================================================================
    let alertFilter = { level: '', acknowledged: '' };

    async function renderAlerts() {
        const content = $('#pageContent');
        content.innerHTML = `
            <div class="filter-bar">
                <select class="form-select" id="alertLevelFilter" style="width:auto;min-width:140px">
                    <option value="">全部级别</option>
                    <option value="critical">严重</option>
                    <option value="warning">警告</option>
                    <option value="info">信息</option>
                </select>
                <select class="form-select" id="alertAckFilter" style="width:auto;min-width:140px">
                    <option value="">全部状态</option>
                    <option value="false">未确认</option>
                    <option value="true">已确认</option>
                </select>
                <button class="btn btn-danger" id="clearAlertsBtn">${icons.trash} 清空全部</button>
            </div>
            <div class="alert-list" id="alertList">${loadingState('加载告警...')}</div>
        `;

        $('#alertLevelFilter').value = alertFilter.level;
        $('#alertAckFilter').value = alertFilter.acknowledged;
        $('#alertLevelFilter').addEventListener('change', (e) => {
            alertFilter.level = e.target.value;
            loadAlerts();
        });
        $('#alertAckFilter').addEventListener('change', (e) => {
            alertFilter.acknowledged = e.target.value;
            loadAlerts();
        });
        $('#clearAlertsBtn').addEventListener('click', async () => {
            if (!confirm('确定要清空所有告警吗？此操作不可撤销。')) return;
            try {
                await SkyGuardAPI.alerts.deleteAll();
                toast('所有告警已清空', 'success');
                loadAlerts();
                updateAlertBadge();
            } catch (err) {
                toast('清空失败: ' + err.message, 'error');
            }
        });

        await loadAlerts();
    }

    async function loadAlerts() {
        const list = $('#alertList');
        if (!list) return;
        list.innerHTML = loadingState('加载告警...');

        try {
            const params = { limit: 100 };
            if (alertFilter.level) params.level = alertFilter.level;
            if (alertFilter.acknowledged !== '') params.acknowledged = alertFilter.acknowledged === 'true';
            const alerts = await SkyGuardAPI.alerts.list(params);

            if (!alerts || alerts.length === 0) {
                list.innerHTML = emptyState('暂无告警', '系统运行正常，没有待处理的告警');
                return;
            }

            const iconMap = {
                critical: icons.alert,
                warning: icons.alert,
                info: icons.info,
            };

            list.innerHTML = alerts.map((a) => `
                <div class="alert-item ${a.acknowledged ? '' : 'unacknowledged'} level-${a.level}">
                    <div class="alert-icon ${a.level}">${iconMap[a.level] || icons.info}</div>
                    <div class="alert-content">
                        <div class="alert-message">${a.message}</div>
                        <div class="alert-meta">
                            <span class="alert-meta-item">${alertLevelBadge(a.level)}</span>
                            <span class="alert-meta-item">${alertTypeLabel(a.type)}</span>
                            ${a.device_id ? `<span class="alert-meta-item">${icons.server} ${a.device_id.substring(0, 8)}</span>` : ''}
                            <span class="alert-meta-item">${icons.clock} ${timeAgo(a.timestamp)}</span>
                        </div>
                    </div>
                    <div class="alert-actions">
                        ${!a.acknowledged
                            ? `<button class="btn btn-ghost btn-sm" onclick="SkyGuardApp.acknowledgeAlert('${a.id}')" title="确认">${icons.check}</button>`
                            : ''
                        }
                        <button class="btn btn-ghost btn-sm" onclick="SkyGuardApp.deleteAlert('${a.id}')" style="color:var(--danger)" title="删除">${icons.trash}</button>
                    </div>
                </div>
            `).join('');
        } catch (err) {
            list.innerHTML = `<div class="card">${emptyState('加载失败', err.message)}</div>`;
        }
    }

    async function acknowledgeAlert(id) {
        try {
            await SkyGuardAPI.alerts.acknowledge(id);
            toast('告警已确认', 'success');
            loadAlerts();
            updateAlertBadge();
        } catch (err) {
            toast('操作失败: ' + err.message, 'error');
        }
    }

    async function deleteAlert(id) {
        try {
            await SkyGuardAPI.alerts.delete(id);
            toast('告警已删除', 'success');
            loadAlerts();
            updateAlertBadge();
        } catch (err) {
            toast('删除失败: ' + err.message, 'error');
        }
    }

    async function updateAlertBadge() {
        try {
            const result = await SkyGuardAPI.alerts.getCount({ acknowledged: false });
            const count = result.count || 0;
            const badge = $('#alertBadge');
            if (count > 0) {
                badge.textContent = count > 99 ? '99+' : count;
                badge.style.display = 'flex';
            } else {
                badge.style.display = 'none';
            }
        } catch {
            // Silent fail
        }
    }

    // ====================================================================
    // PAGE: CONFIG
    // ====================================================================
    async function renderConfig() {
        const content = $('#pageContent');
        content.innerHTML = `
            <div class="section-header">
                <div>
                    <div class="section-title">系统配置</div>
                    <div class="section-subtitle">管理摄像头、云台和禁飞区配置</div>
                </div>
                <button class="btn btn-secondary" id="reloadConfigBtn">${icons.power} 重新加载</button>
            </div>
            <div id="configContainer">${loadingState('加载配置...')}</div>
        `;

        $('#reloadConfigBtn').addEventListener('click', async () => {
            try {
                await SkyGuardAPI.config.reload();
                toast('配置已重新加载', 'success');
                renderConfig();
            } catch (err) {
                toast('加载失败: ' + err.message, 'error');
            }
        });

        try {
            const [systemCfg, cameras, ptzs, zones] = await Promise.all([
                SkyGuardAPI.config.getSystem(),
                SkyGuardAPI.config.listCameras(),
                SkyGuardAPI.config.listPTZ(),
                SkyGuardAPI.config.listZones(),
            ]);

            const container = $('#configContainer');
            container.innerHTML = `
                ${renderConfigSection('system', '系统设置', renderSystemConfig(systemCfg))}
                ${renderConfigSection('cameras', `摄像头配置 (${cameras.length})`, renderCamerasConfig(cameras))}
                ${renderConfigSection('ptz', `云台配置 (${ptzs.length})`, renderPTZConfig(ptzs))}
                ${renderConfigSection('zones', `禁飞区配置 (${zones.length})`, renderZonesConfig(zones))}
            `;

            // Attach collapse handlers
            $$('.config-section-header').forEach((header) => {
                header.addEventListener('click', () => {
                    header.parentElement.classList.toggle('collapsed');
                });
            });
        } catch (err) {
            $('#configContainer').innerHTML = `<div class="card">${emptyState('加载失败', err.message)}</div>`;
        }
    }

    function renderConfigSection(id, title, bodyHtml) {
        return `
            <div class="config-section" id="section-${id}">
                <div class="config-section-header">
                    <span style="font-size:15px;font-weight:600">${title}</span>
                    <svg class="config-section-chevron" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="6 9 12 15 18 9"/>
                    </svg>
                </div>
                <div class="config-section-body">${bodyHtml}</div>
            </div>
        `;
    }

    function renderSystemConfig(cfg) {
        const sections = [
            { label: '应用', data: cfg.app },
            { label: '计算设备', data: cfg.compute },
            { label: '检测', data: cfg.detection },
            { label: '跟踪', data: cfg.tracking },
            { label: 'Web 服务', data: cfg.web },
            { label: '数据库', data: cfg.database },
        ];
        return sections.map((sec) => {
            const rows = Object.entries(sec.data || {}).map(([k, v]) => {
                let display = v;
                if (typeof v === 'object') display = JSON.stringify(v);
                if (typeof v === 'boolean') display = v ? '是' : '否';
                return `<div style="display:flex;justify-content:space-between;padding:6px 0;font-size:13px;border-bottom:1px solid var(--border-subtle)">
                    <span style="color:var(--text-muted)">${k}</span>
                    <span style="font-family:var(--font-mono);color:var(--text-primary);max-width:60%;text-align:right;word-break:break-all">${display}</span>
                </div>`;
            }).join('');
            return `<div style="margin-bottom:16px">
                <div style="font-size:12px;font-weight:600;color:var(--accent);text-transform:uppercase;letter-spacing:0.05em;margin-bottom:8px">${sec.label}</div>
                ${rows}
            </div>`;
        }).join('');
    }

    function renderCamerasConfig(cameras) {
        if (!cameras || cameras.length === 0) {
            return emptyState('暂无摄像头配置', '通过 API 添加摄像头配置');
        }
        return `<div class="table-container"><table>
            <thead><tr><th>名称</th><th>地址</th><th>分辨率</th><th>FPS</th><th>状态</th></tr></thead>
            <tbody>
                ${cameras.map((c) => `
                    <tr>
                        <td>${c.name}</td>
                        <td class="mono">${c.host}:${c.port}</td>
                        <td class="mono">${c.resolution}</td>
                        <td class="mono">${c.fps}</td>
                        <td>${c.enabled ? '<span class="badge badge-success">启用</span>' : '<span class="badge badge-neutral">禁用</span>'}</td>
                    </tr>
                `).join('')}
            </tbody>
        </table></div>`;
    }

    function renderPTZConfig(ptzs) {
        if (!ptzs || ptzs.length === 0) {
            return emptyState('暂无云台配置', '通过 API 添加云台配置');
        }
        return `<div class="table-container"><table>
            <thead><tr><th>名称</th><th>地址</th><th>端口</th><th>自动跟踪</th><th>状态</th></tr></thead>
            <tbody>
                ${ptzs.map((p) => `
                    <tr>
                        <td>${p.name}</td>
                        <td class="mono">${p.host}</td>
                        <td class="mono">${p.port}</td>
                        <td>${p.auto_tracking ? '<span class="badge badge-success">是</span>' : '<span class="badge badge-neutral">否</span>'}</td>
                        <td>${p.enabled ? '<span class="badge badge-success">启用</span>' : '<span class="badge badge-neutral">禁用</span>'}</td>
                    </tr>
                `).join('')}
            </tbody>
        </table></div>`;
    }

    function renderZonesConfig(zones) {
        if (!zones || zones.length === 0) {
            return emptyState('暂无禁飞区配置', '通过 API 添加地理围栏区域');
        }
        return `<div class="table-container"><table>
            <thead><tr><th>名称</th><th>坐标点数</th><th>入侵告警</th><th>离开告警</th><th>状态</th></tr></thead>
            <tbody>
                ${zones.map((z) => `
                    <tr>
                        <td>${z.name}</td>
                        <td class="mono">${z.coordinates ? z.coordinates.length : 0}</td>
                        <td>${z.alert_on_entry ? '<span class="badge badge-danger">开</span>' : '<span class="badge badge-neutral">关</span>'}</td>
                        <td>${z.alert_on_exit ? '<span class="badge badge-warning">开</span>' : '<span class="badge badge-neutral">关</span>'}</td>
                        <td>${z.enabled ? '<span class="badge badge-success">启用</span>' : '<span class="badge badge-neutral">禁用</span>'}</td>
                    </tr>
                `).join('')}
            </tbody>
        </table></div>`;
    }

    // ====================================================================
    // ROUTING
    // ====================================================================
    const pages = {
        dashboard: { title: '仪表盘', render: renderDashboard },
        devices: { title: '设备管理', render: renderDevices },
        alerts: { title: '告警中心', render: renderAlerts },
        config: { title: '系统配置', render: renderConfig },
    };

    function navigate(page) {
        if (!pages[page]) page = 'dashboard';
        state.currentPage = page;

        // Update nav
        $$('.nav-item').forEach((item) => {
            item.classList.toggle('active', item.dataset.page === page);
        });

        // Update title
        $('#pageTitle').textContent = pages[page].title;

        // Render page
        pages[page].render();

        // Update hash
        if (location.hash !== '#' + page) {
            history.replaceState(null, '', '#' + page);
        }
    }

    // ====================================================================
    // INITIALIZATION
    // ====================================================================
    function startClock() {
        function update() {
            const now = new Date();
            const time = now.toLocaleTimeString('zh-CN', { hour12: false });
            const date = now.toLocaleDateString('zh-CN');
            const clock = $('#systemClock');
            if (clock) clock.textContent = `${date} ${time}`;
        }
        update();
        setInterval(update, 1000);
    }

    function startAutoRefresh() {
        if (state.refreshTimer) clearInterval(state.refreshTimer);
        state.refreshTimer = setInterval(() => {
            if (state.currentPage === 'dashboard') {
                renderDashboard();
            }
            updateAlertBadge();
        }, state.refreshInterval);
    }

    function stopAutoRefresh() {
        if (state.refreshTimer) {
            clearInterval(state.refreshTimer);
            state.refreshTimer = null;
        }
    }

    function connectAlertWebSocket() {
        if (state.wsAlert) state.wsAlert.close();
        state.wsAlert = SkyGuardAPI.connectWebSocket('alert', (data) => {
            // New alert received
            toast('新告警: ' + (data.data?.message || '未知'), data.data?.level === 'critical' ? 'error' : 'warning');
            updateAlertBadge();
            if (state.currentPage === 'alerts') {
                loadAlerts();
            }
        }, (status) => {
            const connStatus = $('#connStatus');
            if (connStatus) {
                const dot = connStatus.querySelector('.status-dot');
                const text = connStatus.querySelector('.status-text');
                if (status === 'connected') {
                    dot.className = 'status-dot online';
                    text.textContent = '已连接';
                } else {
                    dot.className = 'status-dot offline';
                    text.textContent = '已断开';
                }
            }
        });
    }

    function init() {
        // Sidebar toggle
        $('#sidebarToggle').addEventListener('click', () => {
            $('#sidebar').classList.toggle('collapsed');
        });

        // Nav items
        $$('.nav-item').forEach((item) => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                navigate(item.dataset.page);
            });
        });

        // Hash change routing (back/forward, direct URL)
        window.addEventListener('hashchange', () => {
            const hash = location.hash.substring(1);
            if (hash && pages[hash] && hash !== state.currentPage) {
                navigate(hash);
            }
        });

        // Refresh button
        $('#refreshBtn').addEventListener('click', async () => {
            const btn = $('#refreshBtn');
            btn.classList.add('spinning');
            await pages[state.currentPage].render();
            updateAlertBadge();
            setTimeout(() => btn.classList.remove('spinning'), 600);
            toast('数据已刷新', 'info', 1500);
        });

        // Initial route from hash
        const hash = location.hash.substring(1);
        navigate(hash || 'dashboard');

        // Start background tasks
        startClock();
        startAutoRefresh();
        updateAlertBadge();
        connectAlertWebSocket();
    }

    // --- Expose for inline event handlers --------------------------------
    window.SkyGuardApp = {
        saveDevice,
        editDevice,
        deleteDevice,
        connectDevice,
        disconnectDevice,
        acknowledgeAlert,
        deleteAlert,
    };

    // Start when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
