/**
 * AI Training For DumDums - Frontend Application
 * Professional-grade JavaScript with WebSocket support and toast notifications
 *
 * @version 2.0.0
 */

// =============================================================================
// Constants & Configuration
// =============================================================================

const CONFIG = Object.freeze({
    POLLING_INTERVAL: 1000,
    TOAST_DURATION: 5000,
    TOAST_DURATION_ERROR: 8000,
    MAX_VISIBLE_TOASTS: 3,
    RECONNECT_DELAY: 2000,
    MAX_RECONNECT_ATTEMPTS: 5
});

const TOAST_TYPES = Object.freeze({
    SUCCESS: 'success',
    ERROR: 'error',
    WARNING: 'warning',
    INFO: 'info'
});

// =============================================================================
// State Management
// =============================================================================

class AppState {
    constructor() {
        this.parameters = {};
        this.presets = {};
        this.currentConfig = {};
        this.isConnected = false;
        this.reconnectAttempts = 0;
    }

    setParameters(params) {
        this.parameters = params;
    }

    setPresets(presets) {
        this.presets = presets;
    }

    updateConfig(key, value) {
        this.currentConfig[key] = value;
    }

    getConfig() {
        return { ...this.currentConfig };
    }

    initConfigFromDefaults() {
        Object.entries(this.parameters).forEach(([key, param]) => {
            if (this.currentConfig[key] === undefined) {
                this.currentConfig[key] = param.default;
            }
        });
    }
}

const state = new AppState();

// =============================================================================
// Toast Notification System
// =============================================================================

class ToastManager {
    constructor() {
        this.container = null;
        this.toasts = [];
        this.init();
    }

    init() {
        this.container = document.createElement('div');
        this.container.className = 'toast-container';
        this.container.setAttribute('role', 'alert');
        this.container.setAttribute('aria-live', 'polite');
        document.body.appendChild(this.container);
    }

    show(message, type = TOAST_TYPES.INFO, duration = CONFIG.TOAST_DURATION) {
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;

        const icon = this.getIcon(type);
        toast.innerHTML = `
            <span class="toast-icon">${icon}</span>
            <span class="toast-message">${this.escapeHtml(message)}</span>
            <button class="toast-close" aria-label="Close notification">&times;</button>
        `;

        // Add close functionality
        const closeBtn = toast.querySelector('.toast-close');
        closeBtn.addEventListener('click', () => this.remove(toast));

        // Add to container
        this.container.appendChild(toast);
        this.toasts.push(toast);

        // Trigger animation
        requestAnimationFrame(() => {
            toast.classList.add('toast-visible');
        });

        // Auto remove
        const timeoutDuration = type === TOAST_TYPES.ERROR ? CONFIG.TOAST_DURATION_ERROR : duration;
        setTimeout(() => this.remove(toast), timeoutDuration);

        // Limit visible toasts
        this.enforceMaxToasts();

        return toast;
    }

    remove(toast) {
        if (!toast || !toast.parentNode) return;

        toast.classList.remove('toast-visible');
        toast.classList.add('toast-hiding');

        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
            this.toasts = this.toasts.filter(t => t !== toast);
        }, 300);
    }

    enforceMaxToasts() {
        while (this.toasts.length > CONFIG.MAX_VISIBLE_TOASTS) {
            this.remove(this.toasts[0]);
        }
    }

    getIcon(type) {
        const icons = {
            [TOAST_TYPES.SUCCESS]: '&#10003;',
            [TOAST_TYPES.ERROR]: '&#10007;',
            [TOAST_TYPES.WARNING]: '&#9888;',
            [TOAST_TYPES.INFO]: '&#8505;'
        };
        return icons[type] || icons[TOAST_TYPES.INFO];
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    success(message) {
        return this.show(message, TOAST_TYPES.SUCCESS);
    }

    error(message) {
        return this.show(message, TOAST_TYPES.ERROR);
    }

    warning(message) {
        return this.show(message, TOAST_TYPES.WARNING);
    }

    info(message) {
        return this.show(message, TOAST_TYPES.INFO);
    }
}

const toast = new ToastManager();

// =============================================================================
// WebSocket Connection Manager
// =============================================================================

class WebSocketManager {
    constructor() {
        this.socket = null;
        this.pollInterval = null;
        this.isWebSocketAvailable = false;
    }

    connect() {
        try {
            // Check if Socket.IO is available
            if (typeof io === 'undefined') {
                console.log('Socket.IO not available, falling back to polling');
                this.isWebSocketAvailable = false;
                return;
            }

            this.socket = io({
                transports: ['websocket', 'polling'],
                reconnection: true,
                reconnectionDelay: CONFIG.RECONNECT_DELAY,
                reconnectionAttempts: CONFIG.MAX_RECONNECT_ATTEMPTS
            });

            this.setupEventHandlers();
            this.isWebSocketAvailable = true;
        } catch (error) {
            console.warn('WebSocket connection failed, using polling fallback:', error);
            this.isWebSocketAvailable = false;
        }
    }

    setupEventHandlers() {
        if (!this.socket) return;

        this.socket.on('connect', () => {
            state.isConnected = true;
            state.reconnectAttempts = 0;
            console.log('WebSocket connected');
            this.updateConnectionStatus(true);
        });

        this.socket.on('disconnect', () => {
            state.isConnected = false;
            console.log('WebSocket disconnected');
            this.updateConnectionStatus(false);
        });

        this.socket.on('connect_error', (error) => {
            console.warn('WebSocket connection error:', error);
            state.reconnectAttempts++;
            if (state.reconnectAttempts >= CONFIG.MAX_RECONNECT_ATTEMPTS) {
                toast.warning('Real-time updates unavailable. Using polling.');
                this.fallbackToPolling();
            }
        });

        this.socket.on('training_update', (data) => {
            this.handleTrainingUpdate(data);
        });

        this.socket.on('training_log', (entry) => {
            this.handleLogEntry(entry);
        });

        // Loss data for real-time chart
        this.socket.on('loss_data', (data) => {
            console.log('Received loss_data:', data);
            if (typeof lossChart !== 'undefined' && lossChart) {
                console.log('lossChart exists, adding point');
                lossChart.addDataPoint(
                    data.step,
                    data.loss,
                    data.eval_loss || null,
                    data.learning_rate || null
                );
            } else {
                console.log('lossChart not available:', typeof lossChart);
            }
        });
    }

    handleTrainingUpdate(data) {
        updateProgress(data);

        if (!data.is_training && this.pollInterval) {
            this.stopPolling();
            resetTrainingUI();
        }
    }

    handleLogEntry(entry) {
        addLogEntry(entry.message, entry.level, entry.time);
    }

    startPolling() {
        if (this.pollInterval) return;

        this.pollInterval = setInterval(async () => {
            try {
                const response = await fetch('/api/status');
                if (!response.ok) throw new Error('Status request failed');

                const status = await response.json();
                updateProgress(status);
                updateLogs(status.logs);

                if (!status.is_training) {
                    this.stopPolling();
                    resetTrainingUI();
                }
            } catch (error) {
                console.error('Status poll failed:', error);
            }
        }, CONFIG.POLLING_INTERVAL);
    }

    stopPolling() {
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
            this.pollInterval = null;
        }
    }

    fallbackToPolling() {
        if (this.socket) {
            this.socket.disconnect();
            this.socket = null;
        }
        this.isWebSocketAvailable = false;
    }

    updateConnectionStatus(connected) {
        const indicator = document.getElementById('connection-status');
        if (indicator) {
            indicator.className = `connection-status ${connected ? 'connected' : 'disconnected'}`;
            indicator.title = connected ? 'Real-time updates active' : 'Reconnecting...';
        }
    }

    requestStatus() {
        if (this.socket && this.socket.connected) {
            this.socket.emit('request_status');
        }
    }
}

const wsManager = new WebSocketManager();

// =============================================================================
// API Client
// =============================================================================

class APIClient {
    static async get(endpoint) {
        const response = await fetch(endpoint);
        if (!response.ok) {
            const error = await response.json().catch(() => ({ error: 'Request failed' }));
            throw new Error(error.error || 'Request failed');
        }
        return response.json();
    }

    static async post(endpoint, data = {}) {
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || 'Request failed');
        }

        return result;
    }

    static async getParameters() {
        return this.get('/api/parameters');
    }

    static async getPresets() {
        return this.get('/api/presets');
    }

    static async startTraining(config) {
        return this.post('/api/train', config);
    }

    static async stopTraining() {
        return this.post('/api/stop');
    }

    static async getStatus() {
        return this.get('/api/status');
    }
}

// =============================================================================
// Initialization
// =============================================================================

document.addEventListener('DOMContentLoaded', async () => {
    try {
        await Promise.all([
            loadParameters(),
            loadPresets()
        ]);

        setupEventListeners();
        wsManager.connect();

        // Check for existing training session
        const status = await APIClient.getStatus();
        if (status.is_training) {
            showTrainingUI();
            updateProgress(status);
            updateLogs(status.logs);

            if (!wsManager.isWebSocketAvailable) {
                wsManager.startPolling();
            }
        }
    } catch (error) {
        console.error('Initialization failed:', error);
        toast.error('Failed to initialize application. Please refresh the page.');
    }
});

async function loadParameters() {
    try {
        const params = await APIClient.getParameters();
        state.setParameters(params);
        state.initConfigFromDefaults();
        renderParameters();
    } catch (error) {
        console.error('Failed to load parameters:', error);
        toast.error('Failed to load parameter definitions');
        throw error;
    }
}

async function loadPresets() {
    try {
        const presets = await APIClient.getPresets();
        state.setPresets(presets);
        renderPresets();
    } catch (error) {
        console.error('Failed to load presets:', error);
        // Non-critical, don't throw
    }
}

function setupEventListeners() {
    const trainingDataEl = document.getElementById('training-data');
    if (trainingDataEl) {
        trainingDataEl.addEventListener('input', updateDataStats);
        // Initialize stats
        updateDataStats();
    }

    // Keyboard shortcuts
    document.addEventListener('keydown', handleKeyboardShortcuts);

    // Modal close on outside click
    document.addEventListener('click', handleModalClick);
}

// =============================================================================
// Rendering Functions
// =============================================================================

function renderPresets() {
    const container = document.getElementById('presets-container');
    if (!container) return;

    const presetsHtml = Object.entries(state.presets).map(([key, preset]) => `
        <div class="preset-card" data-preset="${key}" onclick="applyPreset('${key}')"
             role="button" tabindex="0" aria-label="Apply ${preset.name} preset">
            <h3>${escapeHtml(preset.name)}</h3>
            <p>${escapeHtml(preset.description)}</p>
        </div>
    `).join('');

    container.innerHTML = presetsHtml;

    // Add keyboard support for presets
    container.querySelectorAll('.preset-card').forEach(card => {
        card.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                applyPreset(card.dataset.preset);
            }
        });
    });
}

function renderParameters() {
    const groups = {
        'model-params': ['model_name'],
        'training-params': ['learning_rate', 'epochs', 'batch_size', 'max_length'],
        'advanced-params': ['warmup_steps', 'weight_decay', 'gradient_accumulation', 'optimizer', 'mixed_precision', 'gradient_checkpointing', 'attn_implementation', 'neftune_alpha', 'save_steps', 'logging_steps', 'seed'],
        'lora-params': ['use_lora', 'lora_r', 'lora_alpha', 'lora_dropout', 'use_dora', 'lora_bias']
    };

    Object.entries(groups).forEach(([containerId, paramKeys]) => {
        renderParamGroup(containerId, paramKeys);
    });
}

function renderParamGroup(containerId, paramKeys) {
    const container = document.getElementById(containerId);
    if (!container) return;

    const html = paramKeys
        .filter(key => state.parameters[key])
        .map(key => renderParameter(key, state.parameters[key]))
        .join('');

    container.innerHTML = html;
}

function renderParameter(key, param) {
    let inputHtml = '';

    switch (param.type) {
        case 'slider':
            inputHtml = renderSlider(key, param);
            break;
        case 'select':
            inputHtml = renderSelect(key, param);
            break;
        case 'select_with_custom':
            inputHtml = renderSelectWithCustom(key, param);
            break;
        case 'checkbox':
            inputHtml = renderCheckbox(key, param);
            break;
        case 'number':
            inputHtml = renderNumber(key, param);
            break;
        default:
            inputHtml = `<input type="text" id="${key}" value="${param.default}" onchange="updateValue('${key}', this.value)">`;
    }

    const dependsAttr = param.depends_on ? `data-depends-on="${param.depends_on}"` : '';
    const valueDisplay = (param.type === 'slider' || param.type === 'number')
        ? `<span class="param-value" id="${key}-display">${formatValue(param.default, param)}</span>`
        : '';

    return `
        <div class="param-group" id="param-${key}" ${dependsAttr}>
            <div class="param-header">
                <label class="param-label" for="${key}">${escapeHtml(param.name)}</label>
                ${valueDisplay}
            </div>
            <p class="param-explanation">${escapeHtml(param.explanation)}</p>
            ${param.beginner_tip ? `<div class="param-tip">${escapeHtml(param.beginner_tip)}</div>` : ''}
            ${inputHtml}
        </div>
    `;
}

function renderSlider(key, param) {
    const step = param.step || 1;
    const displayFormat = param.display_format || 'normal';
    const currentValue = state.currentConfig[key] ?? param.default;

    return `
        <input
            type="range"
            id="${key}"
            min="${param.min}"
            max="${param.max}"
            step="${step}"
            value="${currentValue}"
            oninput="updateSliderValue('${key}', this.value, '${displayFormat}')"
            aria-describedby="${key}-display"
        >
    `;
}

function renderSelect(key, param) {
    const currentValue = state.currentConfig[key] ?? param.default;
    const options = param.options.map(opt =>
        `<option value="${opt.value}" ${opt.value === currentValue ? 'selected' : ''}>${escapeHtml(opt.label)}</option>`
    ).join('');

    return `
        <select id="${key}" onchange="updateValue('${key}', this.value)">
            ${options}
        </select>
    `;
}

function renderSelectWithCustom(key, param) {
    const currentValue = state.currentConfig[key] ?? param.default;
    const isCustom = currentValue === 'custom' || (currentValue && !param.options.some(o => o.value === currentValue));

    const options = param.options.map(opt =>
        `<option value="${opt.value}" ${opt.value === currentValue ? 'selected' : ''}>${escapeHtml(opt.label)}</option>`
    ).join('');

    // If current value is a custom path (not in options), select 'custom' and show the path
    const customPath = isCustom && currentValue !== 'custom' ? currentValue : '';
    const showCustomInput = isCustom ? 'display: block;' : 'display: none;';

    return `
        <select id="${key}" onchange="handleSelectWithCustom('${key}', this.value)">
            ${options}
        </select>
        <input
            type="text"
            id="${key}-custom"
            placeholder="Enter model path (e.g., ./Outputs/final_merged)"
            value="${customPath}"
            style="margin-top: 8px; ${showCustomInput}"
            onchange="updateValue('${key}', this.value)"
        >
    `;
}

function handleSelectWithCustom(key, value) {
    const customInput = document.getElementById(`${key}-custom`);
    if (customInput) {
        if (value === 'custom') {
            customInput.style.display = 'block';
            customInput.focus();
            // Don't update config yet - wait for user to enter path
        } else {
            customInput.style.display = 'none';
            updateValue(key, value);
        }
    }
}

function renderCheckbox(key, param) {
    const currentValue = state.currentConfig[key] ?? param.default;
    return `
        <label class="checkbox-wrapper">
            <input
                type="checkbox"
                id="${key}"
                ${currentValue ? 'checked' : ''}
                onchange="updateCheckbox('${key}', this.checked)"
            >
            <span>Enable ${escapeHtml(param.name)}</span>
        </label>
    `;
}

function renderNumber(key, param) {
    const currentValue = state.currentConfig[key] ?? param.default;
    return `
        <input
            type="number"
            id="${key}"
            min="${param.min || 0}"
            max="${param.max || 999999}"
            value="${currentValue}"
            onchange="updateNumberValue('${key}', this.value)"
            aria-describedby="${key}-display"
        >
    `;
}

// =============================================================================
// Value Update Functions
// =============================================================================

function updateSliderValue(key, value, format) {
    const numValue = parseFloat(value);
    state.updateConfig(key, numValue);

    const displayEl = document.getElementById(`${key}-display`);
    if (displayEl) {
        displayEl.textContent = formatValue(numValue, { display_format: format });
    }
}

function updateValue(key, value) {
    state.updateConfig(key, value);
}

function updateNumberValue(key, value) {
    const numValue = parseInt(value, 10);
    state.updateConfig(key, numValue);

    const displayEl = document.getElementById(`${key}-display`);
    if (displayEl) {
        displayEl.textContent = numValue.toString();
    }
}

function updateCheckbox(key, checked) {
    state.updateConfig(key, checked);

    // Handle LoRA visibility
    if (key === 'use_lora') {
        const loraSection = document.getElementById('lora-section');
        if (loraSection) {
            loraSection.style.display = checked ? 'block' : 'none';
        }
    }
}

function formatValue(value, param) {
    if (param.display_format === 'scientific') {
        return value.toExponential(1);
    }
    if (Number.isInteger(value)) {
        return value.toString();
    }
    return value.toFixed(2);
}

// =============================================================================
// Preset Application
// =============================================================================

function applyPreset(presetKey) {
    const preset = state.presets[presetKey];
    if (!preset) return;

    // Update visual selection
    document.querySelectorAll('.preset-card').forEach(card => {
        card.classList.remove('active');
        card.setAttribute('aria-pressed', 'false');
    });

    const selectedCard = document.querySelector(`[data-preset="${presetKey}"]`);
    if (selectedCard) {
        selectedCard.classList.add('active');
        selectedCard.setAttribute('aria-pressed', 'true');
    }

    // Apply settings
    Object.entries(preset.settings).forEach(([key, value]) => {
        state.updateConfig(key, value);

        const element = document.getElementById(key);
        if (element) {
            if (element.type === 'checkbox') {
                element.checked = value;
                updateCheckbox(key, value);
            } else {
                element.value = value;
            }

            // Update display values
            const displayEl = document.getElementById(`${key}-display`);
            if (displayEl && state.parameters[key]) {
                displayEl.textContent = formatValue(value, state.parameters[key]);
            }
        }
    });

    toast.success(`Applied "${preset.name}" preset`);
}

// =============================================================================
// Training Controls
// =============================================================================

async function startTraining() {
    const trainingDataEl = document.getElementById('training-data');
    const trainingData = trainingDataEl ? trainingDataEl.value : '';

    // Get base config
    let config = {
        ...state.getConfig(),
        training_data: trainingData
    };

    // Merge power user features if available
    if (window.powerUserFeatures && typeof window.powerUserFeatures.getConfiguration === 'function') {
        const powerConfig = window.powerUserFeatures.getConfiguration();

        // Map power user config to backend expected keys
        if (powerConfig.weightInit && powerConfig.weightInit.method !== 'default') {
            config.init_method = powerConfig.weightInit.method;
            config.init_range = powerConfig.weightInit.initRange;
            config.init_modules = powerConfig.weightInit.modules;
            if (powerConfig.weightInit.method === 'sparse') {
                config.init_sparsity = powerConfig.weightInit.sparsity;
            }
        }

        // Adam hyperparameters
        if (powerConfig.optimizer) {
            config.adam_beta1 = powerConfig.optimizer.adamBeta1;
            config.adam_beta2 = powerConfig.optimizer.adamBeta2;
            config.adam_epsilon = powerConfig.optimizer.adamEpsilon;
        }

        // LoRA+ settings
        if (powerConfig.loraPlus && powerConfig.loraPlus.enabled) {
            config.lora_plus_enabled = true;
            config.lora_b_lr_ratio = powerConfig.loraPlus.lrRatio;
        }

        // RS-LoRA settings
        if (powerConfig.rsLora && powerConfig.rsLora.enabled) {
            config.rslora_enabled = true;
        }

        // Manual LoRA modules override
        if (powerConfig.manualModules && powerConfig.manualModules.enabled && powerConfig.manualModules.modules.length > 0) {
            config.manual_target_modules = powerConfig.manualModules.modules;
        }

        // Scheduler warmup ratio
        if (powerConfig.scheduler && powerConfig.scheduler.warmupType === 'ratio') {
            config.warmup_ratio = powerConfig.scheduler.warmupRatio;
        }

        // QAT (Quantization-Aware Training) settings
        if (powerConfig.qat && powerConfig.qat.enabled) {
            config.use_qat = true;
            config.qat_bits = powerConfig.qat.bits;
            config.qat_group_size = powerConfig.qat.groupSize;
            config.qat_calibration_samples = powerConfig.qat.calibrationSamples;
            config.qat_warmup_steps = powerConfig.qat.warmupSteps;
            config.qat_symmetric = powerConfig.qat.symmetric;
            config.qat_quantize_embeddings = powerConfig.qat.quantizeEmbeddings;
        }
    }

    // Disable start button immediately
    const startBtn = document.getElementById('start-btn');
    if (startBtn) {
        startBtn.disabled = true;
        startBtn.textContent = 'Starting...';
    }

    try {
        await APIClient.startTraining(config);

        showTrainingUI();
        toast.success('Training started successfully');

        // Start polling if WebSocket not available
        if (!wsManager.isWebSocketAvailable) {
            wsManager.startPolling();
        }

    } catch (error) {
        console.error('Failed to start training:', error);
        toast.error(error.message || 'Failed to start training');
        resetTrainingUI();
    }
}

async function stopTraining() {
    try {
        await APIClient.stopTraining();
        toast.info('Stop signal sent. Training will stop after current step.');
        addLogEntry('Stop requested by user...', 'warning');
    } catch (error) {
        console.error('Failed to stop training:', error);
        toast.error('Failed to stop training');
    }
}

function showTrainingUI() {
    const startBtn = document.getElementById('start-btn');
    const stopBtn = document.getElementById('stop-btn');
    const progressSection = document.getElementById('progress-section');
    const logOutput = document.getElementById('log-output');

    if (startBtn) startBtn.style.display = 'none';
    if (stopBtn) stopBtn.style.display = 'inline-flex';
    if (progressSection) progressSection.style.display = 'block';
    if (logOutput) logOutput.innerHTML = '';

    // Reset progress display
    updateProgressDisplay(0, 'Initializing...');
}

function resetTrainingUI() {
    const startBtn = document.getElementById('start-btn');
    const stopBtn = document.getElementById('stop-btn');

    if (startBtn) {
        startBtn.style.display = 'inline-flex';
        startBtn.disabled = false;
        startBtn.textContent = 'Start Training';
    }
    if (stopBtn) {
        stopBtn.style.display = 'none';
    }

    lastLogCount = 0;
}

// =============================================================================
// Progress & Log Updates
// =============================================================================

let lastLogCount = 0;

function updateProgress(status) {
    updateProgressDisplay(status.progress, formatStatusMessage(status));
}

function updateProgressDisplay(progress, statusText) {
    const progressFill = document.getElementById('progress-fill');
    const progressPercent = document.getElementById('progress-percent');
    const progressStatus = document.getElementById('progress-status');

    if (progressFill) progressFill.style.width = `${progress}%`;
    if (progressPercent) progressPercent.textContent = `${progress}%`;
    if (progressStatus) progressStatus.textContent = statusText;
}

function formatStatusMessage(status) {
    const messages = {
        'idle': 'Idle',
        'initializing': 'Initializing...',
        'loading_model': 'Loading model...',
        'preparing_data': 'Preparing data...',
        'training': `Training (Step ${status.current_step}/${status.total_steps})`,
        'completed': 'Training complete!',
        'error': 'Error occurred',
        'stopping': 'Stopping...'
    };

    return messages[status.status] || status.status;
}

function updateLogs(logs) {
    if (!logs || logs.length === lastLogCount) return;

    for (let i = lastLogCount; i < logs.length; i++) {
        const log = logs[i];
        addLogEntry(log.message, log.level, log.time);
    }

    lastLogCount = logs.length;
}

function addLogEntry(message, level = 'info', time = null) {
    const logContainer = document.getElementById('log-output');
    if (!logContainer) return;

    const timeStr = time || new Date().toLocaleTimeString();

    const entry = document.createElement('div');
    entry.className = `log-entry ${level}`;
    entry.innerHTML = `
        <span class="log-time">[${escapeHtml(timeStr)}]</span>
        <span class="log-message">${escapeHtml(message)}</span>
    `;

    logContainer.appendChild(entry);
    logContainer.scrollTop = logContainer.scrollHeight;
}

// =============================================================================
// UI Helpers
// =============================================================================

function toggleAdvanced() {
    const content = document.getElementById('advanced-settings');
    const icon = document.getElementById('collapse-icon');
    const header = content?.previousElementSibling;

    if (content) {
        const isExpanded = content.classList.toggle('active');
        if (icon) icon.textContent = isExpanded ? '-' : '+';
        if (header) header.setAttribute('aria-expanded', isExpanded.toString());
    }
}

function togglePowerUser() {
    const content = document.getElementById('power-user-content');
    const icon = document.getElementById('power-user-collapse-icon');
    const header = content?.previousElementSibling;

    if (content) {
        const isExpanded = content.classList.toggle('active');
        if (icon) icon.textContent = isExpanded ? '-' : '+';
        if (header) header.setAttribute('aria-expanded', isExpanded.toString());
    }
}

function updateDataStats() {
    const textarea = document.getElementById('training-data');
    if (!textarea) return;

    const text = textarea.value;
    const charCount = document.getElementById('char-count');
    const lineCount = document.getElementById('line-count');

    if (charCount) {
        charCount.textContent = `${text.length.toLocaleString()} characters`;
    }
    if (lineCount) {
        const lines = text.split('\n').filter(l => l.trim()).length;
        lineCount.textContent = `${lines} lines`;
    }
}

// Modal functions
function showGlossary() {
    const modal = document.getElementById('glossary-modal');
    if (modal) {
        modal.classList.add('active');
        modal.setAttribute('aria-hidden', 'false');
        trapFocus(modal);
    }
}

function closeGlossary() {
    const modal = document.getElementById('glossary-modal');
    if (modal) {
        modal.classList.remove('active');
        modal.setAttribute('aria-hidden', 'true');
    }
}

function showHelp() {
    const modal = document.getElementById('help-modal');
    if (modal) {
        modal.classList.add('active');
        modal.setAttribute('aria-hidden', 'false');
        trapFocus(modal);
    }
}

function closeHelp() {
    const modal = document.getElementById('help-modal');
    if (modal) {
        modal.classList.remove('active');
        modal.setAttribute('aria-hidden', 'true');
    }
}

// =============================================================================
// Event Handlers
// =============================================================================

function handleKeyboardShortcuts(e) {
    // Escape closes modals
    if (e.key === 'Escape') {
        document.querySelectorAll('.modal.active').forEach(modal => {
            modal.classList.remove('active');
            modal.setAttribute('aria-hidden', 'true');
        });
    }
}

function handleModalClick(e) {
    if (e.target.classList.contains('modal')) {
        e.target.classList.remove('active');
        e.target.setAttribute('aria-hidden', 'true');
    }
}

function trapFocus(modal) {
    const focusableElements = modal.querySelectorAll(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );

    if (focusableElements.length > 0) {
        focusableElements[0].focus();
    }
}

// =============================================================================
// Utility Functions
// =============================================================================

function escapeHtml(text) {
    if (typeof text !== 'string') return text;
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// =============================================================================
// Inference Playground
// =============================================================================

class InferenceManager {
    constructor() {
        this.isModelLoaded = false;
        this.loadedModelPath = null;
        this.isGenerating = false;
    }

    async getAvailableModels() {
        try {
            const response = await fetch('/api/inference/models');
            if (!response.ok) throw new Error('Failed to fetch models');
            return response.json();
        } catch (error) {
            console.error('Failed to get models:', error);
            throw error;
        }
    }

    async loadModel(modelPath) {
        const response = await fetch('/api/inference/load', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_path: modelPath })
        });

        const result = await response.json();
        if (!response.ok) {
            throw new Error(result.error || 'Failed to load model');
        }

        this.isModelLoaded = true;
        this.loadedModelPath = modelPath;
        return result;
    }

    async unloadModel() {
        const response = await fetch('/api/inference/unload', {
            method: 'POST'
        });

        if (!response.ok) throw new Error('Failed to unload model');

        this.isModelLoaded = false;
        this.loadedModelPath = null;
        return response.json();
    }

    async generate(prompt, params) {
        if (!this.isModelLoaded) {
            throw new Error('No model loaded');
        }

        const response = await fetch('/api/inference/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt,
                ...params
            })
        });

        const result = await response.json();
        if (!response.ok) {
            throw new Error(result.error || 'Generation failed');
        }

        return result;
    }

    async getStatus() {
        const response = await fetch('/api/inference/status');
        if (!response.ok) throw new Error('Failed to get status');
        return response.json();
    }
}

const inferenceManager = new InferenceManager();

// Initialize inference playground on load
document.addEventListener('DOMContentLoaded', () => {
    initializePlayground();
});

async function initializePlayground() {
    await refreshModels();

    // Setup model select change handler
    const modelSelect = document.getElementById('inference-model');
    if (modelSelect) {
        modelSelect.addEventListener('change', () => {
            const loadBtn = document.getElementById('load-model-btn');
            if (loadBtn) {
                loadBtn.disabled = !modelSelect.value;
            }
        });
    }

    // Check if a model is already loaded
    try {
        const status = await inferenceManager.getStatus();
        if (status.model_loaded) {
            inferenceManager.isModelLoaded = true;
            inferenceManager.loadedModelPath = status.loaded_model_path;
            updatePlaygroundUI(true);
            updateModelStatus(`Model loaded: ${status.loaded_model_path}`, 'success');
        }
    } catch (e) {
        console.log('Could not get initial inference status');
    }
}

async function refreshModels() {
    const select = document.getElementById('inference-model');
    const refreshBtn = document.getElementById('refresh-models-btn');

    if (!select) return;

    if (refreshBtn) {
        refreshBtn.disabled = true;
        refreshBtn.textContent = 'Loading...';
    }

    try {
        const data = await inferenceManager.getAvailableModels();
        const models = data.models || [];

        // Clear and repopulate
        select.innerHTML = '';

        if (models.length === 0) {
            select.innerHTML = '<option value="">-- No trained models found --</option>';
            updateModelStatus('No trained models found. Train a model first!', 'info');
        } else {
            select.innerHTML = '<option value="">-- Select a model --</option>';
            models.forEach(model => {
                const option = document.createElement('option');
                option.value = model.path;
                option.textContent = `${model.name} (${model.type})`;
                select.appendChild(option);
            });
            updateModelStatus(`Found ${models.length} trained model(s)`, 'info');
        }

        // If a model was already loaded, select it
        if (data.loaded_model) {
            select.value = data.loaded_model;
            inferenceManager.isModelLoaded = true;
            inferenceManager.loadedModelPath = data.loaded_model;
            updatePlaygroundUI(true);
        }

    } catch (error) {
        console.error('Failed to refresh models:', error);
        select.innerHTML = '<option value="">-- Error loading models --</option>';
        toast.error('Failed to load model list');
    } finally {
        if (refreshBtn) {
            refreshBtn.disabled = false;
            refreshBtn.textContent = 'Refresh';
        }
    }
}

async function loadSelectedModel() {
    const select = document.getElementById('inference-model');
    const loadBtn = document.getElementById('load-model-btn');

    if (!select || !select.value) {
        toast.warning('Please select a model first');
        return;
    }

    const modelPath = select.value;

    if (loadBtn) {
        loadBtn.disabled = true;
        loadBtn.textContent = 'Loading...';
    }

    updateModelStatus('Loading model... This may take a moment.', 'info');

    try {
        await inferenceManager.loadModel(modelPath);
        updatePlaygroundUI(true);
        updateModelStatus('Model loaded and ready!', 'success');
        toast.success('Model loaded successfully!');
    } catch (error) {
        console.error('Failed to load model:', error);
        updateModelStatus(`Failed to load: ${error.message}`, 'error');
        toast.error(error.message || 'Failed to load model');
        updatePlaygroundUI(false);
    } finally {
        if (loadBtn) {
            loadBtn.disabled = false;
            loadBtn.textContent = 'Load Model';
        }
    }
}

async function unloadModel() {
    const unloadBtn = document.getElementById('unload-model-btn');

    if (unloadBtn) {
        unloadBtn.disabled = true;
        unloadBtn.textContent = 'Unloading...';
    }

    try {
        await inferenceManager.unloadModel();
        updatePlaygroundUI(false);
        updateModelStatus('Model unloaded. Select another model to continue.', 'info');
        toast.info('Model unloaded');
    } catch (error) {
        console.error('Failed to unload model:', error);
        toast.error('Failed to unload model');
    } finally {
        if (unloadBtn) {
            unloadBtn.disabled = false;
            unloadBtn.textContent = 'Unload';
        }
    }
}

function updatePlaygroundUI(modelLoaded) {
    const loadBtn = document.getElementById('load-model-btn');
    const unloadBtn = document.getElementById('unload-model-btn');
    const generateBtn = document.getElementById('generate-btn');
    const modelSelect = document.getElementById('inference-model');

    if (modelLoaded) {
        if (loadBtn) loadBtn.style.display = 'none';
        if (unloadBtn) unloadBtn.style.display = 'inline-flex';
        if (generateBtn) generateBtn.disabled = false;
        if (modelSelect) modelSelect.disabled = true;
    } else {
        if (loadBtn) {
            loadBtn.style.display = 'inline-flex';
            loadBtn.disabled = !modelSelect?.value;
        }
        if (unloadBtn) unloadBtn.style.display = 'none';
        if (generateBtn) generateBtn.disabled = true;
        if (modelSelect) modelSelect.disabled = false;
    }
}

function updateModelStatus(message, type = 'info') {
    const statusEl = document.getElementById('model-status');
    if (!statusEl) return;

    statusEl.textContent = message;
    statusEl.className = `model-status status-${type}`;
}

async function generateText() {
    const promptEl = document.getElementById('inference-prompt');
    const generateBtn = document.getElementById('generate-btn');
    const outputContainer = document.getElementById('playground-output-container');
    const outputEl = document.getElementById('inference-output');
    const statsEl = document.getElementById('output-stats');

    if (!promptEl || !promptEl.value.trim()) {
        toast.warning('Please enter a prompt');
        return;
    }

    const prompt = promptEl.value;

    // Get generation parameters
    const params = {
        max_new_tokens: parseInt(document.getElementById('gen-max-tokens')?.value || '100'),
        temperature: parseFloat(document.getElementById('gen-temperature')?.value || '0.7'),
        top_p: parseFloat(document.getElementById('gen-top-p')?.value || '0.9'),
        top_k: parseInt(document.getElementById('gen-top-k')?.value || '50'),
        repetition_penalty: parseFloat(document.getElementById('gen-repetition-penalty')?.value || '1.1'),
        do_sample: true
    };

    // Disable button and show loading state
    if (generateBtn) {
        generateBtn.disabled = true;
        generateBtn.textContent = 'Generating...';
    }

    inferenceManager.isGenerating = true;

    try {
        const startTime = performance.now();
        const result = await inferenceManager.generate(prompt, params);
        const endTime = performance.now();

        // Show output
        if (outputContainer) outputContainer.style.display = 'block';

        if (outputEl) {
            // Highlight the prompt portion vs generated portion
            const promptLength = prompt.length;
            const fullText = result.generated_text;

            // The model returns the full text including the prompt
            const generatedPortion = fullText.slice(promptLength);

            outputEl.innerHTML = `<span class="prompt-text">${escapeHtml(prompt)}</span><span class="generated-text">${escapeHtml(generatedPortion)}</span>`;
        }

        // Show stats
        if (statsEl) {
            const duration = ((endTime - startTime) / 1000).toFixed(2);
            const tokenCount = result.generated_text.split(/\s+/).length;
            statsEl.textContent = `Generated in ${duration}s | ~${tokenCount} tokens | Temperature: ${params.temperature}`;
        }

        toast.success('Text generated!');

    } catch (error) {
        console.error('Generation failed:', error);
        toast.error(error.message || 'Generation failed');

        if (outputContainer) outputContainer.style.display = 'block';
        if (outputEl) {
            outputEl.innerHTML = `<span class="error-text">Error: ${escapeHtml(error.message)}</span>`;
        }
    } finally {
        if (generateBtn) {
            generateBtn.disabled = false;
            generateBtn.textContent = 'Generate Text';
        }
        inferenceManager.isGenerating = false;
    }
}

function clearOutput() {
    const outputContainer = document.getElementById('playground-output-container');
    const outputEl = document.getElementById('inference-output');
    const statsEl = document.getElementById('output-stats');

    if (outputContainer) outputContainer.style.display = 'none';
    if (outputEl) outputEl.innerHTML = '';
    if (statsEl) statsEl.textContent = '';
}

function togglePlaygroundParams() {
    const content = document.getElementById('playground-params-content');
    const icon = document.getElementById('playground-collapse-icon');
    const header = content?.previousElementSibling;

    if (content) {
        const isExpanded = content.classList.toggle('active');
        if (icon) icon.textContent = isExpanded ? '-' : '+';
        if (header) header.setAttribute('aria-expanded', isExpanded.toString());
    }
}

// =============================================================================
// System Info Functions
// =============================================================================

async function loadSystemInfo() {
    try {
        const response = await fetch('/api/system/info');
        const info = await response.json();

        document.getElementById('gpu-info').textContent = info.gpu?.available 
            ? info.gpu.name 
            : 'No GPU (CPU only)';
        
        document.getElementById('vram-info').textContent = info.gpu?.available 
            ? `${info.gpu.memory_free} / ${info.gpu.memory_total}` 
            : 'N/A';
        
        document.getElementById('ram-info').textContent = info.memory?.total 
            ? `${info.memory.available} / ${info.memory.total} (${info.memory.percent_used}% used)` 
            : 'Unknown';
        
        document.getElementById('cpu-info').textContent = info.cpu?.cores 
            ? `${info.cpu.cores} cores` 
            : 'Unknown';

    } catch (error) {
        console.error('Failed to load system info:', error);
        document.getElementById('gpu-info').textContent = 'Error loading';
    }
}

function refreshSystemInfo() {
    loadSystemInfo();
    toast.info('System info refreshed');
}

function toggleSystemInfo() {
    const content = document.getElementById('system-info-content');
    const icon = document.getElementById('system-collapse-icon');
    const header = content?.previousElementSibling;

    if (content) {
        const isExpanded = content.classList.toggle('active');
        if (icon) icon.textContent = isExpanded ? '-' : '+';
        if (header) header.setAttribute('aria-expanded', isExpanded.toString());
        
        // Load system info when expanded for first time
        if (isExpanded && document.getElementById('gpu-info').textContent === 'Loading...') {
            loadSystemInfo();
        }
    }
}

// =============================================================================
// Dataset Management Functions
// =============================================================================

async function refreshDatasets() {
    const select = document.getElementById('dataset-select');
    if (!select) return;

    try {
        const response = await fetch('/api/datasets');
        const data = await response.json();

        select.innerHTML = '<option value="">-- Load saved dataset --</option>';
        
        data.datasets.forEach(dataset => {
            const option = document.createElement('option');
            option.value = dataset.name;
            option.textContent = `${dataset.name} (${dataset.size_human})`;
            select.appendChild(option);
        });

    } catch (error) {
        console.error('Failed to load datasets:', error);
        toast.error('Failed to load datasets');
    }
}

async function loadSelectedDataset() {
    const select = document.getElementById('dataset-select');
    const textarea = document.getElementById('training-data');
    
    if (!select?.value) {
        toast.warning('Please select a dataset first');
        return;
    }

    try {
        const response = await fetch(`/api/datasets/load/${encodeURIComponent(select.value)}`);
        if (!response.ok) throw new Error('Failed to load dataset');
        
        const data = await response.json();
        textarea.value = data.content;
        updateDataStats();
        toast.success(`Loaded ${data.name} (${data.lines} lines)`);

    } catch (error) {
        console.error('Failed to load dataset:', error);
        toast.error('Failed to load dataset');
    }
}

async function saveCurrentDataset() {
    const textarea = document.getElementById('training-data');
    const content = textarea?.value?.trim();

    if (!content) {
        toast.warning('No training data to save');
        return;
    }

    const name = prompt('Enter dataset name:', `dataset_${new Date().toISOString().slice(0,10)}.txt`);
    if (!name) return;

    try {
        const response = await fetch('/api/datasets/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, content })
        });

        if (!response.ok) throw new Error('Failed to save');

        const data = await response.json();
        toast.success(`Saved as ${data.name}`);
        refreshDatasets();

    } catch (error) {
        console.error('Failed to save dataset:', error);
        toast.error('Failed to save dataset');
    }
}

async function uploadDataset(input) {
    const file = input.files?.[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/api/datasets/upload', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Upload failed');

        const data = await response.json();
        toast.success(`Uploaded ${data.name} (${data.lines} lines)`);
        refreshDatasets();

        // Also load it into textarea
        document.getElementById('training-data').value = '';
        const loadResponse = await fetch(`/api/datasets/load/${encodeURIComponent(data.name)}`);
        const loadData = await loadResponse.json();
        document.getElementById('training-data').value = loadData.content;
        updateDataStats();

    } catch (error) {
        console.error('Failed to upload:', error);
        toast.error('Failed to upload dataset');
    }

    input.value = ''; // Reset input
}

function updateDataStats() {
    const textarea = document.getElementById('training-data');
    const text = textarea?.value || '';

    const charCount = document.getElementById('char-count');
    const lineCount = document.getElementById('line-count');
    const tokenEstimate = document.getElementById('token-estimate');

    if (charCount) charCount.textContent = `${text.length.toLocaleString()} characters`;
    if (lineCount) lineCount.textContent = `${text.split('\n').filter(l => l.trim()).length} lines`;
    if (tokenEstimate) tokenEstimate.textContent = `~${Math.ceil(text.length / 4).toLocaleString()} tokens (estimate)`;
}

// =============================================================================
// Drag and Drop File Upload
// =============================================================================

function setupDragAndDrop() {
    const textarea = document.getElementById('training-data');
    if (!textarea) return;

    // Prevent default drag behaviors on the whole document
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        document.body.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
        }, false);
    });

    // Highlight textarea on drag over
    ['dragenter', 'dragover'].forEach(eventName => {
        textarea.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            textarea.classList.add('drag-over');
        }, false);
    });

    // Remove highlight on drag leave/drop
    ['dragleave', 'drop'].forEach(eventName => {
        textarea.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            textarea.classList.remove('drag-over');
        }, false);
    });

    // Handle dropped files
    textarea.addEventListener('drop', async (e) => {
        const files = e.dataTransfer?.files;
        if (!files || files.length === 0) return;

        const file = files[0];
        const validExtensions = ['.txt', '.json', '.jsonl', '.csv'];
        const fileExt = '.' + file.name.split('.').pop().toLowerCase();

        if (!validExtensions.includes(fileExt)) {
            toast.error(`Invalid file type. Supported: ${validExtensions.join(', ')}`);
            return;
        }

        // For text files, we can read directly into the textarea
        if (fileExt === '.txt') {
            const reader = new FileReader();
            reader.onload = (event) => {
                textarea.value = event.target.result;
                updateDataStats();
                toast.success(`Loaded ${file.name}`);
            };
            reader.onerror = () => {
                toast.error('Failed to read file');
            };
            reader.readAsText(file);
        } else {
            // For other formats, upload via API to handle parsing
            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch('/api/datasets/upload', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) throw new Error('Upload failed');

                const data = await response.json();
                toast.success(`Uploaded ${data.name} (${data.lines} lines)`);
                refreshDatasets();

                // Load it into textarea
                const loadResponse = await fetch(`/api/datasets/load/${encodeURIComponent(data.name)}`);
                const loadData = await loadResponse.json();
                textarea.value = loadData.content;
                updateDataStats();
            } catch (error) {
                console.error('Failed to upload:', error);
                toast.error('Failed to upload dataset');
            }
        }
    }, false);
}

// =============================================================================
// Training History Functions
// =============================================================================

async function refreshHistory() {
    const historyList = document.getElementById('history-list');
    if (!historyList) return;

    try {
        const response = await fetch('/api/history');
        const data = await response.json();

        if (data.runs.length === 0) {
            historyList.innerHTML = '<p style="color: var(--text-muted);">No training runs yet. Train a model to see it here!</p>';
            return;
        }

        historyList.innerHTML = data.runs.map(run => `
            <div class="history-item" style="padding: 12px; margin-bottom: 8px; background: var(--bg-secondary); border-radius: 8px; display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="font-weight: 500;">${run.id}</div>
                    <div style="font-size: 0.85rem; color: var(--text-muted);">
                        ${run.has_final ? '✅ Final model' : '⚠️ No final model'} 
                        ${run.checkpoints.length > 0 ? `| ${run.checkpoints.length} checkpoint(s)` : ''}
                    </div>
                </div>
                <div style="display: flex; gap: 8px;">
                    <button class="btn btn-secondary btn-small" onclick="copyModelPath('./output/${run.id}/final')" title="Copy path">📋</button>
                    <button class="btn btn-danger btn-small" onclick="deleteTrainingRun('${run.id}')" title="Delete">🗑️</button>
                </div>
            </div>
        `).join('');

        // Also populate export dropdowns
        populateExportSelects(data.runs);

    } catch (error) {
        console.error('Failed to load history:', error);
        historyList.innerHTML = '<p style="color: var(--error-color);">Failed to load training history</p>';
    }
}

function populateExportSelects(runs) {
    const selects = ['merge-model-select', 'gguf-model-select', 'hf-model-select'];
    
    selects.forEach(selectId => {
        const select = document.getElementById(selectId);
        if (!select) return;

        const currentValue = select.value;
        select.innerHTML = '<option value="">-- Select a model --</option>';

        runs.forEach(run => {
            if (run.has_final) {
                const option = document.createElement('option');
                option.value = `./output/${run.id}/final`;
                option.textContent = `${run.id} (final)`;
                select.appendChild(option);
            }
            run.checkpoints.forEach(cp => {
                const option = document.createElement('option');
                option.value = `./output/${run.id}/${cp}`;
                option.textContent = `${run.id} (${cp})`;
                select.appendChild(option);
            });
        });

        if (currentValue) select.value = currentValue;
    });
}

async function deleteTrainingRun(runId) {
    if (!confirm(`Delete training run "${runId}"? This cannot be undone!`)) return;

    try {
        const response = await fetch(`/api/history/${runId}/delete`, { method: 'DELETE' });
        if (!response.ok) throw new Error('Delete failed');

        toast.success(`Deleted ${runId}`);
        refreshHistory();
        refreshModels();

    } catch (error) {
        console.error('Failed to delete:', error);
        toast.error('Failed to delete training run');
    }
}

function copyModelPath(path) {
    navigator.clipboard.writeText(path)
        .then(() => toast.success('Path copied to clipboard'))
        .catch(() => toast.error('Failed to copy path'));
}

function toggleHistory() {
    const content = document.getElementById('history-content');
    const icon = document.getElementById('history-collapse-icon');
    const header = content?.previousElementSibling;

    if (content) {
        const isExpanded = content.classList.toggle('active');
        if (icon) icon.textContent = isExpanded ? '-' : '+';
        if (header) header.setAttribute('aria-expanded', isExpanded.toString());
        
        if (isExpanded) refreshHistory();
    }
}

// =============================================================================
// Export Functions
// =============================================================================

function toggleExport() {
    const content = document.getElementById('export-content');
    const icon = document.getElementById('export-collapse-icon');
    const header = content?.previousElementSibling;

    if (content) {
        const isExpanded = content.classList.toggle('active');
        if (icon) icon.textContent = isExpanded ? '-' : '+';
        if (header) header.setAttribute('aria-expanded', isExpanded.toString());
        
        if (isExpanded) refreshHistory(); // Populate selects
    }
}

async function mergeLoraModel() {
    const select = document.getElementById('merge-model-select');
    if (!select?.value) {
        toast.warning('Please select a LoRA model');
        return;
    }

    toast.info('Merging LoRA adapter... This may take a moment.');

    try {
        const response = await fetch('/api/export/merge-lora', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ adapter_path: select.value })
        });

        const data = await response.json();

        if (response.ok) {
            toast.success(`Merged! Saved to: ${data.merged_path}`);
            refreshHistory();
        } else {
            toast.error(data.error || 'Merge failed');
        }

    } catch (error) {
        console.error('Merge error:', error);
        toast.error('Failed to merge LoRA adapter');
    }
}

async function exportToGGUF() {
    const modelSelect = document.getElementById('gguf-model-select');
    const quantSelect = document.getElementById('gguf-quant-select');

    if (!modelSelect?.value) {
        toast.warning('Please select a model');
        return;
    }

    try {
        const response = await fetch('/api/export/gguf', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                model_path: modelSelect.value,
                quantization: quantSelect?.value || 'q4_k_m'
            })
        });

        const data = await response.json();

        // Show instructions in a modal or alert
        const instructions = data.steps?.join('\n') || 'No instructions available';
        alert(`GGUF Export Instructions:\n\n${instructions}`);

    } catch (error) {
        console.error('Export error:', error);
        toast.error('Failed to get export instructions');
    }
}

async function getHFInstructions() {
    const modelSelect = document.getElementById('hf-model-select');
    const repoName = document.getElementById('hf-repo-name');

    if (!modelSelect?.value) {
        toast.warning('Please select a model');
        return;
    }

    try {
        const response = await fetch('/api/export/huggingface', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                model_path: modelSelect.value,
                repo_name: repoName?.value || 'my-fine-tuned-model'
            })
        });

        const data = await response.json();

        // Show instructions
        const instructions = data.steps?.join('\n') || 'No instructions available';
        alert(`HuggingFace Upload Instructions:\n\n${instructions}`);

    } catch (error) {
        console.error('Export error:', error);
        toast.error('Failed to get upload instructions');
    }
}

// =============================================================================
// Initialize new features on load
// =============================================================================

document.addEventListener('DOMContentLoaded', () => {
    // Add listener for training data changes
    const trainingData = document.getElementById('training-data');
    if (trainingData) {
        trainingData.addEventListener('input', updateDataStats);
    }

    // Load datasets on page load
    refreshDatasets();

    // Setup drag-and-drop for training data
    setupDragAndDrop();
});

// =============================================================================
// Global Function Exports (for inline handlers)
// =============================================================================

window.applyPreset = applyPreset;
window.startTraining = startTraining;
window.stopTraining = stopTraining;
window.toggleAdvanced = toggleAdvanced;
window.togglePowerUser = togglePowerUser;
window.showGlossary = showGlossary;
window.closeGlossary = closeGlossary;
window.showHelp = showHelp;
window.closeHelp = closeHelp;
window.updateSliderValue = updateSliderValue;
window.updateValue = updateValue;
window.updateNumberValue = updateNumberValue;
window.updateCheckbox = updateCheckbox;
window.handleSelectWithCustom = handleSelectWithCustom;

// Inference exports
window.refreshModels = refreshModels;
window.loadSelectedModel = loadSelectedModel;
window.unloadModel = unloadModel;
window.generateText = generateText;
window.clearOutput = clearOutput;
window.togglePlaygroundParams = togglePlaygroundParams;

// New exports
window.toggleSystemInfo = toggleSystemInfo;
window.refreshSystemInfo = refreshSystemInfo;
window.loadSelectedDataset = loadSelectedDataset;
window.saveCurrentDataset = saveCurrentDataset;
window.uploadDataset = uploadDataset;
window.refreshDatasets = refreshDatasets;
window.toggleHistory = toggleHistory;
window.refreshHistory = refreshHistory;
window.deleteTrainingRun = deleteTrainingRun;
window.copyModelPath = copyModelPath;
window.toggleExport = toggleExport;
window.mergeLoraModel = mergeLoraModel;
window.exportToGGUF = exportToGGUF;
window.getHFInstructions = getHFInstructions;

// =============================================================================
// Loss Chart Management
// =============================================================================

class LossChartManager {
    constructor() {
        this.chart = null;
        this.trainLossData = [];
        this.evalLossData = [];
        this.lrData = [];
        this.isLogScale = false;
        this.initialized = false;
    }

    init() {
        // Retry if Chart.js hasn't loaded from CDN yet
        if (typeof Chart === 'undefined') {
            console.log('Chart.js not loaded yet, retrying in 100ms...');
            setTimeout(() => this.init(), 100);
            return;
        }
        if (this.initialized) return;

        const ctx = document.getElementById('loss-chart');
        if (!ctx) return;

        this.chart = new Chart(ctx, {
            type: 'line',
            data: {
                datasets: [
                    {
                        label: 'Training Loss',
                        data: this.trainLossData,
                        borderColor: '#6366f1',
                        backgroundColor: 'rgba(99, 102, 241, 0.1)',
                        fill: true,
                        tension: 0.3,
                        pointRadius: 2,
                        pointHoverRadius: 5
                    },
                    {
                        label: 'Eval Loss',
                        data: this.evalLossData,
                        borderColor: '#22c55e',
                        backgroundColor: 'rgba(34, 197, 94, 0.1)',
                        fill: false,
                        tension: 0.3,
                        pointRadius: 3,
                        pointHoverRadius: 6,
                        hidden: true
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: {
                    duration: 300
                },
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                plugins: {
                    legend: {
                        labels: {
                            color: '#94a3b8'
                        }
                    },
                    tooltip: {
                        backgroundColor: '#1e293b',
                        titleColor: '#f8fafc',
                        bodyColor: '#94a3b8',
                        borderColor: '#475569',
                        borderWidth: 1
                    }
                },
                scales: {
                    x: {
                        type: 'linear',
                        title: {
                            display: true,
                            text: 'Step',
                            color: '#94a3b8'
                        },
                        ticks: { color: '#64748b' },
                        grid: { color: '#334155' },
                        min: 0,
                        suggestedMax: 10
                    },
                    y: {
                        type: 'linear',
                        title: {
                            display: true,
                            text: 'Loss',
                            color: '#94a3b8'
                        },
                        ticks: { color: '#64748b' },
                        grid: { color: '#334155' },
                        beginAtZero: false,
                        suggestedMin: 0,
                        suggestedMax: 5
                    }
                }
            }
        });

        this.initialized = true;
    }

    addDataPoint(step, loss, evalLoss = null, lr = null) {
        console.log('addDataPoint called:', step, loss);
        if (!this.chart) {
            console.log('Chart not initialized, calling init()');
            this.init();
        }
        if (!this.chart) {
            console.log('Chart still null after init - Chart.js loaded:', typeof Chart !== 'undefined', 'Canvas exists:', !!document.getElementById('loss-chart'));
            return;
        }

        console.log('Adding data point to chart');
        this.trainLossData.push({ x: step, y: loss });

        if (evalLoss !== null) {
            this.evalLossData.push({ x: step, y: evalLoss });
            // Show eval dataset if we have data
            this.chart.data.datasets[1].hidden = false;
        }

        if (lr !== null) {
            this.lrData.push({ x: step, y: lr });
        }

        this.chart.update();  // Full update to recalculate scales
        this.updateStats();
    }

    updateStats() {
        const statsEl = document.getElementById('chart-stats');
        if (!statsEl || this.trainLossData.length === 0) return;

        const latestLoss = this.trainLossData[this.trainLossData.length - 1].y;
        const minLoss = Math.min(...this.trainLossData.map(d => d.y));
        const maxLoss = Math.max(...this.trainLossData.map(d => d.y));

        statsEl.textContent = `Latest: ${latestLoss.toFixed(4)} | Min: ${minLoss.toFixed(4)} | Max: ${maxLoss.toFixed(4)}`;
    }

    toggleLogScale() {
        if (!this.chart) return;

        this.isLogScale = !this.isLogScale;
        this.chart.options.scales.y.type = this.isLogScale ? 'logarithmic' : 'linear';
        this.chart.update();

        toast.info(`Chart scale: ${this.isLogScale ? 'Logarithmic' : 'Linear'}`);
    }

    downloadChart() {
        if (!this.chart) return;

        const link = document.createElement('a');
        link.download = `training-loss-${new Date().toISOString().slice(0, 10)}.png`;
        link.href = this.chart.toBase64Image();
        link.click();

        toast.success('Chart downloaded!');
    }

    reset() {
        this.trainLossData = [];
        this.evalLossData = [];
        this.lrData = [];

        if (this.chart) {
            this.chart.data.datasets[0].data = this.trainLossData;
            this.chart.data.datasets[1].data = this.evalLossData;
            this.chart.data.datasets[1].hidden = true;
            this.chart.update();
        }

        const statsEl = document.getElementById('chart-stats');
        if (statsEl) statsEl.textContent = '';
    }
}

const lossChart = new LossChartManager();

function toggleChartScale() {
    lossChart.toggleLogScale();
}

function downloadChart() {
    lossChart.downloadChart();
}

// =============================================================================
// Checkpoint Resume Functions
// =============================================================================

let selectedCheckpoint = null;

async function refreshCheckpoints() {
    const listEl = document.getElementById('checkpoint-list');
    if (!listEl) return;

    listEl.innerHTML = '<p style="color: var(--text-muted);">Loading...</p>';

    try {
        const response = await fetch('/api/checkpoints');
        const data = await response.json();

        if (data.checkpoints.length === 0) {
            listEl.innerHTML = '<p style="color: var(--text-muted);">No checkpoints found. Train a model first!</p>';
            return;
        }

        listEl.innerHTML = data.checkpoints.map(cp => `
            <div class="checkpoint-item"
                 data-path="${cp.path}"
                 data-config='${JSON.stringify(cp.config || {})}'
                 onclick="selectCheckpoint(this)"
                 style="padding: 12px; margin-bottom: 8px; background: var(--bg-tertiary); border-radius: 8px; cursor: pointer; border: 2px solid transparent; transition: all 0.2s;">
                <div style="font-weight: 500;">${cp.run_id}</div>
                <div style="font-size: 0.85rem; color: var(--text-muted);">
                    ${cp.checkpoint_name} (Step ${cp.step})
                    ${cp.has_config ? '✓ Config available' : ''}
                </div>
            </div>
        `).join('');

    } catch (error) {
        console.error('Failed to load checkpoints:', error);
        listEl.innerHTML = '<p style="color: var(--danger);">Failed to load checkpoints</p>';
    }
}

function selectCheckpoint(element) {
    // Clear previous selection
    document.querySelectorAll('.checkpoint-item').forEach(item => {
        item.style.borderColor = 'transparent';
    });

    // Select this one
    element.style.borderColor = 'var(--primary)';
    selectedCheckpoint = element.dataset.path;

    // Show config info
    const infoEl = document.getElementById('checkpoint-info');
    const configEl = document.getElementById('checkpoint-config');
    const resumeBtn = document.getElementById('resume-btn');

    if (infoEl && configEl) {
        try {
            const config = JSON.parse(element.dataset.config);
            if (Object.keys(config).length > 0) {
                configEl.textContent = JSON.stringify(config, null, 2);
                infoEl.style.display = 'block';
            } else {
                infoEl.style.display = 'none';
            }
        } catch (e) {
            infoEl.style.display = 'none';
        }
    }

    if (resumeBtn) {
        resumeBtn.disabled = false;
    }
}

async function resumeFromCheckpoint() {
    if (!selectedCheckpoint) {
        toast.warning('Please select a checkpoint first');
        return;
    }

    const adjustSettings = document.getElementById('adjust-before-resume')?.checked;

    if (adjustSettings) {
        toast.info('Use the settings above to adjust, then click Resume again');
        document.getElementById('adjust-before-resume').checked = false;
        return;
    }

    const resumeBtn = document.getElementById('resume-btn');
    if (resumeBtn) {
        resumeBtn.disabled = true;
        resumeBtn.textContent = 'Resuming...';
    }

    try {
        const response = await fetch('/api/train/resume', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                checkpoint_path: selectedCheckpoint,
                config_overrides: {
                    additional_epochs: 1,
                    training_data: document.getElementById('training-data')?.value || ''
                }
            })
        });

        const result = await response.json();

        if (response.ok) {
            toast.success('Resuming training from checkpoint!');
            showTrainingUI();
            lossChart.reset();

            if (!wsManager.isWebSocketAvailable) {
                wsManager.startPolling();
            }
        } else {
            toast.error(result.error || 'Failed to resume');
        }

    } catch (error) {
        console.error('Resume failed:', error);
        toast.error('Failed to resume training');
    } finally {
        if (resumeBtn) {
            resumeBtn.disabled = false;
            resumeBtn.textContent = 'Resume Training';
        }
    }
}

function toggleCheckpoints() {
    const content = document.getElementById('checkpoint-content');
    const icon = document.getElementById('checkpoint-collapse-icon');
    const header = content?.previousElementSibling;

    if (content) {
        const isExpanded = content.classList.toggle('active');
        if (icon) icon.textContent = isExpanded ? '-' : '+';
        if (header) header.setAttribute('aria-expanded', isExpanded.toString());

        if (isExpanded) refreshCheckpoints();
    }
}

// =============================================================================
// Format Preview & Auto-Detect
// =============================================================================

async function previewFormat() {
    const textarea = document.getElementById('training-data');
    const outputEl = document.getElementById('format-preview-output');
    const formatSelect = document.getElementById('data_format');

    if (!textarea?.value?.trim()) {
        toast.warning('Enter some training data first');
        return;
    }

    try {
        const response = await fetch('/api/format/preview', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text: textarea.value,
                format: formatSelect?.value || state.currentConfig.data_format || 'completion',
                max_examples: 3
            })
        });

        const data = await response.json();

        if (response.ok && outputEl) {
            const preview = data.formatted_examples.map((ex, i) =>
                `--- Example ${i + 1} ---\n${ex}`
            ).join('\n\n');

            outputEl.textContent = preview || 'No examples to show';
            toast.success(`Previewing ${data.total_examples} examples in "${data.format}" format`);
        } else {
            toast.error(data.error || 'Failed to preview');
        }

    } catch (error) {
        console.error('Preview failed:', error);
        toast.error('Failed to generate preview');
    }
}

async function detectDataFormat() {
    const textarea = document.getElementById('training-data');
    const detectedEl = document.getElementById('detected-format');

    if (!textarea?.value?.trim()) {
        toast.warning('Enter some training data first');
        return;
    }

    try {
        const response = await fetch('/api/format/detect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: textarea.value })
        });

        const data = await response.json();

        if (response.ok && detectedEl) {
            const confidence = Math.round(data.confidence * 100);
            detectedEl.innerHTML = `
                <strong>Detected:</strong> ${data.format} (${confidence}% confidence)<br>
                <span style="font-size: 0.85rem;">${data.reason}</span>
            `;

            // Suggest applying the format
            if (data.suggested_settings?.data_format) {
                const formatSelect = document.getElementById('data_format');
                if (formatSelect && formatSelect.value !== data.suggested_settings.data_format) {
                    if (confirm(`Apply detected format "${data.format}"?`)) {
                        formatSelect.value = data.suggested_settings.data_format;
                        state.updateConfig('data_format', data.suggested_settings.data_format);
                        toast.success(`Format set to "${data.format}"`);
                    }
                }
            }
        }

    } catch (error) {
        console.error('Detection failed:', error);
        toast.error('Failed to detect format');
    }
}

function toggleFormatPreview() {
    const content = document.getElementById('format-content');
    const icon = document.getElementById('format-collapse-icon');
    const header = content?.previousElementSibling;

    if (content) {
        const isExpanded = content.classList.toggle('active');
        if (icon) icon.textContent = isExpanded ? '-' : '+';
        if (header) header.setAttribute('aria-expanded', isExpanded.toString());
    }
}

// =============================================================================
// Custom Template Builder
// =============================================================================

function toggleTemplateBuilder() {
    const content = document.getElementById('template-content');
    const icon = document.getElementById('template-collapse-icon');
    const header = content?.previousElementSibling;

    if (content) {
        const isExpanded = content.classList.toggle('active');
        if (icon) icon.textContent = isExpanded ? '-' : '+';
        if (header) header.setAttribute('aria-expanded', isExpanded.toString());
    }
}

function testCustomTemplate() {
    const templateEl = document.getElementById('custom-template');
    const testInputEl = document.getElementById('template-test-input');
    const outputEl = document.getElementById('template-test-output');

    if (!templateEl?.value || !testInputEl?.value) {
        toast.warning('Enter both a template and test input');
        return;
    }

    const template = templateEl.value;
    const testInput = testInputEl.value;

    // Simple placeholder replacement demo
    let result = template;
    const lines = testInput.split('\n').filter(l => l.trim());

    // Try to fill placeholders
    const placeholders = ['instruction', 'response', 'input', 'output', 'user', 'assistant', 'question', 'answer'];
    placeholders.forEach((ph, i) => {
        if (lines[i % lines.length]) {
            result = result.replace(new RegExp(`\\{${ph}\\}`, 'gi'), lines[i % lines.length]);
        }
    });

    if (outputEl) {
        outputEl.style.display = 'block';
        outputEl.textContent = result;
    }

    toast.success('Template tested!');
}

function saveCustomTemplate() {
    const templateEl = document.getElementById('custom-template');
    const separatorEl = document.getElementById('template-separator');

    if (!templateEl?.value?.trim()) {
        toast.warning('Enter a template first');
        return;
    }

    // Store in localStorage
    const templates = JSON.parse(localStorage.getItem('customTemplates') || '[]');
    const newTemplate = {
        id: Date.now(),
        template: templateEl.value,
        separator: separatorEl?.value || 'newline',
        created: new Date().toISOString()
    };

    templates.push(newTemplate);
    localStorage.setItem('customTemplates', JSON.stringify(templates));

    // Add to format selector
    const formatSelect = document.getElementById('data_format');
    if (formatSelect) {
        const option = document.createElement('option');
        option.value = `custom_${newTemplate.id}`;
        option.textContent = `Custom Template ${templates.length}`;
        formatSelect.appendChild(option);
    }

    toast.success('Custom template saved!');
}

function loadBuiltinTemplates() {
    const templateEl = document.getElementById('custom-template');
    if (!templateEl) return;

    const templates = {
        alpaca: `### Instruction:
{instruction}

### Response:
{response}`,
        chatml: `<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
{assistant}<|im_end|>`,
        llama2: `[INST] {instruction} [/INST] {response}`,
        vicuna: `USER: {user}
ASSISTANT: {assistant}`
    };

    const choice = prompt('Choose template: alpaca, chatml, llama2, vicuna');
    if (choice && templates[choice.toLowerCase()]) {
        templateEl.value = templates[choice.toLowerCase()];
        toast.success(`Loaded ${choice} template`);
    }
}

// =============================================================================
// Memory Calculator
// =============================================================================

async function calculateMemory() {
    const modelSelect = document.getElementById('model_name');
    const batchSize = state.currentConfig.batch_size || 4;
    const maxLength = state.currentConfig.max_length || 128;
    const modelName = modelSelect?.value || state.currentConfig.model_name || 'gpt2';

    try {
        const response = await fetch('/api/memory/estimate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model_name: modelName,
                batch_size: batchSize,
                max_length: maxLength
            })
        });

        const data = await response.json();

        if (response.ok) {
            updateMemoryDisplay(data);
            toast.success('Memory estimates calculated!');
        } else {
            toast.error('Failed to calculate memory');
        }

    } catch (error) {
        console.error('Memory calculation failed:', error);
        toast.error('Failed to calculate memory');
    }
}

function updateMemoryDisplay(data) {
    document.getElementById('mem-fp32').textContent = `${data.estimates.fp32.vram_gb} GB`;
    document.getElementById('mem-fp16').textContent = `${data.estimates.fp16.vram_gb} GB`;
    document.getElementById('mem-8bit').textContent = `${data.estimates.int8.vram_gb} GB`;
    document.getElementById('mem-4bit').textContent = `${data.estimates.int4_qlora.vram_gb} GB`;

    const explanationEl = document.getElementById('memory-explanation');
    if (explanationEl) {
        let html = `<p><strong>${data.model_name}</strong> (${data.parameters_human} parameters)</p>`;
        html += `<p>With batch size ${data.batch_size} and max length ${data.max_length}:</p>`;
        html += `<ul style="margin: 8px 0; padding-left: 20px;">`;

        data.recommendations.forEach(rec => {
            html += `<li style="color: var(--success);">${rec}</li>`;
        });

        if (data.recommendations.length === 0) {
            html += `<li>This model should work on most modern GPUs</li>`;
        }

        html += `</ul>`;
        html += `<p style="margin-top: 12px;"><strong>4-bit QLoRA saves ${data.savings_percent}%</strong> compared to full precision training!</p>`;

        explanationEl.innerHTML = html;
    }
}

function toggleMemoryCalc() {
    const content = document.getElementById('memory-content');
    const icon = document.getElementById('memory-collapse-icon');
    const header = content?.previousElementSibling;

    if (content) {
        const isExpanded = content.classList.toggle('active');
        if (icon) icon.textContent = isExpanded ? '-' : '+';
        if (header) header.setAttribute('aria-expanded', isExpanded.toString());

        if (isExpanded) calculateMemory();
    }
}

// =============================================================================
// Additional Section Toggles
// =============================================================================

function togglePresets() {
    const content = document.getElementById('presets-content');
    const icon = document.getElementById('presets-collapse-icon');
    const header = content?.previousElementSibling;

    if (content) {
        const isExpanded = content.classList.toggle('active');
        if (icon) icon.textContent = isExpanded ? '-' : '+';
        if (header) header.setAttribute('aria-expanded', isExpanded.toString());
    }
}

function toggleTrainingData() {
    const content = document.getElementById('data-content');
    const icon = document.getElementById('data-collapse-icon');
    const header = content?.previousElementSibling;

    if (content) {
        const isExpanded = content.classList.toggle('active');
        if (icon) icon.textContent = isExpanded ? '-' : '+';
        if (header) header.setAttribute('aria-expanded', isExpanded.toString());
    }
}

function toggleModelSettings() {
    const content = document.getElementById('model-content');
    const icon = document.getElementById('model-collapse-icon');
    const header = content?.previousElementSibling;

    if (content) {
        const isExpanded = content.classList.toggle('active');
        if (icon) icon.textContent = isExpanded ? '-' : '+';
        if (header) header.setAttribute('aria-expanded', isExpanded.toString());
    }
}

function toggleTrainingSettings() {
    const content = document.getElementById('training-content');
    const icon = document.getElementById('training-collapse-icon');
    const header = content?.previousElementSibling;

    if (content) {
        const isExpanded = content.classList.toggle('active');
        if (icon) icon.textContent = isExpanded ? '-' : '+';
        if (header) header.setAttribute('aria-expanded', isExpanded.toString());
    }
}

function togglePlayground() {
    const content = document.getElementById('playground-content');
    const icon = document.getElementById('playground-main-collapse-icon');
    const header = content?.previousElementSibling;

    if (content) {
        const isExpanded = content.classList.toggle('active');
        if (icon) icon.textContent = isExpanded ? '-' : '+';
        if (header) header.setAttribute('aria-expanded', isExpanded.toString());
    }
}

function toggleDataTools() {
    const content = document.getElementById('data-tools-content');
    const icon = document.getElementById('data-tools-collapse-icon');
    const header = content?.previousElementSibling;

    if (content) {
        const isExpanded = content.classList.toggle('active');
        if (icon) icon.textContent = isExpanded ? '-' : '+';
        if (header) header.setAttribute('aria-expanded', isExpanded.toString());
    }
}

function toggleAdvancedInference() {
    const content = document.getElementById('adv-inference-content');
    const icon = document.getElementById('adv-inference-collapse-icon');
    const header = content?.previousElementSibling;

    if (content) {
        const isExpanded = content.classList.toggle('active');
        if (icon) icon.textContent = isExpanded ? '-' : '+';
        if (header) header.setAttribute('aria-expanded', isExpanded.toString());
    }
}

// =============================================================================
// Model Architecture Info
// =============================================================================

async function fetchModelInfo(modelName) {
    if (!modelName || modelName === 'custom') {
        updateModelInfoDisplay(null);
        return;
    }

    try {
        const response = await fetch('/api/model/info', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_name: modelName })
        });

        const data = await response.json();
        if (response.ok) {
            updateModelInfoDisplay(data);
        }
    } catch (error) {
        console.error('Failed to fetch model info:', error);
    }
}

function updateModelInfoDisplay(data) {
    const container = document.getElementById('model-info-display');
    if (!container) return;

    if (!data || !data.found) {
        container.innerHTML = data?.note
            ? `<p style="color: var(--text-muted); font-size: 0.85rem;">${data.note}</p>`
            : '';
        container.style.display = data?.note ? 'block' : 'none';
        return;
    }

    container.style.display = 'block';
    container.innerHTML = `
        <div class="model-info-grid">
            <div class="model-info-item">
                <span class="model-info-label">Architecture</span>
                <span class="model-info-value">${data.architecture}</span>
            </div>
            <div class="model-info-item">
                <span class="model-info-label">Type</span>
                <span class="model-info-value">${data.type}</span>
            </div>
            <div class="model-info-item">
                <span class="model-info-label">Parameters</span>
                <span class="model-info-value">${data.parameters_human}</span>
            </div>
            <div class="model-info-item">
                <span class="model-info-label">Layers</span>
                <span class="model-info-value">${data.layers || 'N/A'}</span>
            </div>
            <div class="model-info-item">
                <span class="model-info-label">Hidden Size</span>
                <span class="model-info-value">${data.hidden_size?.toLocaleString() || 'N/A'}</span>
            </div>
            <div class="model-info-item">
                <span class="model-info-label">Attention Heads</span>
                <span class="model-info-value">${data.attention_heads || 'N/A'}</span>
            </div>
            <div class="model-info-item">
                <span class="model-info-label">Vocab Size</span>
                <span class="model-info-value">${data.vocab_size?.toLocaleString() || 'N/A'}</span>
            </div>
            <div class="model-info-item">
                <span class="model-info-label">Context Length</span>
                <span class="model-info-value">${data.context_length?.toLocaleString() || 'N/A'}</span>
            </div>
            <div class="model-info-item">
                <span class="model-info-label">Creator</span>
                <span class="model-info-value">${data.creator || 'Unknown'}</span>
            </div>
            <div class="model-info-item">
                <span class="model-info-label">LoRA Targets</span>
                <span class="model-info-value" style="font-family: monospace; font-size: 0.8rem;">${data.lora_targets?.join(', ') || 'N/A'}</span>
            </div>
        </div>
    `;
}

// Hook into model selection changes
function setupModelInfoListener() {
    // Watch for model selection changes
    const observer = new MutationObserver(() => {
        const modelSelect = document.getElementById('model_name');
        if (modelSelect && !modelSelect.dataset.infoListenerAttached) {
            modelSelect.dataset.infoListenerAttached = 'true';
            modelSelect.addEventListener('change', (e) => {
                fetchModelInfo(e.target.value);
            });
            // Fetch initial model info
            fetchModelInfo(modelSelect.value);
        }
    });

    observer.observe(document.body, { childList: true, subtree: true });
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', setupModelInfoListener);

// =============================================================================
// WebSocket Loss Data Handler (handled in WebSocketManager.setupEventHandlers)
// =============================================================================

// Initialize chart when training starts
const originalShowTrainingUI = showTrainingUI;
window.showTrainingUI = function() {
    originalShowTrainingUI();
    lossChart.init();
    lossChart.reset();
};

// =============================================================================
// Additional Exports
// =============================================================================

window.toggleChartScale = toggleChartScale;
window.downloadChart = downloadChart;
window.toggleCheckpoints = toggleCheckpoints;
window.refreshCheckpoints = refreshCheckpoints;
window.selectCheckpoint = selectCheckpoint;
window.resumeFromCheckpoint = resumeFromCheckpoint;
window.toggleFormatPreview = toggleFormatPreview;
window.previewFormat = previewFormat;
window.detectDataFormat = detectDataFormat;
window.toggleTemplateBuilder = toggleTemplateBuilder;
window.testCustomTemplate = testCustomTemplate;
window.saveCustomTemplate = saveCustomTemplate;
window.loadBuiltinTemplates = loadBuiltinTemplates;
window.toggleMemoryCalc = toggleMemoryCalc;
window.calculateMemory = calculateMemory;
