/**
 * π/2 Features Frontend Module
 *
 * Handles: Rotational Trainer, Dataset Registry, Rotation Tokenizer, Universal Encoder
 *
 * @version 1.0.0
 */

// =============================================================================
// State Management
// =============================================================================

const piFeatureState = {
    // Trainer state
    trainer: {
        initialized: false,
        rotationMode: 'full_cycle',
        epochs: 100,
        batchSize: 32,
        learningRate: 0.0001,
        isTraining: false
    },

    // Registry state
    registry: {
        datasets: [],
        filteredDatasets: [],
        downloading: false,
        currentDownload: null
    },

    // Tokenizer state
    tokenizer: {
        initialized: false,
        vocabSize: 50257,
        phaseTokenBase: 50256,
        selectedFile: null,
        fileType: null // 'tokenized' or 'raw'
    },

    // Encoder state
    encoder: {
        initialized: false,
        selectedFile: null,
        selectedDirectory: null,
        isEncoding: false,
        fileType: null
    }
};

// Phase constants (use existing PI if already declared by pi_quant.js)
const PI_FEATURES = typeof PI !== 'undefined' ? PI : Math.PI;
const PHASES = ['0', 'π/2', 'π', '3π/2'];
const PHASE_COLORS = ['var(--funk-gold)', 'var(--groove-teal)', 'var(--funk-orange)', 'var(--groove-sage)'];

// =============================================================================
// Toggle Functions
// =============================================================================

function togglePiTrainer() {
    const content = document.getElementById('pi-trainer-content');
    const icon = document.getElementById('pi-trainer-collapse-icon');
    const header = content?.previousElementSibling;

    if (content) {
        const isExpanded = content.classList.toggle('active');
        if (icon) icon.textContent = isExpanded ? '-' : '+';
        if (header) header.setAttribute('aria-expanded', isExpanded.toString());

        if (isExpanded) {
            previewTrainerSchedule();
        }
    }
}

function togglePiRegistry() {
    const content = document.getElementById('pi-registry-content');
    const icon = document.getElementById('pi-registry-collapse-icon');
    const header = content?.previousElementSibling;

    if (content) {
        const isExpanded = content.classList.toggle('active');
        if (icon) icon.textContent = isExpanded ? '-' : '+';
        if (header) header.setAttribute('aria-expanded', isExpanded.toString());

        if (isExpanded && piFeatureState.registry.datasets.length === 0) {
            loadRegistryDatasets();
        }
    }
}

function togglePiTokenizer() {
    const content = document.getElementById('pi-tokenizer-content');
    const icon = document.getElementById('pi-tokenizer-collapse-icon');
    const header = content?.previousElementSibling;

    if (content) {
        const isExpanded = content.classList.toggle('active');
        if (icon) icon.textContent = isExpanded ? '-' : '+';
        if (header) header.setAttribute('aria-expanded', isExpanded.toString());
    }
}

function togglePiEncoder() {
    const content = document.getElementById('pi-encoder-content');
    const icon = document.getElementById('pi-encoder-collapse-icon');
    const header = content?.previousElementSibling;

    if (content) {
        const isExpanded = content.classList.toggle('active');
        if (icon) icon.textContent = isExpanded ? '-' : '+';
        if (header) header.setAttribute('aria-expanded', isExpanded.toString());
    }
}

function toggleEncoderAudio() {
    const content = document.getElementById('encoder-audio-content');
    const icon = document.getElementById('encoder-audio-icon');

    if (content) {
        const isExpanded = content.classList.toggle('active');
        if (icon) icon.textContent = isExpanded ? '-' : '+';
    }
}

function toggleEncoderImage() {
    const content = document.getElementById('encoder-image-content');
    const icon = document.getElementById('encoder-image-icon');

    if (content) {
        const isExpanded = content.classList.toggle('active');
        if (icon) icon.textContent = isExpanded ? '-' : '+';
    }
}

// =============================================================================
// π/2 Rotational Trainer Functions
// =============================================================================

function selectRotationMode(mode) {
    piFeatureState.trainer.rotationMode = mode;

    // Update UI
    document.querySelectorAll('.rotation-mode-card').forEach(card => {
        card.style.border = '2px solid transparent';
        if (card.dataset.mode === mode) {
            card.style.border = '2px solid var(--funk-orange)';
        }
    });

    previewTrainerSchedule();
}

function previewTrainerSchedule() {
    const mode = piFeatureState.trainer.rotationMode;
    const epochs = parseInt(document.getElementById('trainer-epochs')?.value || 100);
    const preview = document.getElementById('trainer-schedule-preview');

    if (!preview) return;

    let schedule = [];

    for (let i = 0; i < Math.min(epochs, 8); i++) {
        let phaseIdx;

        switch (mode) {
            case 'full_cycle':
                phaseIdx = i % 4;
                break;
            case 'alternating':
                phaseIdx = (i % 2 === 0) ? 1 : 3;
                break;
            case 'opposition':
                phaseIdx = (i % 2 === 0) ? 0 : 2;
                break;
            case 'random':
                phaseIdx = Math.floor(Math.random() * 4);
                break;
            default:
                phaseIdx = i % 4;
        }

        schedule.push({
            epoch: i + 1,
            phase: PHASES[phaseIdx],
            color: PHASE_COLORS[phaseIdx]
        });
    }

    let html = '<div style="display: flex; flex-wrap: wrap; gap: 6px; font-family: monospace; font-size: 0.85rem;">';

    schedule.forEach(item => {
        html += `<span class="phase-chip" style="padding: 4px 8px; background: ${item.color}; color: var(--bg-primary); border-radius: 4px;">E${item.epoch}: ${item.phase}</span>`;
    });

    if (epochs > 8) {
        html += `<span style="color: var(--text-muted);">... (${epochs} total epochs)</span>`;
    }

    html += '</div>';
    preview.innerHTML = html;
}

async function initializeTrainer() {
    const statusDiv = document.getElementById('trainer-status');
    const statusText = document.getElementById('trainer-status-text');
    const startBtn = document.getElementById('start-rotational-btn');

    if (statusDiv) statusDiv.style.display = 'block';
    if (statusText) statusText.innerHTML = 'Initializing trainer...';

    const config = {
        rotation_mode: piFeatureState.trainer.rotationMode,
        batch_size: parseInt(document.getElementById('trainer-batch-size')?.value || 32),
        learning_rate: parseFloat(document.getElementById('trainer-lr')?.value || 0.0001)
    };

    try {
        const response = await fetch('/api/trainer/init', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });

        const data = await response.json();

        if (response.ok) {
            piFeatureState.trainer.initialized = true;
            if (startBtn) startBtn.disabled = false;
            if (statusText) {
                statusText.innerHTML = `<span style="color: var(--success);">✓ Trainer initialized</span><br>
                    Mode: <strong>${config.rotation_mode}</strong> |
                    Batch: <strong>${config.batch_size}</strong> |
                    LR: <strong>${config.learning_rate}</strong>`;
            }
            if (typeof toast !== 'undefined') toast.success('Trainer initialized successfully');
        } else {
            if (statusText) statusText.innerHTML = `<span style="color: var(--error);">Error: ${data.error || 'Failed to initialize'}</span>`;
            if (typeof toast !== 'undefined') toast.error(data.error || 'Failed to initialize trainer');
        }
    } catch (error) {
        console.error('Trainer init error:', error);
        if (statusText) statusText.innerHTML = `<span style="color: var(--error);">Network error: ${error.message}</span>`;
        if (typeof toast !== 'undefined') toast.error('Network error initializing trainer');
    }
}

async function startRotationalTraining() {
    if (!piFeatureState.trainer.initialized) {
        if (typeof toast !== 'undefined') toast.warning('Please initialize the trainer first');
        return;
    }

    const epochs = parseInt(document.getElementById('trainer-epochs')?.value || 100);
    const statusDiv = document.getElementById('trainer-status');
    const statusText = document.getElementById('trainer-status-text');
    const progressDiv = document.getElementById('trainer-progress');
    const startBtn = document.getElementById('start-rotational-btn');

    if (statusDiv) statusDiv.style.display = 'block';
    if (progressDiv) progressDiv.style.display = 'block';
    if (startBtn) {
        startBtn.disabled = true;
        startBtn.textContent = 'Training...';
    }

    piFeatureState.trainer.isTraining = true;

    try {
        const response = await fetch('/api/trainer/train', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ epochs })
        });

        const data = await response.json();

        if (response.ok) {
            if (statusText) statusText.innerHTML = `<span style="color: var(--success);">✓ Training complete!</span>`;
            document.getElementById('trainer-progress-fill').style.width = '100%';
            if (typeof toast !== 'undefined') toast.success('Rotational training complete!');
        } else {
            if (statusText) statusText.innerHTML = `<span style="color: var(--error);">Error: ${data.error}</span>`;
            if (typeof toast !== 'undefined') toast.error(data.error || 'Training failed');
        }
    } catch (error) {
        console.error('Training error:', error);
        if (statusText) statusText.innerHTML = `<span style="color: var(--error);">Error: ${error.message}</span>`;
        if (typeof toast !== 'undefined') toast.error('Training error');
    } finally {
        piFeatureState.trainer.isTraining = false;
        if (startBtn) {
            startBtn.disabled = false;
            startBtn.textContent = 'Start Rotational Training';
        }
    }
}

// =============================================================================
// π/2 Dataset Registry Functions
// =============================================================================

async function loadRegistryDatasets() {
    const container = document.getElementById('registry-datasets');

    if (!container) return;

    container.innerHTML = '<div class="registry-loading" style="grid-column: 1 / -1; text-align: center; padding: 40px; color: var(--text-muted);">Loading datasets...</div>';

    try {
        const response = await fetch('/api/datasets/list');
        const data = await response.json();

        if (response.ok && data.datasets) {
            piFeatureState.registry.datasets = data.datasets;
            piFeatureState.registry.filteredDatasets = data.datasets;
            renderRegistryDatasets();
        } else {
            container.innerHTML = `<div style="grid-column: 1 / -1; text-align: center; padding: 40px; color: var(--error);">Failed to load datasets: ${data.error || 'Unknown error'}</div>`;
        }
    } catch (error) {
        console.error('Failed to load datasets:', error);
        container.innerHTML = `<div style="grid-column: 1 / -1; text-align: center; padding: 40px; color: var(--error);">Network error loading datasets</div>`;
    }
}

function filterRegistryDatasets() {
    const category = document.getElementById('registry-category')?.value || 'all';
    const phase = document.getElementById('registry-phase')?.value || 'all';

    piFeatureState.registry.filteredDatasets = piFeatureState.registry.datasets.filter(ds => {
        const categoryMatch = category === 'all' || ds.category === category;
        const phaseMatch = phase === 'all' || ds.phase === phase;
        return categoryMatch && phaseMatch;
    });

    renderRegistryDatasets();
}

function renderRegistryDatasets() {
    const container = document.getElementById('registry-datasets');
    if (!container) return;

    const datasets = piFeatureState.registry.filteredDatasets;

    if (datasets.length === 0) {
        container.innerHTML = '<div style="grid-column: 1 / -1; text-align: center; padding: 40px; color: var(--text-muted);">No datasets match your filters</div>';
        return;
    }

    const categoryColors = {
        code: 'var(--groove-teal)',
        music: 'var(--funk-orange)',
        math: 'var(--info)',
        science: 'var(--groove-sage)',
        image: 'var(--funk-gold)',
        text: 'var(--primary)',
        quantum: 'var(--warning)',
        multimodal: 'var(--success)'
    };

    let html = '';

    datasets.forEach(ds => {
        const color = categoryColors[ds.category] || 'var(--text-muted)';

        html += `
            <div class="registry-dataset-card" style="padding: 15px; background: var(--bg-tertiary); border-radius: 8px; border-left: 4px solid ${color};">
                <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 8px;">
                    <h4 style="margin: 0; color: var(--text-primary); font-size: 0.95rem;">${ds.name}</h4>
                    <span style="padding: 2px 8px; background: ${color}; color: var(--bg-primary); border-radius: 4px; font-size: 0.7rem; text-transform: uppercase;">${ds.category}</span>
                </div>
                <p style="margin: 0 0 10px 0; font-size: 0.8rem; color: var(--text-muted);">${ds.description || 'No description'}</p>
                <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.75rem;">
                    <span style="color: var(--text-muted);">${ds.size_estimate || 'Size unknown'}</span>
                    <span style="color: var(--funk-gold);">${ds.phase || 'Phase 1'}</span>
                </div>
                <div style="display: flex; gap: 8px; margin-top: 12px;">
                    <button class="btn btn-secondary" onclick="downloadDataset('${ds.name}')" style="flex: 1; padding: 6px; font-size: 0.8rem;">
                        Download
                    </button>
                    <button class="btn btn-primary" onclick="downloadAndEncode('${ds.name}')" style="flex: 1; padding: 6px; font-size: 0.8rem;">
                        Download & π/2 Encode
                    </button>
                </div>
            </div>
        `;
    });

    container.innerHTML = html;
}

async function downloadDataset(name) {
    const statusDiv = document.getElementById('registry-download-status');
    const statusText = document.getElementById('registry-download-text');
    const progressFill = document.getElementById('registry-download-progress');

    if (statusDiv) statusDiv.style.display = 'block';
    if (statusText) statusText.textContent = `Downloading ${name}...`;
    if (progressFill) progressFill.style.width = '0%';

    piFeatureState.registry.downloading = true;
    piFeatureState.registry.currentDownload = name;

    try {
        const response = await fetch('/api/datasets/download', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name })
        });

        const data = await response.json();

        if (response.ok) {
            if (progressFill) progressFill.style.width = '100%';
            if (statusText) statusText.innerHTML = `<span style="color: var(--success);">✓ Downloaded ${name}</span><br>Path: ${data.path || 'See console'}`;
            if (typeof toast !== 'undefined') toast.success(`Downloaded ${name}`);
        } else {
            if (statusText) statusText.innerHTML = `<span style="color: var(--error);">Error: ${data.error}</span>`;
            if (typeof toast !== 'undefined') toast.error(data.error || 'Download failed');
        }
    } catch (error) {
        console.error('Download error:', error);
        if (statusText) statusText.innerHTML = `<span style="color: var(--error);">Network error</span>`;
        if (typeof toast !== 'undefined') toast.error('Network error');
    } finally {
        piFeatureState.registry.downloading = false;
        piFeatureState.registry.currentDownload = null;
    }
}

async function downloadAndEncode(name) {
    const statusDiv = document.getElementById('registry-download-status');
    const statusText = document.getElementById('registry-download-text');
    const progressFill = document.getElementById('registry-download-progress');

    if (statusDiv) statusDiv.style.display = 'block';
    if (statusText) statusText.textContent = `Downloading and π/2-encoding ${name}...`;
    if (progressFill) progressFill.style.width = '0%';

    try {
        const response = await fetch('/api/datasets/download-and-encode', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name })
        });

        const data = await response.json();

        if (response.ok) {
            if (progressFill) progressFill.style.width = '100%';
            if (statusText) statusText.innerHTML = `<span style="color: var(--success);">✓ Downloaded & encoded ${name}</span><br>Output: ${data.output_path || 'See console'}`;
            if (typeof toast !== 'undefined') toast.success(`${name} ready for π/2 training!`);
        } else {
            if (statusText) statusText.innerHTML = `<span style="color: var(--error);">Error: ${data.error}</span>`;
            if (typeof toast !== 'undefined') toast.error(data.error || 'Operation failed');
        }
    } catch (error) {
        console.error('Download/encode error:', error);
        if (statusText) statusText.innerHTML = `<span style="color: var(--error);">Network error</span>`;
    }
}

// =============================================================================
// π/2 Rotation Tokenizer Functions
// =============================================================================

async function initializeTokenizer() {
    const statusDiv = document.getElementById('tokenizer-status');
    const statusText = document.getElementById('tokenizer-status-text');
    const processBtn = document.getElementById('process-tokenizer-btn');

    if (statusDiv) statusDiv.style.display = 'block';
    if (statusText) statusText.textContent = 'Initializing tokenizer...';

    const config = {
        vocab_size: parseInt(document.getElementById('tokenizer-vocab-size')?.value || 50257),
        phase_token_base: parseInt(document.getElementById('tokenizer-phase-base')?.value || 50256),
        rotate_token_ids: document.getElementById('tokenizer-rotate-ids')?.checked ?? true,
        use_phase_prefix: document.getElementById('tokenizer-phase-prefix')?.checked ?? true,
        include_phase_metadata: document.getElementById('tokenizer-include-meta')?.checked ?? true
    };

    try {
        const response = await fetch('/api/rotation/init', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });

        const data = await response.json();

        if (response.ok) {
            piFeatureState.tokenizer.initialized = true;
            piFeatureState.tokenizer.vocabSize = config.vocab_size;
            if (processBtn && piFeatureState.tokenizer.selectedFile) processBtn.disabled = false;
            if (statusText) {
                statusText.innerHTML = `<span style="color: var(--success);">✓ Tokenizer initialized</span><br>
                    Vocab: <strong>${config.vocab_size.toLocaleString()}</strong> |
                    Phase Base: <strong>${config.phase_token_base}</strong>`;
            }
            if (typeof toast !== 'undefined') toast.success('Tokenizer initialized');
        } else {
            if (statusText) statusText.innerHTML = `<span style="color: var(--error);">Error: ${data.error}</span>`;
        }
    } catch (error) {
        console.error('Tokenizer init error:', error);
        if (statusText) statusText.innerHTML = `<span style="color: var(--error);">Network error</span>`;
    }
}

function selectTokenizerFile(type) {
    piFeatureState.tokenizer.fileType = type;

    // Create a file input
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = type === 'tokenized' ? '.jsonl,.json' : '.txt,.csv,.json';

    input.onchange = (e) => {
        const file = e.target.files[0];
        if (file) {
            piFeatureState.tokenizer.selectedFile = file;

            const fileInfo = document.getElementById('tokenizer-file-info');
            const fileName = document.getElementById('tokenizer-file-name');
            const processBtn = document.getElementById('process-tokenizer-btn');

            if (fileInfo) fileInfo.style.display = 'block';
            if (fileName) fileName.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
            if (processBtn && piFeatureState.tokenizer.initialized) processBtn.disabled = false;

            if (typeof toast !== 'undefined') toast.success(`Selected: ${file.name}`);
        }
    };

    input.click();
}

async function previewTokenRotation() {
    const input = document.getElementById('tokenizer-preview-input')?.value;
    const outputDiv = document.getElementById('tokenizer-preview-output');

    if (!input) {
        if (typeof toast !== 'undefined') toast.warning('Enter text to preview');
        return;
    }

    try {
        const response = await fetch('/api/rotation/preview', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: input })
        });

        const data = await response.json();

        if (response.ok && data.rotations) {
            if (outputDiv) outputDiv.style.display = 'block';

            // Update each phase preview
            ['0', '90', '180', '270'].forEach((deg, idx) => {
                const el = document.getElementById(`preview-phase-${deg}`);
                if (el && data.rotations[idx]) {
                    el.textContent = Array.isArray(data.rotations[idx])
                        ? data.rotations[idx].slice(0, 10).join(', ') + '...'
                        : data.rotations[idx];
                }
            });
        } else {
            if (typeof toast !== 'undefined') toast.error(data.error || 'Preview failed');
        }
    } catch (error) {
        console.error('Preview error:', error);
        if (typeof toast !== 'undefined') toast.error('Network error');
    }
}

async function processTokenizerData() {
    if (!piFeatureState.tokenizer.initialized) {
        if (typeof toast !== 'undefined') toast.warning('Initialize tokenizer first');
        return;
    }

    if (!piFeatureState.tokenizer.selectedFile) {
        if (typeof toast !== 'undefined') toast.warning('Select a file first');
        return;
    }

    const statusDiv = document.getElementById('tokenizer-status');
    const statusText = document.getElementById('tokenizer-status-text');
    const processBtn = document.getElementById('process-tokenizer-btn');

    if (statusDiv) statusDiv.style.display = 'block';
    if (statusText) statusText.textContent = 'Processing and rotating data...';
    if (processBtn) {
        processBtn.disabled = true;
        processBtn.textContent = 'Processing...';
    }

    const formData = new FormData();
    formData.append('file', piFeatureState.tokenizer.selectedFile);
    formData.append('type', piFeatureState.tokenizer.fileType);

    try {
        const endpoint = piFeatureState.tokenizer.fileType === 'tokenized'
            ? '/api/rotation/process'
            : '/api/rotation/process-raw';

        const response = await fetch(endpoint, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            if (statusText) {
                statusText.innerHTML = `<span style="color: var(--success);">✓ Processing complete!</span><br>
                    Input: <strong>${data.stats?.input_records || 'N/A'}</strong> records<br>
                    Output: <strong>${data.stats?.output_records || 'N/A'}</strong> rotated records (4x expansion)<br>
                    File: ${data.output_path || 'See console'}`;
            }
            if (typeof toast !== 'undefined') toast.success('Data rotated successfully!');
        } else {
            if (statusText) statusText.innerHTML = `<span style="color: var(--error);">Error: ${data.error}</span>`;
            if (typeof toast !== 'undefined') toast.error(data.error || 'Processing failed');
        }
    } catch (error) {
        console.error('Processing error:', error);
        if (statusText) statusText.innerHTML = `<span style="color: var(--error);">Network error</span>`;
    } finally {
        if (processBtn) {
            processBtn.disabled = false;
            processBtn.textContent = 'Process & Rotate Data';
        }
    }
}

// =============================================================================
// π/2 Universal Encoder Functions
// =============================================================================

async function initializeEncoder() {
    const statusDiv = document.getElementById('encoder-status');
    const statusText = document.getElementById('encoder-status-text');
    const startBtn = document.getElementById('start-encoding-btn');

    if (statusDiv) statusDiv.style.display = 'block';
    if (statusText) statusText.textContent = 'Initializing encoder...';

    const config = {
        audio: {
            sample_rate: parseInt(document.getElementById('encoder-sample-rate')?.value || 22050),
            chunk_ms: parseInt(document.getElementById('encoder-chunk-ms')?.value || 25),
            n_fft: parseInt(document.getElementById('encoder-n-fft')?.value || 512)
        },
        image: {
            width: parseInt(document.getElementById('encoder-img-width')?.value || 224),
            height: parseInt(document.getElementById('encoder-img-height')?.value || 224),
            patch_size: parseInt(document.getElementById('encoder-patch-size')?.value || 16),
            use_fft: document.getElementById('encoder-use-fft')?.checked ?? true
        }
    };

    try {
        const response = await fetch('/api/universal-encoder/init', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });

        const data = await response.json();

        if (response.ok) {
            piFeatureState.encoder.initialized = true;
            if (startBtn && (piFeatureState.encoder.selectedFile || piFeatureState.encoder.selectedDirectory)) {
                startBtn.disabled = false;
            }
            if (statusText) {
                statusText.innerHTML = `<span style="color: var(--success);">✓ Encoder initialized</span><br>
                    Ready to encode: Text, Audio, Images, Video`;
            }
            if (typeof toast !== 'undefined') toast.success('Universal encoder initialized');
        } else {
            if (statusText) statusText.innerHTML = `<span style="color: var(--error);">Error: ${data.error}</span>`;
        }
    } catch (error) {
        console.error('Encoder init error:', error);
        if (statusText) statusText.innerHTML = `<span style="color: var(--error);">Network error</span>`;
    }
}

function selectEncoderFile() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.txt,.csv,.json,.jsonl,.wav,.mp3,.flac,.ogg,.png,.jpg,.jpeg,.bmp,.webp,.mp4,.avi,.mov,.mkv';

    input.onchange = (e) => {
        const file = e.target.files[0];
        if (file) {
            piFeatureState.encoder.selectedFile = file;
            piFeatureState.encoder.selectedDirectory = null;

            // Determine file type
            const ext = file.name.split('.').pop().toLowerCase();
            const textExts = ['txt', 'csv', 'json', 'jsonl'];
            const audioExts = ['wav', 'mp3', 'flac', 'ogg', 'm4a'];
            const imageExts = ['png', 'jpg', 'jpeg', 'bmp', 'webp', 'gif'];
            const videoExts = ['mp4', 'avi', 'mov', 'mkv', 'webm'];

            let fileType = 'unknown';
            if (textExts.includes(ext)) fileType = 'text';
            else if (audioExts.includes(ext)) fileType = 'audio';
            else if (imageExts.includes(ext)) fileType = 'image';
            else if (videoExts.includes(ext)) fileType = 'video';

            piFeatureState.encoder.fileType = fileType;

            const fileInfo = document.getElementById('encoder-file-info');
            const fileName = document.getElementById('encoder-file-name');
            const fileTypeSpan = document.getElementById('encoder-file-type');
            const startBtn = document.getElementById('start-encoding-btn');

            if (fileInfo) fileInfo.style.display = 'block';
            if (fileName) fileName.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
            if (fileTypeSpan) {
                fileTypeSpan.textContent = fileType.toUpperCase();
                const colors = { text: 'var(--funk-gold)', audio: 'var(--groove-teal)', image: 'var(--funk-orange)', video: 'var(--groove-sage)' };
                fileTypeSpan.style.background = colors[fileType] || 'var(--text-muted)';
            }
            if (startBtn && piFeatureState.encoder.initialized) startBtn.disabled = false;

            if (typeof toast !== 'undefined') toast.success(`Selected: ${file.name}`);
        }
    };

    input.click();
}

function selectEncoderDirectory() {
    // Note: Directory selection requires webkitdirectory which has limited browser support
    const input = document.createElement('input');
    input.type = 'file';
    input.webkitdirectory = true;
    input.multiple = true;

    input.onchange = (e) => {
        const files = Array.from(e.target.files);
        if (files.length > 0) {
            piFeatureState.encoder.selectedDirectory = files;
            piFeatureState.encoder.selectedFile = null;

            const fileInfo = document.getElementById('encoder-file-info');
            const fileName = document.getElementById('encoder-file-name');
            const fileTypeSpan = document.getElementById('encoder-file-type');
            const startBtn = document.getElementById('start-encoding-btn');

            if (fileInfo) fileInfo.style.display = 'block';
            if (fileName) fileName.textContent = `${files.length} files selected`;
            if (fileTypeSpan) {
                fileTypeSpan.textContent = 'BATCH';
                fileTypeSpan.style.background = 'var(--info)';
            }
            if (startBtn && piFeatureState.encoder.initialized) startBtn.disabled = false;

            if (typeof toast !== 'undefined') toast.success(`Selected ${files.length} files`);
        }
    };

    input.click();
}

async function startEncoding() {
    if (!piFeatureState.encoder.initialized) {
        if (typeof toast !== 'undefined') toast.warning('Initialize encoder first');
        return;
    }

    if (!piFeatureState.encoder.selectedFile && !piFeatureState.encoder.selectedDirectory) {
        if (typeof toast !== 'undefined') toast.warning('Select a file or directory first');
        return;
    }

    const statusDiv = document.getElementById('encoder-status');
    const statusText = document.getElementById('encoder-status-text');
    const statsDiv = document.getElementById('encoder-stats');
    const progressBar = document.getElementById('encoder-progress-bar');
    const progressFill = document.getElementById('encoder-progress-fill');
    const startBtn = document.getElementById('start-encoding-btn');
    const outputPath = document.getElementById('encoder-output')?.value || 'output/pi_encoded.jsonl';

    if (statusDiv) statusDiv.style.display = 'block';
    if (statusText) statusText.textContent = 'Encoding to π/2...';
    if (progressBar) progressBar.style.display = 'block';
    if (progressFill) progressFill.style.width = '0%';
    if (startBtn) {
        startBtn.disabled = true;
        startBtn.textContent = 'Encoding...';
    }

    piFeatureState.encoder.isEncoding = true;

    const formData = new FormData();
    formData.append('output_path', outputPath);

    if (piFeatureState.encoder.selectedFile) {
        formData.append('file', piFeatureState.encoder.selectedFile);
    } else if (piFeatureState.encoder.selectedDirectory) {
        piFeatureState.encoder.selectedDirectory.forEach((file, idx) => {
            formData.append(`files`, file);
        });
    }

    try {
        const endpoint = piFeatureState.encoder.selectedDirectory
            ? '/api/universal-encoder/encode-directory'
            : '/api/universal-encoder/encode';

        const response = await fetch(endpoint, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            if (progressFill) progressFill.style.width = '100%';
            if (statsDiv) statsDiv.style.display = 'block';

            // Update stats
            if (data.stats) {
                document.getElementById('stat-files').textContent = data.stats.files_processed || 0;
                document.getElementById('stat-text').textContent = data.stats.text_files || 0;
                document.getElementById('stat-audio').textContent = data.stats.audio_files || 0;
                document.getElementById('stat-image').textContent = data.stats.image_files || 0;
                document.getElementById('stat-tokens').textContent = (data.stats.total_tokens || 0).toLocaleString();
            }

            if (statusText) {
                statusText.innerHTML = `<span style="color: var(--success);">✓ Encoding complete!</span><br>
                    Output: ${data.output_path || outputPath}`;
            }
            if (typeof toast !== 'undefined') toast.success('Encoding complete!');
        } else {
            if (statusText) statusText.innerHTML = `<span style="color: var(--error);">Error: ${data.error}</span>`;
            if (typeof toast !== 'undefined') toast.error(data.error || 'Encoding failed');
        }
    } catch (error) {
        console.error('Encoding error:', error);
        if (statusText) statusText.innerHTML = `<span style="color: var(--error);">Network error</span>`;
    } finally {
        piFeatureState.encoder.isEncoding = false;
        if (startBtn) {
            startBtn.disabled = false;
            startBtn.textContent = 'Encode to π/2';
        }
    }
}

// =============================================================================
// Initialization
// =============================================================================

document.addEventListener('DOMContentLoaded', () => {
    // Set up initial event listeners if needed
    console.log('π/2 Features module loaded');
});

// =============================================================================
// Global Exports (for HTML onclick handlers)
// =============================================================================

window.togglePiTrainer = togglePiTrainer;
window.togglePiRegistry = togglePiRegistry;
window.togglePiTokenizer = togglePiTokenizer;
window.togglePiEncoder = togglePiEncoder;
window.toggleEncoderAudio = toggleEncoderAudio;
window.toggleEncoderImage = toggleEncoderImage;

// Registry functions
window.loadRegistryDatasets = loadRegistryDatasets;
window.filterRegistryDatasets = filterRegistryDatasets;
window.renderRegistryDatasets = renderRegistryDatasets;
window.downloadDataset = downloadDataset;
window.downloadAndEncode = downloadAndEncode;

// Tokenizer functions
window.initializeTokenizer = initializeTokenizer;
window.selectTokenizerFile = selectTokenizerFile;
window.previewTokenRotation = previewTokenRotation;
window.processTokenizerData = processTokenizerData;

// Encoder functions
window.initializeEncoder = initializeEncoder;
window.selectEncoderFile = selectEncoderFile;
window.selectEncoderDirectory = selectEncoderDirectory;
window.startEncoding = startEncoding;

// Trainer functions
window.selectRotationMode = selectRotationMode;
window.previewTrainerSchedule = previewTrainerSchedule;
window.initializeTrainer = initializeTrainer;
window.startRotationalTraining = startRotationalTraining;
