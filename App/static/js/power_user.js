/**
 * Power User Features - Frontend Integration
 * Advanced controls for experienced users
 * Works with advanced_training.py backend
 */

class PowerUserFeatures {
    constructor() {
        this.initialized = false;
        this.currentConfig = {};
    }

    // =========================================================================
    // PANEL CREATION
    // =========================================================================

    createWeightInitPanel() {
        return `
        <div class="power-user-panel" id="weight-init-panel">
            <h3>Weight Initialization</h3>
            <div class="init-controls">
                <select id="init-method" class="form-control">
                    <option value="default">Keep Pretrained Weights (Default)</option>
                    <option value="xavier_uniform">Xavier Uniform (Balanced)</option>
                    <option value="xavier_normal">Xavier Normal (Gaussian)</option>
                    <option value="kaiming_uniform">Kaiming/He Uniform (ReLU nets)</option>
                    <option value="kaiming_normal">Kaiming/He Normal (ReLU nets)</option>
                    <option value="normal">Normal Distribution</option>
                    <option value="truncated_normal">Truncated Normal (Stable)</option>
                    <option value="orthogonal">Orthogonal (Preserves norms)</option>
                    <option value="sparse">Sparse (Mostly zeros)</option>
                </select>

                <div class="init-params" style="display:none;">
                    <label>Init Range:
                        <input type="number" id="init-range" value="0.02" step="0.001" min="0.001" max="1.0">
                        <small>Standard deviation for normal/uniform init</small>
                    </label>

                    <label>Modules to Initialize:
                        <textarea id="init-modules" rows="3" placeholder="lm_head&#10;embed_tokens&#10;(one per line)"></textarea>
                        <small>Leave empty to use defaults</small>
                    </label>

                    <label class="sparse-only" style="display:none;">Sparsity Factor:
                        <input type="number" id="init-sparsity" value="0.1" step="0.01" min="0" max="0.9">
                    </label>
                </div>
            </div>
        </div>
        `;
    }

    createOptimizerPanel() {
        return `
        <div class="power-user-panel" id="optimizer-hyperparams-panel">
            <h3>Optimizer Hyperparameters</h3>

            <!-- Adam/AdamW Parameters -->
            <div class="adam-params optimizer-params">
                <h4>Adam/AdamW Parameters</h4>
                <div class="param-grid">
                    <label>Beta1 (Momentum):
                        <input type="number" id="adam-beta1" value="0.9" step="0.01" min="0" max="0.999">
                        <small>First moment decay (default: 0.9)</small>
                    </label>

                    <label>Beta2 (RMSprop):
                        <input type="number" id="adam-beta2" value="0.999" step="0.001" min="0.9" max="0.9999">
                        <small>Second moment decay (default: 0.999)</small>
                    </label>

                    <label>Epsilon:
                        <input type="number" id="adam-epsilon" value="1e-8" step="1e-9" min="1e-10" max="1e-6">
                        <small>Numerical stability (default: 1e-8)</small>
                    </label>
                </div>
            </div>

            <!-- SGD Parameters -->
            <div class="sgd-params optimizer-params" style="display:none;">
                <h4>SGD Parameters</h4>
                <div class="param-grid">
                    <label>Momentum:
                        <input type="number" id="sgd-momentum" value="0.9" step="0.01" min="0" max="0.999">
                    </label>

                    <label>Dampening:
                        <input type="number" id="sgd-dampening" value="0" step="0.01" min="0" max="1">
                    </label>

                    <label>
                        <input type="checkbox" id="sgd-nesterov">
                        Use Nesterov momentum
                    </label>
                </div>
            </div>

            <!-- Adafactor Parameters -->
            <div class="adafactor-params optimizer-params" style="display:none;">
                <h4>Adafactor Parameters</h4>
                <div class="param-grid">
                    <label>
                        <input type="checkbox" id="adafactor-scale" checked>
                        Scale parameters
                    </label>

                    <label>
                        <input type="checkbox" id="adafactor-relative">
                        Relative step size
                    </label>

                    <label>Clip threshold:
                        <input type="number" id="adafactor-clip" value="1.0" step="0.1" min="0.1" max="10">
                    </label>
                </div>
            </div>
        </div>
        `;
    }

    createWeightDecayGroupsPanel() {
        return `
        <div class="power-user-panel" id="weight-decay-groups">
            <h3>Per-Layer Weight Decay</h3>

            <div class="decay-toggle">
                <label>
                    <input type="checkbox" id="enable-per-layer-decay">
                    Enable per-layer weight decay
                </label>
            </div>

            <div class="decay-groups" style="display:none;">
                <div class="param-grid">
                    <label>Attention Layers:
                        <input type="number" id="attention-weight-decay" value="0.01" step="0.001" min="0" max="0.3">
                    </label>

                    <label>MLP/FFN Layers:
                        <input type="number" id="mlp-weight-decay" value="0.01" step="0.001" min="0" max="0.3">
                    </label>

                    <label>Embedding Layers:
                        <input type="number" id="embedding-weight-decay" value="0.0" step="0.001" min="0" max="0.1">
                    </label>
                </div>

                <div class="no-decay-patterns">
                    <label>No Decay Patterns:
                        <textarea id="no-decay-patterns" rows="3">bias
LayerNorm
layer_norm</textarea>
                        <small>Parameters matching these patterns get zero weight decay</small>
                    </label>
                </div>
            </div>
        </div>
        `;
    }

    createSchedulerAdvancedPanel() {
        return `
        <div class="power-user-panel" id="scheduler-advanced">
            <h3>Advanced Scheduler Config</h3>

            <div class="warmup-config">
                <h4>Warmup Mode</h4>
                <label>
                    <input type="radio" name="warmup-type" value="steps" checked>
                    Use warmup steps (from main config)
                </label>
                <label>
                    <input type="radio" name="warmup-type" value="ratio">
                    Use warmup ratio:
                    <input type="number" id="warmup-ratio" value="0.1" step="0.01" min="0" max="0.5" style="width: 80px; margin-left: 8px;">
                </label>
                
                <h4 style="margin-top: 12px;">Warmup Type</h4>
                <select id="warmup-type-select" class="form-control">
                    <option value="linear">Linear (Default)</option>
                    <option value="cosine">Cosine</option>
                    <option value="constant">Constant</option>
                </select>
                <small>How the learning rate ramps up during warmup</small>
            </div>

            <!-- Cosine with Restarts -->
            <div class="cosine-restarts-params scheduler-specific" style="display:none;">
                <label>Number of Cycles:
                    <input type="number" id="num-cycles" value="1" min="0.5" max="10" step="0.5">
                    <small>Number of cosine cycles (restarts). 0.5 = half cycle</small>
                </label>
            </div>

            <!-- Polynomial Decay -->
            <div class="polynomial-params scheduler-specific" style="display:none;">
                <label>Power:
                    <input type="number" id="poly-power" value="1.0" step="0.1" min="0.5" max="5">
                    <small>1.0 = linear, 2.0 = quadratic, 3.0 = cubic</small>
                </label>

                <label>End Learning Rate:
                    <input type="number" id="lr-end" value="1e-7" step="1e-8" min="0">
                </label>
            </div>

            <div class="scheduler-preview">
                <canvas id="scheduler-preview-chart" width="400" height="150"></canvas>
            </div>
        </div>
        `;
    }

    createLossScalingPanel() {
        return `
        <div class="power-user-panel" id="loss-scaling-panel">
            <h3>Loss Scaling (Mixed Precision)</h3>

            <div class="loss-scaling-strategy">
                <select id="loss-scaling-strategy" class="form-control">
                    <option value="dynamic">Dynamic (Auto-adjust) - Recommended</option>
                    <option value="static">Static (Fixed scale)</option>
                    <option value="none">None (No scaling)</option>
                </select>
            </div>

            <div class="loss-scaling-params" style="margin-top: 12px;">
                <div class="dynamic-params scaling-specific">
                    <div class="param-grid">
                        <label>Initial Scale:
                            <input type="number" id="init-scale" value="32768" min="1">
                        </label>

                        <label>Growth Factor:
                            <input type="number" id="growth-factor" value="2.0" step="0.1" min="1.1">
                        </label>

                        <label>Backoff Factor:
                            <input type="number" id="backoff-factor" value="0.5" step="0.1" min="0.1" max="0.9">
                        </label>

                        <label>Growth Interval:
                            <input type="number" id="growth-interval" value="2000" min="100">
                            <small>Steps between scale increases</small>
                        </label>
                    </div>
                </div>
            </div>
        </div>
        `;
    }

    createLoRAPlusPanel() {
        return `
        <div class="power-user-panel" id="lora-plus-panel">
            <h3>LoRA+ & Advanced LoRA</h3>

            <div class="lora-plus-toggle">
                <label>
                    <input type="checkbox" id="enable-lora-plus">
                    Enable LoRA+ (Different LR for A/B matrices)
                </label>
                <small>B matrix learns faster than A matrix</small>
            </div>

            <div class="lora-plus-params" style="display:none; margin-top: 12px;">
                <label>LR Ratio (B/A):
                    <input type="number" id="lora-lr-ratio" value="16" step="1" min="1" max="64">
                    <small>B matrix LR = A matrix LR x ratio</small>
                </label>
            </div>

            <div class="rs-lora-section" style="margin-top: 16px; padding-top: 16px; border-top: 1px solid var(--border-color);">
                <label>
                    <input type="checkbox" id="enable-rs-lora">
                    Enable RS-LoRA (Rank Stabilization)
                </label>
                <small>Scales by alpha/sqrt(rank) for better stability across ranks</small>
            </div>

            <div class="qlora-compute-dtype" style="margin-top: 16px; padding-top: 16px; border-top: 1px solid var(--border-color);">
                <label>QLoRA Compute Dtype:
                    <select id="qlora-compute-dtype" class="form-control">
                        <option value="float16">Float16 (Default, wider support)</option>
                        <option value="bfloat16">BFloat16 (Better for large values, RTX 30/40+)</option>
                    </select>
                </label>
                <small>Precision for QLoRA compute operations. BF16 needs Ampere+ GPU.</small>
            </div>

            <div class="manual-target-modules" style="margin-top: 16px; padding-top: 16px; border-top: 1px solid var(--border-color);">
                <label>
                    <input type="checkbox" id="manual-modules-toggle">
                    Override automatic module detection
                </label>

                <div class="manual-modules-input" style="display:none; margin-top: 8px;">
                    <textarea id="manual-lora-modules" rows="4" placeholder="q_proj&#10;v_proj&#10;k_proj&#10;o_proj&#10;(one per line)"></textarea>
                    <button id="validate-modules" class="btn btn-secondary" style="margin-top: 8px;">Validate Modules</button>
                    <div id="module-validation-result" style="margin-top: 8px;"></div>
                </div>
            </div>
        </div>
        `;
    }

    createLRFinderPanel() {
        return `
        <div class="power-user-panel" id="lr-finder-panel">
            <h3>Automatic LR Finder</h3>

            <div class="lr-finder-toggle">
                <label>
                    <input type="checkbox" id="enable-lr-finder">
                    Run LR range test before training
                </label>
                <small>Automatically finds optimal learning rate</small>
            </div>

            <div class="lr-finder-params" style="display:none; margin-top: 12px;">
                <div class="param-grid">
                    <label>Test Iterations:
                        <input type="number" id="lr-finder-iterations" value="100" min="50" max="300" step="25">
                    </label>

                    <label>Start LR:
                        <input type="number" id="lr-finder-start" value="1e-7" step="1e-8" min="1e-10">
                    </label>

                    <label>End LR:
                        <input type="number" id="lr-finder-end" value="10" step="0.1" min="0.001">
                    </label>
                </div>
            </div>
        </div>
        `;
    }

    createDataPipelinePanel() {
        return `
        <div class="power-user-panel" id="data-pipeline-panel">
            <h3>Data Pipeline Control</h3>

            <div class="token-control" style="margin-bottom: 16px;">
                <h4>Special Tokens</h4>
                <label>
                    <input type="checkbox" id="add-bos-token" checked>
                    Add BOS (Beginning of Sequence) token
                </label>
                <label>
                    <input type="checkbox" id="add-eos-token" checked>
                    Add EOS (End of Sequence) token
                </label>
                <small>Control whether special tokens are added to training examples</small>
            </div>

            <div class="loss-masking" style="margin-bottom: 16px; padding-top: 16px; border-top: 1px solid var(--border-color);">
                <label>
                    <input type="checkbox" id="completion-only-loss">
                    Completion-only loss masking
                </label>
                <small>Only compute loss on response/completion tokens, not input/instruction tokens. Useful for instruction tuning.</small>
            </div>

            <div class="padding-control" style="margin-bottom: 16px; padding-top: 16px; border-top: 1px solid var(--border-color);">
                <label>
                    <input type="checkbox" id="dynamic-padding" checked>
                    Dynamic padding (batch-level)
                </label>
                <small>Pad to longest sequence in batch instead of max_length. Faster training.</small>
            </div>

            <div class="sample-generation" style="padding-top: 16px; border-top: 1px solid var(--border-color);">
                <h4>Sample Output Generation</h4>
                <label>Generate samples every N steps:
                    <input type="number" id="generate-samples-steps" value="0" min="0" max="10000" step="100">
                </label>
                <small>0 = disabled. Set to e.g. 500 to generate sample outputs during training.</small>
                
                <div class="sample-prompts-input" style="margin-top: 12px;">
                    <label>Sample prompts (one per line):</label>
                    <textarea id="sample-prompts" rows="3" placeholder="Tell me about\nOnce upon a time\nThe meaning of life is"></textarea>
                    <small>These prompts will be used to generate samples during training</small>
                </div>
            </div>
        </div>
        `;
    }

    createContextArchitecturePanel() {
        return `
        <div class="power-user-panel" id="context-arch-panel">
            <h3>Context & Architecture</h3>

            <div class="rope-scaling" style="margin-bottom: 16px;">
                <h4>RoPE Scaling (Context Extension)</h4>
                <select id="rope-scaling-type" class="form-control">
                    <option value="none">None (Use model default)</option>
                    <option value="linear">Linear - Simple scaling</option>
                    <option value="dynamic">Dynamic - Adjusts based on sequence length</option>
                    <option value="yarn">YaRN - Yet another RoPE extension (best quality)</option>
                </select>
                
                <div class="rope-params" style="display:none; margin-top: 12px;">
                    <label>Scaling Factor:
                        <input type="number" id="rope-scaling-factor" value="2.0" min="1.0" max="8.0" step="0.5">
                    </label>
                    <small>2.0 = double context, 4.0 = quadruple, etc.</small>
                </div>
            </div>

            <div class="layer-control" style="margin-bottom: 16px; padding-top: 16px; border-top: 1px solid var(--border-color);">
                <h4>Layer Control</h4>
                <label>
                    <input type="checkbox" id="freeze-embeddings">
                    Freeze embedding layer
                </label>
                <small>Don't train the embedding layer (saves memory)</small>
                
                <div style="margin-top: 8px;">
                    <label>Freeze first N layers:
                        <input type="number" id="freeze-layers" value="0" min="0" max="48" step="1">
                    </label>
                    <small>0 = train all layers. Freezing early layers can speed up training.</small>
                </div>
            </div>

            <div class="torch-compile" style="padding-top: 16px; border-top: 1px solid var(--border-color);">
                <label>
                    <input type="checkbox" id="enable-torch-compile">
                    Enable torch.compile() (PyTorch 2.0+)
                </label>
                <small>Can provide 10-30% speedup but has startup overhead. Best for long training runs.</small>
            </div>
        </div>
        `;
    }

    createQATPanel() {
        return `
        <div class="power-user-panel" id="qat-panel">
            <h3>🔬 Quantization-Aware Training (EfficientQAT)</h3>
            <p class="panel-desc">Train models that stay accurate even when compressed to 2-4 bits. Perfect for deployment on limited hardware.</p>

            <div class="qat-toggle">
                <label>
                    <input type="checkbox" id="enable-efficient-qat">
                    Enable EfficientQAT
                </label>
                <small>Think of it like teaching your model to work with a smaller vocabulary - it learns to be just as smart while using less memory!</small>
            </div>

            <div class="qat-params" style="display:none; margin-top: 16px;">

                <!-- Target Bits -->
                <div class="qat-section" style="margin-bottom: 16px;">
                    <h4>Target Quantization (How much to compress)</h4>
                    <div class="bit-selector">
                        <label>
                            <input type="radio" name="qat-bits" value="8"> 8-bit
                            <small class="bit-desc">Like switching from HD to slightly-less-HD video - barely noticeable!</small>
                        </label>
                        <label>
                            <input type="radio" name="qat-bits" value="4" checked> 4-bit
                            <small class="bit-desc">Sweet spot! Great quality, 4x smaller. Most popular choice.</small>
                        </label>
                        <label>
                            <input type="radio" name="qat-bits" value="3"> 3-bit
                            <small class="bit-desc">More aggressive - some quality loss but much smaller</small>
                        </label>
                        <label>
                            <input type="radio" name="qat-bits" value="2"> 2-bit
                            <small class="bit-desc">Extreme compression - 8x smaller, experimental</small>
                        </label>
                    </div>

                    <div id="qat-memory-estimate" class="memory-estimate" style="margin-top: 12px; padding: 8px; background: var(--bg-tertiary); border-radius: 4px;">
                        <span class="loading">Calculating memory...</span>
                    </div>
                </div>

                <!-- Group Size -->
                <div class="qat-section" style="margin-bottom: 16px; padding-top: 16px; border-top: 1px solid var(--border-color);">
                    <h4>Group Size</h4>
                    <select id="qat-group-size" class="form-control">
                        <option value="32">32 (Most accurate, largest)</option>
                        <option value="64">64 (Balanced)</option>
                        <option value="128" selected>128 (Default - good balance)</option>
                        <option value="256">256 (Smallest, fastest)</option>
                    </select>
                    <small>Think of groups like organizing items into boxes - smaller boxes = more precision but more overhead.</small>
                </div>

                <!-- Calibration Settings -->
                <div class="qat-section" style="margin-bottom: 16px; padding-top: 16px; border-top: 1px solid var(--border-color);">
                    <h4>Calibration (Teaching the compression)</h4>

                    <div class="param-grid">
                        <label>Calibration Samples:
                            <input type="number" id="qat-calibration-samples" value="128" min="16" max="1024" step="16">
                        </label>
                    </div>
                    <small>Like showing the model example data to figure out the best way to compress - more samples = better but slower.</small>

                    <div style="margin-top: 12px;">
                        <label>Warmup Steps:
                            <input type="number" id="qat-warmup-steps" value="100" min="0" max="1000" step="50">
                        </label>
                        <small>How many training steps before starting quantization. Like warming up before exercise!</small>
                    </div>
                </div>

                <!-- Advanced Options -->
                <div class="qat-section" style="padding-top: 16px; border-top: 1px solid var(--border-color);">
                    <h4>Advanced Options</h4>

                    <div style="margin-top: 8px;">
                        <label>
                            <input type="checkbox" id="qat-symmetric" checked>
                            Symmetric Quantization
                        </label>
                        <small>Usually better for weights - keeps zero at zero.</small>
                    </div>

                    <div style="margin-top: 8px;">
                        <label>
                            <input type="checkbox" id="qat-quantize-embeddings">
                            Quantize Embeddings
                        </label>
                        <small>Also compress the word lookup table. Saves more space but may reduce quality.</small>
                    </div>
                </div>
            </div>
        </div>
        `;
    }

    // =========================================================================
    // INITIALIZATION
    // =========================================================================

    init() {
        if (this.initialized) return;

        this.insertPanels();
        this.setupEventHandlers();
        this.loadConfiguration();

        this.initialized = true;
        console.log('Power User Features initialized');
    }

    insertPanels() {
        // Try to find existing container in HTML first
        let panelsContainer = document.getElementById('power-user-panels');

        if (panelsContainer) {
            // Use the existing HTML container
            panelsContainer.innerHTML = `
                ${this.createWeightInitPanel()}
                ${this.createOptimizerPanel()}
                ${this.createWeightDecayGroupsPanel()}
                ${this.createSchedulerAdvancedPanel()}
                ${this.createLossScalingPanel()}
                ${this.createLoRAPlusPanel()}
                ${this.createDataPipelinePanel()}
                ${this.createContextArchitecturePanel()}
                ${this.createLRFinderPanel()}
                ${this.createQATPanel()}
            `;
            return;
        }

        // Fallback: create section dynamically (for backwards compatibility)
        let powerUserSection = document.getElementById('power-user-section');
        if (powerUserSection) return; // Already exists

        powerUserSection = document.createElement('section');
        powerUserSection.id = 'power-user-section';
        powerUserSection.className = 'card';
        powerUserSection.innerHTML = `
            <div class="collapsible-header" onclick="togglePowerUser()" role="button" tabindex="0" aria-expanded="false">
                <h2 style="margin: 0;">Power User Features</h2>
                <span class="collapse-icon" id="power-user-collapse-icon">+</span>
            </div>
            <div class="collapsible-content" id="power-user-content">
                <p class="section-desc">Advanced controls for experienced users. Most users can ignore this section.</p>
                ${this.createWeightInitPanel()}
                ${this.createOptimizerPanel()}
                ${this.createWeightDecayGroupsPanel()}
                ${this.createSchedulerAdvancedPanel()}
                ${this.createLossScalingPanel()}
                ${this.createLoRAPlusPanel()}
                ${this.createDataPipelinePanel()}
                ${this.createContextArchitecturePanel()}
                ${this.createLRFinderPanel()}
                ${this.createQATPanel()}
            </div>
        `;

        // Insert after Advanced Settings section
        const advancedSection = document.getElementById('advanced-settings')?.closest('.card');
        if (advancedSection) {
            advancedSection.parentNode.insertBefore(powerUserSection, advancedSection.nextSibling);
        } else {
            // Fallback: insert before LoRA section
            const loraSection = document.getElementById('lora-section');
            if (loraSection) {
                loraSection.parentNode.insertBefore(powerUserSection, loraSection);
            }
        }
    }

    setupEventHandlers() {
        // Weight initialization
        document.getElementById('init-method')?.addEventListener('change', (e) => {
            this.handleInitMethodChange(e.target.value);
        });

        // Per-layer weight decay toggle
        document.getElementById('enable-per-layer-decay')?.addEventListener('change', (e) => {
            const decayGroups = document.querySelector('.decay-groups');
            if (decayGroups) decayGroups.style.display = e.target.checked ? 'block' : 'none';
        });

        // Loss scaling strategy change
        document.getElementById('loss-scaling-strategy')?.addEventListener('change', (e) => {
            const dynamicParams = document.querySelector('.dynamic-params');
            if (dynamicParams) dynamicParams.style.display = e.target.value === 'dynamic' ? 'block' : 'none';
        });

        // LoRA+ toggle
        document.getElementById('enable-lora-plus')?.addEventListener('change', (e) => {
            const params = document.querySelector('.lora-plus-params');
            if (params) params.style.display = e.target.checked ? 'block' : 'none';
        });

        // Manual modules toggle
        document.getElementById('manual-modules-toggle')?.addEventListener('change', (e) => {
            const input = document.querySelector('.manual-modules-input');
            if (input) input.style.display = e.target.checked ? 'block' : 'none';
        });

        // LR Finder toggle
        document.getElementById('enable-lr-finder')?.addEventListener('change', (e) => {
            const params = document.querySelector('.lr-finder-params');
            if (params) params.style.display = e.target.checked ? 'block' : 'none';
        });

        // Validate modules button
        document.getElementById('validate-modules')?.addEventListener('click', () => {
            this.validateLoRAModules();
        });

        // RoPE scaling type change
        document.getElementById('rope-scaling-type')?.addEventListener('change', (e) => {
            const ropeParams = document.querySelector('.rope-params');
            if (ropeParams) ropeParams.style.display = e.target.value !== 'none' ? 'block' : 'none';
        });

        // QAT toggle and event handlers
        document.getElementById('enable-efficient-qat')?.addEventListener('change', (e) => {
            const params = document.querySelector('.qat-params');
            if (params) params.style.display = e.target.checked ? 'block' : 'none';
            if (e.target.checked) {
                this.updateQATMemoryEstimate();
            }
        });

        // QAT bit selection changes
        document.querySelectorAll('input[name="qat-bits"]').forEach(radio => {
            radio.addEventListener('change', () => this.updateQATMemoryEstimate());
        });

        // Scheduler preview
        this.setupSchedulerPreview();

        // Watch for optimizer type changes in main config
        this.watchOptimizerType();
    }

    watchOptimizerType() {
        // Watch for changes to the main optimizer select
        const observer = new MutationObserver(() => {
            const optimizerSelect = document.getElementById('optimizer');
            if (optimizerSelect && !optimizerSelect.dataset.powerUserListenerAttached) {
                optimizerSelect.dataset.powerUserListenerAttached = 'true';
                optimizerSelect.addEventListener('change', (e) => {
                    this.handleOptimizerTypeChange(e.target.value);
                });
                // Initial check
                this.handleOptimizerTypeChange(optimizerSelect.value);
            }
        });

        observer.observe(document.body, { childList: true, subtree: true });
    }

    handleInitMethodChange(method) {
        const paramsDiv = document.querySelector('.init-params');
        const sparseOnly = document.querySelector('.sparse-only');

        if (method === 'default') {
            if (paramsDiv) paramsDiv.style.display = 'none';
        } else {
            if (paramsDiv) paramsDiv.style.display = 'block';
            if (sparseOnly) sparseOnly.style.display = method === 'sparse' ? 'block' : 'none';
        }
    }

    handleOptimizerTypeChange(type) {
        document.querySelectorAll('.optimizer-params').forEach(div => {
            div.style.display = 'none';
        });

        if (type?.includes('adam')) {
            const adamParams = document.querySelector('.adam-params');
            if (adamParams) adamParams.style.display = 'block';
        } else if (type === 'sgd') {
            const sgdParams = document.querySelector('.sgd-params');
            if (sgdParams) sgdParams.style.display = 'block';
        } else if (type?.includes('adafactor')) {
            const adafactorParams = document.querySelector('.adafactor-params');
            if (adafactorParams) adafactorParams.style.display = 'block';
        }
    }

    setupSchedulerPreview() {
        // Watch for scheduler changes in main config
        const observer = new MutationObserver(() => {
            const schedulerSelect = document.getElementById('lr_scheduler');
            if (schedulerSelect && !schedulerSelect.dataset.powerUserListenerAttached) {
                schedulerSelect.dataset.powerUserListenerAttached = 'true';
                schedulerSelect.addEventListener('change', (e) => {
                    this.handleSchedulerTypeChange(e.target.value);
                    this.previewScheduler();
                });
            }
        });

        observer.observe(document.body, { childList: true, subtree: true });

        // Update preview when params change
        ['warmup-ratio', 'num-cycles', 'poly-power', 'lr-end'].forEach(id => {
            document.getElementById(id)?.addEventListener('change', () => this.previewScheduler());
        });
    }

    handleSchedulerTypeChange(type) {
        document.querySelectorAll('.scheduler-specific').forEach(div => {
            div.style.display = 'none';
        });

        if (type === 'cosine_with_restarts') {
            const cosineParams = document.querySelector('.cosine-restarts-params');
            if (cosineParams) cosineParams.style.display = 'block';
        } else if (type === 'polynomial') {
            const polyParams = document.querySelector('.polynomial-params');
            if (polyParams) polyParams.style.display = 'block';
        }
    }

    async validateLoRAModules() {
        const modules = document.getElementById('manual-lora-modules')?.value || '';
        const resultDiv = document.getElementById('module-validation-result');
        if (!resultDiv) return;

        const moduleList = modules.split(/[,\n]/).map(m => m.trim()).filter(m => m);

        if (moduleList.length === 0) {
            resultDiv.innerHTML = '<p style="color: var(--warning);">No modules specified</p>';
            return;
        }

        // For now, just show what was parsed (actual validation would need model info)
        resultDiv.innerHTML = `
            <p style="color: var(--success);">Parsed ${moduleList.length} modules:</p>
            <code style="font-size: 0.85rem;">${moduleList.join(', ')}</code>
        `;
    }

    previewScheduler() {
        const canvas = document.getElementById('scheduler-preview-chart');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const schedulerSelect = document.getElementById('lr_scheduler');
        const type = schedulerSelect?.value || 'cosine';
        const steps = 1000;

        const warmupType = document.querySelector('input[name="warmup-type"]:checked')?.value;
        let warmupSteps = 100;
        if (warmupType === 'ratio') {
            const ratio = parseFloat(document.getElementById('warmup-ratio')?.value) || 0.1;
            warmupSteps = Math.floor(steps * ratio);
        }

        const data = this.generateSchedulerData(type, steps, warmupSteps);

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        this.drawSchedulerChart(ctx, data, canvas.width, canvas.height);
    }

    generateSchedulerData(type, totalSteps, warmupSteps) {
        const data = [];
        const baseLR = 1.0;

        for (let step = 0; step <= totalSteps; step++) {
            let lr = baseLR;

            if (step < warmupSteps) {
                lr = baseLR * (step / warmupSteps);
            } else {
                const progress = (step - warmupSteps) / (totalSteps - warmupSteps);

                switch (type) {
                    case 'linear':
                        lr = baseLR * (1 - progress);
                        break;
                    case 'cosine':
                        lr = baseLR * 0.5 * (1 + Math.cos(Math.PI * progress));
                        break;
                    case 'cosine_with_restarts':
                        const cycles = parseInt(document.getElementById('num-cycles')?.value) || 1;
                        const cycleProgress = (progress * cycles) % 1;
                        lr = baseLR * 0.5 * (1 + Math.cos(Math.PI * cycleProgress));
                        break;
                    case 'polynomial':
                        const power = parseFloat(document.getElementById('poly-power')?.value) || 1.0;
                        lr = baseLR * Math.pow(1 - progress, power);
                        break;
                    case 'constant':
                    case 'constant_with_warmup':
                        lr = baseLR;
                        break;
                }
            }

            data.push({ step, lr });
        }

        return data;
    }

    drawSchedulerChart(ctx, data, width, height) {
        const padding = 25;
        const chartWidth = width - 2 * padding;
        const chartHeight = height - 2 * padding;

        // Background
        ctx.fillStyle = 'var(--bg-tertiary)';
        ctx.fillRect(0, 0, width, height);

        // Axes
        ctx.strokeStyle = 'var(--text-muted)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(padding, padding);
        ctx.lineTo(padding, height - padding);
        ctx.lineTo(width - padding, height - padding);
        ctx.stroke();

        // Data line
        ctx.strokeStyle = 'var(--primary)';
        ctx.lineWidth = 2;
        ctx.beginPath();

        data.forEach((point, i) => {
            const x = padding + (i / data.length) * chartWidth;
            const y = height - padding - (point.lr * chartHeight);

            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });

        ctx.stroke();

        // Labels
        ctx.fillStyle = 'var(--text-muted)';
        ctx.font = '10px sans-serif';
        ctx.fillText('LR', 5, padding);
        ctx.fillText('Steps', width - padding - 25, height - 5);
    }

    toggleSection() {
        const content = document.getElementById('power-user-content');
        const icon = document.getElementById('power-user-collapse-icon');
        const header = content?.previousElementSibling;

        if (content) {
            const isExpanded = content.classList.toggle('active');
            if (icon) icon.textContent = isExpanded ? '-' : '+';
            if (header) header.setAttribute('aria-expanded', isExpanded.toString());

            if (isExpanded) {
                this.previewScheduler();
            }
        }
    }

    // =========================================================================
    // CONFIGURATION
    // =========================================================================

    getConfiguration() {
        return {
            weightInit: {
                method: document.getElementById('init-method')?.value || 'default',
                initRange: parseFloat(document.getElementById('init-range')?.value) || 0.02,
                modules: (document.getElementById('init-modules')?.value || '').split(/[,\n]/).filter(m => m.trim()),
                sparsity: parseFloat(document.getElementById('init-sparsity')?.value) || 0.1
            },
            optimizer: {
                adamBeta1: parseFloat(document.getElementById('adam-beta1')?.value) || 0.9,
                adamBeta2: parseFloat(document.getElementById('adam-beta2')?.value) || 0.999,
                adamEpsilon: parseFloat(document.getElementById('adam-epsilon')?.value) || 1e-8,
                sgdMomentum: parseFloat(document.getElementById('sgd-momentum')?.value) || 0.9,
                sgdDampening: parseFloat(document.getElementById('sgd-dampening')?.value) || 0,
                sgdNesterov: document.getElementById('sgd-nesterov')?.checked || false
            },
            weightDecay: {
                perLayerEnabled: document.getElementById('enable-per-layer-decay')?.checked || false,
                attention: parseFloat(document.getElementById('attention-weight-decay')?.value) || 0.01,
                mlp: parseFloat(document.getElementById('mlp-weight-decay')?.value) || 0.01,
                embedding: parseFloat(document.getElementById('embedding-weight-decay')?.value) || 0.0,
                noDecayPatterns: (document.getElementById('no-decay-patterns')?.value || '').split('\n').filter(p => p.trim())
            },
            scheduler: {
                warmupMode: document.querySelector('input[name="warmup-type"]:checked')?.value || 'steps',
                warmupRatio: parseFloat(document.getElementById('warmup-ratio')?.value) || 0.1,
                warmupType: document.getElementById('warmup-type-select')?.value || 'linear',
                numCycles: parseFloat(document.getElementById('num-cycles')?.value) || 1,
                polyPower: parseFloat(document.getElementById('poly-power')?.value) || 1.0,
                lrEnd: parseFloat(document.getElementById('lr-end')?.value) || 1e-7
            },
            lossScaling: {
                strategy: document.getElementById('loss-scaling-strategy')?.value || 'dynamic',
                initScale: parseInt(document.getElementById('init-scale')?.value) || 32768,
                growthFactor: parseFloat(document.getElementById('growth-factor')?.value) || 2.0,
                backoffFactor: parseFloat(document.getElementById('backoff-factor')?.value) || 0.5,
                growthInterval: parseInt(document.getElementById('growth-interval')?.value) || 2000
            },
            loraPlus: {
                enabled: document.getElementById('enable-lora-plus')?.checked || false,
                lrRatio: parseFloat(document.getElementById('lora-lr-ratio')?.value) || 16
            },
            rsLora: {
                enabled: document.getElementById('enable-rs-lora')?.checked || false
            },
            qloraComputeDtype: document.getElementById('qlora-compute-dtype')?.value || 'float16',
            manualModules: {
                enabled: document.getElementById('manual-modules-toggle')?.checked || false,
                modules: (document.getElementById('manual-lora-modules')?.value || '').split(/[,\n]/).filter(m => m.trim())
            },
            lrFinder: {
                enabled: document.getElementById('enable-lr-finder')?.checked || false,
                iterations: parseInt(document.getElementById('lr-finder-iterations')?.value) || 100,
                startLr: parseFloat(document.getElementById('lr-finder-start')?.value) || 1e-7,
                endLr: parseFloat(document.getElementById('lr-finder-end')?.value) || 10
            },
            dataPipeline: {
                addBosToken: document.getElementById('add-bos-token')?.checked ?? true,
                addEosToken: document.getElementById('add-eos-token')?.checked ?? true,
                completionOnlyLoss: document.getElementById('completion-only-loss')?.checked || false,
                dynamicPadding: document.getElementById('dynamic-padding')?.checked ?? true,
                generateSamplesSteps: parseInt(document.getElementById('generate-samples-steps')?.value) || 0,
                samplePrompts: (document.getElementById('sample-prompts')?.value || '').split('\n').filter(p => p.trim())
            },
            contextArch: {
                ropeScalingType: document.getElementById('rope-scaling-type')?.value || 'none',
                ropeScalingFactor: parseFloat(document.getElementById('rope-scaling-factor')?.value) || 2.0,
                freezeEmbeddings: document.getElementById('freeze-embeddings')?.checked || false,
                freezeLayers: parseInt(document.getElementById('freeze-layers')?.value) || 0,
                torchCompile: document.getElementById('enable-torch-compile')?.checked || false
            },
            qat: {
                enabled: document.getElementById('enable-efficient-qat')?.checked || false,
                bits: parseInt(document.querySelector('input[name="qat-bits"]:checked')?.value || '4'),
                groupSize: parseInt(document.getElementById('qat-group-size')?.value) || 128,
                calibrationSamples: parseInt(document.getElementById('qat-calibration-samples')?.value) || 128,
                warmupSteps: parseInt(document.getElementById('qat-warmup-steps')?.value) || 100,
                symmetric: document.getElementById('qat-symmetric')?.checked ?? true,
                quantizeEmbeddings: document.getElementById('qat-quantize-embeddings')?.checked || false
            }
        };
    }

    async updateQATMemoryEstimate() {
        const estimateDiv = document.getElementById('qat-memory-estimate');
        if (!estimateDiv) return;

        const targetBits = parseInt(document.querySelector('input[name="qat-bits"]:checked')?.value || '4');

        // Get model size from main config (default to 7B if not set)
        const modelSelect = document.getElementById('model_name');
        let modelSizeB = 7; // Default

        if (modelSelect) {
            const modelName = modelSelect.value.toLowerCase();
            if (modelName.includes('0.5b')) modelSizeB = 0.5;
            else if (modelName.includes('1b') || modelName.includes('1.1b')) modelSizeB = 1;
            else if (modelName.includes('1.5b') || modelName.includes('1.6b')) modelSizeB = 1.5;
            else if (modelName.includes('2.7b') || modelName.includes('2b')) modelSizeB = 2.7;
            else if (modelName.includes('7b')) modelSizeB = 7;
            else if (modelName.includes('13b')) modelSizeB = 13;
            else if (modelName.includes('32b')) modelSizeB = 32;
            else if (modelName.includes('70b')) modelSizeB = 70;
        }

        try {
            const response = await fetch('/api/qat/estimate-memory', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model_size_billions: modelSizeB,
                    target_bits: targetBits
                })
            });

            if (response.ok) {
                const data = await response.json();
                estimateDiv.innerHTML = `
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 0.9em;">
                        <div><strong>Training Peak:</strong> ${data.training_peak_gb || data.block_ap_peak_gb} GB</div>
                        <div><strong>Final Model:</strong> ${data.final_quantized_gb} GB</div>
                    </div>
                    <div style="margin-top: 8px; color: var(--success);">
                        ${targetBits}-bit saves ~${Math.round((1 - targetBits/16) * 100)}% vs FP16!
                    </div>
                `;
            } else {
                // Fallback to local calculation
                const fp16Gb = modelSizeB * 2;
                const finalGb = modelSizeB * targetBits / 8;

                estimateDiv.innerHTML = `
                    <div style="font-size: 0.9em;">
                        <div><strong>Estimated Final ${targetBits}-bit Model:</strong> ~${finalGb.toFixed(1)} GB</div>
                        <div style="color: var(--success);">Saves ~${Math.round((1 - targetBits/16) * 100)}% vs FP16!</div>
                    </div>
                `;
            }
        } catch (e) {
            console.error('QAT memory estimate failed:', e);
            estimateDiv.innerHTML = '<span style="color: var(--warning)">Could not estimate memory</span>';
        }
    }

    saveConfiguration() {
        const config = this.getConfiguration();
        localStorage.setItem('powerUserConfig', JSON.stringify(config));
    }

    loadConfiguration() {
        const saved = localStorage.getItem('powerUserConfig');
        if (saved) {
            try {
                const config = JSON.parse(saved);
                this.applyConfiguration(config);
            } catch (e) {
                console.error('Failed to load power user config:', e);
            }
        }
    }

    applyConfiguration(config) {
        // Weight init
        if (config.weightInit) {
            const initMethod = document.getElementById('init-method');
            if (initMethod) initMethod.value = config.weightInit.method;
            const initRange = document.getElementById('init-range');
            if (initRange) initRange.value = config.weightInit.initRange;
        }

        // LoRA+
        if (config.loraPlus) {
            const loraPlus = document.getElementById('enable-lora-plus');
            if (loraPlus) loraPlus.checked = config.loraPlus.enabled;
            const ratio = document.getElementById('lora-lr-ratio');
            if (ratio) ratio.value = config.loraPlus.lrRatio;
        }

        // RS-LoRA
        if (config.rsLora) {
            const rsLora = document.getElementById('enable-rs-lora');
            if (rsLora) rsLora.checked = config.rsLora.enabled;
        }
    }
}

// Initialize on page load
const powerUserFeatures = new PowerUserFeatures();
document.addEventListener('DOMContentLoaded', () => {
    // Delay initialization to let main app load first
    setTimeout(() => powerUserFeatures.init(), 500);
});

// Export for use in main app
window.powerUserFeatures = powerUserFeatures;
