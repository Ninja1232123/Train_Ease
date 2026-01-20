/**
 * Autonomous Mind Frontend Module
 *
 * Interface for the continuous thinking entity.
 * Watch it think. Inject stimuli. Query its state.
 *
 * @version 1.0.0
 */

// =============================================================================
// State
// =============================================================================

const mindState = {
    running: false,
    paused: false,
    pollInterval: null,
    thoughtCount: 0
};

// =============================================================================
// Toggle Functions
// =============================================================================

function toggleMind() {
    const content = document.getElementById('mind-content');
    const icon = document.getElementById('mind-collapse-icon');
    const header = content?.previousElementSibling;

    if (content) {
        const isExpanded = content.classList.toggle('active');
        if (icon) icon.textContent = isExpanded ? '-' : '+';
        if (header) header.setAttribute('aria-expanded', isExpanded.toString());

        if (isExpanded) {
            populateMindModels();
        }
    }
}

function toggleMindMemories() {
    const content = document.getElementById('mind-memories-content');
    const icon = document.getElementById('mind-memories-icon');

    if (content) {
        const isExpanded = content.classList.toggle('active');
        if (icon) icon.textContent = isExpanded ? '-' : '+';
    }
}

function toggleNeuralState() {
    const content = document.getElementById('neural-state-content');
    const icon = document.getElementById('neural-state-icon');

    if (content) {
        const isExpanded = content.classList.toggle('active');
        if (icon) icon.textContent = isExpanded ? '-' : '+';
    }
}

// =============================================================================
// Mode Selection
// =============================================================================

function updateMindModeDescription() {
    const modeSelect = document.getElementById('mind-mode-select');
    const description = document.getElementById('mind-mode-description');
    const intervalGroup = document.getElementById('mind-interval-group');

    if (!modeSelect) return;

    const mode = modeSelect.value;

    if (mode === 'threshold') {
        if (description) {
            description.innerHTML = `
                <strong>Threshold mode:</strong> Thoughts surface when they crystallize.<br>
                The model continuously processes, and only speaks when something<br>
                crosses the internal threshold. More authentic, less artificial.
            `;
        }
        if (intervalGroup) {
            intervalGroup.style.display = 'none';
        }
    } else {
        if (description) {
            description.textContent = 'Timer mode: generates thoughts on a schedule. Simple but artificial.';
        }
        if (intervalGroup) {
            intervalGroup.style.display = 'block';
        }
    }
}

// =============================================================================
// Model Population
// =============================================================================

async function populateMindModels() {
    const select = document.getElementById('mind-model-select');
    if (!select) return;

    try {
        const response = await fetch('/api/inference/models');
        if (response.ok) {
            const data = await response.json();
            const models = data.models || [];

            select.innerHTML = '<option value="">-- Select trained model --</option>';

            models.forEach(model => {
                const option = document.createElement('option');
                option.value = model.path || model.name;
                option.textContent = model.name || model.path;
                select.appendChild(option);
            });
        }
    } catch (error) {
        console.error('Failed to load models:', error);
    }
}

// =============================================================================
// Mind Control
// =============================================================================

async function awakenMind() {
    const modelSelect = document.getElementById('mind-model-select');
    const intervalSlider = document.getElementById('mind-interval');
    const modeSelect = document.getElementById('mind-mode-select');

    const modelPath = modelSelect?.value;
    const thinkInterval = parseFloat(intervalSlider?.value || 3);
    const mode = modeSelect?.value || 'timer';

    if (!modelPath) {
        if (typeof toast !== 'undefined') toast.warning('Select a model first');
        return;
    }

    const awakenBtn = document.getElementById('awaken-btn');
    const pauseBtn = document.getElementById('pause-btn');
    const stopBtn = document.getElementById('stop-mind-btn');
    const stateDisplay = document.getElementById('mind-state-display');

    if (awakenBtn) {
        awakenBtn.disabled = true;
        awakenBtn.textContent = 'Awakening...';
    }

    try {
        const response = await fetch('/api/mind/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model_path: modelPath,
                think_interval: thinkInterval,
                mode: mode
            })
        });

        const data = await response.json();

        if (response.ok) {
            mindState.running = true;
            mindState.paused = false;

            if (awakenBtn) {
                awakenBtn.textContent = 'Awake';
                awakenBtn.disabled = true;
            }
            if (pauseBtn) pauseBtn.disabled = false;
            if (stopBtn) stopBtn.disabled = false;
            if (stateDisplay) stateDisplay.style.display = 'block';

            // Start polling for state updates
            startMindPolling();

            // Clear thought stream
            const thoughtStream = document.getElementById('thought-stream');
            if (thoughtStream) {
                thoughtStream.innerHTML = '<div style="color: var(--groove-teal); font-style: italic;">Mind awakened... waiting for first thought...</div>';
            }

            updateMindStatus('Thinking', 'var(--groove-teal)');

            if (typeof toast !== 'undefined') toast.success('Mind awakened');
        } else {
            if (awakenBtn) {
                awakenBtn.disabled = false;
                awakenBtn.textContent = 'Awaken';
            }
            if (typeof toast !== 'undefined') toast.error(data.error || 'Failed to awaken mind');
        }
    } catch (error) {
        console.error('Error awakening mind:', error);
        if (awakenBtn) {
            awakenBtn.disabled = false;
            awakenBtn.textContent = 'Awaken';
        }
        if (typeof toast !== 'undefined') toast.error('Network error');
    }
}

async function pauseMind() {
    const pauseBtn = document.getElementById('pause-btn');

    try {
        if (mindState.paused) {
            // Resume
            await fetch('/api/mind/resume', { method: 'POST' });
            mindState.paused = false;
            if (pauseBtn) pauseBtn.textContent = 'Pause';
            updateMindStatus('Thinking', 'var(--groove-teal)');
        } else {
            // Pause
            await fetch('/api/mind/pause', { method: 'POST' });
            mindState.paused = true;
            if (pauseBtn) pauseBtn.textContent = 'Resume';
            updateMindStatus('Paused', 'var(--warning)');
        }
    } catch (error) {
        console.error('Error pausing mind:', error);
    }
}

async function stopMind() {
    const awakenBtn = document.getElementById('awaken-btn');
    const pauseBtn = document.getElementById('pause-btn');
    const stopBtn = document.getElementById('stop-mind-btn');

    try {
        await fetch('/api/mind/stop', { method: 'POST' });

        mindState.running = false;
        mindState.paused = false;

        // Stop polling
        if (mindState.pollInterval) {
            clearInterval(mindState.pollInterval);
            mindState.pollInterval = null;
        }

        if (awakenBtn) {
            awakenBtn.disabled = false;
            awakenBtn.textContent = 'Awaken';
        }
        if (pauseBtn) {
            pauseBtn.disabled = true;
            pauseBtn.textContent = 'Pause';
        }
        if (stopBtn) stopBtn.disabled = true;

        updateMindStatus('Dormant', 'var(--text-muted)');

        if (typeof toast !== 'undefined') toast.success('Mind at rest');
    } catch (error) {
        console.error('Error stopping mind:', error);
    }
}

// =============================================================================
// Polling & Updates
// =============================================================================

function startMindPolling() {
    if (mindState.pollInterval) {
        clearInterval(mindState.pollInterval);
    }

    // Poll every second
    mindState.pollInterval = setInterval(async () => {
        if (!mindState.running) return;

        try {
            // Get mind state
            const stateResponse = await fetch('/api/mind/state');
            const state = await stateResponse.json();

            if (state.error) return;

            updateMindUI(state);

            // Get neural state
            const neuralResponse = await fetch('/api/mind/neural');
            const neural = await neuralResponse.json();

            if (!neural.error) {
                updateNeuralUI(neural);
            }
        } catch (error) {
            console.error('Error polling mind state:', error);
        }
    }, 1000);
}

function updateMindUI(state) {
    // Update status
    if (state.running && !state.paused) {
        updateMindStatus('Thinking', 'var(--groove-teal)');
    } else if (state.paused) {
        updateMindStatus('Paused', 'var(--warning)');
    }

    // Update focus
    const focusEl = document.getElementById('mind-focus');
    if (focusEl && state.focus) {
        const focusNames = {
            'idle': 'Wandering',
            'self_reflection': 'Self-Reflecting',
            'memory_recall': 'Remembering',
            'external_input': 'Processing Input',
            'curiosity': 'Curious',
            'consolidation': 'Consolidating'
        };
        focusEl.textContent = focusNames[state.focus] || state.focus;
    }

    // Update mood
    const moodEl = document.getElementById('mind-mood');
    if (moodEl && typeof state.emotional_valence === 'number') {
        const valence = state.emotional_valence;
        if (valence > 0.3) moodEl.textContent = '😊';
        else if (valence > 0.1) moodEl.textContent = '🙂';
        else if (valence < -0.3) moodEl.textContent = '😔';
        else if (valence < -0.1) moodEl.textContent = '😕';
        else moodEl.textContent = '😐';
    }

    // Update arousal
    const arousalEl = document.getElementById('mind-arousal');
    if (arousalEl && typeof state.arousal === 'number') {
        arousalEl.textContent = state.arousal.toFixed(2);
        // Color based on arousal level
        if (state.arousal > 0.7) {
            arousalEl.style.color = 'var(--funk-orange)';
        } else if (state.arousal < 0.3) {
            arousalEl.style.color = 'var(--groove-sage)';
        } else {
            arousalEl.style.color = 'var(--text-primary)';
        }
    }

    // Update thought count
    const countEl = document.getElementById('mind-thought-count');
    if (countEl && state.stats) {
        countEl.textContent = state.stats.thoughts_generated || 0;
    }

    // Update thought stream
    if (state.recent_thoughts && state.recent_thoughts.length > 0) {
        updateThoughtStream(state.recent_thoughts);
    }

    // Update current thought
    if (state.current_thought) {
        updateCurrentThought(state.current_thought);
    }

    // Update memories
    if (state.long_term_memories) {
        updateMemories(state.long_term_memories);
    }
}

function updateMindStatus(status, color) {
    const statusEl = document.getElementById('mind-status');
    if (statusEl) {
        statusEl.textContent = status;
        statusEl.style.color = color;
    }
}

function updateThoughtStream(thoughts) {
    const stream = document.getElementById('thought-stream');
    if (!stream) return;

    // Check if we have new thoughts
    const newCount = thoughts.length;
    if (newCount === mindState.thoughtCount) return;

    mindState.thoughtCount = newCount;

    // Build HTML for thoughts
    let html = '';
    thoughts.forEach((thought, idx) => {
        const isLatest = idx === thoughts.length - 1;
        const focusColor = getFocusColor(thought.focus);
        const time = thought.timestamp ? new Date(thought.timestamp * 1000).toLocaleTimeString() : '';

        html += `
            <div style="margin-bottom: 12px; padding: 10px; background: ${isLatest ? 'var(--bg-tertiary)' : 'transparent'}; border-radius: 6px; ${isLatest ? 'border-left: 3px solid ' + focusColor : ''}">
                <div style="font-size: 0.7rem; color: var(--text-muted); margin-bottom: 4px;">
                    <span style="color: ${focusColor};">${thought.focus || 'idle'}</span>
                    ${time ? ' • ' + time : ''}
                    ${thought.emotional_valence ? ' • ' + getMoodEmoji(thought.emotional_valence) : ''}
                </div>
                <div style="color: ${isLatest ? 'var(--text-primary)' : 'var(--text-secondary)'};">${thought.content}</div>
            </div>
        `;
    });

    stream.innerHTML = html;

    // Scroll to bottom
    stream.scrollTop = stream.scrollHeight;
}

function updateCurrentThought(thought) {
    const box = document.getElementById('current-thought-box');
    const textEl = document.getElementById('current-thought-text');
    const altsEl = document.getElementById('thought-alternatives');

    if (!box || !textEl) return;

    box.style.display = 'block';
    textEl.textContent = thought.content;

    // Show alternatives if available
    if (altsEl && thought.top_alternatives && thought.top_alternatives.length > 0) {
        const alts = thought.top_alternatives.slice(0, 5).map(([token, prob]) =>
            `"${token}" (${(prob * 100).toFixed(1)}%)`
        ).join(', ');
        altsEl.textContent = `Considered: ${alts}`;
    }
}

function updateMemories(memories) {
    const countEl = document.getElementById('memory-count');
    const listEl = document.getElementById('memories-list');

    if (countEl) countEl.textContent = memories.length;

    if (listEl && memories.length > 0) {
        let html = '';
        memories.forEach(mem => {
            const time = mem.timestamp ? new Date(mem.timestamp * 1000).toLocaleTimeString() : '';
            html += `
                <div style="margin-bottom: 10px; padding: 8px; background: var(--bg-tertiary); border-radius: 4px;">
                    <div style="font-size: 0.75rem; color: var(--text-muted);">${time}</div>
                    <div>${mem.content}</div>
                </div>
            `;
        });
        listEl.innerHTML = html;
    }
}

function updateNeuralUI(neural) {
    const entropyEl = document.getElementById('neural-entropy');
    const normEl = document.getElementById('neural-norm');
    const cacheEl = document.getElementById('neural-cache');

    if (entropyEl) {
        entropyEl.textContent = neural.attention_entropy?.toFixed(4) || '-';
    }
    if (normEl) {
        normEl.textContent = neural.hidden_state_norm?.toFixed(4) || '-';
    }
    if (cacheEl) {
        cacheEl.textContent = neural.kv_cache_size || '-';
    }
}

// =============================================================================
// Interaction
// =============================================================================

async function injectToMind() {
    const input = document.getElementById('mind-inject-input');
    const text = input?.value?.trim();

    if (!text) {
        if (typeof toast !== 'undefined') toast.warning('Enter something to inject');
        return;
    }

    try {
        const response = await fetch('/api/mind/inject', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });

        if (response.ok) {
            if (input) input.value = '';
            if (typeof toast !== 'undefined') toast.success('Stimulus injected');
        }
    } catch (error) {
        console.error('Error injecting:', error);
    }
}

async function queryMind() {
    const input = document.getElementById('mind-query-input');
    const responseEl = document.getElementById('mind-query-response');
    const question = input?.value?.trim();

    if (!question) {
        if (typeof toast !== 'undefined') toast.warning('Enter a question');
        return;
    }

    if (responseEl) {
        responseEl.style.display = 'block';
        responseEl.innerHTML = '<span style="color: var(--text-muted);">Thinking...</span>';
    }

    try {
        const response = await fetch('/api/mind/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question })
        });

        const data = await response.json();

        if (response.ok && responseEl) {
            responseEl.innerHTML = `<strong>Response:</strong><br>${data.response || 'No response'}`;
        }
    } catch (error) {
        console.error('Error querying:', error);
        if (responseEl) {
            responseEl.innerHTML = '<span style="color: var(--error);">Error querying mind</span>';
        }
    }
}

// =============================================================================
// Helpers
// =============================================================================

function getFocusColor(focus) {
    const colors = {
        'idle': 'var(--text-muted)',
        'self_reflection': 'var(--groove-plum)',
        'memory_recall': 'var(--info)',
        'external_input': 'var(--funk-orange)',
        'curiosity': 'var(--funk-gold)',
        'consolidation': 'var(--groove-sage)'
    };
    return colors[focus] || 'var(--text-muted)';
}

function getMoodEmoji(valence) {
    if (valence > 0.3) return '😊';
    if (valence > 0.1) return '🙂';
    if (valence < -0.3) return '😔';
    if (valence < -0.1) return '😕';
    return '😐';
}

// =============================================================================
// Initialization
// =============================================================================

document.addEventListener('DOMContentLoaded', () => {
    // Check if mind is already running on page load
    fetch('/api/mind/state')
        .then(res => res.json())
        .then(state => {
            if (state.running) {
                mindState.running = true;
                mindState.paused = state.paused || false;

                const awakenBtn = document.getElementById('awaken-btn');
                const pauseBtn = document.getElementById('pause-btn');
                const stopBtn = document.getElementById('stop-mind-btn');
                const stateDisplay = document.getElementById('mind-state-display');

                if (awakenBtn) {
                    awakenBtn.textContent = 'Awake';
                    awakenBtn.disabled = true;
                }
                if (pauseBtn) {
                    pauseBtn.disabled = false;
                    pauseBtn.textContent = mindState.paused ? 'Resume' : 'Pause';
                }
                if (stopBtn) stopBtn.disabled = false;
                if (stateDisplay) stateDisplay.style.display = 'block';

                startMindPolling();
            }
        })
        .catch(() => {
            // Mind not running, that's fine
        });

    console.log('Autonomous Mind module loaded');
});
