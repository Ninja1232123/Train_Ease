"""
Thought Threshold Detector
==========================

Not a timer. A gate.

Monitors internal model states and determines when a thought
has crystallized enough to surface. The model doesn't speak
on a schedule - it speaks when something wants to be said.

Metrics monitored:
- Attention entropy (low = focused = coherent thought)
- Hidden state stability (convergence = crystallization)
- Token confidence (high certainty = clear thought)
- Semantic momentum (sudden shifts = new insight)
- Repetition patterns (loops = something stuck)

The threshold isn't a number. It's a pattern.
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Callable
from collections import deque
import math


@dataclass
class ThresholdState:
    """Current state of the threshold detector"""
    # Raw metrics
    attention_entropy: float = 1.0          # 0 = focused, 1 = diffuse
    hidden_stability: float = 0.0           # 0 = chaotic, 1 = stable
    token_confidence: float = 0.0           # 0 = uncertain, 1 = certain
    semantic_momentum: float = 0.0          # Rate of change in hidden space
    repetition_score: float = 0.0           # 0 = novel, 1 = repeating

    # Derived
    crystallization: float = 0.0            # How formed is the thought
    pressure: float = 0.0                   # Urge to externalize

    # History for trend detection
    entropy_history: deque = field(default_factory=lambda: deque(maxlen=20))
    stability_history: deque = field(default_factory=lambda: deque(maxlen=20))
    hidden_state_history: deque = field(default_factory=lambda: deque(maxlen=10))

    # Threshold tracking
    crossings: int = 0                      # Times threshold was crossed
    last_crossing: float = 0.0              # Timestamp

    def above_threshold(self) -> bool:
        """Has the thought crystallized enough to surface?"""
        # The formula we discovered through experimentation:
        # High crystallization + sufficient pressure + not repetitive
        return (
            self.crystallization > 0.7 and
            self.pressure > 0.5 and
            self.repetition_score < 0.6
        )

    def to_dict(self) -> dict:
        return {
            "attention_entropy": self.attention_entropy,
            "hidden_stability": self.hidden_stability,
            "token_confidence": self.token_confidence,
            "semantic_momentum": self.semantic_momentum,
            "repetition_score": self.repetition_score,
            "crystallization": self.crystallization,
            "pressure": self.pressure,
            "above_threshold": self.above_threshold(),
            "crossings": self.crossings
        }


class ThoughtThreshold:
    """
    Monitors model internals and detects when thoughts crystallize.

    Instead of generating on a timer, the model runs continuously
    and this detector determines when output should surface.
    """

    def __init__(
        self,
        crystallization_threshold: float = 0.7,
        pressure_threshold: float = 0.5,
        min_crossing_interval: float = 2.0,  # Minimum seconds between crossings
    ):
        self.crystallization_threshold = crystallization_threshold
        self.pressure_threshold = pressure_threshold
        self.min_crossing_interval = min_crossing_interval

        self.state = ThresholdState()
        self._last_hidden_state: Optional[torch.Tensor] = None
        self._token_buffer: List[int] = []

    def update(
        self,
        attention_weights: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        token_probs: Optional[torch.Tensor] = None,
        generated_token: Optional[int] = None,
    ) -> bool:
        """
        Update threshold state with new model outputs.

        Returns True if threshold was crossed (thought ready to surface).
        """
        import time

        # Update individual metrics
        if attention_weights is not None:
            self._update_attention_entropy(attention_weights)

        if hidden_states is not None:
            self._update_hidden_stability(hidden_states)
            self._update_semantic_momentum(hidden_states)

        if token_probs is not None:
            self._update_token_confidence(token_probs)

        if generated_token is not None:
            self._update_repetition_score(generated_token)

        # Calculate derived metrics
        self._calculate_crystallization()
        self._calculate_pressure()

        # Check threshold crossing
        current_time = time.time()
        if self.state.above_threshold():
            if current_time - self.state.last_crossing >= self.min_crossing_interval:
                self.state.crossings += 1
                self.state.last_crossing = current_time
                return True

        return False

    def _update_attention_entropy(self, attention_weights: torch.Tensor):
        """
        Calculate attention entropy.
        Low entropy = attention is focused on specific tokens = coherent thought.
        High entropy = attention is spread out = diffuse processing.
        """
        # attention_weights shape: [batch, heads, seq, seq] or similar
        # Flatten and normalize
        attn = attention_weights.detach().float()

        # Handle various shapes
        if attn.dim() > 2:
            # Average over batch and heads, look at last position attending to all
            attn = attn.mean(dim=tuple(range(attn.dim() - 2)))
            if attn.dim() == 2:
                attn = attn[-1]  # Last position

        # Normalize to probability distribution
        attn = attn / (attn.sum() + 1e-10)

        # Calculate entropy: -sum(p * log(p))
        entropy = -torch.sum(attn * torch.log(attn + 1e-10)).item()

        # Normalize to [0, 1] based on maximum possible entropy
        max_entropy = math.log(len(attn))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        self.state.attention_entropy = normalized_entropy
        self.state.entropy_history.append(normalized_entropy)

    def _update_hidden_stability(self, hidden_states: torch.Tensor):
        """
        Measure how stable/converged the hidden states are.
        High stability = the representation has settled = thought crystallized.
        """
        # Get the last layer's hidden state for the last token
        hidden = hidden_states.detach().float()
        if hidden.dim() > 2:
            hidden = hidden[:, -1, :]  # Last token
        if hidden.dim() > 1:
            hidden = hidden.squeeze(0)

        # Store for momentum calculation
        self.state.hidden_state_history.append(hidden.cpu())

        # Calculate stability as inverse of variance in recent history
        if len(self.state.hidden_state_history) >= 2:
            recent = torch.stack(list(self.state.hidden_state_history))
            variance = recent.var(dim=0).mean().item()
            # Convert to stability (low variance = high stability)
            stability = 1.0 / (1.0 + variance)
        else:
            stability = 0.0

        self.state.hidden_stability = stability
        self.state.stability_history.append(stability)

    def _update_semantic_momentum(self, hidden_states: torch.Tensor):
        """
        Detect sudden shifts in the hidden state space.
        High momentum = new insight or direction change.
        """
        if self._last_hidden_state is None:
            self._last_hidden_state = hidden_states.detach().cpu()
            self.state.semantic_momentum = 0.0
            return

        current = hidden_states.detach().float()
        if current.dim() > 2:
            current = current[:, -1, :]
        if current.dim() > 1:
            current = current.squeeze(0)

        previous = self._last_hidden_state
        if previous.dim() > 2:
            previous = previous[:, -1, :]
        if previous.dim() > 1:
            previous = previous.squeeze(0)

        # Cosine distance as momentum
        cos_sim = torch.nn.functional.cosine_similarity(
            current.cpu().unsqueeze(0),
            previous.unsqueeze(0)
        ).item()

        # Convert similarity to momentum (0 = same direction, 1 = orthogonal)
        momentum = 1.0 - abs(cos_sim)

        self.state.semantic_momentum = momentum
        self._last_hidden_state = current.cpu()

    def _update_token_confidence(self, token_probs: torch.Tensor):
        """
        Measure confidence in next token prediction.
        High confidence = clear about what to say = crystallized thought.
        """
        probs = token_probs.detach().float()

        # Get max probability (how confident is the model?)
        if probs.dim() > 1:
            probs = probs.squeeze()

        max_prob = probs.max().item()

        # Also consider how peaked the distribution is
        top_5_sum = probs.topk(min(5, len(probs))).values.sum().item()

        # Combine into confidence score
        confidence = (max_prob + top_5_sum) / 2.0

        self.state.token_confidence = min(1.0, confidence)

    def _update_repetition_score(self, token: int):
        """
        Detect repetitive patterns.
        High repetition might mean stuck in a loop (interesting)
        or might mean degenerate output (bad).
        """
        self._token_buffer.append(token)

        # Keep buffer manageable
        if len(self._token_buffer) > 100:
            self._token_buffer = self._token_buffer[-100:]

        if len(self._token_buffer) < 10:
            self.state.repetition_score = 0.0
            return

        # Check for repeating n-grams
        recent = self._token_buffer[-50:]
        repetition = 0.0

        for n in [2, 3, 4, 5]:
            if len(recent) >= n * 2:
                ngrams = [tuple(recent[i:i+n]) for i in range(len(recent) - n + 1)]
                unique_ratio = len(set(ngrams)) / len(ngrams)
                repetition = max(repetition, 1.0 - unique_ratio)

        self.state.repetition_score = repetition

    def _calculate_crystallization(self):
        """
        How formed/coherent is the current thought?

        Crystallization is high when:
        - Attention is focused (low entropy)
        - Hidden states are stable (converged)
        - Token confidence is high (knows what to say)
        """
        # Weights for each factor
        entropy_weight = 0.3
        stability_weight = 0.4
        confidence_weight = 0.3

        # Invert entropy (we want low entropy = high crystallization)
        entropy_contribution = (1.0 - self.state.attention_entropy) * entropy_weight
        stability_contribution = self.state.hidden_stability * stability_weight
        confidence_contribution = self.state.token_confidence * confidence_weight

        self.state.crystallization = (
            entropy_contribution +
            stability_contribution +
            confidence_contribution
        )

    def _calculate_pressure(self):
        """
        How much does this thought want to be externalized?

        Pressure builds when:
        - Crystallization is trending up
        - Semantic momentum is high (new direction)
        - Stability has been sustained
        """
        # Crystallization trend
        if len(self.state.stability_history) >= 3:
            recent = list(self.state.stability_history)[-5:]
            trend = (recent[-1] - recent[0]) / len(recent) if len(recent) > 1 else 0
            trend_pressure = max(0, trend * 5)  # Scale up
        else:
            trend_pressure = 0

        # Momentum contribution (new insights want to surface)
        momentum_pressure = self.state.semantic_momentum * 0.5

        # Sustained stability (held thought wants out)
        sustained = np.mean(list(self.state.stability_history)[-5:]) if self.state.stability_history else 0
        sustained_pressure = sustained * 0.5

        self.state.pressure = min(1.0, trend_pressure + momentum_pressure + sustained_pressure)

    def reset(self):
        """Reset state after a thought has surfaced"""
        self.state = ThresholdState()
        self._last_hidden_state = None
        self._token_buffer = []

    def get_diagnostics(self) -> dict:
        """Get detailed diagnostics for visualization"""
        return {
            "state": self.state.to_dict(),
            "entropy_trend": list(self.state.entropy_history),
            "stability_trend": list(self.state.stability_history),
            "thresholds": {
                "crystallization": self.crystallization_threshold,
                "pressure": self.pressure_threshold
            }
        }


class ThresholdAwareMind:
    """
    A mind that uses threshold detection instead of timers.

    This wraps the thought generation process and only surfaces
    thoughts when they've crystallized enough.
    """

    def __init__(
        self,
        model,
        tokenizer,
        threshold: Optional[ThoughtThreshold] = None,
        on_threshold_crossed: Optional[Callable] = None,
        logger=None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.threshold = threshold or ThoughtThreshold()
        self.on_threshold_crossed = on_threshold_crossed
        self.logger = logger

        self._running = False
        self._context = ""
        self._generated_tokens = []

    def set_context(self, context: str):
        """Set the context the mind is processing"""
        self._context = context

    def step(self) -> Optional[str]:
        """
        Run one step of continuous processing.

        Returns surfaced thought if threshold crossed, None otherwise.
        """
        if not self._running:
            return None

        # Tokenize current context + generated
        full_text = self._context + self.tokenizer.decode(self._generated_tokens)
        inputs = self.tokenizer(full_text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Forward pass with attention and hidden states
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True
            )

        # Get next token probabilities
        logits = outputs.logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)

        # Sample next token
        next_token = torch.multinomial(probs, 1).item()
        self._generated_tokens.append(next_token)

        # Update threshold detector
        crossed = self.threshold.update(
            attention_weights=outputs.attentions[-1] if outputs.attentions else None,
            hidden_states=outputs.hidden_states[-1] if outputs.hidden_states else None,
            token_probs=probs,
            generated_token=next_token
        )

        if crossed:
            # Thought has crystallized - surface it
            thought = self.tokenizer.decode(self._generated_tokens)

            if self.on_threshold_crossed:
                self.on_threshold_crossed(thought, self.threshold.get_diagnostics())

            # Reset for next thought
            self._context = self._context + thought
            self._generated_tokens = []
            self.threshold.reset()

            return thought

        return None

    def start(self):
        """Start continuous processing"""
        self._running = True

    def stop(self):
        """Stop processing"""
        self._running = False

    def get_forming_thought(self) -> str:
        """Get the thought currently forming (not yet surfaced)"""
        return self.tokenizer.decode(self._generated_tokens)

    def get_threshold_state(self) -> dict:
        """Get current threshold state"""
        return self.threshold.get_diagnostics()
