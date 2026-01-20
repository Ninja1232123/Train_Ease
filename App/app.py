"""
AI Training For DumDums - Main Application
A beginner-friendly interface for training AI models.

Professional-grade implementation with:
- Thread-safe state management
- Comprehensive input validation
- Structured logging
- WebSocket real-time updates
- Health monitoring endpoints
- Type hints throughout
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import threading
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

from flask import Flask, render_template, request, jsonify, Response
from flask_socketio import SocketIO, emit
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import random

# EfficientQAT integration
from efficient_qat import add_qat_routes, validate_qat_config, QATConfig, EfficientQATTrainer, integrate_qat_in_training

# Universal Data Loader (CSV, JSON, JSONL, Parquet, TXT → tokenized JSONL)
from data_loader import add_data_loader_routes, DataLoader, DataFormat


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class Config:
    """Application configuration with sensible defaults."""

    # Server settings
    HOST: str = field(default_factory=lambda: os.getenv("HOST", "127.0.0.1"))
    PORT: int = field(default_factory=lambda: int(os.getenv("PORT", "5000")))
    DEBUG: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")
    SECRET_KEY: str = field(default_factory=lambda: os.getenv("SECRET_KEY", os.urandom(24).hex()))

    # Training limits
    MAX_TRAINING_DATA_SIZE: int = field(default_factory=lambda: int(os.getenv("MAX_TRAINING_DATA_SIZE", "1000000")))  # 1MB
    MAX_TRAINING_TIMEOUT: int = field(default_factory=lambda: int(os.getenv("MAX_TRAINING_TIMEOUT", "3600")))  # 1 hour
    MAX_EPOCHS: int = field(default_factory=lambda: int(os.getenv("MAX_EPOCHS", "20")))
    MAX_BATCH_SIZE: int = field(default_factory=lambda: int(os.getenv("MAX_BATCH_SIZE", "32")))

    # Logging
    LOG_LEVEL: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    LOG_FORMAT: str = field(default_factory=lambda: os.getenv(
        "LOG_FORMAT",
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    ))


def create_config() -> Config:
    """Factory function to create configuration."""
    return Config()


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(config: Config) -> logging.Logger:
    """Configure structured logging."""
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL.upper()),
        format=config.LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger("ai_trainer")
    logger.setLevel(getattr(logging, config.LOG_LEVEL.upper()))
    return logger


# =============================================================================
# ENUMS & DATA CLASSES
# =============================================================================

class TrainingStatus(str, Enum):
    """Training status enumeration."""
    IDLE = "idle"
    INITIALIZING = "initializing"
    LOADING_MODEL = "loading_model"
    PREPARING_DATA = "preparing_data"
    TRAINING = "training"
    STOPPING = "stopping"
    COMPLETED = "completed"
    ERROR = "error"


class LogLevel(str, Enum):
    """Log level enumeration."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"


@dataclass
class LogEntry:
    """Structured log entry."""
    time: str
    level: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class TrainingState:
    """Thread-safe training state container."""
    is_training: bool = False
    progress: int = 0
    status: TrainingStatus = TrainingStatus.IDLE
    logs: list[LogEntry] = field(default_factory=list)
    current_step: int = 0
    total_steps: int = 0
    started_at: Optional[datetime] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_training": self.is_training,
            "progress": self.progress,
            "status": self.status.value,
            "logs": [log.to_dict() for log in self.logs],
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "started_at": self.started_at.isoformat() if self.started_at else None
        }


# =============================================================================
# TRAINING STATE MANAGER (Thread-Safe)
# =============================================================================

class TrainingStateManager:
    """Thread-safe manager for training state."""

    def __init__(self, socketio: Optional[SocketIO] = None):
        self._state = TrainingState()
        self._lock = threading.RLock()
        self._socketio = socketio
        self._logger = logging.getLogger("ai_trainer.state")

    @property
    def state(self) -> TrainingState:
        with self._lock:
            return self._state

    def is_training(self) -> bool:
        with self._lock:
            return self._state.is_training

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            return self._state.to_dict()

    def get_logs(self) -> list[dict[str, str]]:
        with self._lock:
            return [log.to_dict() for log in self._state.logs]

    def reset(self) -> None:
        """Reset state for new training run."""
        with self._lock:
            self._state = TrainingState(
                is_training=True,
                status=TrainingStatus.INITIALIZING,
                started_at=datetime.now()
            )
            self._emit_update()

    def set_status(self, status: TrainingStatus) -> None:
        with self._lock:
            self._state.status = status
            self._emit_update()

    def update_progress(self, current_step: int, total_steps: int) -> None:
        with self._lock:
            self._state.current_step = current_step
            self._state.total_steps = total_steps
            self._state.progress = int(100 * current_step / max(1, total_steps))
            self._emit_update()

    def add_log(self, message: str, level: LogLevel = LogLevel.INFO) -> None:
        with self._lock:
            timestamp = datetime.now().strftime("%H:%M:%S")
            entry = LogEntry(time=timestamp, level=level.value, message=message)
            self._state.logs.append(entry)
            self._logger.info(f"[{level.value.upper()}] {message}")
            self._emit_log(entry)

    def complete(self, success: bool = True) -> None:
        with self._lock:
            self._state.is_training = False
            self._state.status = TrainingStatus.COMPLETED if success else TrainingStatus.ERROR
            self._state.progress = 100 if success else self._state.progress
            self._emit_update()

    def request_stop(self) -> None:
        with self._lock:
            if self._state.is_training:
                self._state.status = TrainingStatus.STOPPING
                self._emit_update()

    def should_stop(self) -> bool:
        with self._lock:
            return self._state.status == TrainingStatus.STOPPING

    def _emit_update(self) -> None:
        """Emit state update via WebSocket."""
        if self._socketio:
            try:
                self._socketio.emit("training_update", self._state.to_dict())
            except Exception as e:
                self._logger.warning(f"Failed to emit WebSocket update: {e}")

    def _emit_log(self, entry: LogEntry) -> None:
        """Emit log entry via WebSocket."""
        if self._socketio:
            try:
                self._socketio.emit("training_log", entry.to_dict())
            except Exception as e:
                self._logger.warning(f"Failed to emit WebSocket log: {e}")

    def emit_loss_data(self, step: int, loss: float, eval_loss: Optional[float] = None, lr: Optional[float] = None) -> None:
        """Emit loss data point for real-time charting."""
        if self._socketio:
            try:
                data = {
                    "step": step,
                    "loss": loss,
                    "timestamp": datetime.now().isoformat()
                }
                if eval_loss is not None:
                    data["eval_loss"] = eval_loss
                if lr is not None:
                    data["learning_rate"] = lr
                self._socketio.emit("loss_data", data)
            except Exception as e:
                self._logger.warning(f"Failed to emit loss data: {e}")


# =============================================================================
# INPUT VALIDATION
# =============================================================================

class ValidationError(Exception):
    """Custom validation error."""
    pass


class InputValidator:
    """Validates training configuration inputs."""

    ALLOWED_MODELS = {
        "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "distilgpt2",
        "facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3b",
        "EleutherAI/pythia-160m", "EleutherAI/pythia-410m", "EleutherAI/pythia-1b",
        "microsoft/phi-1_5", "microsoft/phi-2",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "Qwen/Qwen2-0.5B", "Qwen/Qwen2-1.5B",
        "stabilityai/stablelm-2-zephyr-1_6b"
    }

    def __init__(self, config: Config):
        self.config = config
        self._logger = logging.getLogger("ai_trainer.validator")

    def validate_config(self, data: dict[str, Any]) -> dict[str, Any]:
        """Validate and sanitize training configuration."""
        validated = {}

        # Model name - allow HuggingFace models OR local paths
        model_name = data.get("model_name", "gpt2")
        if model_name not in self.ALLOWED_MODELS:
            # Check if it's a valid local path
            import os
            if os.path.isdir(model_name) and (
                os.path.exists(os.path.join(model_name, "config.json")) or
                os.path.exists(os.path.join(model_name, "adapter_config.json"))
            ):
                self._logger.info(f"Using custom local model: {model_name}")
            else:
                raise ValidationError(f"Invalid model: {model_name}. Must be a HuggingFace model ID or a valid local model path.")
        validated["model_name"] = model_name

        # Learning rate
        learning_rate = float(data.get("learning_rate", 5e-5))
        if not (1e-7 <= learning_rate <= 1e-2):
            raise ValidationError("Learning rate must be between 1e-7 and 1e-2")
        validated["learning_rate"] = learning_rate

        # Epochs
        epochs = int(data.get("epochs", 3))
        if not (1 <= epochs <= self.config.MAX_EPOCHS):
            raise ValidationError(f"Epochs must be between 1 and {self.config.MAX_EPOCHS}")
        validated["epochs"] = epochs

        # Batch size
        batch_size = int(data.get("batch_size", 4))
        if not (1 <= batch_size <= self.config.MAX_BATCH_SIZE):
            raise ValidationError(f"Batch size must be between 1 and {self.config.MAX_BATCH_SIZE}")
        validated["batch_size"] = batch_size

        # Max length
        max_length = int(data.get("max_length", 128))
        if not (16 <= max_length <= 2048):
            raise ValidationError("Max length must be between 16 and 2048")
        validated["max_length"] = max_length

        # Warmup steps
        warmup_steps = int(data.get("warmup_steps", 100))
        if not (0 <= warmup_steps <= 10000):
            raise ValidationError("Warmup steps must be between 0 and 10000")
        validated["warmup_steps"] = warmup_steps

        # Weight decay
        weight_decay = float(data.get("weight_decay", 0.01))
        if not (0 <= weight_decay <= 1):
            raise ValidationError("Weight decay must be between 0 and 1")
        validated["weight_decay"] = weight_decay

        # Per-parameter-group weight decay
        validated["per_layer_weight_decay"] = bool(data.get("per_layer_weight_decay", False))
        if validated["per_layer_weight_decay"]:
            # Attention layer weight decay
            attention_weight_decay = float(data.get("attention_weight_decay", 0.01))
            if not (0 <= attention_weight_decay <= 0.3):
                raise ValidationError("Attention weight decay must be between 0 and 0.3")
            validated["attention_weight_decay"] = attention_weight_decay

            # MLP/FFN layer weight decay
            mlp_weight_decay = float(data.get("mlp_weight_decay", 0.01))
            if not (0 <= mlp_weight_decay <= 0.3):
                raise ValidationError("MLP weight decay must be between 0 and 0.3")
            validated["mlp_weight_decay"] = mlp_weight_decay

            # Embedding layer weight decay
            embedding_weight_decay = float(data.get("embedding_weight_decay", 0.0))
            if not (0 <= embedding_weight_decay <= 0.1):
                raise ValidationError("Embedding weight decay must be between 0 and 0.1")
            validated["embedding_weight_decay"] = embedding_weight_decay

            # No decay patterns (e.g., bias, LayerNorm)
            no_decay_patterns = data.get("no_decay_patterns", ["bias", "LayerNorm", "layer_norm"])
            if isinstance(no_decay_patterns, str):
                no_decay_patterns = [p.strip() for p in no_decay_patterns.split("\n") if p.strip()]
            validated["no_decay_patterns"] = no_decay_patterns

        # Gradient accumulation
        gradient_accumulation = int(data.get("gradient_accumulation", 1))
        if not (1 <= gradient_accumulation <= 64):
            raise ValidationError("Gradient accumulation must be between 1 and 64")
        validated["gradient_accumulation"] = gradient_accumulation

        # Learning rate scheduler
        lr_scheduler = data.get("lr_scheduler", "cosine")
        valid_schedulers = ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]
        if lr_scheduler not in valid_schedulers:
            raise ValidationError(f"Invalid scheduler. Must be one of: {valid_schedulers}")
        validated["lr_scheduler"] = lr_scheduler

        # Early stopping
        validated["early_stopping"] = bool(data.get("early_stopping", False))
        if validated["early_stopping"]:
            validated["early_stopping_patience"] = max(1, min(10, int(data.get("early_stopping_patience", 3))))
            validated["early_stopping_threshold"] = max(0, min(1, float(data.get("early_stopping_threshold", 0.01))))

        # Eval split
        eval_split = float(data.get("eval_split", 0))
        if not (0 <= eval_split <= 0.5):
            raise ValidationError("Eval split must be between 0 and 0.5")
        validated["eval_split"] = eval_split
        validated["stratified_split"] = bool(data.get("stratified_split", False))

        # Label smoothing
        label_smoothing = float(data.get("label_smoothing", 0.0))
        if not (0 <= label_smoothing <= 0.3):
            raise ValidationError("Label smoothing must be between 0 and 0.3")
        validated["label_smoothing"] = label_smoothing

        # Gradient clipping
        max_grad_norm = float(data.get("max_grad_norm", 1.0))
        if not (0.1 <= max_grad_norm <= 10):
            raise ValidationError("Max gradient norm must be between 0.1 and 10")
        validated["max_grad_norm"] = max_grad_norm

        # QLoRA / Quantization
        validated["use_qlora"] = bool(data.get("use_qlora", False))
        if validated["use_qlora"]:
            validated["qlora_bits"] = int(data.get("qlora_bits", 4))
            if validated["qlora_bits"] not in [4, 8]:
                raise ValidationError("QLoRA bits must be 4 or 8")
            validated["qlora_double_quant"] = bool(data.get("qlora_double_quant", True))

        # Data format
        data_format = data.get("data_format", "completion")
        valid_formats = ["completion", "instruction", "chat", "qa"]
        if data_format not in valid_formats:
            raise ValidationError(f"Invalid data format. Must be one of: {valid_formats}")
        validated["data_format"] = data_format

        # LoRA settings
        validated["use_lora"] = bool(data.get("use_lora", False))

        if validated["use_lora"] or validated["use_qlora"]:
            lora_r = int(data.get("lora_r", 8))
            if not (2 <= lora_r <= 128):
                raise ValidationError("LoRA rank must be between 2 and 128")
            validated["lora_r"] = lora_r

            lora_alpha = int(data.get("lora_alpha", 32))
            if not (4 <= lora_alpha <= 256):
                raise ValidationError("LoRA alpha must be between 4 and 256")
            validated["lora_alpha"] = lora_alpha
            
            lora_dropout = float(data.get("lora_dropout", 0.1))
            if not (0 <= lora_dropout <= 0.5):
                raise ValidationError("LoRA dropout must be between 0 and 0.5")
            validated["lora_dropout"] = lora_dropout

            # DoRA
            validated["use_dora"] = bool(data.get("use_dora", False))

            # RS-LoRA (Rank Stabilization)
            validated["use_rs_lora"] = bool(data.get("use_rs_lora", False))

            # LoRA+ (Different learning rates for A/B matrices)
            validated["use_lora_plus"] = bool(data.get("use_lora_plus", False))
            if validated["use_lora_plus"]:
                lora_lr_ratio = float(data.get("lora_lr_ratio", 16.0))
                if not (1.0 <= lora_lr_ratio <= 64.0):
                    raise ValidationError("LoRA+ LR ratio must be between 1.0 and 64.0")
                validated["lora_lr_ratio"] = lora_lr_ratio

            # LoRA bias
            lora_bias = data.get("lora_bias", "none")
            if lora_bias not in ["none", "all", "lora_only"]:
                raise ValidationError("LoRA bias must be 'none', 'all', or 'lora_only'")
            validated["lora_bias"] = lora_bias

        # NEFTune
        neftune_alpha = float(data.get("neftune_alpha", 0))
        if not (0 <= neftune_alpha <= 15):
            raise ValidationError("NEFTune alpha must be between 0 and 15")
        validated["neftune_alpha"] = neftune_alpha

        # Cosine restarts num_cycles
        cosine_num_cycles = float(data.get("cosine_num_cycles", 1.0))
        if not (0.5 <= cosine_num_cycles <= 10):
            raise ValidationError("Cosine num_cycles must be between 0.5 and 10")
        validated["cosine_num_cycles"] = cosine_num_cycles

        # Polynomial decay power
        polynomial_power = float(data.get("polynomial_power", 1.0))
        if not (0.5 <= polynomial_power <= 5.0):
            raise ValidationError("Polynomial power must be between 0.5 and 5.0")
        validated["polynomial_power"] = polynomial_power

        # Warmup type
        warmup_type = data.get("warmup_type", "linear")
        if warmup_type not in ["linear", "cosine", "constant"]:
            raise ValidationError("Warmup type must be linear, cosine, or constant")
        validated["warmup_type"] = warmup_type

        # BOS/EOS token control
        validated["add_bos_token"] = bool(data.get("add_bos_token", True))
        validated["add_eos_token"] = bool(data.get("add_eos_token", True))

        # QLoRA compute dtype
        if validated.get("use_qlora", False):
            qlora_compute_dtype = data.get("qlora_compute_dtype", "float16")
            if qlora_compute_dtype not in ["float16", "bfloat16"]:
                raise ValidationError("QLoRA compute dtype must be float16 or bfloat16")
            validated["qlora_compute_dtype"] = qlora_compute_dtype

        # Completion-only loss masking
        validated["completion_only_loss"] = bool(data.get("completion_only_loss", False))

        # Dynamic vs static padding
        validated["dynamic_padding"] = bool(data.get("dynamic_padding", True))

        # Sample output generation during training
        validated["generate_samples_every_n_steps"] = int(data.get("generate_samples_every_n_steps", 0))
        validated["sample_prompts"] = data.get("sample_prompts", [])

        # Attention implementation
        attn_impl = data.get("attn_implementation", "auto")
        if attn_impl not in ["auto", "sdpa", "flash_attention_2", "eager"]:
            raise ValidationError("Invalid attention implementation")
        validated["attn_implementation"] = attn_impl

        # RoPE scaling for context extension
        rope_scaling_type = data.get("rope_scaling_type", "none")
        if rope_scaling_type not in ["none", "linear", "dynamic", "yarn"]:
            raise ValidationError("RoPE scaling type must be none, linear, dynamic, or yarn")
        validated["rope_scaling_type"] = rope_scaling_type
        if rope_scaling_type != "none":
            rope_factor = float(data.get("rope_scaling_factor", 2.0))
            if not (1.0 <= rope_factor <= 8.0):
                raise ValidationError("RoPE scaling factor must be between 1.0 and 8.0")
            validated["rope_scaling_factor"] = rope_factor

        # Layer freezing
        validated["freeze_embeddings"] = bool(data.get("freeze_embeddings", False))
        freeze_layers = int(data.get("freeze_layers", 0))
        if not (0 <= freeze_layers <= 48):
            raise ValidationError("Freeze layers must be between 0 and 48")
        validated["freeze_layers"] = freeze_layers

        # Torch compile
        validated["torch_compile"] = bool(data.get("torch_compile", False))

        # Save/logging steps
        validated["save_steps"] = max(50, min(10000, int(data.get("save_steps", 500))))
        validated["logging_steps"] = max(1, min(1000, int(data.get("logging_steps", 10))))
        validated["eval_steps"] = max(10, min(5000, int(data.get("eval_steps", 100))))

        # Mixed precision
        mixed_precision = data.get("mixed_precision", "fp16")
        valid_precision = ["no", "fp16", "bf16"]
        if mixed_precision not in valid_precision:
            raise ValidationError(f"Invalid mixed precision. Must be one of: {valid_precision}")
        validated["mixed_precision"] = mixed_precision
        # Backwards compatibility: also set fp16/bf16 flags
        validated["fp16"] = mixed_precision == "fp16"
        validated["bf16"] = mixed_precision == "bf16"

        # Optimizer
        optimizer = data.get("optimizer", "adamw_torch")
        valid_optimizers = ["adamw_torch", "adamw_hf", "adam8bit", "adamw_bnb_8bit", "adafactor", "sgd"]
        if optimizer not in valid_optimizers:
            raise ValidationError(f"Invalid optimizer. Must be one of: {valid_optimizers}")
        validated["optimizer"] = optimizer

        # Gradient checkpointing
        validated["gradient_checkpointing"] = bool(data.get("gradient_checkpointing", False))

        # Sequence packing
        validated["use_sequence_packing"] = bool(data.get("use_sequence_packing", False))

        # Seed
        seed = int(data.get("seed", 42))
        if not (0 <= seed <= 2**32 - 1):
            raise ValidationError("Seed must be a valid 32-bit unsigned integer")
        validated["seed"] = seed

        # Training data
        training_data = data.get("training_data", "")
        if isinstance(training_data, str):
            training_data = self._sanitize_text(training_data)
            if len(training_data.encode('utf-8')) > self.config.MAX_TRAINING_DATA_SIZE:
                raise ValidationError(
                    f"Training data exceeds maximum size of {self.config.MAX_TRAINING_DATA_SIZE / 1024:.0f}KB"
                )
        validated["training_data"] = training_data

        # QAT (Quantization-Aware Training) settings
        validated["use_qat"] = bool(data.get("use_qat", False))
        if validated["use_qat"]:
            # QAT bit width
            qat_bits = int(data.get("qat_bits", 4))
            if qat_bits not in [2, 3, 4, 8]:
                raise ValidationError("QAT bits must be 2, 3, 4, or 8")
            validated["qat_bits"] = qat_bits

            # QAT group size
            qat_group_size = int(data.get("qat_group_size", 128))
            if qat_group_size not in [32, 64, 128, 256]:
                raise ValidationError("QAT group size must be 32, 64, 128, or 256")
            validated["qat_group_size"] = qat_group_size

            # QAT quantize embeddings
            validated["qat_quantize_embeddings"] = bool(data.get("qat_quantize_embeddings", False))

            # QAT symmetric quantization
            validated["qat_symmetric"] = bool(data.get("qat_symmetric", True))

            # QAT calibration samples
            qat_calibration_samples = int(data.get("qat_calibration_samples", 128))
            if not (16 <= qat_calibration_samples <= 1024):
                raise ValidationError("QAT calibration samples must be between 16 and 1024")
            validated["qat_calibration_samples"] = qat_calibration_samples

            # QAT warmup steps before quantization kicks in
            qat_warmup_steps = int(data.get("qat_warmup_steps", 100))
            if not (0 <= qat_warmup_steps <= 1000):
                raise ValidationError("QAT warmup steps must be between 0 and 1000")
            validated["qat_warmup_steps"] = qat_warmup_steps

        self._logger.info(f"Validated config: model={validated['model_name']}, epochs={validated['epochs']}")
        return validated

    def _sanitize_text(self, text: str) -> str:
        """Sanitize training text input."""
        # Remove null bytes and other control characters (except newlines and tabs)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        return text.strip()


# =============================================================================
# PARAMETER DEFINITIONS
# =============================================================================

TRAINING_PARAMETERS: dict[str, dict[str, Any]] = {
    "model_name": {
        "name": "Base Model",
        "type": "select_with_custom",
        "default": "gpt2",
        "options": [
            {"value": "custom", "label": "📁 Custom Path / Previously Trained Model"},
            {"value": "distilgpt2", "label": "DistilGPT-2 (82M) - Super fast, good for testing"},
            {"value": "gpt2", "label": "GPT-2 Small (124M) - Fast, good balance"},
            {"value": "gpt2-medium", "label": "GPT-2 Medium (355M) - Better quality"},
            {"value": "gpt2-large", "label": "GPT-2 Large (774M) - High quality, slower"},
            {"value": "gpt2-xl", "label": "GPT-2 XL (1.5B) - Best GPT-2, needs good GPU"},
            {"value": "facebook/opt-125m", "label": "OPT-125M - Meta's small model"},
            {"value": "facebook/opt-350m", "label": "OPT-350M - Meta's medium model"},
            {"value": "facebook/opt-1.3b", "label": "OPT-1.3B - Meta's larger model"},
            {"value": "EleutherAI/pythia-160m", "label": "Pythia-160M - Open research model"},
            {"value": "EleutherAI/pythia-410m", "label": "Pythia-410M - Larger Pythia"},
            {"value": "EleutherAI/pythia-1b", "label": "Pythia-1B - Big Pythia"},
            {"value": "microsoft/phi-1_5", "label": "Phi-1.5 (1.3B) - Microsoft's efficient model"},
            {"value": "microsoft/phi-2", "label": "Phi-2 (2.7B) - Microsoft's best small model"},
            {"value": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "label": "TinyLlama (1.1B) - Fast chat model"},
            {"value": "Qwen/Qwen2-0.5B", "label": "Qwen2-0.5B - Alibaba's tiny model"},
            {"value": "Qwen/Qwen2-1.5B", "label": "Qwen2-1.5B - Alibaba's small model"},
            {"value": "stabilityai/stablelm-2-zephyr-1_6b", "label": "StableLM-2 (1.6B) - Stability AI"},
        ],
        "explanation": "Choose a base model or use a custom path. Select 'Custom Path' to continue training a previously fine-tuned model.",
        "beginner_tip": "Start with GPT-2 Small or DistilGPT-2. To continue training a model you already trained, select 'Custom Path' and enter the path to your model (e.g., ./Outputs/final_merged)."
    },
    "learning_rate": {
        "name": "Learning Rate",
        "type": "slider",
        "default": 5e-5,
        "min": 1e-6,
        "max": 1e-3,
        "step": 1e-6,
        "display_format": "scientific",
        "explanation": "How big of steps the AI takes when learning. Too big = overshoots and learns garbage. Too small = takes forever and might get stuck.",
        "beginner_tip": "The default (5e-5 or 0.00005) works great for most cases. Only change this if training seems weird.",
        "presets": {"conservative": 1e-5, "balanced": 5e-5, "aggressive": 2e-4}
    },
    "epochs": {
        "name": "Training Rounds (Epochs)",
        "type": "slider",
        "default": 3,
        "min": 1,
        "max": 20,
        "step": 1,
        "explanation": "How many times the AI reads through ALL your training data. More rounds = learns better, but too many = memorizes instead of understanding (overfitting).",
        "beginner_tip": "3-5 epochs is usually the sweet spot. Watch your loss - if it stops going down, you can stop early.",
        "presets": {"quick_test": 1, "balanced": 3, "thorough": 5}
    },
    "batch_size": {
        "name": "Batch Size",
        "type": "slider",
        "default": 4,
        "min": 1,
        "max": 32,
        "step": 1,
        "explanation": "How many examples the AI looks at before updating its brain. Bigger = faster training but needs more memory. If you get 'out of memory' errors, make this smaller!",
        "beginner_tip": "Start with 4 or 8. Getting crashes? Try 2 or even 1. Got a beefy GPU? Try 16 or 32.",
        "presets": {"low_memory": 1, "balanced": 4, "high_memory": 16}
    },
    "max_length": {
        "name": "Maximum Text Length",
        "type": "slider",
        "default": 128,
        "min": 32,
        "max": 1024,
        "step": 32,
        "explanation": "The longest piece of text (in tokens, roughly words) the AI will process at once. Longer = can understand bigger context but uses way more memory.",
        "beginner_tip": "128-256 is good for most tasks. Only go higher if you need the AI to understand long documents.",
        "presets": {"short_text": 64, "balanced": 128, "long_text": 512}
    },
    "warmup_steps": {
        "name": "Warmup Steps",
        "type": "slider",
        "default": 100,
        "min": 0,
        "max": 1000,
        "step": 10,
        "explanation": "Number of baby steps before going full speed. The AI starts with tiny learning steps, then gradually speeds up. Helps prevent early mistakes from ruining training.",
        "beginner_tip": "100-500 is usually fine. More warmup = safer but slower start. For tiny datasets, use less warmup.",
        "presets": {"no_warmup": 0, "gentle": 100, "careful": 500}
    },
    "weight_decay": {
        "name": "Weight Decay",
        "type": "slider",
        "default": 0.01,
        "min": 0,
        "max": 0.3,
        "step": 0.01,
        "explanation": "A gentle force that keeps the AI's 'brain weights' from getting too extreme. Helps prevent overfitting (memorizing instead of learning).",
        "beginner_tip": "0.01 is a good default. If your AI seems to memorize your data but can't handle new stuff, try increasing this.",
        "presets": {"none": 0, "light": 0.01, "strong": 0.1}
    },
    "gradient_accumulation": {
        "name": "Gradient Accumulation Steps",
        "type": "slider",
        "default": 1,
        "min": 1,
        "max": 16,
        "step": 1,
        "explanation": "A trick to simulate bigger batch sizes without needing more memory. If you want batch size 16 but only have memory for 4, set batch=4 and accumulation=4.",
        "beginner_tip": "Leave at 1 unless you need bigger effective batch sizes but keep running out of memory.",
        "presets": {"normal": 1, "memory_saver": 4, "maximum_saver": 8}
    },
    "use_lora": {
        "name": "Use LoRA (Efficient Fine-tuning)",
        "type": "checkbox",
        "default": False,
        "explanation": "LoRA = Low-Rank Adaptation. Instead of updating the ENTIRE model, only update small adapter layers. Uses way less memory and is faster, with nearly the same quality!",
        "beginner_tip": "Turn this ON if you're running on limited hardware or want faster training. It's basically a cheat code for efficiency."
    },
    "lora_r": {
        "name": "LoRA Rank",
        "type": "slider",
        "default": 8,
        "min": 4,
        "max": 64,
        "step": 4,
        "depends_on": "use_lora",
        "explanation": "How 'complex' the LoRA adapters are. Higher = can learn more complex stuff but uses more memory. Lower = more efficient but might miss nuances.",
        "beginner_tip": "8 or 16 works great for most tasks. Only go higher for really complex fine-tuning.",
        "presets": {"efficient": 4, "balanced": 8, "expressive": 16}
    },
    "lora_alpha": {
        "name": "LoRA Alpha",
        "type": "slider",
        "default": 32,
        "min": 8,
        "max": 128,
        "step": 8,
        "depends_on": "use_lora",
        "explanation": "Controls how much the LoRA adapters influence the model. Usually set to 2-4x the LoRA rank. Higher = stronger adaptation effect.",
        "beginner_tip": "A good rule: set this to 2-4 times your LoRA Rank. So if rank=8, alpha=16-32 is good."
    },
    "lora_dropout": {
        "name": "LoRA Dropout",
        "type": "slider",
        "default": 0.1,
        "min": 0,
        "max": 0.5,
        "step": 0.05,
        "depends_on": "use_lora",
        "explanation": "Randomly 'turns off' some LoRA connections during training. Helps prevent overfitting.",
        "beginner_tip": "0.05-0.1 is usually good. Increase to 0.2 if you notice overfitting (train loss low but outputs are bad)."
    },
    "use_dora": {
        "name": "Use DoRA (Weight-Decomposed LoRA)",
        "type": "checkbox",
        "default": False,
        "depends_on": "use_lora",
        "explanation": "DoRA decomposes weights into magnitude and direction, often outperforming standard LoRA. Slight memory/speed overhead.",
        "beginner_tip": "Try it! Often gives better results than regular LoRA with minimal extra cost."
    },
    "lora_bias": {
        "name": "LoRA Bias Training",
        "type": "select",
        "default": "none",
        "options": [
            {"value": "none", "label": "None - Don't train biases (default)"},
            {"value": "all", "label": "All - Train all bias parameters"},
            {"value": "lora_only", "label": "LoRA Only - Train biases in LoRA layers"}
        ],
        "depends_on": "use_lora",
        "explanation": "Whether to train bias parameters alongside LoRA. 'All' can improve quality but uses more memory.",
        "beginner_tip": "Start with 'none'. Try 'all' if you want slightly better results and have memory to spare."
    },
    "neftune_alpha": {
        "name": "NEFTune Noise Alpha",
        "type": "slider",
        "default": 0,
        "min": 0,
        "max": 15,
        "step": 1,
        "explanation": "Adds noise to embeddings during training. 0 = disabled. 5-10 often improves instruction-following. Higher = more regularization.",
        "beginner_tip": "Try 5 for instruction tuning. Set to 0 to disable. Values above 10 are aggressive."
    },
    "attn_implementation": {
        "name": "Attention Implementation",
        "type": "select",
        "default": "auto",
        "options": [
            {"value": "auto", "label": "Auto - Let transformers decide"},
            {"value": "sdpa", "label": "SDPA - PyTorch native, good balance"},
            {"value": "flash_attention_2", "label": "Flash Attention 2 - Fastest (requires flash-attn)"},
            {"value": "eager", "label": "Eager - Standard, most compatible"}
        ],
        "explanation": "Which attention algorithm to use. Flash Attention 2 is fastest but requires separate install. SDPA is built into PyTorch 2.0+.",
        "beginner_tip": "Auto works fine. Use Flash Attention 2 if installed for best speed on long sequences."
    },
    "lr_scheduler": {
        "name": "Learning Rate Scheduler",
        "type": "select",
        "default": "cosine",
        "options": [
            {"value": "cosine", "label": "Cosine - Smoothly decreases (recommended)"},
            {"value": "linear", "label": "Linear - Steadily decreases"},
            {"value": "constant", "label": "Constant - Never changes"},
            {"value": "constant_with_warmup", "label": "Constant + Warmup - Flat after warmup"},
            {"value": "cosine_with_restarts", "label": "Cosine Restarts - Periodic resets"},
            {"value": "polynomial", "label": "Polynomial - Curved decrease"}
        ],
        "explanation": "How the learning rate changes during training. Starting high and decreasing usually helps the model fine-tune better in later stages.",
        "beginner_tip": "Cosine is great for most cases. Linear is simpler but works well too. Constant is good for very short training runs."
    },
    "early_stopping": {
        "name": "Enable Early Stopping",
        "type": "checkbox",
        "default": False,
        "explanation": "Automatically stop training when the model stops improving. Saves time and prevents overfitting!",
        "beginner_tip": "Turn this ON if you're not sure how many epochs to use. It'll stop when it's learned enough."
    },
    "early_stopping_patience": {
        "name": "Early Stopping Patience",
        "type": "slider",
        "default": 3,
        "min": 1,
        "max": 10,
        "step": 1,
        "depends_on": "early_stopping",
        "explanation": "How many evaluation checks to wait before stopping. Higher = more patient, might train longer.",
        "beginner_tip": "3 is a good default. Use 5+ if your loss is noisy (jumps around a lot)."
    },
    "early_stopping_threshold": {
        "name": "Early Stopping Threshold",
        "type": "slider",
        "default": 0.01,
        "min": 0,
        "max": 0.1,
        "step": 0.005,
        "display_format": "decimal",
        "depends_on": "early_stopping",
        "explanation": "Minimum improvement required to count as 'getting better'. Smaller = more sensitive to tiny improvements.",
        "beginner_tip": "0.01 works for most cases. Lower it if training stops too early."
    },
    "eval_split": {
        "name": "Evaluation Split",
        "type": "slider",
        "default": 0.1,
        "min": 0,
        "max": 0.3,
        "step": 0.05,
        "display_format": "percent",
        "explanation": "Percentage of data to hold back for testing. The model never trains on this data, so it shows how well it generalizes.",
        "beginner_tip": "10% (0.1) is standard. Set to 0 to use ALL data for training (faster but no eval metrics)."
    },
    "stratified_split": {
        "name": "Stratified Split",
        "type": "checkbox",
        "default": False,
        "depends_on": "eval_split",
        "explanation": "Ensures train/eval sets have similar distributions. Useful when data has distinct categories or patterns you want equally represented.",
        "beginner_tip": "Turn ON if your data has clear categories (like different topics or styles). Leave OFF for general text data."
    },
    "label_smoothing": {
        "name": "Label Smoothing",
        "type": "slider",
        "default": 0.0,
        "min": 0,
        "max": 0.2,
        "step": 0.01,
        "display_format": "decimal",
        "explanation": "Softens target labels slightly, preventing overconfidence. Helps the model generalize better and reduces overfitting.",
        "beginner_tip": "Try 0.05-0.1 if your model is overfitting. Leave at 0 for standard training."
    },
    "eval_steps": {
        "name": "Evaluate Every N Steps",
        "type": "slider",
        "default": 100,
        "min": 10,
        "max": 1000,
        "step": 10,
        "explanation": "How often to run evaluation on the held-out data. More frequent = better monitoring but slower training.",
        "beginner_tip": "100-200 is good. Lower for short runs, higher for long runs."
    },
    "max_grad_norm": {
        "name": "Gradient Clipping",
        "type": "slider",
        "default": 1.0,
        "min": 0.1,
        "max": 5.0,
        "step": 0.1,
        "explanation": "Limits how big gradient updates can be. Prevents 'exploding gradients' that can ruin training.",
        "beginner_tip": "1.0 is the safe default. Lower to 0.5 if you see NaN losses. Raise to 2+ if training seems too slow."
    },
    "use_qlora": {
        "name": "Use QLoRA (4-bit Training)",
        "type": "checkbox",
        "default": False,
        "explanation": "Loads the model in 4-bit precision, using ~75% less VRAM! Enables training large models on consumer GPUs. Requires bitsandbytes library.",
        "beginner_tip": "Turn ON if you're running out of memory. This is how people train 7B+ models on 8GB GPUs!"
    },
    "qlora_bits": {
        "name": "Quantization Bits",
        "type": "select",
        "default": 4,
        "options": [
            {"value": 4, "label": "4-bit - Maximum memory savings"},
            {"value": 8, "label": "8-bit - Better quality, still saves memory"}
        ],
        "depends_on": "use_qlora",
        "explanation": "How much to compress the model. 4-bit uses less memory, 8-bit has slightly better quality.",
        "beginner_tip": "4-bit is usually fine. Try 8-bit if you have memory to spare and want slightly better results."
    },
    "qlora_double_quant": {
        "name": "Double Quantization",
        "type": "checkbox",
        "default": True,
        "depends_on": "use_qlora",
        "explanation": "Quantizes the quantization constants too. Saves a bit more memory with minimal quality impact.",
        "beginner_tip": "Leave ON - it's free memory savings with almost no downside."
    },
    "data_format": {
        "name": "Data Format",
        "type": "select",
        "default": "completion",
        "options": [
            {"value": "completion", "label": "Completion - Raw text, model learns to continue"},
            {"value": "instruction", "label": "Instruction - 'Instruction:' and 'Response:' pairs"},
            {"value": "chat", "label": "Chat - Multi-turn conversation format"},
            {"value": "qa", "label": "Q&A - Question and Answer pairs"}
        ],
        "explanation": "How your training data is structured. The app will format it correctly for the model.",
        "beginner_tip": "Use 'Completion' for raw text. Use 'Instruction' for task-following data. Use 'Chat' for conversations."
    },
    "save_steps": {
        "name": "Save Every N Steps",
        "type": "slider",
        "default": 500,
        "min": 100,
        "max": 5000,
        "step": 100,
        "explanation": "How often to save a checkpoint of your model. Useful if training crashes - you won't lose everything!",
        "beginner_tip": "500-1000 is good. For long training runs, save more often. For quick tests, save less to save disk space."
    },
    "logging_steps": {
        "name": "Log Every N Steps",
        "type": "slider",
        "default": 10,
        "min": 1,
        "max": 100,
        "step": 1,
        "explanation": "How often to print training stats (loss, etc). More frequent = see more detail but slightly slower.",
        "beginner_tip": "10-50 is good. Set to 1 if you want to watch every single update (good for debugging)."
    },
    "mixed_precision": {
        "name": "Mixed Precision Mode",
        "type": "select",
        "default": "fp16",
        "options": [
            {"value": "no", "label": "None (FP32) - Full precision, most compatible"},
            {"value": "fp16", "label": "FP16 - Half precision, 2x faster (recommended)"},
            {"value": "bf16", "label": "BF16 - Brain float, better for large models (Ampere+ GPUs)"},
        ],
        "explanation": "Controls numerical precision during training. Lower precision = faster + less memory. FP16 is standard. BF16 handles large values better but needs newer GPUs (RTX 30xx+, A100).",
        "beginner_tip": "FP16 works great on most GPUs. Use BF16 if you have an RTX 30/40 series or A100. Use None if you get NaN errors."
    },
    "optimizer": {
        "name": "Optimizer",
        "type": "select",
        "default": "adamw_torch",
        "options": [
            {"value": "adamw_torch", "label": "AdamW (PyTorch) - Standard, reliable"},
            {"value": "adamw_hf", "label": "AdamW (HuggingFace) - Similar to PyTorch"},
            {"value": "adam8bit", "label": "Adam 8-bit - 30% less memory (requires bitsandbytes)"},
            {"value": "adamw_bnb_8bit", "label": "AdamW 8-bit - Memory efficient AdamW"},
            {"value": "adafactor", "label": "Adafactor - Very memory efficient, no momentum"},
            {"value": "sgd", "label": "SGD - Simple, fast, needs tuning"},
        ],
        "explanation": "The algorithm that updates model weights. AdamW is the gold standard. 8-bit variants save ~30% memory. Adafactor saves even more but can be unstable.",
        "beginner_tip": "AdamW (PyTorch) is the safe choice. Try 8-bit Adam if running low on memory. Adafactor for extreme memory savings."
    },
    "gradient_checkpointing": {
        "name": "Gradient Checkpointing",
        "type": "checkbox",
        "default": False,
        "explanation": "Trades compute for memory by recomputing activations during backward pass. Reduces memory by ~60% but training is ~30% slower. Essential for large models on limited VRAM.",
        "beginner_tip": "Turn ON if you're running out of memory. The slowdown is worth it to train larger models or use bigger batch sizes."
    },
    "use_sequence_packing": {
        "name": "Sequence Packing",
        "type": "checkbox",
        "default": False,
        "explanation": "Combines multiple short training examples into a single sequence to maximize GPU utilization. Can speed up training 2-5x when you have many short examples. Best for datasets with varied-length texts.",
        "beginner_tip": "Turn ON if your training data has lots of short examples (like Q&A pairs or short dialogues). Significantly speeds up training by reducing padding waste."
    },
    "seed": {
        "name": "Random Seed",
        "type": "number",
        "default": 42,
        "min": 0,
        "max": 999999,
        "explanation": "A number that controls randomness. Using the same seed = same results every time. Useful for reproducibility.",
        "beginner_tip": "42 is the classic default (Hitchhiker's Guide reference!). Change it if you want different random variations."
    }
}

# Preset configurations
PRESETS: dict[str, dict[str, Any]] = {
    "quick_test": {
        "name": "Quick Test",
        "description": "Fast training to see if everything works. Low quality but quick results.",
        "settings": {
            "model_name": "distilgpt2",
            "learning_rate": 5e-5,
            "epochs": 1,
            "batch_size": 4,
            "max_length": 64,
            "warmup_steps": 50,
            "use_lora": True,
            "lr_scheduler": "constant",
            "fp16": True
        }
    },
    "balanced": {
        "name": "Balanced",
        "description": "Good balance of speed and quality. Recommended for most users.",
        "settings": {
            "model_name": "gpt2",
            "learning_rate": 5e-5,
            "epochs": 3,
            "batch_size": 4,
            "max_length": 128,
            "warmup_steps": 100,
            "weight_decay": 0.01,
            "use_lora": False,
            "fp16": True
        }
    },
    "quality": {
        "name": "Quality Focus",
        "description": "Slower but better results. Use if you have time and decent hardware.",
        "settings": {
            "model_name": "gpt2-medium",
            "learning_rate": 2e-5,
            "epochs": 5,
            "batch_size": 4,
            "max_length": 256,
            "warmup_steps": 200,
            "weight_decay": 0.01,
            "use_lora": False,
            "fp16": True
        }
    },
    "low_memory": {
        "name": "Low Memory Mode",
        "description": "For computers with limited RAM/VRAM. Slower but won't crash!",
        "settings": {
            "model_name": "distilgpt2",
            "learning_rate": 5e-5,
            "epochs": 3,
            "batch_size": 1,
            "max_length": 64,
            "gradient_accumulation": 4,
            "use_lora": True,
            "lora_r": 4,
            "fp16": True
        }
    },
    "lora_efficient": {
        "name": "LoRA Efficient",
        "description": "Uses LoRA for fast, memory-efficient fine-tuning with good quality.",
        "settings": {
            "model_name": "gpt2",
            "learning_rate": 1e-4,
            "epochs": 3,
            "batch_size": 4,
            "max_length": 128,
            "warmup_steps": 100,
            "use_lora": True,
            "lora_r": 8,
            "lora_alpha": 32,
            "fp16": True
        }
    },
    "phi2_quality": {
        "name": "Phi-2 Quality",
        "description": "Microsoft's Phi-2 with LoRA. Excellent quality for its size. Needs 8GB+ VRAM.",
        "settings": {
            "model_name": "microsoft/phi-2",
            "learning_rate": 2e-4,
            "epochs": 3,
            "batch_size": 2,
            "max_length": 256,
            "warmup_steps": 100,
            "weight_decay": 0.01,
            "gradient_accumulation": 4,
            "use_lora": True,
            "lora_r": 16,
            "lora_alpha": 32,
            "fp16": True
        }
    },
    "tinyllama_chat": {
        "name": "TinyLlama Chat",
        "description": "Fast chat model training. Good for conversational AI. Needs 6GB+ VRAM.",
        "settings": {
            "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "learning_rate": 2e-4,
            "epochs": 3,
            "batch_size": 2,
            "max_length": 512,
            "warmup_steps": 100,
            "weight_decay": 0.01,
            "gradient_accumulation": 4,
            "use_lora": True,
            "lora_r": 16,
            "lora_alpha": 32,
            "fp16": True
        }
    },
    "qwen_efficient": {
        "name": "Qwen2 Efficient",
        "description": "Alibaba's Qwen2 0.5B. Very fast, surprisingly capable. Good for limited hardware.",
        "settings": {
            "model_name": "Qwen/Qwen2-0.5B",
            "learning_rate": 1e-4,
            "epochs": 3,
            "batch_size": 4,
            "max_length": 256,
            "warmup_steps": 50,
            "weight_decay": 0.01,
            "use_lora": True,
            "lora_r": 8,
            "lora_alpha": 16,
            "fp16": True
        }
    }
}


# =============================================================================
# APPLICATION FACTORY
# =============================================================================

def create_app(config: Optional[Config] = None) -> tuple[Flask, SocketIO, TrainingStateManager]:
    """Application factory for creating Flask app with dependencies."""
    if config is None:
        config = create_config()

    app = Flask(__name__)
    app.config["SECRET_KEY"] = config.SECRET_KEY

    # Initialize SocketIO for real-time updates
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

    # Initialize state manager
    state_manager = TrainingStateManager(socketio)

    # Initialize validator
    validator = InputValidator(config)

    # Setup logging
    logger = setup_logging(config)

    # Store in app context
    app.config["APP_CONFIG"] = config
    app.config["STATE_MANAGER"] = state_manager
    app.config["VALIDATOR"] = validator

    # Register routes
    register_routes(app, state_manager, validator, config, logger)
    register_socketio_events(socketio, state_manager)

    # Register additional feature routes
    add_data_processing_routes(app, state_manager, validator, config, logger)
    add_monitoring_routes(app, state_manager, config, logger)
    add_advanced_inference_routes(app, logger)
    add_chart_comparison_routes(app, logger)

    # Register EfficientQAT routes
    add_qat_routes(app, state_manager, validator, logger)

    # Register Universal Data Loader routes (any format → tokenized JSONL)
    add_data_loader_routes(app, logger)

    return app, socketio, state_manager


def register_routes(
    app: Flask,
    state_manager: TrainingStateManager,
    validator: InputValidator,
    config: Config,
    logger: logging.Logger
) -> None:
    """Register all application routes."""

    @app.route('/')
    def index() -> str:
        """Serve main page."""
        return render_template('index.html')

    @app.route('/api/health')
    def health_check() -> tuple[Response, int]:
        """Health check endpoint - verify the app and dependencies are working."""
        health = {
            "status": "healthy",
            "app": "AI Training For DumDums",
            "checks": {}
        }

        # Check PyTorch
        try:
            import torch
            health["checks"]["pytorch"] = {
                "available": True,
                "version": torch.__version__,
                "cuda": torch.cuda.is_available(),
                "cuda_devices": torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
        except ImportError:
            health["checks"]["pytorch"] = {"available": False}
            health["status"] = "degraded"

        # Check Transformers
        try:
            import transformers
            health["checks"]["transformers"] = {
                "available": True,
                "version": transformers.__version__
            }
        except ImportError:
            health["checks"]["transformers"] = {"available": False}
            health["status"] = "degraded"

        # Check PEFT
        try:
            import peft
            health["checks"]["peft"] = {"available": True, "version": peft.__version__}
        except ImportError:
            health["checks"]["peft"] = {"available": False, "note": "QLoRA/LoRA unavailable"}

        # Check BitsAndBytes
        try:
            import bitsandbytes
            health["checks"]["bitsandbytes"] = {"available": True}
        except ImportError:
            health["checks"]["bitsandbytes"] = {"available": False, "note": "4-bit quantization unavailable"}

        status_code = 200 if health["status"] == "healthy" else 503
        return jsonify(health), status_code

    @app.route('/api/parameters')
    def get_parameters() -> Response:
        """Return all parameter definitions for the UI."""
        return jsonify(TRAINING_PARAMETERS)

    @app.route('/api/presets')
    def get_presets() -> Response:
        """Return all preset configurations."""
        return jsonify(PRESETS)

    @app.route('/api/train', methods=['POST'])
    def start_training() -> tuple[Response, int]:
        """Start a training run."""
        if state_manager.is_training():
            return jsonify({"error": "Training already in progress"}), 409

        try:
            raw_config = request.get_json()
            if not raw_config:
                return jsonify({"error": "No configuration provided"}), 400

            validated_config = validator.validate_config(raw_config)
        except ValidationError as e:
            logger.warning(f"Validation error: {e}")
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            logger.error(f"Unexpected error during validation: {e}")
            return jsonify({"error": "Invalid configuration format"}), 400

        # Reset state and start training
        state_manager.reset()

        # Start training in background thread
        thread = threading.Thread(
            target=run_training,
            args=(validated_config, state_manager, config, logger),
            daemon=True
        )
        thread.start()

        return jsonify({"status": "started", "message": "Training initiated"}), 202

    @app.route('/api/status')
    def get_status() -> Response:
        """Get current training status."""
        return jsonify(state_manager.get_status())

    @app.route('/api/stop', methods=['POST'])
    def stop_training() -> Response:
        """Request training stop."""
        state_manager.request_stop()
        return jsonify({"status": "stop_requested", "message": "Stop signal sent"})

    @app.route('/api/logs')
    def get_logs() -> Response:
        """Get training logs."""
        return jsonify({"logs": state_manager.get_logs()})

    # Health check endpoints
    @app.route('/health')
    def basic_health_check() -> Response:
        """Basic health check endpoint."""
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "2.0.0"
        })

    @app.route('/health/ready')
    def readiness_check() -> tuple[Response, int]:
        """Readiness probe - checks if app can accept traffic."""
        try:
            # Could add more checks here (DB, external services, etc.)
            return jsonify({
                "status": "ready",
                "training_active": state_manager.is_training()
            }), 200
        except Exception as e:
            return jsonify({"status": "not_ready", "error": str(e)}), 503

    @app.route('/health/live')
    def liveness_check() -> Response:
        """Liveness probe - checks if app is alive."""
        return jsonify({"status": "alive"})

    @app.route('/api/metrics')
    def get_metrics() -> Response:
        """Basic metrics endpoint."""
        status = state_manager.get_status()
        return jsonify({
            "training": {
                "is_active": status["is_training"],
                "status": status["status"],
                "progress": status["progress"],
                "current_step": status["current_step"],
                "total_steps": status["total_steps"]
            },
            "system": {
                "timestamp": datetime.now().isoformat()
            }
        })

    # =========================================================================
    # SYSTEM INFO ENDPOINTS
    # =========================================================================

    @app.route('/api/system/info')
    def get_system_info() -> Response:
        """Get system hardware information."""
        info = {
            "python_version": sys.version,
            "platform": sys.platform,
            "gpu": {"available": False, "name": None, "memory_total": None, "memory_free": None},
            "cpu": {"cores": os.cpu_count()},
            "memory": {}
        }

        try:
            import torch
            if torch.cuda.is_available():
                info["gpu"]["available"] = True
                info["gpu"]["name"] = torch.cuda.get_device_name(0)
                info["gpu"]["memory_total"] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
                info["gpu"]["memory_free"] = f"{(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3:.1f} GB"
                info["gpu"]["cuda_version"] = torch.version.cuda
        except ImportError:
            pass

        try:
            import psutil
            mem = psutil.virtual_memory()
            info["memory"]["total"] = f"{mem.total / 1024**3:.1f} GB"
            info["memory"]["available"] = f"{mem.available / 1024**3:.1f} GB"
            info["memory"]["percent_used"] = mem.percent
        except ImportError:
            pass

        return jsonify(info)

    @app.route('/api/system/gpu-memory')
    def get_gpu_memory() -> Response:
        """Get current GPU memory usage."""
        try:
            import torch
            if torch.cuda.is_available():
                return jsonify({
                    "allocated": f"{torch.cuda.memory_allocated(0) / 1024**3:.2f} GB",
                    "reserved": f"{torch.cuda.memory_reserved(0) / 1024**3:.2f} GB",
                    "total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
                })
            return jsonify({"error": "No GPU available"}), 400
        except ImportError:
            return jsonify({"error": "PyTorch not installed"}), 400

    # =========================================================================
    # DATASET MANAGEMENT ENDPOINTS
    # =========================================================================

    @app.route('/api/datasets')
    def list_datasets() -> Response:
        """List saved datasets."""
        datasets_dir = "./datasets"
        datasets = []

        if os.path.exists(datasets_dir):
            for filename in os.listdir(datasets_dir):
                filepath = os.path.join(datasets_dir, filename)
                if os.path.isfile(filepath) and filename.endswith(('.txt', '.json', '.jsonl')):
                    stat = os.stat(filepath)
                    datasets.append({
                        "name": filename,
                        "path": filepath,
                        "size": stat.st_size,
                        "size_human": f"{stat.st_size / 1024:.1f} KB" if stat.st_size < 1024*1024 else f"{stat.st_size / 1024**2:.1f} MB",
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })

        return jsonify({"datasets": sorted(datasets, key=lambda x: x["modified"], reverse=True)})

    @app.route('/api/datasets/save', methods=['POST'])
    def save_dataset() -> tuple[Response, int]:
        """Save training data as a dataset."""
        data = request.get_json()
        if not data or "content" not in data:
            return jsonify({"error": "content is required"}), 400

        name = data.get("name", f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        if not name.endswith(('.txt', '.json', '.jsonl')):
            name += '.txt'

        # Sanitize filename
        name = re.sub(r'[^\w\-_\.]', '_', name)

        datasets_dir = "./datasets"
        os.makedirs(datasets_dir, exist_ok=True)

        filepath = os.path.join(datasets_dir, name)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(data["content"])

        return jsonify({
            "status": "saved",
            "name": name,
            "path": filepath,
            "size": len(data["content"])
        }), 201

    @app.route('/api/datasets/load/<path:filename>')
    def load_dataset(filename: str) -> tuple[Response, int]:
        """Load a saved dataset."""
        # Sanitize and validate path
        filename = os.path.basename(filename)
        filepath = os.path.join("./datasets", filename)

        if not os.path.exists(filepath):
            return jsonify({"error": "Dataset not found"}), 404

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            return jsonify({
                "name": filename,
                "content": content,
                "size": len(content),
                "lines": len(content.splitlines())
            }), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/datasets/delete/<path:filename>', methods=['DELETE'])
    def delete_dataset(filename: str) -> tuple[Response, int]:
        """Delete a saved dataset."""
        filename = os.path.basename(filename)
        filepath = os.path.join("./datasets", filename)

        if not os.path.exists(filepath):
            return jsonify({"error": "Dataset not found"}), 404

        try:
            os.remove(filepath)
            return jsonify({"status": "deleted", "name": filename}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/datasets/upload', methods=['POST'])
    def upload_dataset() -> tuple[Response, int]:
        """Upload a dataset file."""
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Validate file extension
        allowed_extensions = {'.txt', '.json', '.jsonl', '.csv'}
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in allowed_extensions:
            return jsonify({"error": f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"}), 400

        # Sanitize filename
        filename = re.sub(r'[^\w\-_\.]', '_', file.filename)

        datasets_dir = "./datasets"
        os.makedirs(datasets_dir, exist_ok=True)

        filepath = os.path.join(datasets_dir, filename)
        file.save(filepath)

        # Read and return info
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        return jsonify({
            "status": "uploaded",
            "name": filename,
            "path": filepath,
            "size": len(content),
            "lines": len(content.splitlines())
        }), 201

    @app.route('/api/datasets/download-url', methods=['POST'])
    def download_dataset_from_url() -> tuple[Response, int]:
        """Download a dataset from URL (useful for RunPod)."""
        import urllib.request
        import urllib.error
        import urllib.parse
        import ipaddress

        data = request.get_json()
        if not data or "url" not in data:
            return jsonify({"error": "URL is required"}), 400

        url = data["url"]

        # Validate URL to prevent SSRF attacks
        try:
            parsed = urllib.parse.urlparse(url)

            # Only allow http and https schemes
            if parsed.scheme not in ['http', 'https']:
                return jsonify({"error": "Only HTTP and HTTPS URLs are allowed"}), 400

            # Block internal/private IP addresses
            if parsed.hostname:
                try:
                    ip = ipaddress.ip_address(parsed.hostname)
                    if ip.is_private or ip.is_loopback or ip.is_link_local:
                        return jsonify({"error": "Access to private/internal IPs is not allowed"}), 400
                except ValueError:
                    # Not an IP address, it's a hostname - allow it
                    pass
        except Exception as e:
            return jsonify({"error": f"Invalid URL: {str(e)}"}), 400

        filename = data.get("filename") or url.split("/")[-1].split("?")[0]

        # Sanitize filename
        filename = re.sub(r'[^\w\-_\.]', '_', filename)
        if not filename:
            filename = "downloaded_dataset.jsonl"

        # Ensure valid extension
        allowed_extensions = {'.txt', '.json', '.jsonl', '.csv'}
        ext = os.path.splitext(filename)[1].lower()
        if ext not in allowed_extensions:
            filename += '.jsonl'

        datasets_dir = "./datasets"
        os.makedirs(datasets_dir, exist_ok=True)
        filepath = os.path.join(datasets_dir, filename)

        try:
            logger.info(f"Downloading dataset from {url}")
            urllib.request.urlretrieve(url, filepath)

            # Get file info
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            return jsonify({
                "status": "downloaded",
                "name": filename,
                "path": filepath,
                "size": len(content),
                "lines": len(content.splitlines()),
                "source": url
            }), 201

        except urllib.error.URLError as e:
            return jsonify({"error": f"Failed to download: {str(e)}"}), 400
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/models/download-hf', methods=['POST'])
    def download_model_from_hf() -> tuple[Response, int]:
        """Download a model from HuggingFace Hub."""
        data = request.get_json()
        if not data or "model_id" not in data:
            return jsonify({"error": "model_id is required"}), 400

        model_id = data["model_id"]
        output_dir = data.get("output_dir", f"./models/{model_id.replace('/', '_')}")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info(f"Downloading model {model_id} from HuggingFace")

            # Download tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            tokenizer.save_pretrained(output_dir)

            # Download model (just config and weights info, not full weights yet)
            # This caches it for later use
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype="auto",
                device_map="auto",
                low_cpu_mem_usage=True
            )
            model.save_pretrained(output_dir)

            return jsonify({
                "status": "downloaded",
                "model_id": model_id,
                "output_dir": output_dir,
                "message": f"Model {model_id} downloaded successfully"
            }), 200

        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/models/list')
    def list_local_models() -> Response:
        """List locally available models."""
        models_dir = "./models"
        trained_dir = "./trained_models"

        models = []

        for dir_path in [models_dir, trained_dir]:
            if os.path.exists(dir_path):
                for name in os.listdir(dir_path):
                    full_path = os.path.join(dir_path, name)
                    if os.path.isdir(full_path):
                        # Check if it's a valid model directory
                        has_config = os.path.exists(os.path.join(full_path, "config.json"))
                        has_adapter = os.path.exists(os.path.join(full_path, "adapter_config.json"))

                        if has_config or has_adapter:
                            models.append({
                                "name": name,
                                "path": full_path,
                                "type": "adapter" if has_adapter else "full",
                                "size": sum(
                                    os.path.getsize(os.path.join(full_path, f))
                                    for f in os.listdir(full_path)
                                    if os.path.isfile(os.path.join(full_path, f))
                                )
                            })

        return jsonify({"models": models})

    @app.route('/api/models/package', methods=['POST'])
    def package_model_for_download() -> tuple[Response, int]:
        """Package a trained model as a zip for easy download."""
        import zipfile
        import shutil

        data = request.get_json()
        if not data or "model_path" not in data:
            return jsonify({"error": "model_path is required"}), 400

        model_path = data["model_path"]

        if not os.path.exists(model_path):
            return jsonify({"error": "Model path does not exist"}), 404

        try:
            # Create descriptive zip file name
            base_name = os.path.basename(model_path.rstrip('/'))
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            # Try to extract model name from path or use base name
            model_name = base_name.replace(' ', '_').replace('/', '_')
            zip_name = f"{model_name}_{timestamp}.zip"
            zip_path = os.path.join("./exports", zip_name)
            os.makedirs("./exports", exist_ok=True)

            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(model_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, model_path)
                        zipf.write(file_path, arcname)

            zip_size = os.path.getsize(zip_path)

            return jsonify({
                "status": "packaged",
                "zip_path": zip_path,
                "zip_name": zip_name,
                "size_bytes": zip_size,
                "size_mb": round(zip_size / (1024 * 1024), 2),
                "download_url": f"/exports/{zip_name}"
            }), 200

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/exports/<path:filename>')
    def download_export(filename: str) -> Response:
        """Serve exported files for download."""
        from flask import send_from_directory
        return send_from_directory('./exports', filename, as_attachment=True)

    # =========================================================================
    # TRAINING HISTORY ENDPOINTS
    # =========================================================================

    @app.route('/api/history')
    def get_training_history() -> Response:
        """Get training run history."""
        output_dir = "./output"
        runs = []

        if os.path.exists(output_dir):
            for run_dir in os.listdir(output_dir):
                run_path = os.path.join(output_dir, run_dir)
                if os.path.isdir(run_path):
                    # Check for training info
                    info = {
                        "id": run_dir,
                        "path": run_path,
                        "has_final": os.path.exists(os.path.join(run_path, "final")),
                        "checkpoints": []
                    }

                    # Find checkpoints
                    for item in os.listdir(run_path):
                        if item.startswith("checkpoint-"):
                            info["checkpoints"].append(item)

                    # Get timestamp from directory name
                    try:
                        info["timestamp"] = datetime.strptime(run_dir, "%Y%m%d_%H%M%S").isoformat()
                    except ValueError:
                        info["timestamp"] = None

                    runs.append(info)

        return jsonify({"runs": sorted(runs, key=lambda x: x["id"], reverse=True)})

    @app.route('/api/history/<run_id>/delete', methods=['DELETE'])
    def delete_training_run(run_id: str) -> tuple[Response, int]:
        """Delete a training run."""
        import shutil

        run_id = os.path.basename(run_id)
        run_path = os.path.join("./output", run_id)

        if not os.path.exists(run_path):
            return jsonify({"error": "Training run not found"}), 404

        try:
            shutil.rmtree(run_path)
            return jsonify({"status": "deleted", "run_id": run_id}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # =========================================================================
    # INFERENCE ENDPOINTS
    # =========================================================================

    @app.route('/api/inference/models')
    def get_inference_models() -> Response:
        """Get list of available trained models for inference."""
        models = inference_manager.get_available_models()
        loaded_model = inference_manager.get_loaded_model()
        return jsonify({
            "models": models,
            "loaded_model": loaded_model
        })

    @app.route('/api/inference/load', methods=['POST'])
    def load_inference_model() -> tuple[Response, int]:
        """Load a model for inference."""
        data = request.get_json()
        if not data or "model_path" not in data:
            return jsonify({"error": "model_path is required"}), 400

        model_path = data["model_path"]

        # Validate path is within allowed directories (check multiple case variations)
        abs_path = os.path.abspath(model_path)
        allowed_dirs = [
            os.path.abspath("./output"), os.path.abspath("./Output"), os.path.abspath("./Outputs"),
            os.path.abspath("./trained_models"), os.path.abspath("./Trained_Models"),
            os.path.abspath("./models"), os.path.abspath("./Models"),
        ]

        is_valid_path = any(abs_path.startswith(d) for d in allowed_dirs if os.path.exists(d))
        if not is_valid_path:
            return jsonify({"error": "Invalid model path. Must be in output/, trained_models/, or models/"}), 400

        if not os.path.isdir(abs_path):
            return jsonify({"error": "Model path does not exist"}), 404

        try:
            inference_manager.load_model(abs_path)
            return jsonify({
                "status": "loaded",
                "model_path": abs_path,
                "message": "Model loaded successfully"
            }), 200
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return jsonify({"error": f"Failed to load model: {str(e)}"}), 500

    @app.route('/api/inference/unload', methods=['POST'])
    def unload_inference_model() -> Response:
        """Unload the current inference model."""
        inference_manager.unload_model()
        return jsonify({"status": "unloaded", "message": "Model unloaded"})

    @app.route('/api/inference/generate', methods=['POST'])
    def generate_text() -> tuple[Response, int]:
        """Generate text using the loaded model."""
        if inference_manager.get_loaded_model() is None:
            return jsonify({"error": "No model loaded. Load a model first."}), 400

        data = request.get_json()
        if not data or "prompt" not in data:
            return jsonify({"error": "prompt is required"}), 400

        prompt = data.get("prompt", "")
        if not prompt.strip():
            return jsonify({"error": "Prompt cannot be empty"}), 400

        # Validate and sanitize parameters
        max_new_tokens = min(500, max(1, int(data.get("max_new_tokens", 100))))
        temperature = min(2.0, max(0.1, float(data.get("temperature", 0.7))))
        top_p = min(1.0, max(0.1, float(data.get("top_p", 0.9))))
        top_k = min(100, max(1, int(data.get("top_k", 50))))
        repetition_penalty = min(2.0, max(1.0, float(data.get("repetition_penalty", 1.1))))
        do_sample = bool(data.get("do_sample", True))

        try:
            generated = inference_manager.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                num_return_sequences=1
            )

            return jsonify({
                "generated_text": generated[0] if generated else "",
                "prompt": prompt,
                "parameters": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "repetition_penalty": repetition_penalty,
                    "do_sample": do_sample
                }
            }), 200

        except Exception as e:
            logger.error(f"Generation error: {e}")
            return jsonify({"error": f"Generation failed: {str(e)}"}), 500

    @app.route('/api/inference/status')
    def get_inference_status() -> Response:
        """Get current inference status."""
        loaded_model = inference_manager.get_loaded_model()
        return jsonify({
            "model_loaded": loaded_model is not None,
            "loaded_model_path": loaded_model,
            "available_models_count": len(inference_manager.get_available_models())
        })

    # =========================================================================
    # MODEL EXPORT ENDPOINTS
    # =========================================================================

    @app.route('/api/export/gguf', methods=['POST'])
    def export_to_gguf() -> tuple[Response, int]:
        """Export model to GGUF format for llama.cpp."""
        data = request.get_json()
        if not data or "model_path" not in data:
            return jsonify({"error": "model_path is required"}), 400

        model_path = data["model_path"]
        quantization = data.get("quantization", "q4_k_m")

        # Validate path
        abs_path = os.path.abspath(model_path)
        output_abs = os.path.abspath("./output")
        if not abs_path.startswith(output_abs):
            return jsonify({"error": "Invalid model path"}), 400

        if not os.path.isdir(abs_path):
            return jsonify({"error": "Model path does not exist"}), 404

        # Check for llama.cpp convert script
        try:
            import subprocess
            result = subprocess.run(
                ["which", "python3"],
                capture_output=True,
                text=True
            )

            # For now, return instructions since GGUF conversion requires llama.cpp
            return jsonify({
                "status": "instructions",
                "message": "GGUF export requires llama.cpp. Here's how to do it manually:",
                "steps": [
                    "1. Clone llama.cpp: git clone https://github.com/ggerganov/llama.cpp",
                    "2. Install requirements: pip install -r llama.cpp/requirements.txt",
                    f"3. Convert: python llama.cpp/convert.py {abs_path} --outtype f16",
                    f"4. Quantize: ./llama.cpp/quantize {abs_path}/ggml-model-f16.gguf {abs_path}/model-{quantization}.gguf {quantization}"
                ],
                "model_path": abs_path,
                "quantization": quantization
            }), 200

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/export/merge-lora', methods=['POST'])
    def merge_lora_adapter() -> tuple[Response, int]:
        """Merge LoRA adapter with base model."""
        data = request.get_json()
        if not data or "adapter_path" not in data:
            return jsonify({"error": "adapter_path is required"}), 400

        adapter_path = data["adapter_path"]

        # Validate path
        abs_path = os.path.abspath(adapter_path)
        output_abs = os.path.abspath("./output")
        if not abs_path.startswith(output_abs):
            return jsonify({"error": "Invalid adapter path"}), 400

        if not os.path.isdir(abs_path):
            return jsonify({"error": "Adapter path does not exist"}), 404

        # Check if it's a LoRA model
        adapter_config = os.path.join(abs_path, "adapter_config.json")
        if not os.path.exists(adapter_config):
            return jsonify({"error": "Not a LoRA adapter (no adapter_config.json)"}), 400

        try:
            import torch
            from peft import PeftModel, PeftConfig
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info(f"Merging LoRA adapter from {abs_path}")

            # Load config to get base model
            peft_config = PeftConfig.from_pretrained(abs_path)
            base_model_name = peft_config.base_model_name_or_path

            # Load base model and adapter
            base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
            model = PeftModel.from_pretrained(base_model, abs_path)

            # Merge and unload
            merged_model = model.merge_and_unload()

            # Save merged model
            merged_path = abs_path + "_merged"
            merged_model.save_pretrained(merged_path)

            # Also save tokenizer
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            tokenizer.save_pretrained(merged_path)

            return jsonify({
                "status": "merged",
                "merged_path": merged_path,
                "base_model": base_model_name,
                "message": f"LoRA adapter merged successfully. Saved to {merged_path}"
            }), 200

        except ImportError as e:
            return jsonify({"error": f"Missing dependency: {e}. Install with: pip install peft"}), 500
        except Exception as e:
            logger.error(f"Merge error: {e}")
            return jsonify({"error": str(e)}), 500

    # =========================================================================
    # CHECKPOINT RESUME ENDPOINTS
    # =========================================================================

    @app.route('/api/checkpoints')
    def get_checkpoints() -> Response:
        """Get all available checkpoints for resuming."""
        checkpoints = []
        output_dir = "./output"

        if os.path.exists(output_dir):
            for run_dir in sorted(os.listdir(output_dir), reverse=True):
                run_path = os.path.join(output_dir, run_dir)
                if not os.path.isdir(run_path):
                    continue

                # Find checkpoints in this run
                for item in os.listdir(run_path):
                    if item.startswith("checkpoint-"):
                        checkpoint_path = os.path.join(run_path, item)
                        if os.path.isdir(checkpoint_path):
                            # Try to load config
                            config_path = os.path.join(run_path, "training_config.json")
                            config = None
                            if os.path.exists(config_path):
                                try:
                                    with open(config_path, 'r') as f:
                                        config = json.load(f)
                                except Exception:
                                    pass

                            checkpoints.append({
                                "id": f"{run_dir}/{item}",
                                "path": checkpoint_path,
                                "run_id": run_dir,
                                "checkpoint_name": item,
                                "step": int(item.split("-")[1]) if "-" in item else 0,
                                "has_config": config is not None,
                                "config": config
                            })

        return jsonify({"checkpoints": checkpoints})

    @app.route('/api/checkpoints/<path:checkpoint_id>/config')
    def get_checkpoint_config(checkpoint_id: str) -> tuple[Response, int]:
        """Get the training config for a checkpoint."""
        # Extract run_id from checkpoint_id (format: run_id/checkpoint-xxx)
        parts = checkpoint_id.split("/")
        if len(parts) < 2:
            return jsonify({"error": "Invalid checkpoint ID"}), 400

        run_id = parts[0]
        config_path = os.path.join("./output", run_id, "training_config.json")

        if not os.path.exists(config_path):
            return jsonify({"error": "No config found for this checkpoint"}), 404

        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return jsonify(config), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/train/resume', methods=['POST'])
    def resume_training() -> tuple[Response, int]:
        """Resume training from a checkpoint."""
        if state_manager.is_training():
            return jsonify({"error": "Training already in progress"}), 409

        data = request.get_json()
        if not data or "checkpoint_path" not in data:
            return jsonify({"error": "checkpoint_path is required"}), 400

        checkpoint_path = data["checkpoint_path"]

        # Validate path
        abs_path = os.path.abspath(checkpoint_path)
        output_abs = os.path.abspath("./output")
        if not abs_path.startswith(output_abs):
            return jsonify({"error": "Invalid checkpoint path"}), 400

        if not os.path.isdir(abs_path):
            return jsonify({"error": "Checkpoint path does not exist"}), 404

        # Get config overrides
        config_overrides = data.get("config_overrides", {})

        # Reset state
        state_manager.reset()

        # Start resume training in background
        thread = threading.Thread(
            target=run_resume_training,
            args=(abs_path, config_overrides, state_manager, config, logger),
            daemon=True
        )
        thread.start()

        return jsonify({"status": "resuming", "checkpoint": abs_path}), 202

    # =========================================================================
    # FORMAT PREVIEW & DETECTION ENDPOINTS
    # =========================================================================

    @app.route('/api/format/preview', methods=['POST'])
    def preview_format() -> tuple[Response, int]:
        """Preview how data will be formatted."""
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "text is required"}), 400

        text = data["text"]
        data_format = data.get("format", "completion")
        max_examples = data.get("max_examples", 3)

        # Format the data
        formatted = format_training_data(text, data_format)

        # Return first N examples
        preview = formatted[:max_examples]

        return jsonify({
            "formatted_examples": preview,
            "total_examples": len(formatted),
            "format": data_format
        }), 200

    @app.route('/api/format/detect', methods=['POST'])
    def detect_format() -> tuple[Response, int]:
        """Auto-detect the format of training data."""
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "text is required"}), 400

        text = data["text"]
        detected = auto_detect_format(text)

        return jsonify(detected), 200

    # =========================================================================
    # MEMORY ESTIMATION ENDPOINT
    # =========================================================================

    @app.route('/api/memory/estimate', methods=['POST'])
    def estimate_memory() -> tuple[Response, int]:
        """Estimate VRAM usage for different configurations."""
        data = request.get_json()
        model_name = data.get("model_name", "gpt2") if data else "gpt2"
        batch_size = data.get("batch_size", 4) if data else 4
        max_length = data.get("max_length", 128) if data else 128

        estimates = calculate_memory_estimate(model_name, batch_size, max_length)
        return jsonify(estimates), 200

    # =========================================================================
    # MODEL ARCHITECTURE INFO ENDPOINT
    # =========================================================================

    @app.route('/api/model/info', methods=['POST'])
    def get_model_info() -> tuple[Response, int]:
        """Get architecture information for a model."""
        data = request.get_json()
        model_name = data.get("model_name", "gpt2") if data else "gpt2"

        info = get_model_architecture_info(model_name)
        return jsonify(info), 200

    @app.route('/api/export/huggingface', methods=['POST'])
    def export_to_huggingface() -> tuple[Response, int]:
        """Get instructions for uploading to HuggingFace Hub."""
        import shlex

        data = request.get_json()
        if not data or "model_path" not in data:
            return jsonify({"error": "model_path is required"}), 400

        model_path = data["model_path"]
        repo_name = data.get("repo_name", "my-fine-tuned-model")

        abs_path = os.path.abspath(model_path)

        # Sanitize values for safe display in shell commands
        safe_repo_name = shlex.quote(repo_name)
        safe_abs_path = shlex.quote(abs_path)

        return jsonify({
            "status": "instructions",
            "message": "To upload to HuggingFace Hub:",
            "steps": [
                "1. Install huggingface_hub: pip install huggingface_hub",
                "2. Login: huggingface-cli login",
                f"3. Upload: huggingface-cli upload {safe_repo_name} {safe_abs_path} --repo-type model",
                "Or use Python:",
                "```python",
                "from huggingface_hub import HfApi",
                "api = HfApi()",
                f"api.upload_folder(folder_path={safe_abs_path}, repo_id='your-username/{safe_repo_name}', repo_type='model')",
                "```"
            ],
            "model_path": abs_path,
            "repo_name": repo_name
        }), 200

    # =========================================================================
    # MULTI-CHANNEL ENCODING ROUTES
    # =========================================================================

    @app.route('/api/multichannel/encode', methods=['POST'])
    def multichannel_encode() -> tuple[Response, int]:
        """Generate multi-channel encoded training data."""
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        messages = data.get("messages", [])
        instructions = data.get("instructions", [])

        if not messages:
            return jsonify({"error": "No messages to encode"}), 400

        # Import encoder
        try:
            from multichannel_training import (
                encode_message, decode_letters, generate_paragraph,
                WORD_BANK, ENCODING_CONTENT
            )
        except ImportError:
            return jsonify({"error": "multichannel_training module not found"}), 500

        examples = []
        default_instructions = [
            "Explain how multi-channel encoding works.",
            "Describe the structure of hidden channels in text.",
            "What makes multi-channel training data effective?",
            "How do 5-bit encodings map letters to bytes?",
        ]

        for i, msg in enumerate(messages):
            instruction = instructions[i] if i < len(instructions) else default_instructions[i % len(default_instructions)]

            # Get required first letters
            required_letters = encode_message(msg)

            # Generate paragraphs
            paragraphs = []
            for j, letter in enumerate(required_letters):
                starters = WORD_BANK.get(letter.upper(), ['The pattern shows'])
                starter = starters[j % len(starters)] if isinstance(starters, list) else starters
                content = ENCODING_CONTENT[j % len(ENCODING_CONTENT)]
                para = f"{starter}, {content}."
                paragraphs.append(para)

            response = "\n\n".join(paragraphs)

            examples.append({
                "instruction": instruction,
                "input": "",
                "output": response,
                "_encoded": msg,
                "_letters": ''.join(required_letters)
            })

        return jsonify({
            "status": "success",
            "count": len(examples),
            "examples": examples
        }), 200

    @app.route('/api/multichannel/decode', methods=['POST'])
    def multichannel_decode() -> tuple[Response, int]:
        """Decode multi-channel encoded text to extract hidden message."""
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "No text provided"}), 400

        text = data["text"]

        try:
            from multichannel_training import decode_letters
        except ImportError:
            return jsonify({"error": "multichannel_training module not found"}), 500

        # Extract first letters from paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if not paragraphs:
            paragraphs = [p.strip() for p in text.split('\n') if p.strip()]

        first_letters = [p[0].upper() for p in paragraphs if p]
        decoded = decode_letters(first_letters)

        return jsonify({
            "status": "success",
            "first_letters": ''.join(first_letters),
            "decoded_message": decoded,
            "paragraph_count": len(paragraphs)
        }), 200

    @app.route('/api/multichannel/generate-dataset', methods=['POST'])
    def multichannel_generate_dataset() -> tuple[Response, int]:
        """Generate a full multi-channel training dataset."""
        data = request.get_json()

        # Default curriculum about encoding itself
        default_curriculum = [
            ("5BIT", "Explain how the 5-bit letter encoding system works."),
            ("BYTES", "Describe how letters convert to bytes."),
            ("LAYER", "What are the different encoding layers?"),
            ("MULTI", "How does multi-channel encoding work?"),
            ("TRAIN", "How does this affect model training?"),
            ("LEARN", "What do models learn from structured data?"),
            ("CODE", "How does bytecode emerge from text?"),
            ("TEST", "How can we verify encoding transfer?"),
        ]

        curriculum = data.get("curriculum", default_curriculum) if data else default_curriculum
        output_filename = data.get("filename", "multichannel_training.jsonl") if data else "multichannel_training.jsonl"

        try:
            from multichannel_training import (
                encode_message, WORD_BANK, ENCODING_CONTENT
            )
        except ImportError:
            return jsonify({"error": "multichannel_training module not found"}), 500

        examples = []
        for msg, instruction in curriculum:
            required_letters = encode_message(msg)

            paragraphs = []
            for j, letter in enumerate(required_letters):
                starters = WORD_BANK.get(letter.upper(), ['The pattern shows'])
                starter = starters[j % len(starters)] if isinstance(starters, list) else starters
                content = ENCODING_CONTENT[j % len(ENCODING_CONTENT)]
                para = f"{starter}, {content}."
                paragraphs.append(para)

            response = "\n\n".join(paragraphs)

            examples.append({
                "instruction": instruction,
                "input": "",
                "output": response
            })

        # Save to data folder
        output_path = os.path.join("data", output_filename)
        os.makedirs("data", exist_ok=True)

        with open(output_path, 'w') as f:
            for ex in examples:
                f.write(json.dumps(ex) + '\n')

        return jsonify({
            "status": "success",
            "count": len(examples),
            "output_path": output_path,
            "message": f"Generated {len(examples)} multi-channel training examples"
        }), 200


def register_socketio_events(socketio: SocketIO, state_manager: TrainingStateManager) -> None:
    """Register WebSocket event handlers."""

    @socketio.on('connect')
    def handle_connect() -> None:
        """Handle client connection."""
        emit('connected', {'status': 'connected'})
        # Send current state to newly connected client
        emit('training_update', state_manager.get_status())

    @socketio.on('request_status')
    def handle_status_request() -> None:
        """Handle status request from client."""
        emit('training_update', state_manager.get_status())


# =============================================================================
# DATA PROCESSING ROUTES
# =============================================================================

def add_data_processing_routes(app: Flask, state_manager: TrainingStateManager, validator: InputValidator, config: Config, logger: logging.Logger) -> None:
    """Add comprehensive data processing routes to the Flask app."""

    @app.route('/api/data/preview', methods=['POST'])
    def preview_data():
        """Preview first N examples of training data with formatting."""
        try:
            data = request.get_json()
            text = data.get("text", "")
            n_examples = data.get("n_examples", 5)
            data_format = data.get("format", "completion")

            if not text.strip():
                return jsonify({"error": "No data to preview"}), 400

            # Format and preview
            formatted = format_training_data(text, data_format)
            preview_examples = formatted[:n_examples]

            # Calculate statistics
            lines = text.split('\n')
            words = text.split()
            unique_words = len(set(words))
            avg_line_length = sum(len(line) for line in lines) / max(1, len(lines))

            # Check for duplicates
            duplicate_lines = len(lines) - len(set(lines))

            # Encoding warnings
            warnings = []
            try:
                text.encode('utf-8')
            except UnicodeEncodeError as e:
                warnings.append(f"Encoding issue detected: {str(e)}")

            if duplicate_lines > 0:
                warnings.append(f"Found {duplicate_lines} duplicate lines")

            if avg_line_length < 10:
                warnings.append("Very short lines detected - may need more content")

            if len(formatted) < 10:
                warnings.append("Very few examples - consider adding more training data")

            return jsonify({
                "preview": preview_examples,
                "total_examples": len(formatted),
                "statistics": {
                    "total_lines": len(lines),
                    "total_words": len(words),
                    "unique_words": unique_words,
                    "avg_line_length": round(avg_line_length, 1),
                    "duplicate_lines": duplicate_lines,
                    "estimated_tokens": len(text) // 4
                },
                "warnings": warnings
            }), 200

        except Exception as e:
            logger.error(f"Data preview error: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/data/validate', methods=['POST'])
    def validate_data():
        """Validate training data and provide detailed feedback."""
        try:
            data = request.get_json()
            text = data.get("text", "")
            data_format = data.get("format", "completion")

            issues = []
            suggestions = []

            if not text.strip():
                issues.append("No training data provided")

            lines = [l for l in text.split('\n') if l.strip()]

            if len(lines) < 10:
                issues.append(f"Only {len(lines)} lines of data - recommend at least 100")
                suggestions.append("Add more diverse training examples")

            unique_lines = set(lines)
            if len(unique_lines) < len(lines):
                duplicate_count = len(lines) - len(unique_lines)
                issues.append(f"{duplicate_count} duplicate lines found")
                suggestions.append("Remove duplicates or add more variety")

            try:
                text.encode('utf-8')
            except UnicodeEncodeError:
                issues.append("Non-UTF8 characters detected")
                suggestions.append("Clean text encoding or remove special characters")

            if data_format == "instruction" or data_format == "chat":
                if len(lines) % 2 != 0:
                    issues.append("Odd number of lines - instruction/response should be paired")
                    suggestions.append("Ensure each instruction has a corresponding response")

            token_lengths = [len(line.split()) for line in lines]
            avg_len = 0
            min_len = 0
            max_len = 0
            if token_lengths:
                min_len = min(token_lengths)
                max_len = max(token_lengths)
                avg_len = sum(token_lengths) / len(token_lengths)

                if max_len > 512:
                    issues.append(f"Very long examples detected ({max_len} tokens)")
                    suggestions.append("Consider splitting long examples or increasing max_length")

                if min_len < 3:
                    issues.append("Very short examples detected")
                    suggestions.append("Ensure examples have meaningful content")

            if len(unique_lines) < 50:
                suggestions.append("Consider data augmentation:")
                suggestions.append("- Shuffle sentence order in examples")
                suggestions.append("- Use synonym replacement")
                suggestions.append("- Add slight variations of existing examples")

            severity = "ok"
            if len(issues) > 3:
                severity = "critical"
            elif len(issues) > 0:
                severity = "warning"

            return jsonify({
                "status": severity,
                "issues": issues,
                "suggestions": suggestions,
                "stats": {
                    "total_lines": len(lines),
                    "unique_lines": len(unique_lines),
                    "avg_tokens_per_line": round(avg_len, 1) if token_lengths else 0,
                    "min_tokens": min_len if token_lengths else 0,
                    "max_tokens": max_len if token_lengths else 0
                }
            }), 200

        except Exception as e:
            logger.error(f"Data validation error: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/data/augment', methods=['POST'])
    def augment_data():
        """Simple data augmentation techniques."""
        try:
            data = request.get_json()
            text = data.get("text", "")
            augmentation_type = data.get("type", "shuffle")
            strength = data.get("strength", 0.3)

            lines = [l for l in text.split('\n') if l.strip()]
            augmented = []

            if augmentation_type == "shuffle":
                for line in lines:
                    sentences = line.split('. ')
                    if len(sentences) > 1 and random.random() < strength:
                        random.shuffle(sentences)
                        augmented.append('. '.join(sentences))
                    else:
                        augmented.append(line)

            elif augmentation_type == "synonym":
                synonyms = {
                    # Quality
                    "good": ["great", "excellent", "fine", "nice", "wonderful", "fantastic"],
                    "bad": ["poor", "terrible", "awful", "negative", "horrible", "dreadful"],
                    "best": ["greatest", "finest", "top", "optimal", "ideal", "perfect"],
                    "worst": ["poorest", "lowest", "terrible", "awful", "bottom"],
                    # Size
                    "big": ["large", "huge", "enormous", "vast", "massive", "substantial"],
                    "small": ["tiny", "little", "mini", "petite", "compact", "minor"],
                    "large": ["big", "huge", "enormous", "substantial", "sizable"],
                    "tiny": ["small", "little", "minuscule", "minute", "microscopic"],
                    # Speed
                    "fast": ["quick", "rapid", "swift", "speedy", "hasty", "prompt"],
                    "slow": ["sluggish", "gradual", "leisurely", "unhurried", "delayed"],
                    "quick": ["fast", "rapid", "swift", "speedy", "prompt", "instant"],
                    # Emotion
                    "happy": ["glad", "joyful", "pleased", "delighted", "cheerful", "content"],
                    "sad": ["unhappy", "sorrowful", "down", "depressed", "melancholy", "gloomy"],
                    "angry": ["mad", "furious", "upset", "annoyed", "irritated", "enraged"],
                    # Actions
                    "make": ["create", "build", "produce", "generate", "construct", "form"],
                    "get": ["obtain", "acquire", "receive", "fetch", "retrieve", "gain"],
                    "use": ["utilize", "employ", "apply", "leverage", "operate"],
                    "help": ["assist", "aid", "support", "facilitate", "guide"],
                    "show": ["display", "present", "demonstrate", "reveal", "exhibit"],
                    "give": ["provide", "offer", "supply", "deliver", "grant"],
                    "take": ["grab", "seize", "acquire", "obtain", "remove"],
                    "start": ["begin", "initiate", "launch", "commence", "kick off"],
                    "stop": ["halt", "cease", "end", "terminate", "discontinue"],
                    "run": ["execute", "operate", "perform", "function", "work"],
                    "work": ["function", "operate", "perform", "run", "labor"],
                    # Descriptors
                    "new": ["fresh", "recent", "modern", "novel", "latest", "updated"],
                    "old": ["ancient", "aged", "previous", "former", "outdated"],
                    "important": ["significant", "crucial", "vital", "essential", "key"],
                    "easy": ["simple", "straightforward", "effortless", "basic", "trivial"],
                    "hard": ["difficult", "challenging", "tough", "complex", "demanding"],
                    "different": ["distinct", "unique", "various", "diverse", "alternative"],
                    "same": ["identical", "equal", "equivalent", "similar", "matching"],
                    # Common verbs
                    "think": ["believe", "consider", "suppose", "assume", "reckon"],
                    "know": ["understand", "recognize", "realize", "comprehend", "grasp"],
                    "want": ["desire", "wish", "need", "require", "seek"],
                    "need": ["require", "demand", "want", "necessitate"],
                    "try": ["attempt", "endeavor", "strive", "aim", "seek"],
                    "look": ["appear", "seem", "glance", "observe", "view"],
                    "see": ["observe", "notice", "view", "witness", "spot"],
                    "say": ["state", "mention", "express", "declare", "assert"],
                    "tell": ["inform", "notify", "advise", "explain", "describe"],
                    # Tech terms
                    "error": ["bug", "issue", "problem", "fault", "defect", "glitch"],
                    "fix": ["repair", "resolve", "correct", "patch", "remedy"],
                    "data": ["information", "content", "records", "input", "values"],
                    "code": ["program", "script", "software", "logic", "instructions"],
                    "test": ["check", "verify", "validate", "examine", "evaluate"],
                    "update": ["modify", "change", "revise", "alter", "refresh"],
                }

                for line in lines:
                    if random.random() < strength:
                        words = line.split()
                        for i, word in enumerate(words):
                            word_lower = word.lower().strip('.,!?')
                            if word_lower in synonyms and random.random() < 0.3:
                                replacement = random.choice(synonyms[word_lower])
                                words[i] = word.replace(word_lower, replacement)
                        augmented.append(' '.join(words))
                    else:
                        augmented.append(line)

            elif augmentation_type == "duplicate_vary":
                for line in lines:
                    augmented.append(line)
                    if random.random() < strength:
                        variations = [
                            line.replace('.', '!'),
                            line.replace(',', ' and'),
                            "In other words, " + line.lower(),
                            line + " This is important.",
                        ]
                        augmented.append(random.choice(variations))

            else:
                augmented = lines

            combined = lines + augmented
            random.shuffle(combined)

            return jsonify({
                "augmented_text": '\n'.join(combined),
                "original_count": len(lines),
                "augmented_count": len(combined),
                "augmentation_type": augmentation_type
            }), 200

        except Exception as e:
            logger.error(f"Data augmentation error: {e}")
            return jsonify({"error": str(e)}), 500


# =============================================================================
# MONITORING ROUTES
# =============================================================================

def add_monitoring_routes(app: Flask, state_manager: TrainingStateManager, config: Config, logger: logging.Logger) -> None:
    """Add comprehensive monitoring endpoints."""

    @app.route('/api/monitoring/gpu')
    def get_gpu_monitoring():
        """Get detailed GPU stats during training."""
        try:
            import torch

            if not torch.cuda.is_available():
                return jsonify({"error": "No GPU available"}), 400

            try:
                import nvidia_ml_py3 as nvml
                nvml.nvmlInit()
                handle = nvml.nvmlDeviceGetHandleByIndex(0)

                temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                power = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                util = nvml.nvmlDeviceGetUtilizationRates(handle)
                mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)

                processes = nvml.nvmlDeviceGetComputeRunningProcesses(handle)
                process_info = []
                for p in processes:
                    try:
                        process_info.append({
                            "pid": p.pid,
                            "memory_mb": p.usedGpuMemory / 1024**2,
                        })
                    except:
                        pass

                return jsonify({
                    "temperature": f"{temp}°C",
                    "power_draw": f"{power:.1f}W",
                    "gpu_utilization": f"{util.gpu}%",
                    "memory_utilization": f"{util.memory}%",
                    "memory_used": f"{mem_info.used / 1024**3:.2f} GB",
                    "memory_total": f"{mem_info.total / 1024**3:.2f} GB",
                    "memory_free": f"{mem_info.free / 1024**3:.2f} GB",
                    "processes": process_info,
                    "cuda_version": torch.version.cuda,
                }), 200

            except Exception as e:
                return jsonify({
                    "memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024**3:.2f} GB",
                    "memory_reserved": f"{torch.cuda.memory_reserved(0) / 1024**3:.2f} GB",
                    "memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB",
                    "error": "nvidia-ml-py not available for detailed stats"
                }), 200

        except ImportError:
            return jsonify({"error": "PyTorch not installed"}), 400

    @app.route('/api/monitoring/training-speed')
    def get_training_speed():
        """Get training speed metrics."""
        status = state_manager.get_status()

        if not status["is_training"]:
            return jsonify({"error": "Not currently training"}), 400

        if status["started_at"]:
            start = datetime.fromisoformat(status["started_at"])
            elapsed = (datetime.now() - start).total_seconds()

            current_step = status.get("current_step", 0)
            total_steps = status.get("total_steps", 1)

            if elapsed > 0 and current_step > 0:
                steps_per_sec = current_step / elapsed
                samples_per_sec = steps_per_sec * 4

                remaining_steps = total_steps - current_step
                eta_seconds = remaining_steps / max(0.001, steps_per_sec)

                hours = int(eta_seconds // 3600)
                minutes = int((eta_seconds % 3600) // 60)
                seconds = int(eta_seconds % 60)

                return jsonify({
                    "steps_per_second": round(steps_per_sec, 2),
                    "samples_per_second": round(samples_per_sec, 2),
                    "elapsed_time": f"{int(elapsed)}s",
                    "estimated_time_remaining": f"{hours}h {minutes}m {seconds}s",
                    "current_step": current_step,
                    "total_steps": total_steps,
                    "progress_percent": round(100 * current_step / total_steps, 1)
                }), 200

        return jsonify({"error": "Training metrics not available"}), 400

    @app.route('/api/monitoring/perplexity', methods=['POST'])
    def calculate_perplexity():
        """Calculate perplexity on provided text using a loaded model."""
        try:
            data = request.get_json()
            text = data.get("text", "")
            model_path = data.get("model_path", None)

            if not text.strip():
                return jsonify({"error": "No text provided"}), 400

            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import math

            # Use global inference manager if model is loaded, otherwise load specified model
            if model_path:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForCausalLM.from_pretrained(model_path)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model.to(device)
            elif inference_manager.get_loaded_model():
                model = inference_manager._model
                tokenizer = inference_manager._tokenizer
                device = inference_manager._device
            else:
                return jsonify({"error": "No model loaded. Load a model first."}), 400

            model.eval()

            # Tokenize the text
            encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            input_ids = encodings.input_ids.to(device)

            # Calculate perplexity
            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss.item()
                perplexity = math.exp(loss)

            # Calculate token-level stats
            num_tokens = input_ids.size(1)

            return jsonify({
                "perplexity": round(perplexity, 4),
                "loss": round(loss, 6),
                "num_tokens": num_tokens,
                "interpretation": (
                    "Excellent" if perplexity < 10 else
                    "Good" if perplexity < 50 else
                    "Moderate" if perplexity < 100 else
                    "Poor" if perplexity < 500 else
                    "Very Poor"
                ),
                "note": "Lower perplexity indicates the model is more confident/accurate on this text"
            }), 200

        except Exception as e:
            logger.error(f"Perplexity calculation error: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/monitoring/loss-spikes')
    def detect_loss_spikes():
        """Detect and report loss spikes from recent training."""
        try:
            # Get recent loss data from training runs
            spikes = []
            spike_threshold = float(request.args.get("threshold", 1.5))  # 50% increase

            for run_id, run_data in _training_runs_data.items():
                loss_data = run_data.get("loss_data", [])
                if len(loss_data) < 3:
                    continue

                # Look for spikes (sudden increases)
                for i in range(2, len(loss_data)):
                    prev_avg = (loss_data[i-1].get("loss", 0) + loss_data[i-2].get("loss", 0)) / 2
                    current = loss_data[i].get("loss", 0)

                    if prev_avg > 0 and current > prev_avg * spike_threshold:
                        spikes.append({
                            "run_id": run_id,
                            "step": loss_data[i].get("step", i),
                            "previous_avg_loss": round(prev_avg, 4),
                            "spike_loss": round(current, 4),
                            "increase_percent": round((current - prev_avg) / prev_avg * 100, 1),
                            "severity": "high" if current > prev_avg * 2 else "medium"
                        })

            return jsonify({
                "spikes_detected": len(spikes),
                "threshold_used": spike_threshold,
                "spikes": spikes[:20],  # Limit to recent 20
                "recommendations": [
                    "Loss spikes often indicate learning rate is too high",
                    "Try reducing learning rate or increasing warmup steps",
                    "Check for data quality issues (very long or short sequences)",
                    "Consider gradient clipping (max_grad_norm)"
                ] if spikes else []
            }), 200

        except Exception as e:
            logger.error(f"Loss spike detection error: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/monitoring/lr-finder', methods=['POST'])
    def run_lr_finder():
        """Run automatic learning rate range test."""
        try:
            data = request.get_json() or {}
            model_path = data.get("model_path")
            num_iterations = int(data.get("num_iterations", 100))
            start_lr = float(data.get("start_lr", 1e-7))
            end_lr = float(data.get("end_lr", 10))

            if not model_path:
                return jsonify({"error": "model_path required"}), 400

            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from torch.utils.data import DataLoader, TensorDataset

            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Load model
            model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model.to(device)
            model.train()

            # Create dummy data for LR test
            dummy_text = ["The quick brown fox"] * 32
            encodings = tokenizer(dummy_text, return_tensors="pt", padding=True, truncation=True, max_length=64)
            dataset = TensorDataset(encodings.input_ids, encodings.attention_mask)
            loader = DataLoader(dataset, batch_size=4, shuffle=True)

            # Run LR range test
            optimizer = torch.optim.AdamW(model.parameters(), lr=start_lr)
            
            import numpy as np
            lr_schedule = np.logspace(np.log10(start_lr), np.log10(end_lr), num_iterations)
            
            lrs = []
            losses = []
            smoothed_loss = 0
            best_loss = float('inf')
            data_iter = iter(loader)

            for i, lr in enumerate(lr_schedule):
                if i >= num_iterations:
                    break

                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(loader)
                    batch = next(data_iter)

                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                input_ids = batch[0].to(device)
                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                smoothed_loss = 0.98 * smoothed_loss + 0.02 * loss_val if i > 0 else loss_val

                lrs.append(float(lr))
                losses.append(float(smoothed_loss))

                if smoothed_loss > 4 * best_loss or np.isnan(smoothed_loss):
                    break
                if smoothed_loss < best_loss:
                    best_loss = smoothed_loss

            # Find suggested LR (steepest descent)
            suggested_lr = start_lr * 10
            if len(losses) > 10:
                gradients = np.gradient(losses)
                search_range = len(gradients) // 2
                if search_range > 0:
                    min_idx = int(np.argmin(gradients[:search_range]))
                    suggested_idx = max(0, min_idx - len(lrs) // 10)
                    suggested_lr = lrs[suggested_idx]

            # Cleanup
            del model, optimizer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            return jsonify({
                "lrs": lrs,
                "losses": losses,
                "suggested_lr": suggested_lr,
                "iterations_run": len(lrs),
                "note": "Use a learning rate slightly before the steepest descent point"
            }), 200

        except Exception as e:
            logger.error(f"LR finder error: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/monitoring/packing-efficiency', methods=['POST'])
    def calculate_packing_efficiency():
        """Calculate sequence packing efficiency for the dataset."""
        try:
            data = request.get_json() or {}
            text = data.get("text", "")
            max_length = int(data.get("max_length", 512))

            if not text.strip():
                return jsonify({"error": "No text provided"}), 400

            lines = [l.strip() for l in text.split('\n') if l.strip()]
            if not lines:
                return jsonify({"error": "No valid lines"}), 400

            # Estimate token counts (rough: ~4 chars per token)
            token_counts = [len(line) // 4 + 1 for line in lines]

            # Without packing: each example padded to max_length
            total_without_packing = len(lines) * max_length
            actual_tokens = sum(min(tc, max_length) for tc in token_counts)
            padding_waste = total_without_packing - actual_tokens

            # With packing: combine sequences up to max_length
            packed_sequences = 0
            current_packed = 0
            for tc in token_counts:
                tc = min(tc, max_length)
                if current_packed + tc <= max_length:
                    current_packed += tc
                else:
                    packed_sequences += 1
                    current_packed = tc
            if current_packed > 0:
                packed_sequences += 1

            total_with_packing = packed_sequences * max_length
            packing_savings = total_without_packing - total_with_packing

            efficiency = (actual_tokens / total_without_packing) * 100 if total_without_packing > 0 else 0
            packing_efficiency = (actual_tokens / total_with_packing) * 100 if total_with_packing > 0 else 0

            return jsonify({
                "num_examples": len(lines),
                "total_tokens": actual_tokens,
                "avg_tokens_per_example": round(actual_tokens / len(lines), 1),
                "max_length": max_length,
                "without_packing": {
                    "total_padded_tokens": total_without_packing,
                    "padding_waste": padding_waste,
                    "efficiency_percent": round(efficiency, 1)
                },
                "with_packing": {
                    "packed_sequences": packed_sequences,
                    "total_padded_tokens": total_with_packing,
                    "efficiency_percent": round(packing_efficiency, 1)
                },
                "packing_benefit": {
                    "sequences_saved": len(lines) - packed_sequences,
                    "tokens_saved": packing_savings,
                    "speedup_estimate": f"{len(lines) / packed_sequences:.1f}x" if packed_sequences > 0 else "N/A"
                }
            }), 200

        except Exception as e:
            logger.error(f"Packing efficiency error: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/monitoring/cost-estimate')
    def estimate_training_cost():
        """Estimate cloud training cost based on current usage."""
        try:
            gpu_type = request.args.get("gpu_type", "a100")
            region = request.args.get("region", "us-east-1")

            gpu_costs = {
                "a100": 3.67,
                "v100": 3.06,
                "t4": 0.526,
                "a10g": 1.006,
                "rtx3090": 1.50,
            }

            cost_per_hour = gpu_costs.get(gpu_type.lower(), 2.0)

            status = state_manager.get_status()
            if status["is_training"] and status["started_at"]:
                start = datetime.fromisoformat(status["started_at"])
                elapsed_hours = (datetime.now() - start).total_seconds() / 3600

                current_cost = elapsed_hours * cost_per_hour

                progress = status.get("progress", 0)
                if progress > 0:
                    total_hours_estimate = elapsed_hours * (100 / progress)
                    total_cost_estimate = total_hours_estimate * cost_per_hour
                else:
                    total_hours_estimate = 0
                    total_cost_estimate = 0

                return jsonify({
                    "gpu_type": gpu_type,
                    "cost_per_hour": f"${cost_per_hour:.2f}",
                    "elapsed_hours": round(elapsed_hours, 2),
                    "current_cost": f"${current_cost:.2f}",
                    "estimated_total_hours": round(total_hours_estimate, 2),
                    "estimated_total_cost": f"${total_cost_estimate:.2f}",
                    "region": region,
                    "note": "Estimates based on typical cloud GPU pricing"
                }), 200

            return jsonify({
                "gpu_type": gpu_type,
                "cost_per_hour": f"${cost_per_hour:.2f}",
                "status": "Not currently training",
                "note": "Start training to see cost estimates"
            }), 200

        except Exception as e:
            logger.error(f"Cost estimation error: {e}")
            return jsonify({"error": str(e)}), 500


# =============================================================================
# ADVANCED INFERENCE ROUTES
# =============================================================================

# Store chat histories per session
_chat_histories: dict = {}

def add_advanced_inference_routes(app: Flask, logger: logging.Logger) -> None:
    """Add advanced inference features like chat mode and batch generation."""

    @app.route('/api/inference/chat', methods=['POST'])
    def chat_mode():
        """Multi-turn chat with conversation history."""
        try:
            data = request.get_json()
            session_id = data.get("session_id", "default")
            message = data.get("message", "")
            clear_history = data.get("clear_history", False)

            if not message.strip():
                return jsonify({"error": "Message cannot be empty"}), 400

            inference_manager = app.config.get("INFERENCE_MANAGER")
            if not inference_manager or not inference_manager.isModelLoaded:
                return jsonify({"error": "No model loaded"}), 400

            if clear_history or session_id not in _chat_histories:
                _chat_histories[session_id] = []

            history = _chat_histories[session_id]
            history.append({"role": "user", "content": message})

            prompt = ""
            for turn in history[-10:]:
                if turn["role"] == "user":
                    prompt += f"User: {turn['content']}\n"
                else:
                    prompt += f"Assistant: {turn['content']}\n"
            prompt += "Assistant: "

            response = inference_manager.generate(
                prompt=prompt,
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )

            if response and response[0]:
                assistant_response = response[0].replace(prompt, "").strip()
                history.append({"role": "assistant", "content": assistant_response})

                if len(history) > 50:
                    history = history[-40:]
                    _chat_histories[session_id] = history

                return jsonify({
                    "response": assistant_response,
                    "session_id": session_id,
                    "turn_count": len(history) // 2,
                    "history": history[-6:]
                }), 200
            else:
                return jsonify({"error": "Failed to generate response"}), 500

        except Exception as e:
            logger.error(f"Chat mode error: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/inference/batch', methods=['POST'])
    def batch_generate():
        """Generate responses for multiple prompts at once."""
        try:
            data = request.get_json()
            prompts = data.get("prompts", [])
            params = data.get("params", {})

            if not prompts:
                return jsonify({"error": "No prompts provided"}), 400

            inference_manager = app.config.get("INFERENCE_MANAGER")
            if not inference_manager or not inference_manager.isModelLoaded:
                return jsonify({"error": "No model loaded"}), 400

            results = []
            for i, prompt in enumerate(prompts[:10]):
                try:
                    response = inference_manager.generate(
                        prompt=prompt,
                        max_new_tokens=params.get("max_new_tokens", 100),
                        temperature=params.get("temperature", 0.7),
                        top_p=params.get("top_p", 0.9),
                        do_sample=params.get("do_sample", True)
                    )

                    results.append({
                        "index": i,
                        "prompt": prompt,
                        "response": response[0] if response else "",
                        "status": "success"
                    })
                except Exception as e:
                    results.append({
                        "index": i,
                        "prompt": prompt,
                        "response": "",
                        "status": "error",
                        "error": str(e)
                    })

            return jsonify({
                "results": results,
                "total": len(prompts),
                "processed": len(results),
                "parameters": params
            }), 200

        except Exception as e:
            logger.error(f"Batch generation error: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/inference/presets')
    def get_inference_presets():
        """Get sampling presets for different generation styles."""
        presets = {
            "creative": {
                "name": "Creative",
                "description": "High creativity, more surprising outputs",
                "temperature": 1.2,
                "top_p": 0.95,
                "top_k": 100,
                "repetition_penalty": 1.2,
                "do_sample": True
            },
            "balanced": {
                "name": "Balanced",
                "description": "Good balance of creativity and coherence",
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.1,
                "do_sample": True
            },
            "precise": {
                "name": "Precise",
                "description": "More predictable, coherent outputs",
                "temperature": 0.3,
                "top_p": 0.7,
                "top_k": 20,
                "repetition_penalty": 1.05,
                "do_sample": True
            },
            "deterministic": {
                "name": "Deterministic",
                "description": "Same output every time (greedy decoding)",
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": 0,
                "repetition_penalty": 1.0,
                "do_sample": False
            }
        }
        return jsonify(presets), 200

    @app.route('/api/inference/compare', methods=['POST'])
    def compare_outputs():
        """Generate outputs from multiple models/settings for comparison."""
        try:
            data = request.get_json()
            prompt = data.get("prompt", "")
            configs = data.get("configs", [])

            if not prompt.strip():
                return jsonify({"error": "Prompt cannot be empty"}), 400

            inference_manager = app.config.get("INFERENCE_MANAGER")
            if not inference_manager or not inference_manager.isModelLoaded:
                return jsonify({"error": "No model loaded"}), 400

            if not configs:
                configs = [
                    {"name": "Conservative", "temperature": 0.3},
                    {"name": "Balanced", "temperature": 0.7},
                    {"name": "Creative", "temperature": 1.2}
                ]

            results = []
            for config in configs[:5]:
                try:
                    response = inference_manager.generate(
                        prompt=prompt,
                        temperature=config.get("temperature", 0.7),
                        max_new_tokens=config.get("max_new_tokens", 100),
                        top_p=config.get("top_p", 0.9),
                        do_sample=config.get("do_sample", True)
                    )

                    results.append({
                        "name": config.get("name", "Unknown"),
                        "config": config,
                        "output": response[0] if response else "",
                        "status": "success"
                    })
                except Exception as e:
                    results.append({
                        "name": config.get("name", "Unknown"),
                        "config": config,
                        "output": "",
                        "status": "error",
                        "error": str(e)
                    })

            return jsonify({
                "prompt": prompt,
                "comparisons": results,
                "count": len(results)
            }), 200

        except Exception as e:
            logger.error(f"Comparison error: {e}")
            return jsonify({"error": str(e)}), 500


# =============================================================================
# CHART COMPARISON ROUTES
# =============================================================================

# Store training run data
_training_runs_data: dict = {}

def add_chart_comparison_routes(app: Flask, logger: logging.Logger) -> None:
    """Add routes for comparing multiple training runs on the same chart."""

    @app.route('/api/charts/save-run', methods=['POST'])
    def save_training_run_data():
        """Save loss data from a training run for later comparison."""
        try:
            data = request.get_json()
            run_id = data.get("run_id")
            loss_data = data.get("loss_data", [])
            metadata = data.get("metadata", {})

            if not run_id:
                return jsonify({"error": "run_id required"}), 400

            _training_runs_data[run_id] = {
                "loss_data": loss_data,
                "metadata": metadata,
                "timestamp": datetime.now().isoformat()
            }

            return jsonify({
                "status": "saved",
                "run_id": run_id,
                "data_points": len(loss_data)
            }), 200

        except Exception as e:
            logger.error(f"Save run data error: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/charts/compare', methods=['POST'])
    def compare_training_runs():
        """Generate comparison chart for multiple training runs."""
        try:
            data = request.get_json()
            run_ids = data.get("run_ids", [])
            chart_type = data.get("chart_type", "line")

            if not run_ids or len(run_ids) < 2:
                return jsonify({"error": "Need at least 2 runs to compare"}), 400

            comparison_data = []
            for run_id in run_ids[:5]:
                if run_id in _training_runs_data:
                    comparison_data.append({
                        "id": run_id,
                        "data": _training_runs_data[run_id]["loss_data"],
                        "metadata": _training_runs_data[run_id]["metadata"]
                    })

            if len(comparison_data) < 2:
                return jsonify({"error": "Not enough valid run data"}), 400

            fig, ax = plt.subplots(figsize=(10, 6))

            colors = ['blue', 'red', 'green', 'orange', 'purple']
            for i, run in enumerate(comparison_data):
                steps = [d.get("step", idx) for idx, d in enumerate(run["data"])]
                losses = [d.get("loss", 0) for d in run["data"]]

                label = run["metadata"].get("name", run["id"])
                color = colors[i % len(colors)]

                if chart_type == "scatter":
                    ax.scatter(steps, losses, label=label, color=color, alpha=0.6, s=10)
                else:
                    ax.plot(steps, losses, label=label, color=color, linewidth=2)

            ax.set_xlabel('Step')
            ax.set_ylabel('Loss')
            ax.set_title('Training Run Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)

            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            chart_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()

            return jsonify({
                "chart": f"data:image/png;base64,{chart_base64}",
                "runs_compared": len(comparison_data),
                "chart_type": chart_type
            }), 200

        except Exception as e:
            logger.error(f"Chart comparison error: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/charts/export/<run_id>')
    def export_chart_data(run_id):
        """Export loss data as CSV or JSON."""
        format_type = request.args.get("format", "json")

        if run_id not in _training_runs_data:
            return jsonify({"error": "Run not found"}), 404

        run_data = _training_runs_data[run_id]

        if format_type == "csv":
            import csv
            import io

            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=["step", "loss", "eval_loss", "learning_rate"])
            writer.writeheader()

            for point in run_data["loss_data"]:
                writer.writerow({
                    "step": point.get("step", ""),
                    "loss": point.get("loss", ""),
                    "eval_loss": point.get("eval_loss", ""),
                    "learning_rate": point.get("learning_rate", "")
                })

            response = Response(output.getvalue(), mimetype='text/csv')
            # Sanitize run_id to prevent header injection
            safe_run_id = re.sub(r'[^\w\-]', '_', run_id)
            response.headers["Content-Disposition"] = f"attachment; filename={safe_run_id}_loss_data.csv"
            return response

        else:
            return jsonify({
                "run_id": run_id,
                "metadata": run_data["metadata"],
                "loss_data": run_data["loss_data"],
                "timestamp": run_data["timestamp"]
            }), 200

    @app.route('/api/charts/list')
    def list_saved_runs():
        """List all saved training runs."""
        runs = []
        for run_id, data in _training_runs_data.items():
            runs.append({
                "run_id": run_id,
                "name": data["metadata"].get("name", run_id),
                "timestamp": data["timestamp"],
                "data_points": len(data["loss_data"])
            })
        return jsonify({"runs": runs}), 200

    @app.route('/api/logs/export')
    def export_training_logs():
        """Export training logs as JSON or CSV."""
        format_type = request.args.get("format", "json").lower()
        run_id = request.args.get("run_id", None)

        # Get logs from state manager
        logs = state_manager.logs if hasattr(state_manager, 'logs') else []

        # If run_id specified, try to get from saved runs
        if run_id and run_id in _training_runs_data:
            run_data = _training_runs_data[run_id]
            export_data = {
                "run_id": run_id,
                "metadata": run_data.get("metadata", {}),
                "loss_data": run_data.get("loss_data", []),
                "timestamp": run_data.get("timestamp", ""),
                "logs": logs
            }
        else:
            # Export current session logs
            export_data = {
                "run_id": "current_session",
                "logs": [{"message": log.get("message", str(log)), "level": log.get("level", "INFO"),
                         "timestamp": log.get("timestamp", "")} if isinstance(log, dict)
                        else {"message": str(log), "level": "INFO"} for log in logs],
                "timestamp": datetime.now().isoformat()
            }

        if format_type == "csv":
            import csv
            import io

            output = io.StringIO()

            # Export loss data as CSV if available
            if "loss_data" in export_data and export_data["loss_data"]:
                writer = csv.DictWriter(output, fieldnames=["step", "loss", "eval_loss", "learning_rate", "timestamp"])
                writer.writeheader()
                for point in export_data["loss_data"]:
                    writer.writerow({
                        "step": point.get("step", ""),
                        "loss": point.get("loss", ""),
                        "eval_loss": point.get("eval_loss", ""),
                        "learning_rate": point.get("learning_rate", ""),
                        "timestamp": point.get("timestamp", "")
                    })
            else:
                # Export logs as CSV
                writer = csv.DictWriter(output, fieldnames=["timestamp", "level", "message"])
                writer.writeheader()
                for log in export_data.get("logs", []):
                    writer.writerow({
                        "timestamp": log.get("timestamp", ""),
                        "level": log.get("level", "INFO"),
                        "message": log.get("message", "")
                    })

            response = Response(output.getvalue(), mimetype='text/csv')
            # Include run_id in filename if available for better tracking
            run_name = run_id if run_id and run_id != "current_session" else "session"
            filename = f"training_{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            response.headers["Content-Disposition"] = f"attachment; filename={filename}"
            return response

        else:
            return jsonify(export_data), 200


# =============================================================================
# TRAINING LOGIC
# =============================================================================

# Data format templates
DATA_FORMAT_TEMPLATES = {
    "completion": "{text}",
    "instruction": "### Instruction:\n{instruction}\n\n### Response:\n{response}",
    "chat": "<|user|>\n{user}\n<|assistant|>\n{assistant}",
    "qa": "Question: {question}\nAnswer: {answer}"
}

def format_training_data(text: str, data_format: str) -> list[str]:
    """Format training data based on selected format."""
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    if data_format == "completion":
        return lines
    
    formatted = []
    if data_format == "instruction":
        # Try to parse instruction/response pairs
        i = 0
        while i < len(lines):
            if i + 1 < len(lines):
                formatted.append(f"### Instruction:\n{lines[i]}\n\n### Response:\n{lines[i+1]}")
                i += 2
            else:
                formatted.append(lines[i])
                i += 1
    elif data_format == "chat":
        # Parse as user/assistant pairs
        i = 0
        while i < len(lines):
            if i + 1 < len(lines):
                formatted.append(f"<|user|>\n{lines[i]}\n<|assistant|>\n{lines[i+1]}")
                i += 2
            else:
                formatted.append(lines[i])
                i += 1
    elif data_format == "qa":
        # Parse as Q&A pairs
        i = 0
        while i < len(lines):
            if i + 1 < len(lines):
                formatted.append(f"Question: {lines[i]}\nAnswer: {lines[i+1]}")
                i += 2
            else:
                formatted.append(lines[i])
                i += 1
    else:
        formatted = lines
    
    return formatted if formatted else lines


def run_training(
    config: dict[str, Any],
    state_manager: TrainingStateManager,
    app_config: Config,
    logger: logging.Logger
) -> None:
    """Execute the training pipeline with full feature support."""
    try:
        state_manager.add_log("Starting training setup...", LogLevel.INFO)

        # Import heavy dependencies here to avoid slow startup
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            Trainer,
            DataCollatorForLanguageModeling,
            TrainerCallback,
            EarlyStoppingCallback
        )
        from datasets import Dataset

        # Import advanced training features
        try:
            from advanced_training import (
                WeightInitializer,
                AdvancedOptimizerFactory,
                OptimizerConfig,
                LoRATargetModulesManager,
                RSLoRAScaling,
                AutomaticLRFinder,
                MixedPrecisionLossScaler,
                AdvancedSchedulerConfig
            )
            ADVANCED_TRAINING_AVAILABLE = True
        except ImportError:
            ADVANCED_TRAINING_AVAILABLE = False
            state_manager.add_log("Advanced training module not found, using defaults", LogLevel.DEBUG)

        # Check device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        state_manager.add_log(f"Using device: {device}", LogLevel.INFO)

        if device == "cpu":
            state_manager.add_log(
                "No GPU detected! Training will be slow. Consider using Google Colab for free GPU access.",
                LogLevel.WARNING
            )

        # Load model and tokenizer
        model_name = config["model_name"]
        state_manager.add_log(f"Loading model: {model_name}...", LogLevel.INFO)
        state_manager.set_status(TrainingStatus.LOADING_MODEL)

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # QLoRA / Quantization loading
        quantization_config = None
        if config.get("use_qlora", False) and device == "cuda":
            try:
                from transformers import BitsAndBytesConfig
                
                bits = config.get("qlora_bits", 4)
                state_manager.add_log(f"Enabling {bits}-bit quantization (QLoRA)...", LogLevel.INFO)
                
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=(bits == 4),
                    load_in_8bit=(bits == 8),
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=config.get("qlora_double_quant", True),
                )
                
                # Build model kwargs with attention implementation
                model_kwargs = {
                    "quantization_config": quantization_config,
                    "device_map": "auto",
                    "trust_remote_code": True
                }
                attn_impl = config.get("attn_implementation", "auto")
                if attn_impl != "auto":
                    model_kwargs["attn_implementation"] = attn_impl
                    state_manager.add_log(f"Using attention: {attn_impl}", LogLevel.INFO)

                model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
                state_manager.add_log(f"Model loaded in {bits}-bit mode!", LogLevel.SUCCESS)
            except ImportError:
                state_manager.add_log("bitsandbytes not installed, loading model normally", LogLevel.WARNING)
                model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        else:
            # Build model kwargs with attention implementation
            model_kwargs = {"trust_remote_code": True}
            attn_impl = config.get("attn_implementation", "auto")
            if attn_impl != "auto":
                model_kwargs["attn_implementation"] = attn_impl
                state_manager.add_log(f"Using attention: {attn_impl}", LogLevel.INFO)
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        # Apply LoRA if requested (works with both regular and QLoRA)
        if config.get("use_lora", False) or config.get("use_qlora", False):
            state_manager.add_log("Applying LoRA adapters...", LogLevel.INFO)
            try:
                from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
                
                # Prepare model for training if using quantization
                if config.get("use_qlora", False):
                    model = prepare_model_for_kbit_training(model)
                
                # Determine target modules based on model architecture
                # Check for manual override first
                manual_modules = config.get("manual_lora_targets", "").strip()
                if manual_modules and ADVANCED_TRAINING_AVAILABLE:
                    target_modules = LoRATargetModulesManager.parse_manual_modules(manual_modules)
                    state_manager.add_log(f"Using manual LoRA targets: {target_modules}", LogLevel.INFO)
                elif ADVANCED_TRAINING_AVAILABLE:
                    target_modules = LoRATargetModulesManager.get_modules_for_architecture(model_name)
                    state_manager.add_log(f"Auto-detected LoRA targets: {target_modules}", LogLevel.INFO)
                else:
                    # Fallback to hardcoded detection
                    target_modules = None
                    if "gpt2" in model_name.lower():
                        target_modules = ["c_attn", "c_proj"]
                    elif "llama" in model_name.lower() or "mistral" in model_name.lower():
                        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
                    elif "phi" in model_name.lower():
                        target_modules = ["q_proj", "v_proj", "k_proj", "dense"]
                    elif "qwen" in model_name.lower():
                        target_modules = ["c_attn", "c_proj", "w1", "w2"]
                
                # Build LoRA config with DoRA and bias options
                use_dora = config.get("use_dora", False)
                lora_bias = config.get("lora_bias", "none")

                lora_kwargs = {
                    "r": config.get("lora_r", 8),
                    "lora_alpha": config.get("lora_alpha", 32),
                    "target_modules": target_modules,
                    "lora_dropout": config.get("lora_dropout", 0.1),
                    "bias": lora_bias,
                    "task_type": "CAUSAL_LM"
                }

                # Add DoRA if enabled (requires peft >= 0.9.0)
                if use_dora:
                    lora_kwargs["use_dora"] = True
                    state_manager.add_log("DoRA enabled (weight-decomposed LoRA)", LogLevel.INFO)

                lora_config = LoraConfig(**lora_kwargs)
                model = get_peft_model(model, lora_config)

                # Apply RS-LoRA scaling if enabled
                if config.get("use_rs_lora", False) and ADVANCED_TRAINING_AVAILABLE:
                    RSLoRAScaling.apply_rslora_scaling(
                        model,
                        rank=config.get("lora_r", 8),
                        alpha=config.get("lora_alpha", 32)
                    )
                    state_manager.add_log("RS-LoRA scaling applied", LogLevel.INFO)

                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                total = sum(p.numel() for p in model.parameters())
                state_manager.add_log(
                    f"LoRA applied! Training {trainable:,} / {total:,} params ({100*trainable/total:.2f}%)",
                    LogLevel.SUCCESS
                )
            except ImportError:
                state_manager.add_log("PEFT not installed. Install with: pip install peft", LogLevel.WARNING)

        # Apply weight initialization if specified
        init_method = config.get("weight_init_method", "default")
        if init_method != "default" and ADVANCED_TRAINING_AVAILABLE:
            WeightInitializer.initialize_weights(
                model,
                method=init_method,
                init_range=config.get("weight_init_range", 0.02)
            )
            state_manager.add_log(f"Applied weight initialization: {init_method}", LogLevel.INFO)

        # Move to device if not using quantization (quantized models handle this automatically)
        if not config.get("use_qlora", False):
            model.to(device)

        # Prepare dataset
        state_manager.add_log("Preparing dataset...", LogLevel.INFO)
        state_manager.set_status(TrainingStatus.PREPARING_DATA)

        training_text = config.get("training_data", "")
        if not training_text.strip():
            training_text = SAMPLE_TRAINING_DATA
            state_manager.add_log("No training data provided, using sample data for demo", LogLevel.INFO)

        # Format data based on selected format
        data_format = config.get("data_format", "completion")
        lines = format_training_data(training_text, data_format)
        state_manager.add_log(f"Using data format: {data_format}", LogLevel.INFO)

        # Tokenize
        max_length = config["max_length"]

        def tokenize_function(examples: dict) -> dict:
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length,
                padding="max_length"
            )

        dataset = Dataset.from_dict({"text": lines})
        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

        # Split into train/eval if requested
        eval_dataset = None
        eval_split = config.get("eval_split", 0)
        if eval_split > 0:
            stratified = config.get("stratified_split", False)
            if stratified and len(tokenized_dataset) >= 10:
                # Stratified split based on sequence length buckets
                try:
                    import numpy as np
                    from sklearn.model_selection import train_test_split as sklearn_split

                    # Create length-based strata (5 buckets)
                    lengths = [sum(1 for t in ex["input_ids"] if t != tokenizer.pad_token_id)
                              for ex in tokenized_dataset]
                    length_bins = np.digitize(lengths, np.percentile(lengths, [20, 40, 60, 80]))

                    indices = list(range(len(tokenized_dataset)))
                    train_idx, eval_idx = sklearn_split(
                        indices,
                        test_size=eval_split,
                        random_state=config.get("seed", 42),
                        stratify=length_bins
                    )

                    train_data = tokenized_dataset.select(train_idx)
                    eval_data = tokenized_dataset.select(eval_idx)
                    tokenized_dataset = train_data
                    eval_dataset = eval_data
                    state_manager.add_log(
                        f"Stratified split: {len(tokenized_dataset)} train, {len(eval_dataset)} eval",
                        LogLevel.INFO
                    )
                except Exception as e:
                    state_manager.add_log(f"Stratified split failed, using random: {e}", LogLevel.WARNING)
                    split = tokenized_dataset.train_test_split(test_size=eval_split, seed=config.get("seed", 42))
                    tokenized_dataset = split["train"]
                    eval_dataset = split["test"]
            else:
                split = tokenized_dataset.train_test_split(test_size=eval_split, seed=config.get("seed", 42))
                tokenized_dataset = split["train"]
                eval_dataset = split["test"]
                state_manager.add_log(
                    f"Dataset split: {len(tokenized_dataset)} train, {len(eval_dataset)} eval",
                    LogLevel.INFO
                )
        else:
            state_manager.add_log(f"Dataset ready: {len(lines)} examples", LogLevel.SUCCESS)

        # Training arguments
        output_dir = f"./output/{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        epochs = config["epochs"]
        batch_size = config["batch_size"]
        grad_accum = config.get("gradient_accumulation", 1)
        steps_per_epoch = max(1, len(tokenized_dataset) // (batch_size * grad_accum))
        total_steps = steps_per_epoch * epochs

        # Enable gradient checkpointing if requested
        if config.get("gradient_checkpointing", False):
            state_manager.add_log("Enabling gradient checkpointing (saves ~60% memory, ~30% slower)", LogLevel.INFO)
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()

        # Determine precision settings
        mixed_precision = config.get("mixed_precision", "fp16")
        use_fp16 = mixed_precision == "fp16" and device == "cuda"
        use_bf16 = mixed_precision == "bf16" and device == "cuda"

        if use_bf16:
            # Check if BF16 is supported
            import torch
            if not torch.cuda.is_bf16_supported():
                state_manager.add_log("BF16 not supported on this GPU, falling back to FP16", LogLevel.WARNING)
                use_bf16 = False
                use_fp16 = True

        # Get optimizer
        optimizer_name = config.get("optimizer", "adamw_torch")
        state_manager.add_log(f"Optimizer: {optimizer_name}", LogLevel.INFO)

        # Map optimizer names for 8-bit variants
        if optimizer_name in ["adam8bit", "adamw_bnb_8bit"]:
            try:
                import bitsandbytes
                state_manager.add_log("Using 8-bit optimizer (saves ~30% memory)", LogLevel.INFO)
            except ImportError:
                state_manager.add_log("bitsandbytes not installed, falling back to AdamW", LogLevel.WARNING)
                optimizer_name = "adamw_torch"

        # Calculate warmup steps (support both steps and ratio)
        warmup_steps = config.get("warmup_steps", 100)
        warmup_ratio = config.get("warmup_ratio", 0.0)
        if warmup_ratio > 0:
            warmup_steps = int(total_steps * warmup_ratio)
            state_manager.add_log(f"Using warmup ratio {warmup_ratio} = {warmup_steps} steps", LogLevel.INFO)

        # Build training arguments
        training_args_dict = {
            "output_dir": output_dir,
            "num_train_epochs": epochs,
            "per_device_train_batch_size": batch_size,
            "gradient_accumulation_steps": grad_accum,
            "learning_rate": config["learning_rate"],
            "weight_decay": config.get("weight_decay", 0.01),
            "warmup_steps": warmup_steps,
            "logging_steps": config.get("logging_steps", 10),
            "save_steps": config.get("save_steps", 500),
            "fp16": use_fp16,
            "bf16": use_bf16,
            "optim": optimizer_name,
            "seed": config.get("seed", 42),
            "report_to": "none",
            "logging_dir": f"{output_dir}/logs",
            "lr_scheduler_type": config.get("lr_scheduler", "cosine"),
            "max_grad_norm": config.get("max_grad_norm", 1.0),
            "gradient_checkpointing": config.get("gradient_checkpointing", False),
        }

        # Add Adam hyperparameters if using Adam-based optimizer
        if "adam" in optimizer_name.lower():
            training_args_dict["adam_beta1"] = config.get("adam_beta1", 0.9)
            training_args_dict["adam_beta2"] = config.get("adam_beta2", 0.999)
            training_args_dict["adam_epsilon"] = config.get("adam_epsilon", 1e-8)
            if config.get("adam_beta1", 0.9) != 0.9 or config.get("adam_beta2", 0.999) != 0.999:
                state_manager.add_log(
                    f"Custom Adam betas: ({config.get('adam_beta1', 0.9)}, {config.get('adam_beta2', 0.999)})",
                    LogLevel.INFO
                )

        # Add gradient checkpointing kwargs for newer transformers
        if config.get("gradient_checkpointing", False):
            training_args_dict["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

        # Label smoothing
        label_smoothing = config.get("label_smoothing", 0.0)
        if label_smoothing > 0:
            training_args_dict["label_smoothing_factor"] = label_smoothing
            state_manager.add_log(f"Label smoothing: {label_smoothing}", LogLevel.INFO)

        # NEFTune (noise embeddings for better generalization)
        neftune_alpha = config.get("neftune_alpha", 0)
        if neftune_alpha > 0:
            training_args_dict["neftune_noise_alpha"] = neftune_alpha
            state_manager.add_log(f"NEFTune enabled (alpha={neftune_alpha})", LogLevel.INFO)

        # Add eval settings if we have eval data
        if eval_dataset is not None:
            # Handle transformers version compatibility (eval_strategy vs evaluation_strategy)
            try:
                import transformers
                if hasattr(transformers, '__version__') and int(transformers.__version__.split('.')[0]) >= 4 and int(transformers.__version__.split('.')[1]) >= 36:
                    training_args_dict["eval_strategy"] = "steps"
                else:
                    training_args_dict["evaluation_strategy"] = "steps"
            except:
                training_args_dict["eval_strategy"] = "steps"  # Default to newer
            training_args_dict["eval_steps"] = config.get("eval_steps", 100)
            training_args_dict["load_best_model_at_end"] = config.get("early_stopping", False)
            training_args_dict["metric_for_best_model"] = "eval_loss"
            training_args_dict["greater_is_better"] = False

        training_args = TrainingArguments(**training_args_dict)

        # Data collator
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        # Build callbacks list
        callbacks = []
        
        # Custom callback for progress updates and loss tracking
        loss_history = []
        
        class ProgressCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs and not state_manager.should_stop():
                    loss = logs.get("loss", None)
                    eval_loss = logs.get("eval_loss", None)
                    lr = logs.get("learning_rate", None)
                    step = state.global_step
                    state_manager.update_progress(step, total_steps)

                    # Build log message
                    msg_parts = [f"Step {step}/{total_steps}"]
                    if loss is not None:
                        msg_parts.append(f"Loss: {loss:.4f}")
                        loss_history.append({"step": step, "loss": loss, "type": "train"})
                        # Emit loss data for real-time chart
                        state_manager.emit_loss_data(step, loss, eval_loss, lr)
                    if eval_loss is not None:
                        msg_parts.append(f"Eval Loss: {eval_loss:.4f}")
                        loss_history.append({"step": step, "loss": eval_loss, "type": "eval"})
                    if lr is not None:
                        msg_parts.append(f"LR: {lr:.2e}")

                    state_manager.add_log(" | ".join(msg_parts), LogLevel.INFO)

                if state_manager.should_stop():
                    control.should_training_stop = True

        callbacks.append(ProgressCallback())

        # Add early stopping if enabled
        if config.get("early_stopping", False) and eval_dataset is not None:
            callbacks.append(EarlyStoppingCallback(
                early_stopping_patience=config.get("early_stopping_patience", 3),
                early_stopping_threshold=config.get("early_stopping_threshold", 0.01)
            ))
            state_manager.add_log(
                f"Early stopping enabled (patience: {config.get('early_stopping_patience', 3)})",
                LogLevel.INFO
            )

        state_manager.add_log(f"LR Scheduler: {config.get('lr_scheduler', 'cosine')}", LogLevel.INFO)

        # Create custom optimizer if using LoRA+ (different LR for A/B matrices)
        custom_optimizer = None
        is_lora = config.get("use_lora", False) or config.get("use_qlora", False)
        if is_lora and config.get("use_lora_plus", False) and ADVANCED_TRAINING_AVAILABLE:
            state_manager.add_log("Creating LoRA+ optimizer (different LR for A/B matrices)...", LogLevel.INFO)

            opt_config = OptimizerConfig(
                optimizer_type=optimizer_name,
                learning_rate=config["learning_rate"],
                weight_decay=config.get("weight_decay", 0.01),
                adam_beta1=config.get("adam_beta1", 0.9),
                adam_beta2=config.get("adam_beta2", 0.999),
                adam_epsilon=config.get("adam_epsilon", 1e-8),
                lora_plus_enabled=True,
                lora_b_lr_ratio=config.get("lora_b_lr_ratio", 16.0)
            )

            param_groups = AdvancedOptimizerFactory.create_parameter_groups(model, opt_config, is_lora=True)
            custom_optimizer = AdvancedOptimizerFactory.create_optimizer(param_groups, opt_config)

            state_manager.add_log(
                f"LoRA+ enabled: B matrix LR = {config['learning_rate'] * config.get('lora_b_lr_ratio', 16.0):.2e}",
                LogLevel.SUCCESS
            )

        # Create trainer
        state_manager.add_log("Starting training...", LogLevel.INFO)
        state_manager.set_status(TrainingStatus.TRAINING)

        # Use SFTTrainer with packing if enabled, otherwise standard Trainer
        use_packing = config.get("use_sequence_packing", False)

        # Prepare optimizer tuple for Trainer (optimizer, lr_scheduler)
        # When custom_optimizer is provided, we pass it; scheduler is handled by Trainer
        optimizers = (custom_optimizer, None) if custom_optimizer is not None else (None, None)

        if use_packing:
            try:
                from trl import SFTTrainer, SFTConfig
                state_manager.add_log("Sequence packing enabled - using SFTTrainer", LogLevel.INFO)

                # Reconstruct text dataset for SFTTrainer (it handles tokenization)
                text_dataset = Dataset.from_dict({"text": lines})
                if eval_split > 0:
                    text_split = text_dataset.train_test_split(test_size=eval_split, seed=config.get("seed", 42))
                    text_dataset = text_split["train"]
                    eval_text_dataset = text_split["test"]
                else:
                    eval_text_dataset = None

                # SFTConfig for newer TRL versions
                sft_config = SFTConfig(
                    output_dir=output_dir,
                    num_train_epochs=epochs,
                    per_device_train_batch_size=batch_size,
                    gradient_accumulation_steps=grad_accum,
                    learning_rate=config["learning_rate"],
                    weight_decay=config.get("weight_decay", 0.01),
                    warmup_steps=config.get("warmup_steps", 100),
                    logging_steps=config.get("logging_steps", 10),
                    save_steps=config.get("save_steps", 500),
                    fp16=use_fp16,
                    bf16=use_bf16,
                    optim=optimizer_name if custom_optimizer is None else "adamw_torch",
                    seed=config.get("seed", 42),
                    report_to="none",
                    max_seq_length=max_length,
                    packing=True,  # Enable sequence packing
                    dataset_text_field="text",
                )

                trainer = SFTTrainer(
                    model=model,
                    args=sft_config,
                    train_dataset=text_dataset,
                    eval_dataset=eval_text_dataset,
                    processing_class=tokenizer,
                    callbacks=callbacks,
                    optimizers=optimizers,
                )
            except ImportError:
                state_manager.add_log(
                    "TRL not installed - sequence packing unavailable. Install with: pip install trl",
                    LogLevel.WARNING
                )
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=tokenized_dataset,
                    eval_dataset=eval_dataset,
                    data_collator=data_collator,
                    callbacks=callbacks,
                    optimizers=optimizers,
                )
            except Exception as e:
                state_manager.add_log(f"SFTTrainer failed ({e}), falling back to standard Trainer", LogLevel.WARNING)
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=tokenized_dataset,
                    eval_dataset=eval_dataset,
                    data_collator=data_collator,
                    callbacks=callbacks,
                    optimizers=optimizers,
                )
        else:
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                callbacks=callbacks,
                optimizers=optimizers,
            )

        # Train
        trainer.train()

        # Save model
        if not state_manager.should_stop():
            state_manager.add_log("Saving model...", LogLevel.INFO)
            trainer.save_model(f"{output_dir}/final")
            tokenizer.save_pretrained(f"{output_dir}/final")
            
            # Save training config for reproducibility
            import json
            config_to_save = {k: v for k, v in config.items() if k != "training_data"}
            config_to_save["loss_history"] = loss_history
            with open(f"{output_dir}/training_config.json", "w") as f:
                json.dump(config_to_save, f, indent=2)

            state_manager.add_log(f"Training complete! Model saved to {output_dir}/final", LogLevel.SUCCESS)
            state_manager.complete(success=True)
        else:
            state_manager.add_log("Training stopped by user", LogLevel.WARNING)
            state_manager.complete(success=False)

    except Exception as e:
        logger.error(f"Training error: {e}")
        logger.error(traceback.format_exc())
        state_manager.add_log(f"Error: {str(e)}", LogLevel.ERROR)
        state_manager.complete(success=False)


# =============================================================================
# AUTO-DETECT FORMAT
# =============================================================================

def auto_detect_format(text: str) -> dict[str, Any]:
    """Auto-detect the format of training data."""
    lines = [l.strip() for l in text.split('\n') if l.strip()]

    if not lines:
        return {"format": "completion", "confidence": 0, "reason": "No data provided"}

    # Check for common patterns
    instruction_patterns = [
        r'^###\s*(instruction|input)',
        r'^instruction:',
        r'^input:',
        r'^\[INST\]',
    ]

    response_patterns = [
        r'^###\s*(response|output)',
        r'^response:',
        r'^output:',
        r'^\[/INST\]',
    ]

    chat_patterns = [
        r'^(user|human|customer):',
        r'^(assistant|ai|agent|bot):',
        r'^<\|user\|>',
        r'^<\|assistant\|>',
    ]

    qa_patterns = [
        r'^(question|q):',
        r'^(answer|a):',
    ]

    text_lower = text.lower()

    # Count pattern matches
    instruction_count = sum(1 for line in lines if any(re.match(p, line.lower()) for p in instruction_patterns))
    response_count = sum(1 for line in lines if any(re.match(p, line.lower()) for p in response_patterns))
    chat_count = sum(1 for line in lines if any(re.match(p, line.lower()) for p in chat_patterns))
    qa_count = sum(1 for line in lines if any(re.match(p, line.lower()) for p in qa_patterns))

    total_lines = len(lines)

    # Determine format based on pattern density
    if instruction_count + response_count > total_lines * 0.2:
        return {
            "format": "instruction",
            "confidence": min(0.95, (instruction_count + response_count) / total_lines),
            "reason": f"Found {instruction_count} instruction and {response_count} response markers",
            "suggested_settings": {"data_format": "instruction"}
        }

    if chat_count > total_lines * 0.2:
        return {
            "format": "chat",
            "confidence": min(0.95, chat_count / total_lines),
            "reason": f"Found {chat_count} chat markers (user/assistant patterns)",
            "suggested_settings": {"data_format": "chat"}
        }

    if qa_count > total_lines * 0.2:
        return {
            "format": "qa",
            "confidence": min(0.95, qa_count / total_lines),
            "reason": f"Found {qa_count} Q&A markers",
            "suggested_settings": {"data_format": "qa"}
        }

    # Check if it looks like JSON/JSONL
    try:
        json.loads(lines[0])
        return {
            "format": "jsonl",
            "confidence": 0.9,
            "reason": "Data appears to be JSON lines format",
            "suggested_settings": {"data_format": "completion"}
        }
    except (json.JSONDecodeError, IndexError):
        pass

    # Default to completion
    return {
        "format": "completion",
        "confidence": 0.7,
        "reason": "No specific format detected - treating as raw text for completion",
        "suggested_settings": {"data_format": "completion"}
    }


# =============================================================================
# MEMORY ESTIMATION
# =============================================================================

# Model parameter counts (approximate)
MODEL_PARAMS = {
    "distilgpt2": 82_000_000,
    "gpt2": 124_000_000,
    "gpt2-medium": 355_000_000,
    "gpt2-large": 774_000_000,
    "gpt2-xl": 1_500_000_000,
    "facebook/opt-125m": 125_000_000,
    "facebook/opt-350m": 350_000_000,
    "facebook/opt-1.3b": 1_300_000_000,
    "EleutherAI/pythia-160m": 160_000_000,
    "EleutherAI/pythia-410m": 410_000_000,
    "EleutherAI/pythia-1b": 1_000_000_000,
    "microsoft/phi-1_5": 1_300_000_000,
    "microsoft/phi-2": 2_700_000_000,
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": 1_100_000_000,
    "Qwen/Qwen2-0.5B": 500_000_000,
    "Qwen/Qwen2-1.5B": 1_500_000_000,
    "stabilityai/stablelm-2-zephyr-1_6b": 1_600_000_000,
}


def calculate_memory_estimate(model_name: str, batch_size: int = 4, max_length: int = 128) -> dict[str, Any]:
    """Estimate VRAM usage for different precision modes."""
    params = MODEL_PARAMS.get(model_name, 124_000_000)  # Default to GPT-2 size

    # Base memory calculations (parameters only)
    # FP32: 4 bytes per param
    # FP16: 2 bytes per param
    # 8-bit: 1 byte per param
    # 4-bit: 0.5 bytes per param

    fp32_model = params * 4 / (1024**3)  # GB
    fp16_model = params * 2 / (1024**3)
    int8_model = params * 1 / (1024**3)
    int4_model = params * 0.5 / (1024**3)

    # Training overhead (optimizer states, gradients, activations)
    # Roughly 4x for AdamW in FP32, 2-3x for FP16, less for quantized
    training_multiplier_fp32 = 4.0
    training_multiplier_fp16 = 2.5
    training_multiplier_8bit = 1.8
    training_multiplier_4bit = 1.5

    # Activation memory (scales with batch_size * max_length)
    activation_base = (batch_size * max_length * params * 2) / (1024**3) * 0.001  # Rough estimate

    fp32_total = fp32_model * training_multiplier_fp32 + activation_base
    fp16_total = fp16_model * training_multiplier_fp16 + activation_base
    int8_total = int8_model * training_multiplier_8bit + activation_base
    int4_total = int4_model * training_multiplier_4bit + activation_base

    # LoRA reduces trainable params significantly
    lora_reduction = 0.1  # LoRA typically trains ~1-10% of params

    fp16_lora = fp16_model + (fp16_model * training_multiplier_fp16 * lora_reduction) + activation_base
    int4_lora = int4_model + (int4_model * training_multiplier_4bit * lora_reduction) + activation_base

    # Determine recommendations
    recommendations = []
    if int4_lora < 6:
        recommendations.append("4-bit QLoRA: Fits on most gaming GPUs (6GB+)")
    if int4_lora < 4:
        recommendations.append("4-bit QLoRA: Works on budget GPUs (4GB)")
    if fp16_lora < 8:
        recommendations.append("FP16 + LoRA: Good balance of speed and quality")
    if fp32_total > 24:
        recommendations.append("Full fine-tuning requires high-end GPU (A100/H100)")

    savings_vs_fp32 = ((fp32_total - int4_lora) / fp32_total) * 100

    return {
        "model_name": model_name,
        "parameters": params,
        "parameters_human": f"{params / 1e9:.1f}B" if params >= 1e9 else f"{params / 1e6:.0f}M",
        "estimates": {
            "fp32": {"vram_gb": round(fp32_total, 1), "label": "Full Precision (FP32)"},
            "fp16": {"vram_gb": round(fp16_total, 1), "label": "Half Precision (FP16)"},
            "fp16_lora": {"vram_gb": round(fp16_lora, 1), "label": "FP16 + LoRA"},
            "int8": {"vram_gb": round(int8_total, 1), "label": "8-bit Quantized"},
            "int4_qlora": {"vram_gb": round(int4_lora, 1), "label": "4-bit QLoRA"},
        },
        "savings_percent": round(savings_vs_fp32, 0),
        "recommendations": recommendations,
        "batch_size": batch_size,
        "max_length": max_length
    }


# =============================================================================
# MODEL ARCHITECTURE INFO
# =============================================================================

# Known model architectures with metadata
MODEL_ARCHITECTURES = {
    "distilgpt2": {
        "architecture": "GPT-2 (Distilled)",
        "type": "Decoder-only Transformer",
        "layers": 6,
        "hidden_size": 768,
        "attention_heads": 12,
        "vocab_size": 50257,
        "context_length": 1024,
        "release": "2019",
        "creator": "Hugging Face",
    },
    "gpt2": {
        "architecture": "GPT-2 Small",
        "type": "Decoder-only Transformer",
        "layers": 12,
        "hidden_size": 768,
        "attention_heads": 12,
        "vocab_size": 50257,
        "context_length": 1024,
        "release": "2019",
        "creator": "OpenAI",
    },
    "gpt2-medium": {
        "architecture": "GPT-2 Medium",
        "type": "Decoder-only Transformer",
        "layers": 24,
        "hidden_size": 1024,
        "attention_heads": 16,
        "vocab_size": 50257,
        "context_length": 1024,
        "release": "2019",
        "creator": "OpenAI",
    },
    "gpt2-large": {
        "architecture": "GPT-2 Large",
        "type": "Decoder-only Transformer",
        "layers": 36,
        "hidden_size": 1280,
        "attention_heads": 20,
        "vocab_size": 50257,
        "context_length": 1024,
        "release": "2019",
        "creator": "OpenAI",
    },
    "gpt2-xl": {
        "architecture": "GPT-2 XL",
        "type": "Decoder-only Transformer",
        "layers": 48,
        "hidden_size": 1600,
        "attention_heads": 25,
        "vocab_size": 50257,
        "context_length": 1024,
        "release": "2019",
        "creator": "OpenAI",
    },
    "facebook/opt-125m": {
        "architecture": "OPT-125M",
        "type": "Decoder-only Transformer",
        "layers": 12,
        "hidden_size": 768,
        "attention_heads": 12,
        "vocab_size": 50272,
        "context_length": 2048,
        "release": "2022",
        "creator": "Meta AI",
    },
    "facebook/opt-350m": {
        "architecture": "OPT-350M",
        "type": "Decoder-only Transformer",
        "layers": 24,
        "hidden_size": 1024,
        "attention_heads": 16,
        "vocab_size": 50272,
        "context_length": 2048,
        "release": "2022",
        "creator": "Meta AI",
    },
    "facebook/opt-1.3b": {
        "architecture": "OPT-1.3B",
        "type": "Decoder-only Transformer",
        "layers": 24,
        "hidden_size": 2048,
        "attention_heads": 32,
        "vocab_size": 50272,
        "context_length": 2048,
        "release": "2022",
        "creator": "Meta AI",
    },
    "EleutherAI/pythia-160m": {
        "architecture": "Pythia-160M",
        "type": "Decoder-only Transformer (GPT-NeoX)",
        "layers": 12,
        "hidden_size": 768,
        "attention_heads": 12,
        "vocab_size": 50304,
        "context_length": 2048,
        "release": "2023",
        "creator": "EleutherAI",
    },
    "EleutherAI/pythia-410m": {
        "architecture": "Pythia-410M",
        "type": "Decoder-only Transformer (GPT-NeoX)",
        "layers": 24,
        "hidden_size": 1024,
        "attention_heads": 16,
        "vocab_size": 50304,
        "context_length": 2048,
        "release": "2023",
        "creator": "EleutherAI",
    },
    "EleutherAI/pythia-1b": {
        "architecture": "Pythia-1B",
        "type": "Decoder-only Transformer (GPT-NeoX)",
        "layers": 16,
        "hidden_size": 2048,
        "attention_heads": 8,
        "vocab_size": 50304,
        "context_length": 2048,
        "release": "2023",
        "creator": "EleutherAI",
    },
    "microsoft/phi-1_5": {
        "architecture": "Phi-1.5",
        "type": "Decoder-only Transformer",
        "layers": 24,
        "hidden_size": 2048,
        "attention_heads": 32,
        "vocab_size": 51200,
        "context_length": 2048,
        "release": "2023",
        "creator": "Microsoft Research",
    },
    "microsoft/phi-2": {
        "architecture": "Phi-2",
        "type": "Decoder-only Transformer",
        "layers": 32,
        "hidden_size": 2560,
        "attention_heads": 32,
        "vocab_size": 51200,
        "context_length": 2048,
        "release": "2023",
        "creator": "Microsoft Research",
    },
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {
        "architecture": "TinyLlama-1.1B",
        "type": "Decoder-only Transformer (Llama)",
        "layers": 22,
        "hidden_size": 2048,
        "attention_heads": 32,
        "vocab_size": 32000,
        "context_length": 2048,
        "release": "2024",
        "creator": "TinyLlama Team",
    },
    "Qwen/Qwen2-0.5B": {
        "architecture": "Qwen2-0.5B",
        "type": "Decoder-only Transformer",
        "layers": 24,
        "hidden_size": 896,
        "attention_heads": 14,
        "vocab_size": 151936,
        "context_length": 32768,
        "release": "2024",
        "creator": "Alibaba",
    },
    "Qwen/Qwen2-1.5B": {
        "architecture": "Qwen2-1.5B",
        "type": "Decoder-only Transformer",
        "layers": 28,
        "hidden_size": 1536,
        "attention_heads": 12,
        "vocab_size": 151936,
        "context_length": 32768,
        "release": "2024",
        "creator": "Alibaba",
    },
    "stabilityai/stablelm-2-zephyr-1_6b": {
        "architecture": "StableLM-2-Zephyr-1.6B",
        "type": "Decoder-only Transformer",
        "layers": 24,
        "hidden_size": 2048,
        "attention_heads": 32,
        "vocab_size": 100352,
        "context_length": 4096,
        "release": "2024",
        "creator": "Stability AI",
    },
}


def get_model_architecture_info(model_name: str) -> dict[str, Any]:
    """Get architecture information for a model."""
    params = MODEL_PARAMS.get(model_name, None)
    arch_info = MODEL_ARCHITECTURES.get(model_name, None)

    if arch_info:
        return {
            "model_name": model_name,
            "found": True,
            "parameters": params,
            "parameters_human": f"{params / 1e9:.1f}B" if params and params >= 1e9 else f"{params / 1e6:.0f}M" if params else "Unknown",
            **arch_info,
            "lora_targets": get_lora_targets_for_model(model_name),
        }
    else:
        # Unknown model - return basic info
        return {
            "model_name": model_name,
            "found": False,
            "parameters": params,
            "parameters_human": f"{params / 1e9:.1f}B" if params and params >= 1e9 else f"{params / 1e6:.0f}M" if params else "Unknown",
            "architecture": "Unknown (Custom Model)",
            "type": "Unknown",
            "note": "This is a custom or unknown model. Architecture details will be loaded when the model is fetched.",
            "lora_targets": get_lora_targets_for_model(model_name),
        }


def get_lora_targets_for_model(model_name: str) -> list[str]:
    """Get recommended LoRA target modules for a model architecture."""
    # Common target modules by model family
    if "gpt2" in model_name.lower() or "distilgpt2" in model_name.lower():
        return ["c_attn", "c_proj"]
    elif "opt" in model_name.lower():
        return ["q_proj", "v_proj", "k_proj", "out_proj"]
    elif "pythia" in model_name.lower() or "gpt-neox" in model_name.lower():
        return ["query_key_value", "dense"]
    elif "phi" in model_name.lower():
        return ["q_proj", "k_proj", "v_proj", "dense"]
    elif "llama" in model_name.lower() or "tinyllama" in model_name.lower():
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    elif "qwen" in model_name.lower():
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    elif "stablelm" in model_name.lower():
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    else:
        # Default for transformer models
        return ["q_proj", "v_proj"]


# =============================================================================
# RESUME TRAINING
# =============================================================================

def run_resume_training(
    checkpoint_path: str,
    config_overrides: dict[str, Any],
    state_manager: TrainingStateManager,
    app_config: Config,
    logger: logging.Logger
) -> None:
    """Resume training from a checkpoint."""
    try:
        state_manager.add_log(f"Resuming training from {checkpoint_path}...", LogLevel.INFO)

        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            Trainer,
            DataCollatorForLanguageModeling,
            TrainerCallback
        )
        from datasets import Dataset

        device = "cuda" if torch.cuda.is_available() else "cpu"
        state_manager.add_log(f"Using device: {device}", LogLevel.INFO)

        # Load the checkpoint config
        run_dir = os.path.dirname(checkpoint_path)
        config_path = os.path.join(run_dir, "training_config.json")

        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                original_config = json.load(f)
            state_manager.add_log("Loaded original training config", LogLevel.INFO)
        else:
            original_config = {}
            state_manager.add_log("No original config found, using defaults", LogLevel.WARNING)

        # Merge with overrides
        config = {**original_config, **config_overrides}

        # Load model from checkpoint
        state_manager.add_log("Loading model from checkpoint...", LogLevel.INFO)
        state_manager.set_status(TrainingStatus.LOADING_MODEL)

        # Check if it's a PEFT/LoRA checkpoint
        adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            from peft import PeftModel, PeftConfig

            peft_config = PeftConfig.from_pretrained(checkpoint_path)
            base_model = AutoModelForCausalLM.from_pretrained(
                peft_config.base_model_name_or_path,
                trust_remote_code=True
            )
            model = PeftModel.from_pretrained(base_model, checkpoint_path)
            tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
            state_manager.add_log("Loaded LoRA checkpoint", LogLevel.SUCCESS)
        else:
            model = AutoModelForCausalLM.from_pretrained(checkpoint_path, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
            state_manager.add_log("Loaded full model checkpoint", LogLevel.SUCCESS)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model.to(device)

        # Prepare dataset
        state_manager.add_log("Preparing dataset...", LogLevel.INFO)
        state_manager.set_status(TrainingStatus.PREPARING_DATA)

        training_text = config.get("training_data", "")
        if not training_text.strip():
            training_text = SAMPLE_TRAINING_DATA
            state_manager.add_log("No training data in config, using sample", LogLevel.INFO)

        data_format = config.get("data_format", "completion")
        lines = format_training_data(training_text, data_format)

        max_length = config.get("max_length", 128)

        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length,
                padding="max_length"
            )

        dataset = Dataset.from_dict({"text": lines})
        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

        # Training arguments - continue from checkpoint
        additional_epochs = config_overrides.get("additional_epochs", 1)

        training_args = TrainingArguments(
            output_dir=run_dir,
            num_train_epochs=additional_epochs,
            per_device_train_batch_size=config.get("batch_size", 4),
            learning_rate=config.get("learning_rate", 5e-5),
            warmup_steps=0,  # No warmup on resume
            logging_steps=config.get("logging_steps", 10),
            save_steps=config.get("save_steps", 500),
            fp16=config.get("fp16", True) and device == "cuda",
            report_to="none",
            resume_from_checkpoint=checkpoint_path
        )

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        # Progress callback
        total_steps = len(tokenized_dataset) * additional_epochs // config.get("batch_size", 4)

        class ProgressCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs and not state_manager.should_stop():
                    loss = logs.get("loss", None)
                    step = state.global_step
                    state_manager.update_progress(step, total_steps)
                    if loss is not None:
                        state_manager.add_log(f"Step {step} | Loss: {loss:.4f}", LogLevel.INFO)
                if state_manager.should_stop():
                    control.should_training_stop = True

        state_manager.add_log(f"Resuming for {additional_epochs} more epoch(s)...", LogLevel.INFO)
        state_manager.set_status(TrainingStatus.TRAINING)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
            callbacks=[ProgressCallback()]
        )

        trainer.train(resume_from_checkpoint=checkpoint_path)

        if not state_manager.should_stop():
            state_manager.add_log("Saving resumed model...", LogLevel.INFO)
            trainer.save_model(f"{run_dir}/final")
            tokenizer.save_pretrained(f"{run_dir}/final")
            state_manager.add_log(f"Resumed training complete! Model saved to {run_dir}/final", LogLevel.SUCCESS)
            state_manager.complete(success=True)
        else:
            state_manager.add_log("Resume training stopped by user", LogLevel.WARNING)
            state_manager.complete(success=False)

    except Exception as e:
        logger.error(f"Resume training error: {e}")
        logger.error(traceback.format_exc())
        state_manager.add_log(f"Error: {str(e)}", LogLevel.ERROR)
        state_manager.complete(success=False)


# Sample training data for demo
SAMPLE_TRAINING_DATA = """The quick brown fox jumps over the lazy dog.
Machine learning is a subset of artificial intelligence.
Python is a popular programming language for data science.
Neural networks are inspired by the human brain.
Deep learning has revolutionized computer vision and NLP.
Training AI models requires quality data and proper hyperparameters.
Transfer learning allows us to build on pre-trained models.
Fine-tuning adapts a general model to specific tasks."""


# =============================================================================
# INFERENCE MANAGER
# =============================================================================

class InferenceManager:
    """Manages model loading and text generation for inference."""

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._current_model_path: Optional[str] = None
        self._lock = threading.RLock()
        self._logger = logging.getLogger("ai_trainer.inference")
        self._device = None

    def get_available_models(self) -> list[dict[str, Any]]:
        """Discover all trained models in output, trained_models, and models directories."""
        models = []
        # Check multiple possible directory names (case variations + common names)
        search_dirs = [
            "./output", "./Output", "./Outputs",
            "./trained_models", "./Trained_Models",
            "./models", "./Models"
        ]

        for base_dir in search_dirs:
            if not os.path.exists(base_dir):
                continue

            for item in sorted(os.listdir(base_dir), reverse=True):
                item_path = os.path.join(base_dir, item)
                if not os.path.isdir(item_path):
                    continue

                # Check if this directory itself is a valid model
                if self._is_valid_model(item_path):
                    models.append({
                        "id": f"{base_dir}/{item}",
                        "path": os.path.abspath(item_path),
                        "name": item,
                        "timestamp": item if item.startswith("20") else "",
                        "type": "adapter" if os.path.exists(os.path.join(item_path, "adapter_config.json")) else "full",
                        "source_dir": base_dir
                    })

                # Check for nested structure (final, final_merged, checkpoint-*, etc)
                try:
                    for sub_item in os.listdir(item_path):
                        sub_path = os.path.join(item_path, sub_item)
                        if not os.path.isdir(sub_path):
                            continue

                        if self._is_valid_model(sub_path):
                            model_type = "checkpoint" if sub_item.startswith("checkpoint-") else "final"
                            models.append({
                                "id": f"{item}/{sub_item}",
                                "path": os.path.abspath(sub_path),
                                "name": f"{item} - {sub_item}",
                                "timestamp": item if item.startswith("20") else "",
                                "type": model_type,
                                "source_dir": base_dir
                            })
                except PermissionError:
                    pass

        return models

    def _is_valid_model(self, path: str) -> bool:
        """Check if a directory contains a valid model."""
        # Check for essential files
        required_files = ["config.json"]
        for f in required_files:
            if not os.path.exists(os.path.join(path, f)):
                return False

        # Check for model weights (pytorch or safetensors)
        has_weights = (
            os.path.exists(os.path.join(path, "pytorch_model.bin")) or
            os.path.exists(os.path.join(path, "model.safetensors")) or
            any(f.startswith("pytorch_model") for f in os.listdir(path)) or
            any(f.endswith(".safetensors") for f in os.listdir(path))
        )

        return has_weights

    def load_model(self, model_path: str) -> bool:
        """Load a model for inference."""
        with self._lock:
            if self._current_model_path == model_path and self._model is not None:
                return True  # Already loaded

            try:
                self._logger.info(f"Loading model from {model_path}")

                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                self._device = "cuda" if torch.cuda.is_available() else "cpu"

                # Check if it's a PEFT/LoRA model
                adapter_config = os.path.join(model_path, "adapter_config.json")
                if os.path.exists(adapter_config):
                    # Load as PEFT model
                    from peft import PeftModel, PeftConfig

                    peft_config = PeftConfig.from_pretrained(model_path)
                    base_model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path)
                    self._model = PeftModel.from_pretrained(base_model, model_path)
                    self._tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
                else:
                    # Load as regular model
                    self._model = AutoModelForCausalLM.from_pretrained(model_path)
                    self._tokenizer = AutoTokenizer.from_pretrained(model_path)

                if self._tokenizer.pad_token is None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token

                self._model.to(self._device)
                self._model.eval()
                self._current_model_path = model_path

                self._logger.info(f"Model loaded successfully on {self._device}")
                return True

            except Exception as e:
                self._logger.error(f"Failed to load model: {e}")
                self._model = None
                self._tokenizer = None
                self._current_model_path = None
                raise

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        num_return_sequences: int = 1
    ) -> list[str]:
        """Generate text from a prompt."""
        with self._lock:
            if self._model is None or self._tokenizer is None:
                raise RuntimeError("No model loaded. Load a model first.")

            import torch

            # Encode input
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self._device)

            # Generate
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if do_sample else 1.0,
                    top_p=top_p if do_sample else 1.0,
                    top_k=top_k if do_sample else 0,
                    repetition_penalty=repetition_penalty,
                    do_sample=do_sample,
                    num_return_sequences=num_return_sequences,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )

            # Decode outputs
            generated_texts = []
            for output in outputs:
                text = self._tokenizer.decode(output, skip_special_tokens=True)
                generated_texts.append(text)

            return generated_texts

    def unload_model(self) -> None:
        """Unload the current model to free memory."""
        with self._lock:
            if self._model is not None:
                del self._model
                self._model = None

            if self._tokenizer is not None:
                del self._tokenizer
                self._tokenizer = None

            self._current_model_path = None

            # Force garbage collection
            import gc
            gc.collect()

            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

            self._logger.info("Model unloaded")

    def get_loaded_model(self) -> Optional[str]:
        """Get the currently loaded model path."""
        with self._lock:
            return self._current_model_path


# Global inference manager instance
inference_manager = InferenceManager()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

# Create application instance
config = create_config()
logger = setup_logging(config)

app, socketio, state_manager = create_app(config)


def main() -> None:
    """Main entry point."""
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║              AI Training For DumDums v2.0                     ║
    ║                                                               ║
    ║   A beginner-friendly interface for training AI models        ║
    ║   Now with real-time WebSocket updates!                       ║
    ║                                                               ║
    ║   Open http://localhost:5000 in your browser to get started   ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)

    socketio.run(
        app,
        debug=config.DEBUG,
        host=config.HOST,
        port=config.PORT,
        allow_unsafe_werkzeug=config.DEBUG
    )


if __name__ == '__main__':
    main()
