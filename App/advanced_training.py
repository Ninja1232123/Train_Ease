"""
Advanced Training Features for AI Training For DumDums
======================================================
Consolidated module implementing power user features:
- Weight initialization options
- Advanced optimizer controls (betas, eps, momentum)
- Per-parameter-group weight decay
- LoRA+ support (different LR for A/B matrices)
- RSLoRA scaling
- Manual LoRA target modules override
- Loss scaling strategies for mixed precision
- Automatic LR finder
- Advanced scheduler configurations
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field
import logging

logger = logging.getLogger("ai_trainer.advanced")


# =============================================================================
# WEIGHT INITIALIZATION
# =============================================================================

class WeightInitializer:
    """Advanced weight initialization strategies for fine-tuning."""

    INIT_METHODS = {
        "default": "Keep pretrained weights as-is",
        "xavier_uniform": "Xavier/Glorot uniform initialization",
        "xavier_normal": "Xavier/Glorot normal initialization",
        "kaiming_uniform": "He uniform initialization (good for ReLU)",
        "kaiming_normal": "He normal initialization (good for ReLU)",
        "normal": "Normal distribution (mean=0, std=0.02)",
        "uniform": "Uniform distribution [-0.02, 0.02]",
        "truncated_normal": "Truncated normal (within 2 std devs)",
        "orthogonal": "Orthogonal initialization (preserves norms)",
        "sparse": "Sparse initialization (mostly zeros)"
    }

    @staticmethod
    def initialize_weights(
        model: nn.Module,
        method: str = "default",
        modules_to_init: Optional[List[str]] = None,
        init_range: float = 0.02,
        sparsity: float = 0.1
    ) -> None:
        """
        Initialize model weights with specified method.

        Args:
            model: The model to initialize
            method: Initialization method from INIT_METHODS
            modules_to_init: Specific modules to reinitialize (None = common output layers)
            init_range: Range for uniform/normal init
            sparsity: Sparsity factor for sparse init
        """
        if method == "default":
            return

        if modules_to_init is None:
            modules_to_init = ["lm_head", "embed_tokens", "wte", "wpe"]

        initialized_count = 0
        for name, module in model.named_modules():
            should_init = any(target in name for target in modules_to_init)

            if should_init and isinstance(module, (nn.Linear, nn.Embedding)):
                if method == "xavier_uniform":
                    nn.init.xavier_uniform_(module.weight)
                elif method == "xavier_normal":
                    nn.init.xavier_normal_(module.weight)
                elif method == "kaiming_uniform":
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                elif method == "kaiming_normal":
                    nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                elif method == "normal":
                    nn.init.normal_(module.weight, mean=0.0, std=init_range)
                elif method == "uniform":
                    nn.init.uniform_(module.weight, -init_range, init_range)
                elif method == "truncated_normal":
                    with torch.no_grad():
                        module.weight.normal_(0, init_range)
                        module.weight.clamp_(-2*init_range, 2*init_range)
                elif method == "orthogonal":
                    if module.weight.dim() >= 2:
                        nn.init.orthogonal_(module.weight)
                elif method == "sparse":
                    if isinstance(module, nn.Linear):
                        nn.init.sparse_(module.weight, sparsity=sparsity)

                # Initialize biases to zero if present
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)

                initialized_count += 1
                logger.debug(f"Initialized {name} with {method}")

        logger.info(f"Weight initialization ({method}): {initialized_count} modules initialized")


# =============================================================================
# OPTIMIZER CONFIGURATION
# =============================================================================

@dataclass
class OptimizerConfig:
    """Comprehensive optimizer configuration."""

    optimizer_type: str = "adamw_torch"
    learning_rate: float = 5e-5
    weight_decay: float = 0.01

    # Adam/AdamW specific
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8

    # SGD specific
    momentum: float = 0.9
    dampening: float = 0.0
    nesterov: bool = False

    # Adafactor specific
    adafactor_scale_parameter: bool = True
    adafactor_relative_step: bool = False
    adafactor_warmup_init: bool = False
    adafactor_clip_threshold: float = 1.0

    # LoRA+ specific
    lora_plus_enabled: bool = False
    lora_b_lr_ratio: float = 16.0

    # RSLoRA
    rslora_enabled: bool = False

    # Per-parameter group decay
    per_param_weight_decay: Optional[Dict[str, float]] = None

    # Mixed precision loss scaling
    loss_scale_type: str = "dynamic"
    static_loss_scale: float = 128.0

    # Performance options
    foreach_optimizer: bool = False
    fused_optimizer: bool = False

    def to_training_args(self) -> Dict[str, Any]:
        """Convert to TrainingArguments parameters."""
        args = {
            "optim": self.optimizer_type,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
        }

        if "adam" in self.optimizer_type.lower():
            args.update({
                "adam_beta1": self.adam_beta1,
                "adam_beta2": self.adam_beta2,
                "adam_epsilon": self.adam_epsilon,
            })

        return args


# =============================================================================
# ADVANCED OPTIMIZER FACTORY
# =============================================================================

class AdvancedOptimizerFactory:
    """Factory for creating optimizers with advanced configurations."""

    NO_DECAY_PATTERNS = ["bias", "LayerNorm.weight", "layer_norm.weight", "layernorm"]

    @classmethod
    def create_parameter_groups(
        cls,
        model: nn.Module,
        config: OptimizerConfig,
        is_lora: bool = False
    ) -> List[Dict[str, Any]]:
        """Create parameter groups with custom configurations."""

        param_groups = []

        if is_lora and config.lora_plus_enabled:
            # LoRA+ configuration: different LR for A and B matrices
            lora_a_params = []
            lora_b_params = []
            other_params_decay = []
            other_params_no_decay = []

            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue

                if "lora_A" in name:
                    lora_a_params.append(param)
                elif "lora_B" in name:
                    lora_b_params.append(param)
                elif any(nd in name for nd in cls.NO_DECAY_PATTERNS):
                    other_params_no_decay.append(param)
                else:
                    other_params_decay.append(param)

            if lora_a_params:
                param_groups.append({
                    "params": lora_a_params,
                    "lr": config.learning_rate,
                    "weight_decay": config.weight_decay,
                    "group_name": "lora_A"
                })

            if lora_b_params:
                param_groups.append({
                    "params": lora_b_params,
                    "lr": config.learning_rate * config.lora_b_lr_ratio,
                    "weight_decay": config.weight_decay,
                    "group_name": "lora_B"
                })

            if other_params_decay:
                param_groups.append({
                    "params": other_params_decay,
                    "lr": config.learning_rate,
                    "weight_decay": config.weight_decay,
                    "group_name": "other_decay"
                })

            if other_params_no_decay:
                param_groups.append({
                    "params": other_params_no_decay,
                    "lr": config.learning_rate,
                    "weight_decay": 0.0,
                    "group_name": "other_no_decay"
                })

        elif config.per_param_weight_decay:
            # Per-parameter-group weight decay by layer type
            grouped_params: Dict[str, List] = {}

            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue

                # Determine group based on parameter name
                has_decay = not any(nd in name for nd in cls.NO_DECAY_PATTERNS)

                # Subdivide by layer type
                if "attention" in name.lower() or "attn" in name.lower():
                    group_key = "attention_decay" if has_decay else "attention_no_decay"
                elif "mlp" in name.lower() or "fc" in name.lower() or "dense" in name.lower():
                    group_key = "mlp_decay" if has_decay else "mlp_no_decay"
                elif "embed" in name.lower():
                    group_key = "embedding_decay" if has_decay else "embedding_no_decay"
                else:
                    group_key = "other_decay" if has_decay else "other_no_decay"

                if group_key not in grouped_params:
                    grouped_params[group_key] = []
                grouped_params[group_key].append(param)

            for group_name, params in grouped_params.items():
                custom_wd = config.per_param_weight_decay.get(
                    group_name,
                    0.0 if "no_decay" in group_name else config.weight_decay
                )

                param_groups.append({
                    "params": params,
                    "lr": config.learning_rate,
                    "weight_decay": custom_wd,
                    "group_name": group_name
                })
        else:
            # Standard parameter groups
            decay_params = []
            no_decay_params = []

            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue

                if any(nd in name for nd in cls.NO_DECAY_PATTERNS):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

            if decay_params:
                param_groups.append({
                    "params": decay_params,
                    "lr": config.learning_rate,
                    "weight_decay": config.weight_decay,
                    "group_name": "decay"
                })
            if no_decay_params:
                param_groups.append({
                    "params": no_decay_params,
                    "lr": config.learning_rate,
                    "weight_decay": 0.0,
                    "group_name": "no_decay"
                })

        # Log parameter group info
        for group in param_groups:
            num_params = sum(p.numel() for p in group["params"])
            logger.info(
                f"Parameter group '{group.get('group_name', 'unnamed')}': "
                f"{num_params:,} params, LR={group['lr']:.2e}, WD={group['weight_decay']}"
            )

        return param_groups

    @staticmethod
    def create_optimizer(
        param_groups: List[Dict[str, Any]],
        config: OptimizerConfig
    ) -> torch.optim.Optimizer:
        """Create optimizer with specified configuration."""

        optimizer_type = config.optimizer_type.lower()

        # Clean param groups for optimizer (remove custom keys)
        clean_groups = []
        for group in param_groups:
            clean_group = {k: v for k, v in group.items() if k in ["params", "lr", "weight_decay"]}
            clean_groups.append(clean_group)

        if optimizer_type in ["adamw", "adamw_torch", "adamw_hf"]:
            return torch.optim.AdamW(
                clean_groups,
                betas=(config.adam_beta1, config.adam_beta2),
                eps=config.adam_epsilon
            )

        elif optimizer_type == "adam":
            return torch.optim.Adam(
                clean_groups,
                betas=(config.adam_beta1, config.adam_beta2),
                eps=config.adam_epsilon
            )

        elif optimizer_type in ["adamw_bnb_8bit", "adam8bit", "adamw_8bit"]:
            try:
                import bitsandbytes as bnb
                return bnb.optim.AdamW8bit(
                    clean_groups,
                    betas=(config.adam_beta1, config.adam_beta2),
                    eps=config.adam_epsilon
                )
            except ImportError:
                logger.warning("bitsandbytes not installed, falling back to AdamW")
                return torch.optim.AdamW(clean_groups)

        elif optimizer_type == "adafactor":
            try:
                from transformers import Adafactor
                return Adafactor(
                    clean_groups,
                    scale_parameter=config.adafactor_scale_parameter,
                    relative_step=config.adafactor_relative_step,
                    warmup_init=config.adafactor_warmup_init,
                    clip_threshold=config.adafactor_clip_threshold
                )
            except ImportError:
                logger.warning("Adafactor not available, falling back to AdamW")
                return torch.optim.AdamW(clean_groups)

        elif optimizer_type == "sgd":
            return torch.optim.SGD(
                clean_groups,
                momentum=config.momentum,
                dampening=config.dampening,
                nesterov=config.nesterov
            )

        elif optimizer_type == "lion":
            try:
                from lion_pytorch import Lion
                return Lion(clean_groups, betas=(config.adam_beta1, config.adam_beta2))
            except ImportError:
                logger.warning("Lion optimizer not available, falling back to AdamW")
                return torch.optim.AdamW(clean_groups)

        elif optimizer_type == "sophia":
            try:
                from sophia import SophiaG
                return SophiaG(
                    clean_groups,
                    betas=(config.adam_beta1, config.adam_beta2),
                    rho=0.04
                )
            except ImportError:
                logger.warning("Sophia optimizer not available, falling back to AdamW")
                return torch.optim.AdamW(clean_groups)

        else:
            logger.warning(f"Unknown optimizer {optimizer_type}, using AdamW")
            return torch.optim.AdamW(clean_groups)


# =============================================================================
# LORA TARGET MODULES MANAGER
# =============================================================================

class LoRATargetModulesManager:
    """Manage LoRA target modules with manual override support."""

    # Common module patterns by architecture
    ARCHITECTURE_MODULES = {
        "llama": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "mistral": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "phi": ["q_proj", "v_proj", "k_proj", "dense", "fc1", "fc2"],
        "gpt2": ["c_attn", "c_proj", "c_fc"],
        "gptj": ["q_proj", "v_proj", "k_proj", "out_proj", "fc_in", "fc_out"],
        "bloom": ["query", "value", "key", "dense", "dense_h_to_4h", "dense_4h_to_h"],
        "opt": ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
        "qwen": ["q_proj", "v_proj", "k_proj", "o_proj", "w1", "w2", "gate"],
        "gemma": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "stablelm": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "pythia": ["query_key_value", "dense"],
        "tinyllama": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    }

    @staticmethod
    def parse_manual_modules(modules_text: str) -> List[str]:
        """Parse manual module input (comma or newline separated)."""
        modules = []
        for line in modules_text.split('\n'):
            for module in line.split(','):
                module = module.strip()
                if module:
                    modules.append(module)
        return modules

    @staticmethod
    def get_modules_for_architecture(model_name: str) -> List[str]:
        """Get recommended LoRA modules for a model architecture."""
        model_lower = model_name.lower()

        for arch, modules in LoRATargetModulesManager.ARCHITECTURE_MODULES.items():
            if arch in model_lower:
                return modules

        # Default fallback
        return ["q_proj", "v_proj"]

    @staticmethod
    def validate_modules(model: nn.Module, target_modules: List[str]) -> Tuple[List[str], List[str]]:
        """Validate that target modules exist in the model."""
        model_module_names = set()
        for name, _ in model.named_modules():
            if '.' in name:
                model_module_names.add(name.split('.')[-1])
            model_module_names.add(name)

        valid = []
        invalid = []

        for module in target_modules:
            if module in model_module_names or any(module in m for m in model_module_names):
                valid.append(module)
            else:
                invalid.append(module)

        return valid, invalid


# =============================================================================
# RS-LORA SCALING
# =============================================================================

class RSLoRAScaling:
    """Implements Rank-Stabilized LoRA scaling."""

    @staticmethod
    def calculate_scaling_factor(rank: int, alpha: float) -> float:
        """
        Calculate RS-LoRA scaling factor.
        RS-LoRA scales by alpha/sqrt(rank) instead of alpha/rank.
        """
        return alpha / np.sqrt(rank) if rank > 0 else alpha

    @staticmethod
    def apply_rslora_scaling(model: nn.Module, rank: int, alpha: float) -> None:
        """Apply RS-LoRA scaling to LoRA modules in the model."""
        scaling_factor = RSLoRAScaling.calculate_scaling_factor(rank, alpha)

        logger.info(f"Applying RS-LoRA scaling: factor={scaling_factor:.4f} (rank={rank}, alpha={alpha})")

        for name, module in model.named_modules():
            if hasattr(module, "scaling"):
                module.scaling = scaling_factor
                logger.debug(f"Set RS-LoRA scaling for {name}")


# =============================================================================
# AUTOMATIC LR FINDER
# =============================================================================

class AutomaticLRFinder:
    """Implements automatic learning rate range test."""

    @staticmethod
    def find_lr(
        model: nn.Module,
        train_loader,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda",
        start_lr: float = 1e-7,
        end_lr: float = 10,
        num_iter: int = 100,
        smooth_factor: float = 0.98
    ) -> Tuple[List[float], List[float], float]:
        """
        Run LR range test and return optimal LR.

        Returns:
            lrs: List of learning rates tested
            losses: List of smoothed losses
            suggested_lr: Suggested optimal learning rate
        """
        logger.info(f"Running LR range test from {start_lr:.2e} to {end_lr:.2e}")

        # Save initial model state
        initial_state = {k: v.clone() for k, v in model.state_dict().items()}
        initial_optimizer_state = optimizer.state_dict()

        # Initialize tracking
        lrs = []
        losses = []
        smoothed_loss = 0
        best_loss = float('inf')

        # Calculate LR schedule (log scale)
        lr_schedule = np.logspace(np.log10(start_lr), np.log10(end_lr), num_iter)

        model.train()
        data_iter = iter(train_loader)

        for i, lr in enumerate(lr_schedule):
            if i >= num_iter:
                break

            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            # Update LR
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Forward pass
            if isinstance(batch, dict):
                inputs = batch['input_ids'].to(device)
                labels = batch.get('labels', inputs).to(device)
                outputs = model(inputs, labels=labels)
                loss = outputs.loss
            else:
                inputs = batch[0].to(device)
                outputs = model(inputs, labels=inputs)
                loss = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track loss with smoothing
            loss_val = loss.item()
            if i == 0:
                smoothed_loss = loss_val
            else:
                smoothed_loss = smooth_factor * smoothed_loss + (1 - smooth_factor) * loss_val

            lrs.append(lr)
            losses.append(smoothed_loss)

            # Check for divergence
            if smoothed_loss > 4 * best_loss or np.isnan(smoothed_loss):
                logger.info(f"Loss diverged at LR={lr:.2e}, stopping test")
                break

            if smoothed_loss < best_loss:
                best_loss = smoothed_loss

        # Restore model state
        model.load_state_dict(initial_state)
        optimizer.load_state_dict(initial_optimizer_state)

        # Find suggested LR (steepest descent point)
        suggested_lr = start_lr * 10  # Default fallback

        if len(losses) > 10:
            gradients = np.gradient(losses)
            # Look for steepest descent in first half
            search_range = len(gradients) // 2
            if search_range > 0:
                min_grad_idx = np.argmin(gradients[:search_range])
                # Suggested LR is slightly before steepest descent
                suggested_idx = max(0, min_grad_idx - len(lrs) // 10)
                suggested_lr = lrs[suggested_idx]

        logger.info(f"LR range test complete. Suggested LR: {suggested_lr:.2e}")

        return lrs, losses, suggested_lr


# =============================================================================
# MIXED PRECISION LOSS SCALER
# =============================================================================

class MixedPrecisionLossScaler:
    """Advanced loss scaling for mixed precision training."""

    STRATEGIES = {
        "none": "No loss scaling",
        "static": "Fixed loss scale value",
        "dynamic": "Automatic adjustment based on gradient overflow"
    }

    def __init__(
        self,
        scale_type: str = "dynamic",
        initial_scale: float = 2**15,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        min_scale: float = 1.0,
        max_scale: float = 2**24
    ):
        self.scale_type = scale_type
        self.scale = initial_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.min_scale = min_scale
        self.max_scale = max_scale

        self._iter = 0
        self._last_overflow_iter = -1

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale the loss for backward pass."""
        if self.scale_type == "none":
            return loss
        return loss * self.scale

    def update_scale(self, overflow: bool) -> None:
        """Update loss scale based on gradient overflow."""
        if self.scale_type != "dynamic":
            return

        self._iter += 1

        if overflow:
            self._last_overflow_iter = self._iter
            self.scale = max(self.min_scale, self.scale * self.backoff_factor)
            logger.debug(f"Gradient overflow! Reducing loss scale to {self.scale}")
        elif (self._iter - self._last_overflow_iter) % self.growth_interval == 0:
            self.scale = min(self.max_scale, self.scale * self.growth_factor)
            logger.debug(f"Increasing loss scale to {self.scale}")

    def unscale_gradients(self, optimizer: torch.optim.Optimizer) -> None:
        """Unscale gradients before optimizer step."""
        if self.scale_type == "none":
            return

        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                if param.grad is not None:
                    param.grad.data.div_(self.scale)

    def check_overflow(self, optimizer: torch.optim.Optimizer) -> bool:
        """Check if gradients contain inf/nan."""
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                if param.grad is not None:
                    if torch.any(torch.isnan(param.grad)) or torch.any(torch.isinf(param.grad)):
                        return True
        return False


# =============================================================================
# ADVANCED SCHEDULER CONFIGURATIONS
# =============================================================================

class AdvancedSchedulerConfig:
    """Advanced learning rate scheduler configurations."""

    @staticmethod
    def create_scheduler(
        scheduler_type: str,
        optimizer: torch.optim.Optimizer,
        num_training_steps: int,
        num_warmup_steps: int = 0,
        warmup_ratio: Optional[float] = None,
        num_cycles: float = 0.5,
        power: float = 1.0,
        lr_end: float = 1e-7
    ):
        """Create learning rate scheduler with advanced configurations."""
        from transformers import get_scheduler

        # Calculate warmup steps from ratio if provided
        if warmup_ratio is not None:
            num_warmup_steps = int(num_training_steps * warmup_ratio)

        if scheduler_type == "cosine_with_restarts":
            return get_scheduler(
                name="cosine_with_restarts",
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                num_cycles=num_cycles
            )

        elif scheduler_type == "polynomial":
            return get_scheduler(
                name="polynomial",
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                power=power,
                lr_end=lr_end
            )

        elif scheduler_type == "inverse_sqrt":
            return get_scheduler(
                name="inverse_sqrt",
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps
            )

        else:
            return get_scheduler(
                name=scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )


# =============================================================================
# FLASK ROUTE PARAMETER DEFINITIONS
# =============================================================================

def get_advanced_training_params() -> Dict[str, Any]:
    """Get parameter definitions for advanced training controls (for Flask routes)."""
    return {
        # Weight Initialization
        "weight_init_method": {
            "name": "Weight Initialization",
            "type": "select",
            "default": "default",
            "options": [
                {"value": "default", "label": "Default - Keep pretrained weights"},
                {"value": "normal", "label": "Normal - Gaussian distribution"},
                {"value": "xavier_uniform", "label": "Xavier Uniform - Good for tanh"},
                {"value": "xavier_normal", "label": "Xavier Normal - Good for tanh"},
                {"value": "kaiming_uniform", "label": "Kaiming Uniform - Good for ReLU"},
                {"value": "kaiming_normal", "label": "Kaiming Normal - Good for ReLU"},
                {"value": "orthogonal", "label": "Orthogonal - Preserves gradient norms"},
            ],
            "explanation": "How to initialize weights for layers being trained. Default keeps pretrained weights.",
            "advanced": True
        },

        # Optimizer Hyperparameters
        "adam_beta1": {
            "name": "Adam Beta1 (Momentum)",
            "type": "slider",
            "default": 0.9,
            "min": 0.0,
            "max": 0.999,
            "step": 0.001,
            "explanation": "First moment decay rate. Controls momentum. 0.9 is standard, lower = less momentum.",
            "advanced": True
        },
        "adam_beta2": {
            "name": "Adam Beta2 (RMSprop)",
            "type": "slider",
            "default": 0.999,
            "min": 0.9,
            "max": 0.9999,
            "step": 0.0001,
            "explanation": "Second moment decay rate. Controls adaptive learning rate. 0.999 works for most cases.",
            "advanced": True
        },
        "adam_epsilon": {
            "name": "Adam Epsilon",
            "type": "slider",
            "default": 1e-8,
            "min": 1e-10,
            "max": 1e-6,
            "step": 1e-10,
            "display_format": "scientific",
            "explanation": "Numerical stability constant. Prevents division by zero. Rarely needs changing.",
            "advanced": True
        },

        # LoRA+ Mode
        "lora_plus_enabled": {
            "name": "LoRA+ Mode",
            "type": "checkbox",
            "default": False,
            "depends_on": "use_lora",
            "explanation": "Use different learning rates for LoRA A and B matrices. Can improve convergence.",
            "advanced": True
        },
        "lora_b_lr_ratio": {
            "name": "LoRA B LR Multiplier",
            "type": "slider",
            "default": 16,
            "min": 1,
            "max": 64,
            "step": 1,
            "depends_on": "lora_plus_enabled",
            "explanation": "Multiplier for LoRA B matrix learning rate. Paper suggests 16x.",
            "advanced": True
        },

        # RS-LoRA
        "rslora_enabled": {
            "name": "RS-LoRA Scaling",
            "type": "checkbox",
            "default": False,
            "depends_on": "use_lora",
            "explanation": "Rank-Stabilized LoRA scaling. Improves stability across different ranks.",
            "advanced": True
        },

        # Manual LoRA Targets
        "manual_lora_targets": {
            "name": "Manual LoRA Target Modules",
            "type": "textarea",
            "default": "",
            "placeholder": "q_proj, v_proj, k_proj, o_proj",
            "depends_on": "use_lora",
            "explanation": "Manually specify LoRA target modules (comma-separated). Leave empty for auto-detection.",
            "advanced": True
        },

        # Per-Layer Weight Decay
        "per_layer_weight_decay": {
            "name": "Per-Layer Weight Decay",
            "type": "checkbox",
            "default": False,
            "explanation": "Use different weight decay for attention, MLP, and embedding layers.",
            "advanced": True
        },
        "attention_weight_decay": {
            "name": "Attention Weight Decay",
            "type": "slider",
            "default": 0.01,
            "min": 0,
            "max": 0.3,
            "step": 0.01,
            "depends_on": "per_layer_weight_decay",
            "explanation": "Weight decay for attention layers.",
            "advanced": True
        },
        "mlp_weight_decay": {
            "name": "MLP Weight Decay",
            "type": "slider",
            "default": 0.01,
            "min": 0,
            "max": 0.3,
            "step": 0.01,
            "depends_on": "per_layer_weight_decay",
            "explanation": "Weight decay for MLP/feedforward layers.",
            "advanced": True
        },
        "embedding_weight_decay": {
            "name": "Embedding Weight Decay",
            "type": "slider",
            "default": 0.0,
            "min": 0,
            "max": 0.1,
            "step": 0.01,
            "depends_on": "per_layer_weight_decay",
            "explanation": "Weight decay for embedding layers. Often set to 0.",
            "advanced": True
        },

        # Scheduler Options
        "warmup_ratio": {
            "name": "Warmup Ratio",
            "type": "slider",
            "default": 0.0,
            "min": 0,
            "max": 0.5,
            "step": 0.01,
            "explanation": "Fraction of training for warmup. Alternative to fixed warmup steps (0 = use steps).",
            "advanced": True
        },
        "cosine_num_cycles": {
            "name": "Cosine Restart Cycles",
            "type": "slider",
            "default": 1,
            "min": 0.5,
            "max": 5,
            "step": 0.5,
            "depends_on": "lr_scheduler=cosine_with_restarts",
            "explanation": "Number of cosine cycles. 0.5 = half cycle, 1+ = restarts.",
            "advanced": True
        },
        "polynomial_power": {
            "name": "Polynomial Decay Power",
            "type": "slider",
            "default": 1.0,
            "min": 0.5,
            "max": 3.0,
            "step": 0.1,
            "depends_on": "lr_scheduler=polynomial",
            "explanation": "Power for polynomial decay. 1.0 = linear, 2.0 = quadratic.",
            "advanced": True
        },

        # Loss Scaling
        "loss_scale_type": {
            "name": "Loss Scaling Strategy",
            "type": "select",
            "default": "dynamic",
            "options": [
                {"value": "none", "label": "None - No loss scaling"},
                {"value": "static", "label": "Static - Fixed scale value"},
                {"value": "dynamic", "label": "Dynamic - Auto-adjust (recommended)"}
            ],
            "explanation": "Loss scaling for mixed precision. Dynamic is recommended.",
            "advanced": True
        },

        # LR Finder
        "auto_find_lr": {
            "name": "Auto Find Learning Rate",
            "type": "checkbox",
            "default": False,
            "explanation": "Run LR range test before training to find optimal learning rate.",
            "advanced": True
        },
        "lr_finder_iterations": {
            "name": "LR Finder Iterations",
            "type": "slider",
            "default": 100,
            "min": 50,
            "max": 300,
            "step": 25,
            "depends_on": "auto_find_lr",
            "explanation": "Number of iterations for LR range test.",
            "advanced": True
        },

        # QLoRA Compute Dtype
        "qlora_compute_dtype": {
            "name": "QLoRA Compute Dtype",
            "type": "select",
            "default": "float16",
            "options": [
                {"value": "float16", "label": "FP16 - Standard half precision"},
                {"value": "bfloat16", "label": "BF16 - Better range (newer GPUs)"}
            ],
            "depends_on": "use_qlora",
            "explanation": "Computation dtype for QLoRA. BF16 needs Ampere+ GPU.",
            "advanced": True
        },
    }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def setup_advanced_training(
    model: nn.Module,
    config: Dict[str, Any],
    num_training_steps: int,
    is_lora: bool = False
) -> Tuple[torch.optim.Optimizer, Any, Optional[MixedPrecisionLossScaler]]:
    """
    Convenience function to set up advanced training components.

    Returns:
        optimizer: Configured optimizer
        scheduler: Learning rate scheduler
        loss_scaler: Loss scaler for mixed precision (or None)
    """
    # Build optimizer config
    opt_config = OptimizerConfig(
        optimizer_type=config.get("optimizer", "adamw_torch"),
        learning_rate=config.get("learning_rate", 5e-5),
        weight_decay=config.get("weight_decay", 0.01),
        adam_beta1=config.get("adam_beta1", 0.9),
        adam_beta2=config.get("adam_beta2", 0.999),
        adam_epsilon=config.get("adam_epsilon", 1e-8),
        lora_plus_enabled=config.get("lora_plus_enabled", False),
        lora_b_lr_ratio=config.get("lora_b_lr_ratio", 16.0),
        rslora_enabled=config.get("rslora_enabled", False),
    )

    # Per-parameter weight decay
    if config.get("per_layer_weight_decay", False):
        opt_config.per_param_weight_decay = {
            "attention_decay": config.get("attention_weight_decay", 0.01),
            "mlp_decay": config.get("mlp_weight_decay", 0.01),
            "embedding_decay": config.get("embedding_weight_decay", 0.0),
        }

    # Create parameter groups and optimizer
    param_groups = AdvancedOptimizerFactory.create_parameter_groups(model, opt_config, is_lora)
    optimizer = AdvancedOptimizerFactory.create_optimizer(param_groups, opt_config)

    # Apply RS-LoRA scaling if enabled
    if is_lora and opt_config.rslora_enabled:
        rank = config.get("lora_r", 8)
        alpha = config.get("lora_alpha", 32)
        RSLoRAScaling.apply_rslora_scaling(model, rank, alpha)

    # Create scheduler
    warmup_steps = config.get("warmup_steps", 0)
    warmup_ratio = config.get("warmup_ratio", 0.0)
    if warmup_ratio > 0:
        warmup_steps = int(num_training_steps * warmup_ratio)

    scheduler = AdvancedSchedulerConfig.create_scheduler(
        scheduler_type=config.get("lr_scheduler", "cosine"),
        optimizer=optimizer,
        num_training_steps=num_training_steps,
        num_warmup_steps=warmup_steps,
        num_cycles=config.get("cosine_num_cycles", 1),
        power=config.get("polynomial_power", 1.0)
    )

    # Create loss scaler for mixed precision
    loss_scaler = None
    if config.get("fp16", False) or config.get("bf16", False):
        loss_scaler = MixedPrecisionLossScaler(
            scale_type=config.get("loss_scale_type", "dynamic")
        )

    return optimizer, scheduler, loss_scaler


if __name__ == "__main__":
    print("Advanced Training Features Module")
    print("=" * 50)

    print("\nWeight Initialization Methods:")
    for method, desc in WeightInitializer.INIT_METHODS.items():
        print(f"  - {method}: {desc}")

    print("\nLoss Scaling Strategies:")
    for strategy, desc in MixedPrecisionLossScaler.STRATEGIES.items():
        print(f"  - {strategy}: {desc}")

    print("\nLoRA Target Modules by Architecture:")
    for arch, modules in LoRATargetModulesManager.ARCHITECTURE_MODULES.items():
        print(f"  - {arch}: {', '.join(modules[:4])}...")

    print("\nAdvanced Parameters Available:")
    params = get_advanced_training_params()
    for key, info in params.items():
        print(f"  - {info['name']}")

    print("\n[OK] Advanced training module ready for integration")
