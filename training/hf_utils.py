"""
Shared utilities for the HF/PEFT Qwen3-VL training and export pipeline.

Contains a manual PyTorch LoRA implementation for the vision tower (so we can
use rank 32 there while the LLM uses PEFT LoRA at rank 64), plus helpers for
applying, saving, loading, and merging those adapters.
"""

import math
from pathlib import Path

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Manual LoRA linear layer (used for vision tower and projector)
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Low-rank adapter wrapped around a base nn.Linear.

    Forward: y = base(x) + (x @ A @ B) * (alpha / r)
    Base weight is [out_features, in_features] (PyTorch convention).
    LoRA matrices: A=[in_features, r], B=[r, out_features].
    """

    def __init__(self, base_layer: nn.Linear, r: int, alpha: int, dropout: float = 0.0):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"LoRALinear only wraps nn.Linear, got {type(base_layer)}")
        self.base_layer = base_layer
        self.r = r
        self.lora_alpha = alpha
        self.scaling = alpha / r

        in_features = base_layer.in_features
        out_features = base_layer.out_features
        device = base_layer.weight.device

        self.lora_A = nn.Parameter(torch.zeros(in_features, r, device=device))
        self.lora_B = nn.Parameter(torch.zeros(r, out_features, device=device))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Initialize A with Kaiming uniform, B with zeros -> initial adapter output is zero
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base_layer(x)
        x_d = self.lora_dropout(x).to(self.lora_A.dtype)
        lora_out = (x_d @ self.lora_A @ self.lora_B).to(out.dtype)
        out = out + lora_out * self.scaling
        return out

    def merge(self, safe_merge: bool = False, adapter_names=None) -> nn.Linear:
        """Return a new nn.Linear with the LoRA update fused into the base weight."""
        with torch.no_grad():
            base_weight = self.base_layer.weight.data  # [out, in]
            delta = (self.lora_A @ self.lora_B).T * self.scaling  # [out, in]
            new_weight = base_weight + delta.to(base_weight.dtype)

        device = self.base_layer.weight.device
        new_layer = nn.Linear(
            self.base_layer.in_features,
            self.base_layer.out_features,
            bias=self.base_layer.bias is not None,
            device=device,
        )
        new_layer.weight.data = new_weight
        if self.base_layer.bias is not None:
            new_layer.bias.data = self.base_layer.bias.data
        return new_layer


# ---------------------------------------------------------------------------
# Apply / save / load / merge manual LoRA adapters
# ---------------------------------------------------------------------------

VISION_LORA_TARGETS = ["attn.qkv", "attn.proj", "mlp.linear_fc1", "mlp.linear_fc2"]
PROJECTOR_LORA_TARGETS = ["linear_fc1", "linear_fc2"]


def _get_vision_module(model: nn.Module):
    vision = getattr(model, "visual", None)
    if vision is None and hasattr(model, "model"):
        vision = getattr(model.model, "visual", None)
    if vision is None:
        raise RuntimeError("Could not find vision tower in model")
    return vision


def apply_vision_block_lora(model: nn.Module, r: int = 32, alpha: int = 32, dropout: float = 0.0, num_blocks: int | None = None):
    """Apply manual LoRA to vision transformer blocks.

    Args:
        num_blocks: If set, only adapt the last N (deepest) blocks.
                    If None, adapt all blocks.
    """
    vision = _get_vision_module(model)
    all_blocks = list(vision.blocks)
    total_blocks = len(all_blocks)
    if num_blocks is not None:
        blocks = all_blocks[-num_blocks:]
        print(f"Vision LoRA: targeting last {num_blocks} of {total_blocks} blocks (indices {total_blocks - num_blocks}–{total_blocks - 1})")
    else:
        blocks = all_blocks
        print(f"Vision LoRA: targeting all {total_blocks} blocks")
    count = 0
    for block in blocks:
        for target in VISION_LORA_TARGETS:
            parts = target.split(".")
            module = block
            for part in parts:
                module = getattr(module, part)
            if isinstance(module, nn.Linear):
                parent = block
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], LoRALinear(module, r, alpha, dropout))
                count += 1
    print(f"Applied custom LoRA (r={r}, alpha={alpha}) to {count} vision-block layers")
    return count


def apply_projector_lora(model: nn.Module, r: int = 64, alpha: int = 64, dropout: float = 0.0):
    """Apply manual LoRA to the vision-language projector (main + deepstack)."""
    vision = _get_vision_module(model)
    count = 0
    for proj in [vision.merger, *vision.deepstack_merger_list]:
        for target in PROJECTOR_LORA_TARGETS:
            module = getattr(proj, target)
            if isinstance(module, nn.Linear):
                setattr(proj, target, LoRALinear(module, r, alpha, dropout))
                count += 1
    print(f"Applied custom LoRA (r={r}, alpha={alpha}) to {count} vision-language projector layers")
    return count


def apply_vision_lora(model: nn.Module, r: int = 32, alpha: int = 32, dropout: float = 0.0):
    """Apply manual LoRA to all vision blocks and the projector (legacy convenience)."""
    return apply_vision_block_lora(model, r, alpha, dropout) + apply_projector_lora(model, r, alpha, dropout)


def save_custom_lora(model: nn.Module, path: Path | str):
    """Save state dict of all LoRALinear modules, stripping PEFT prefixes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {}
    for n, m in model.named_modules():
        if isinstance(m, LoRALinear):
            # Normalize PEFT-wrapped names to base model names.
            key = n
            for prefix in ("base_model.model.model.", "base_model.model.", "base_model."):
                if key.startswith(prefix):
                    # Map PeftModel base_model.model.<rest> -> model.<rest>
                    key = key[len(prefix):]
                    if not key.startswith("model."):
                        key = "model." + key
                    break
            state[key + ".lora_A"] = m.lora_A.data
            state[key + ".lora_B"] = m.lora_B.data
            state[key + ".r"] = torch.tensor(m.r)
            state[key + ".lora_alpha"] = torch.tensor(m.lora_alpha)
    torch.save(state, path)
    print(f"Saved custom LoRA weights to {path} ({len(state)} tensors)")


def load_custom_lora(model: nn.Module, path: Path | str):
    """Load LoRALinear weights into a model that already has LoRALinear layers.

    Handles both base models (module names like ``model.visual...``) and
    PEFT-wrapped models (module names like ``base_model.model.model.visual...``).
    """
    path = Path(path)
    state = torch.load(path, map_location="cpu")

    def normalize_key(key: str) -> str:
        # Strip common PEFT/base-model prefixes and ensure 'model.' prefix.
        for prefix in ("base_model.model.model.", "base_model.model.", "base_model."):
            if key.startswith(prefix):
                key = key[len(prefix):]
                if not key.startswith("model."):
                    key = "model." + key
                break
        return key

    state = {normalize_key(k): v for k, v in state.items() if k.endswith((".lora_A", ".lora_B"))}

    loaded = 0
    for n, m in model.named_modules():
        if isinstance(m, LoRALinear):
            key_a = normalize_key(n) + ".lora_A"
            key_b = normalize_key(n) + ".lora_B"
            if key_a not in state:
                raise KeyError(f"Missing LoRA weight for {n}: {key_a} not in saved state (available prefixes: {list(state.keys())[:3]}...)")
            device = m.lora_A.device
            m.lora_A.data = state[key_a].to(device)
            m.lora_B.data = state[key_b].to(device)
            loaded += 1
    print(f"Loaded custom LoRA weights into {loaded} layers from {path}")
    return loaded


def merge_custom_lora(model: nn.Module) -> nn.Module:
    """Replace every LoRALinear with a merged nn.Linear in-place."""
    merged = 0
    for n, m in model.named_modules():
        if isinstance(m, LoRALinear):
            # Find parent module and attribute name
            parts = n.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], m.merge())
            merged += 1
    print(f"Merged {merged} custom LoRA layers into base weights")
    return model


# ---------------------------------------------------------------------------
# PEFT helpers
# ---------------------------------------------------------------------------

LLM_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def get_peft_lora_config(r: int = 64, alpha: int = 64, dropout: float = 0.0, target_modules=None):
    """Build a PEFT LoraConfig for the language model only."""
    from peft import LoraConfig

    if target_modules is None:
        target_modules = LLM_LORA_TARGETS

    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=list(target_modules),
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def count_trainable_parameters(model: nn.Module) -> int:
    """Return number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_trainable_parameters(model: nn.Module):
    trainable = count_trainable_parameters(model)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"Trainable params: {trainable:,} || "
        f"All params: {total:,} || "
        f"Trainable %: {100 * trainable / total:.4f}"
    )
