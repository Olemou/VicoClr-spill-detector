import math
import numpy as np
import torch.optim as optim
import yaml
import torch
from pathlib import Path
from typing import Dict, Any
import importlib.resources as pkg_resources
from src_utils.logging import get_logger
logger  = get_logger("Learning be setup",force=True)


# -------------------------------------------------
# 1. Split parameters (clean + correct)
# -------------------------------------------------
def split_params(model: torch.nn.Module):
    head_params = []
    backbone_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "head" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    return backbone_params, head_params


# -------------------------------------------------
# 2. LR scaling (sqrt rule)
# -------------------------------------------------

# -------------------------------------------------
# 4. Scheduler (warmup + cosine)
# -------------------------------------------------
def cosine_schedule(
    epoch,
    optimizer,
    warmup_epochs=10,
    max_epochs=100,
    min_lr_ratio=0.01,
    hard_min_lr=5e-6
):
    for pg in optimizer.param_groups:
        base_lr = pg["initial_lr"]
        
        soft_min_lr = base_lr * min_lr_ratio
        min_lr = max(soft_min_lr, hard_min_lr)

        if epoch < warmup_epochs:
            t = epoch / warmup_epochs
            lr = min_lr + (base_lr - min_lr) * (t ** 2)
        else:
            progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
            lr = min_lr + 0.5 * (base_lr - min_lr) * (
                1 + np.cos(np.pi * progress)
            )

        pg["lr"] = lr

    return [pg["lr"] for pg in optimizer.param_groups]



# utils/config.py
def load_config(config_name: str = "base.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    """
    config_path = str(pkg_resources.files("configs").joinpath(config_name))
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_optimizer_params(model, batch_size: int, config: Dict = None):
    """
    Build optimizer parameter groups using config.
    """
    if config is None:
        config = load_config()
    
    training_cfg = config['training']
    
    # Calculate learning rate scaling
    scale = (batch_size / training_cfg['base_batch_size']) ** 0.5
    
    lr_backbone = training_cfg['base_lr_backbone'] * scale
    lr_head     = training_cfg['base_lr_head'] * scale
    
    backbone_params, head_params = split_params(model)   # your existing function
    
    optimizer = optim.AdamW([
        {
            "params": backbone_params,
            "lr": lr_backbone,
            "initial_lr": lr_backbone,
            "weight_decay": training_cfg['weight_decay_backbone']
        },
        {
            "params": head_params,
            "lr": lr_head,
            "initial_lr": lr_head,
            "weight_decay": training_cfg['weight_decay_head']
        }
    ])
    
    return optimizer


def get_gradient_stats(model: torch.nn.Module):
    """
    Returns min, max, mean, and norm of gradients across all parameters.
    """
    grad_stats = {
        "grad_min": float('inf'),
        "grad_max": float('-inf'),
        "grad_mean": 0.0,
        "grad_norm": 0.0,
        "grad_abs_mean": 0.0,
    }
    
    total_elements = 0
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            g = param.grad.data
            
            grad_stats["grad_min"] = min(grad_stats["grad_min"], g.min().item())
            grad_stats["grad_max"] = max(grad_stats["grad_max"], g.max().item())
            
            grad_stats["grad_mean"] += g.mean().item() * g.numel()
            grad_stats["grad_abs_mean"] += g.abs().mean().item() * g.numel()
            grad_stats["grad_norm"] += g.norm(2).item() ** 2
            
            total_elements += g.numel()
    
    if total_elements > 0:
        grad_stats["grad_mean"] /= total_elements
        grad_stats["grad_abs_mean"] /= total_elements
        grad_stats["grad_norm"] = grad_stats["grad_norm"] ** 0.5
    
    return grad_stats


def log_training_status(optimizer, model, epoch, step=None):
    """Combined log: LR + Gradient statistics"""
    # Learning Rates
    lrs = [pg['lr'] for pg in optimizer.param_groups]
    
    # Gradient Stats
    gstats = get_gradient_stats(model)
    
    if step is not None:
        prefix = f"Epoch {epoch+1:3d} | Step {step:5d}"
    else:
        prefix = f"Epoch {epoch+1:3d}"
    
    logger.info(
        f"{prefix} | "
        f"LR_bb: {lrs[0]:.2e} | LR_head: {lrs[1]:.2e} | "
        f"Grad: [{gstats['grad_min']:.2e}, {gstats['grad_max']:.2e}] "
        f"(mean: {gstats['grad_mean']:.2e})"
    )
