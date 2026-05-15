"""
dlf-core: Deep Learning Framework Core

Base classes for building deep learning projects:
- Module: Base model class with config management and checkpointing
- Trainer: Base trainer class with training loop and evaluation
- LoRAModule: LoRA fine-tuning wrapper for HuggingFace transformers
- Utilities: Seed setting, path helpers, dict merging
"""

from .module import Module
from .trainer import Trainer, TrainerConfig
from .lora import LoRAModule
from .utils import set_seed, get_next_idx

__version__ = "0.1.0"

__all__ = [
    "Module",
    "Trainer",
    "TrainerConfig",
    "LoRAModule",
    "set_seed",
    "get_next_idx",
]