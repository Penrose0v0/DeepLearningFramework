import os
from typing import Any, Dict, Optional
import yaml
import warnings

import torch
import torch.nn as nn

from .utils import _deep_merge


class Module(nn.Module):

    cfg_filename = "config.yaml"
    weights_ext = ".pth"

    def __init__(self, **config):
        super().__init__()
        self.module_name = self.__class__.registry_name()
        self.config: Dict[str, Any] = _deep_merge(self.default_config(), config)

    @classmethod
    def default_config(cls) -> Dict[str, Any]:
        return {}

    @classmethod
    def registry_name(cls) -> str:
        return getattr(cls, "_registered_name", cls.__name__)

    # Save
    def save_config(self, save_dir: str):
        cfg_path = os.path.join(save_dir, self.cfg_filename)

        cfg = {}
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                try:
                    cfg = yaml.safe_load(f) or {}
                except yaml.YAMLError:
                    cfg = {}

        if "modules" not in cfg:
            cfg["modules"] = {}

        cfg["modules"][self.module_name] = self.config
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    def save_weights(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        weight_path = os.path.join(save_dir, f"{self.__class__.registry_name()}{self.weights_ext}")
        torch.save(self.state_dict(), weight_path)

    def save_ckpt(self, save_dir: str):
        self.save_config(save_dir)
        self.save_weights(save_dir)

    # Load
    @classmethod
    def from_config(cls, load_dir: str):
        cfg_path = os.path.join(load_dir, cls.cfg_filename)

        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"No config file found at {cfg_path}")
        
        with open(cfg_path, "r", encoding="utf-8") as f:
            root = yaml.safe_load(f) or {}
        mod_cfg = (root.get("modules") or {}).get(cls.registry_name(), {})
        if not isinstance(mod_cfg, dict):
            raise FileNotFoundError(f"Invalid config format in {cfg_path}, expect a dict")

        # Compare with defaults
        defaults = cls.default_config()
        extra_keys = set(mod_cfg.keys()) - set(defaults.keys())
        if extra_keys:
            warnings.warn(f"[{cls.__name__}] Unrecognized config fields: {extra_keys}", UserWarning)
        missing = set(defaults.keys()) - set(mod_cfg.keys())
        if missing:
            warnings.warn(f"[{cls.__name__}] Missing config fields: {missing}, using defaults.", UserWarning)

        cfg = _deep_merge(defaults, mod_cfg)
        inst = cls(**cfg)
        return inst

    @classmethod
    def from_ckpt(cls, load_dir: str):
        """
        Build and load a module entirely from checkpoint:
          1) read YAML config
          2) reconcile with default_config (warn on mismatch)
          3) instantiate model with **config
          4) load weights
        """
        inst = cls.from_config(load_dir=load_dir)

        weight_path = os.path.join(load_dir, f"{cls.registry_name()}{cls.weights_ext}")
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"No checkpoint found at {weight_path}")

        try:
            state = torch.load(weight_path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(weight_path, map_location="cpu")

        inst.load_state_dict(state)

        return inst

    def forward(self, *args, **kwargs):
        raise NotImplementedError
