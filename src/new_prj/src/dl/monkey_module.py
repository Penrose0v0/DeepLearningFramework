# monkey_module.py
from typing import Dict, Any
import torch
import torch.nn as nn

from dlf.module import Module


class MonkeyModule(Module):
    """
    一个简单的两层 MLP 特征提取器
    输入: [N, in_dim]
    输出: [N, out_dim]
    """

    @classmethod
    def default_config(cls) -> Dict[str, Any]:
        return {
            "in_dim": 128,
            "hidden_dim": 256,
            "out_dim": 128,
            "dropout": 0.0,
            "activation": "relu",  # relu|gelu|tanh|silu
        }

    def __init__(self, **config):
        super().__init__(**config)
        cfg = self.config

        act = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
            "silu": nn.SiLU,
        }.get(cfg["activation"], nn.ReLU)

        layers = [
            nn.Linear(cfg["in_dim"], cfg["hidden_dim"]),
            act(),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(cfg["hidden_dim"], cfg["out_dim"]),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, in_dim]
        return self.net(x)