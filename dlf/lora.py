import os
import warnings
from typing import Any, Dict

import torch
import transformers
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from peft.utils import load_peft_weights

from .module import Module


_AUTO_CLASSES = {
    "AutoModel",
    "AutoModelForCausalLM",
    "AutoModelForSeq2SeqLM",
    "AutoModelForSequenceClassification",
    "AutoModelForTokenClassification",
    "AutoModelForQuestionAnswering",
    "AutoModelForMaskedLM",
}


class LoRAModule(Module):
    """
    LoRA fine-tuning wrapper for HuggingFace transformers, built on dlf.Module.

    Override `default_config()` to specify the base model and LoRA hyper-params:

        class MyLoRA(LoRAModule):
            @classmethod
            def default_config(cls):
                return {
                    "base_model": "Qwen/Qwen2.5-7B",
                    "auto_class": "AutoModelForCausalLM",
                    "model_kwargs": {"torch_dtype": "bfloat16"},
                    "lora": {
                        "r": 8,
                        "lora_alpha": 16,
                        "lora_dropout": 0.05,
                        "target_modules": ["q_proj", "v_proj"],
                        "bias": "none",
                        "task_type": "CAUSAL_LM",
                    },
                }

        model = MyLoRA()
        model.print_trainable_parameters()
        model.save_ckpt("out/")              # stores only the adapter
        model = MyLoRA.from_ckpt("out/")     # rebuild base + load adapter
        model.merge_and_save("merged/")      # merge LoRA into base, export full model
    """

    adapter_dirname = "adapter"

    def __init__(self, **config):
        super().__init__(**config)

        if not self.config.get("base_model"):
            raise ValueError(
                f"[{self.__class__.__name__}] `base_model` is required in config "
                "(HF model id or local path)."
            )

        base = self._load_base_model()
        self.model = self._wrap_with_lora(base)

    @classmethod
    def default_config(cls) -> Dict[str, Any]:
        return {
            "base_model": None,
            "auto_class": "AutoModelForCausalLM",
            "model_kwargs": {},
            "lora": {
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "target_modules": ["q_proj", "v_proj"],
                "bias": "none",
                "task_type": "CAUSAL_LM",
            },
        }

    def _resolve_auto_class(self):
        name = self.config.get("auto_class", "AutoModelForCausalLM")
        if name not in _AUTO_CLASSES:
            raise ValueError(
                f"Unsupported auto_class '{name}'. Choose from: {sorted(_AUTO_CLASSES)}"
            )
        return getattr(transformers, name)

    def _resolve_model_kwargs(self) -> Dict[str, Any]:
        kwargs = dict(self.config.get("model_kwargs") or {})
        dtype = kwargs.get("torch_dtype")
        if isinstance(dtype, str):
            kwargs["torch_dtype"] = getattr(torch, dtype)
        return kwargs

    def _load_base_model(self):
        klass = self._resolve_auto_class()
        return klass.from_pretrained(self.config["base_model"], **self._resolve_model_kwargs())

    def _wrap_with_lora(self, base):
        lora_cfg = LoraConfig(**self.config["lora"])
        return get_peft_model(base, lora_cfg)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    # Save / Load (override to persist adapter only, not the full base model)
    def save_weights(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        adapter_dir = os.path.join(save_dir, self.adapter_dirname)
        self.model.save_pretrained(adapter_dir)

    def load_weights(self, load_dir: str):
        adapter_dir = os.path.join(load_dir, self.adapter_dirname)
        if not os.path.isdir(adapter_dir):
            raise FileNotFoundError(f"No LoRA adapter found at {adapter_dir}")
        state = load_peft_weights(adapter_dir)
        return set_peft_model_state_dict(self.model, state)

    @classmethod
    def from_ckpt(cls, load_dir: str):
        inst = cls.from_config(load_dir=load_dir)
        # Load adapter weights into the existing peft model rather than rebuilding
        # the base model (which would double GPU memory transiently).
        inst.load_weights(load_dir)
        return inst

    def merge_and_save(self, save_dir: str, save_tokenizer: bool = True):
        """
        Merge LoRA weights into the base model and save as a standalone HF checkpoint.
        After this call the in-memory `self.model` is replaced with the merged model
        (LoRA layers are gone), so further training is not possible without re-wrapping.
        """
        os.makedirs(save_dir, exist_ok=True)
        merged = self.model.merge_and_unload()
        merged.save_pretrained(save_dir)
        self.model = merged

        if save_tokenizer:
            try:
                tok = transformers.AutoTokenizer.from_pretrained(self.config["base_model"])
                tok.save_pretrained(save_dir)
            except Exception as e:
                warnings.warn(f"[LoRAModule] Skipped saving tokenizer: {e}")
