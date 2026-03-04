import os
import json
import yaml
import math
import numbers
import shutil
import random
import warnings
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple, Iterable

import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from .utils import set_seed, get_next_idx
from .module import Module


@dataclass
class TrainerConfig:
    # Basic hyper parameters
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    grad_clip: float = 0.0

    # Train config
    save_interval: int = 1
    eval_interval: int = 1
    amp: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    grad_accum_steps: int = 1

    # Evaluate config
    monitor: str = "val_loss"
    mode: str = "min"

    # Log & Output
    exp_name: str = "experiment"
    output_dir: str = "./outputs"
    seed: int = 3407
    device: Optional[str] = None

    resume: bool = False
    debug: bool = False


class Trainer:
    """
      1) Rewrite:
         - make_dataset(self) -> (train_set, val_set | None)
         - make_model(self) -> Module
         - make_optimizer(self, model) -> (optimizer, scheduler | None)
         - train_one_epoch(self, epoch, model, train_loader, optimizer, scaler) -> dict
         - evaluate(self, epoch, model, val_loader) -> dict
      2) new_trainer = YourTrainer(TrainerConfig(...))
         new_trainer.run()
    """

    cfg_filename = "config.yaml"

    def __init__(self, cfg: TrainerConfig):
        self.cfg = cfg
        set_seed(cfg.seed)

        # Device
        if cfg.device is not None:
            self.device = torch.device(cfg.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Make dir
        exp_root = os.path.join(cfg.output_dir, cfg.exp_name)
        os.makedirs(exp_root, exist_ok=True)
        if getattr(cfg, "debug", False):
            self.run_dir = os.path.join(exp_root, "_debug")
            if os.path.exists(self.run_dir):
                shutil.rmtree(self.run_dir)
        else:
            run_idx = get_next_idx(exp_root)
            self.run_dir = os.path.join(exp_root, f"{run_idx}")
        self.ckpt_dir = os.path.join(self.run_dir, "checkpoints")
        self.logs_dir = os.path.join(self.run_dir, "logs")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        # Log file
        self.train_log_path = os.path.join(self.logs_dir, "train.jsonl")
        self.eval_log_path = os.path.join(self.logs_dir, "eval.jsonl")
        self.summary_yaml = os.path.join(self.run_dir, "summary.yaml")

        # Components
        self.model: Optional[Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler = None
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.device.type == "cuda" and self.cfg.amp))

        # Variables
        self.global_step = 0
        self.start_epoch = 1
        self.best_value = -math.inf if self.cfg.mode == "max" else math.inf
        self.best_epoch = 0
        self.no_improve_epochs = 0

        # Data
        self.train_set = None
        self.val_set = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None


    def make_dataset(self) -> Tuple[Iterable, Optional[Iterable]]:
        raise NotImplementedError

    def make_dataloaders(self):
        if self.train_set is None and self.val_set is None:
            self.train_set, self.val_set = self.make_dataset()

        if self.train_set is not None:
            self.train_loader = DataLoader(
                self.train_set,
                batch_size=self.cfg.batch_size,
                shuffle=True,
                num_workers=self.cfg.num_workers,
                pin_memory=self.cfg.pin_memory,
                persistent_workers=self.cfg.persistent_workers and self.cfg.num_workers > 0,
                prefetch_factor=self.cfg.prefetch_factor if self.cfg.num_workers > 0 else None,
                drop_last=True,
            )
        if self.val_set is not None:
            self.val_loader = DataLoader(
                self.val_set,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                num_workers=self.cfg.num_workers,
                pin_memory=self.cfg.pin_memory,
                persistent_workers=self.cfg.persistent_workers and self.cfg.num_workers > 0,
                prefetch_factor=self.cfg.prefetch_factor if self.cfg.num_workers > 0 else None,
                drop_last=False,
            )

    def make_model(self) -> Module:
        raise NotImplementedError

    def make_optimizer(self, model: Module):
        raise NotImplementedError

    def train_one_epoch(self, epoch: int, model: Module, train_loader: DataLoader,
                        optimizer, scaler: torch.cuda.amp.GradScaler) -> Dict[str, float]:
        raise NotImplementedError

    @torch.inference_mode()
    def evaluate(self, epoch: int, model: Module, val_loader: Optional[DataLoader]) -> Dict[str, float]:
        raise NotImplementedError

    # Main
    def run(self):
        self.make_dataloaders()
        if self.model is None:
            self.model = self.make_model()
        self.model.to(self.device)
        self.optimizer, self.scheduler = self.make_optimizer(self.model)

        if self.cfg.resume:
            latest = self._latest_ckpt_dir()
            if latest is not None:
                try:
                    self._load_states(latest)
                    print(f"[Trainer] Resumed from {latest}")
                except Exception as e:
                    warnings.warn(f"[Trainer] Failed to resume from {latest}: {e}")

        total_epochs = self.cfg.epochs

        epoch_range = range(self.start_epoch, total_epochs + 1) if not self.cfg.debug else range(1)
        pbar = tqdm(
            epoch_range,
            desc=f"Training ({self.cfg.exp_name})",
            dynamic_ncols=True,
            position=0,
            leave=True
        )

        if not self.cfg.debug:
            self.evaluate(0, self.model, self.val_loader)
        for epoch in pbar:
            self.model.train()
            train_log = self.train_one_epoch(epoch, self.model, self.train_loader, self.optimizer, self.scaler)
            if not isinstance(train_log, dict):
                train_log = {"_note": "train_one_epoch returned non-dict"}

            self._flush_grad_accum_if_needed()

            if self.scheduler is not None and getattr(self.scheduler, "step_on", "epoch") == "epoch":
                self.scheduler.step()

            train_log = {"epoch": epoch, "global_step": self.global_step, **train_log}
            self._append_jsonl(self.train_log_path, train_log)

            do_eval = (self.val_loader is not None) and (epoch % max(1, self.cfg.eval_interval) == 0)
            eval_log = {}
            if do_eval:
                self.model.eval()
                with torch.inference_mode():
                    eval_log = self.evaluate(epoch, self.model, self.val_loader) or {}
                if not isinstance(eval_log, dict):
                    eval_log = {"_note": "evaluate returned non-dict"}
                eval_log = {"epoch": epoch, **eval_log}
                self._append_jsonl(self.eval_log_path, eval_log)
                self._plot_curves_safe()

                if self.cfg.monitor not in eval_log:
                    warnings.warn(f"[Trainer] monitor key '{self.cfg.monitor}' not found in eval_log keys={list(eval_log.keys())}")
                    monitored = None
                else:
                    monitored = float(eval_log[self.cfg.monitor])

                improved = False
                if monitored is not None:
                    if (self.cfg.mode == "min" and monitored < self.best_value) or \
                       (self.cfg.mode == "max" and monitored > self.best_value):
                        improved = True
                        self.best_value = monitored
                        self.best_epoch = epoch
                        self.no_improve_epochs = 0
                    else:
                        self.no_improve_epochs += 1

                if improved:
                    tag_dir = self._ckpt_tag_dir(epoch)
                    self._save_ckpt(tag_dir, epoch)
                    self._update_best_symlink(tag_dir)

            if epoch % max(1, self.cfg.save_interval) == 0:
                tag_dir = self._ckpt_tag_dir(epoch)
                self._save_ckpt(tag_dir, epoch)

            self._write_summary(epoch, train_log, eval_log)

            msg = []
            if "loss" in train_log:
                msg.append(f"loss={train_log['loss']:.4f}")
            if do_eval and self.cfg.monitor in eval_log:
                msg.append(f"{self.cfg.monitor}={eval_log[self.cfg.monitor]:.4f}")
            pbar.set_postfix_str(" | ".join(msg))

        pbar.close()
        print(f"[Trainer] Done. Run dir: {self.run_dir}")


    def _ckpt_tag_dir(self, epoch: int) -> str:
        tag = f"epoch_{epoch:04d}"
        path = os.path.join(self.ckpt_dir, tag)
        os.makedirs(path, exist_ok=True)
        return path

    def _latest_ckpt_dir(self) -> Optional[str]:
        if not os.path.isdir(self.ckpt_dir):
            return None
        tags = [d for d in os.listdir(self.ckpt_dir) if d.startswith("epoch_")]
        if not tags:
            return None
        tags.sort()
        return os.path.join(self.ckpt_dir, tags[-1])

    def _save_ckpt(self, ckpt_dir: str, epoch: int):
        self.model.save_ckpt(ckpt_dir)

        trainer_state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
            "no_improve_epochs": self.no_improve_epochs,
            "device": str(self.device),
            "rng_state": {
                "python": random.getstate(),
                "torch": torch.get_rng_state().tolist(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            "optimizer_state": self.optimizer.state_dict() if self.optimizer is not None else None,
            "scheduler_state": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler_state": self.scaler.state_dict() if self.scaler is not None else None,
        }
        torch.save(trainer_state, os.path.join(ckpt_dir, "trainer_state.pt"))

        self._update_ckpt_config(ckpt_dir)

        latest = os.path.join(self.ckpt_dir, "latest")
        try:
            if os.path.islink(latest) or os.path.exists(latest):
                if os.path.islink(latest):
                    os.unlink(latest)
                else:
                    shutil.rmtree(latest)
            os.symlink(os.path.relpath(ckpt_dir, self.ckpt_dir), latest)
        except Exception:
            if os.path.isdir(latest):
                shutil.rmtree(latest)
            shutil.copytree(ckpt_dir, latest)

    def _update_ckpt_config(self, ckpt_dir: str):
        cfg_path = os.path.join(ckpt_dir, "config.yaml")
        root = {}
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                try:
                    root = yaml.safe_load(f) or {}
                except yaml.YAMLError:
                    root = {}
        root.setdefault("trainer", {})
        root["trainer"]["config"] = asdict(self.cfg)
        root["trainer"]["runtime"] = {
            "run_dir": self.run_dir,
            "ckpt_dir": self.ckpt_dir,
            "logs_dir": self.logs_dir,
        }
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(root, f, sort_keys=False, allow_unicode=True)

    def _load_states(self, ckpt_dir: str):
        # Load model
        if self.model is None:
            self.model = Module.from_ckpt(ckpt_dir)
        else:
            weight_path = os.path.join(
                ckpt_dir, f"{self.model.__class__.registry_name()}{getattr(self.model, 'weights_ext', '.pth')}"
            )
            state = torch.load(weight_path, map_location="cpu")
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            if missing or unexpected:
                warnings.warn(f"[Trainer] load_state: missing={missing}, unexpected={unexpected}")
        self.model.to(self.device)

        # Load trainer
        state_path = os.path.join(ckpt_dir, "trainer_state.pt")
        if not os.path.exists(state_path):
            raise FileNotFoundError(f"trainer_state not found: {state_path}")
        state = torch.load(state_path, map_location="cpu")

        if self.optimizer is None or self.scheduler is None:
            self.optimizer, self.scheduler = self.make_optimizer(self.model)

        if state.get("optimizer_state") is not None:
            self.optimizer.load_state_dict(state["optimizer_state"])
            self._move_optimizer_state_to_device()
        if state.get("scheduler_state") is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(state["scheduler_state"])
        if state.get("scaler_state") is not None and self.scaler is not None:
            self.scaler.load_state_dict(state["scaler_state"])

        self.start_epoch = int(state.get("epoch", 0)) + 1
        self.global_step = int(state.get("global_step", 0))
        self.best_value = state.get("best_value", self.best_value)
        self.best_epoch = state.get("best_epoch", self.best_epoch)
        self.no_improve_epochs = state.get("no_improve_epochs", 0)

        rng = state.get("rng_state", {})
        if "python" in rng and rng["python"] is not None:
            random.setstate(rng["python"])
        if "torch" in rng and rng["torch"] is not None:
            torch.set_rng_state(torch.tensor(rng["torch"], dtype=torch.uint8))
        if torch.cuda.is_available() and rng.get("cuda"):
            try:
                torch.cuda.set_rng_state_all(rng["cuda"])
            except Exception:
                pass

    def _move_optimizer_state_to_device(self):
        if self.optimizer is None: return
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)

    # log
    def _root_cfg_path(self) -> str:
        return os.path.join(self.run_dir, self.cfg_filename)

    def _append_jsonl(self, path: str, record: Dict[str, Any]):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _plot_curves_safe(self):
        try:
            self._plot_curves()
        except Exception as e:
            warnings.warn(f"[Trainer] plot curves failed: {e}")

    def _plot_curves(self):

        def _read_jsonl(path):
            if not os.path.exists(path):
                return []
            data = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line.strip())
                        if isinstance(obj, dict):
                            data.append(obj)
                    except Exception:
                        continue
            return data

        train_data = _read_jsonl(self.train_log_path)
        eval_data = _read_jsonl(self.eval_log_path)

        def _plot_file(data, name_prefix, x_key="epoch"):
            if not data:
                return
            all_keys = set()
            for d in data:
                all_keys.update(d.keys())
            numeric_keys = [k for k in all_keys if all(isinstance(d.get(k), numbers.Number) for d in data if k in d)]
            if not numeric_keys:
                return

            if any("global_step" in d for d in data):
                x_key = "global_step"
            elif any("epoch" in d for d in data):
                x_key = "epoch"
            else:
                x_key = None

            for key in numeric_keys:
                if key == x_key:
                    continue
                xs = [d.get(x_key) for d in data if x_key in d and key in d] if x_key else list(range(len(data)))
                ys = [d.get(key) for d in data if key in d]
                if not xs or not ys or len(xs) != len(ys):
                    continue
                plt.figure()
                plt.plot(xs, ys)
                plt.xlabel(x_key if x_key else "index")
                plt.ylabel(key)
                plt.title(f"{name_prefix} {key}")
                plt.tight_layout()
                fname = f"{name_prefix.lower()}_{key}.png"
                plt.savefig(os.path.join(self.logs_dir, fname))
                plt.close()

        _plot_file(train_data, "Train")
        _plot_file(eval_data, "Eval")


    def _write_summary(self, epoch: int, train_log: Dict[str, Any], eval_log: Dict[str, Any]):
        summary = {
            "last_epoch": epoch,
            "best_epoch": self.best_epoch,
            "best_value": float(self.best_value) if isinstance(self.best_value, (int, float)) else self.best_value,
            "monitor": self.cfg.monitor,
            "mode": self.cfg.mode,
            "latest_ckpt": os.path.relpath(self._latest_ckpt_dir() or "", self.run_dir),
            "train_log_last": train_log,
            "eval_log_last": eval_log,
        }
        with open(self.summary_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(summary, f, sort_keys=False, allow_unicode=True)

    def _update_best_symlink(self, target_dir: str):
        best = os.path.join(self.ckpt_dir, "best")
        try:
            if os.path.islink(best) or os.path.exists(best):
                if os.path.islink(best):
                    os.unlink(best)
                else:
                    shutil.rmtree(best)
            os.symlink(os.path.relpath(target_dir, self.ckpt_dir), best)
        except Exception:
            if os.path.isdir(best):
                shutil.rmtree(best)
            shutil.copytree(target_dir, best)

    @classmethod
    def from_config(cls, run_dir: str):
        ckpt_root = os.path.join(run_dir, "checkpoints")
        if not os.path.isdir(ckpt_root):
            raise FileNotFoundError(f"No checkpoints dir at {ckpt_root}")

        latest = os.path.join(ckpt_root, "latest")
        if os.path.exists(latest):
            ckpt_dir = latest
        else:
            tags = [d for d in os.listdir(ckpt_root) if d.startswith("epoch_")]
            if not tags:
                raise FileNotFoundError(f"No checkpoint tags under {ckpt_root}")
            tags.sort()
            ckpt_dir = os.path.join(ckpt_root, tags[-1])

        cfg_path = os.path.join(ckpt_dir, cls.cfg_filename)
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"No config at {cfg_path}")

        with open(cfg_path, "r", encoding="utf-8") as f:
            root = yaml.safe_load(f) or {}
        trainer_cfg = ((root.get("trainer") or {}).get("config") or {})
        cfg = TrainerConfig(**trainer_cfg)

        inst = cls(cfg)
        inst.run_dir = run_dir
        inst.ckpt_dir = os.path.join(run_dir, "checkpoints")
        inst.logs_dir = os.path.join(run_dir, "logs")
        return inst

    @classmethod
    def from_ckpt(cls, ckpt_dir: str):
        run_dir = os.path.abspath(os.path.join(ckpt_dir, "..", ".."))
        inst = cls.from_config(run_dir)
        inst.model = inst.make_model()
        inst.model.to(inst.device)
        inst.optimizer, inst.scheduler = inst.make_optimizer(inst.model)
        inst._load_states(ckpt_dir)
        return inst
    
    def _flush_grad_accum_if_needed(self):
        # Flush remaining accumulated grads at epoch end (or early stop)
        if max(1, self.cfg.grad_accum_steps) <= 1:
            return
        if self.optimizer is None:
            return
        # If already aligned, nothing to do
        if (self.global_step % self.cfg.grad_accum_steps) == 0:
            return

        # Check whether there is any gradient accumulated
        has_grad = False
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    has_grad = True
                    break
            if has_grad:
                break
        if not has_grad:
            return

        if self.scaler.is_enabled():
            if self.cfg.grad_clip and self.cfg.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            if self.cfg.grad_clip and self.cfg.grad_clip > 0:
                clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.grad_clip)
            self.optimizer.step()

        self.optimizer.zero_grad(set_to_none=True)

    def step_and_update(self, loss: torch.Tensor) -> bool:
        updated = False
        loss = loss / max(1, self.cfg.grad_accum_steps)
        
        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        if (self.global_step + 1) % self.cfg.grad_accum_steps == 0:
            if self.scaler.is_enabled():
                if self.cfg.grad_clip and self.cfg.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                if self.cfg.grad_clip and self.cfg.grad_clip > 0:
                    clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.grad_clip)
                self.optimizer.step()

            self.optimizer.zero_grad(set_to_none=True)
            updated = True

        self.global_step += 1
        return updated