# This implementation is adapted from the AstroCLIP framework
# (Parker et al. 2024), with additional modules and modifications specific to SpecCLIP.
# Original AstroCLIP code: https://github.com/PolymathicAI/AstroCLIP

from typing import Any, Dict, Optional,Union
import matplotlib.pyplot as plt
import wandb
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.cli import SaveConfigCallback
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping
from omegaconf import OmegaConf
import torch
import os

class SpecificEpochSaver(Callback):
    def __init__(self, epochs_to_save, dirpath=None, filename=None):
        """
        Save checkpoints at specific epochs.
        
        Args:
            epochs_to_save: List of epochs where checkpoints should be saved
            dirpath: Directory to save checkpoints to
            filename: Custom filename format. Uses {epoch} and {step} if provided
        """
        super().__init__()
        self.epochs_to_save = epochs_to_save if isinstance(epochs_to_save, list) else [epochs_to_save]
        self.dirpath = dirpath
        self.filename_format = filename or "epoch={epoch:03d}-step={step:d}.ckpt"
        
    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        
        if epoch in self.epochs_to_save:
            # Get directory path
            dirpath = self.dirpath or os.path.join(trainer.default_root_dir, "checkpoints")
            os.makedirs(dirpath, exist_ok=True)
            
            # Create filename
            filename = self.filename_format.format(
                epoch=epoch,
                step=trainer.global_step,
                val_loss=trainer.callback_metrics.get("val_epoch_loss", 0.0)
            )
            
            # Save checkpoint
            checkpoint_path = os.path.join(dirpath, filename)
            trainer.save_checkpoint(checkpoint_path)
            print(f"✓ Saved checkpoint at epoch {epoch} to {checkpoint_path}")

class EpochLossCallback(Callback):
    """Enhanced epoch loss callback with cleaner distributed logging"""
    def __init__(self, log_train_path, log_val_path):
        super().__init__()
        self.log_train_path = log_train_path
        self.log_val_path = log_val_path
        self.val_loss_history = []
        self.train_loss_history = []
        self.min_delta = 0.0001  # Minimum improvement threshold

    def on_validation_epoch_end(self, trainer, pl_module):
        # Only process on global rank 0 to avoid duplicate logging
        if not trainer.is_global_zero:
            return

        # Access the logged validation loss
        val_loss = trainer.callback_metrics.get('val_epoch_loss')

        if val_loss is not None:
            # Convert to float value
            val_loss_value = val_loss.item() if torch.is_tensor(val_loss) else float(val_loss)

            # Track history and check for improvement
            if self.val_loss_history:
                prev_best = min(self.val_loss_history)
                improvement = prev_best - val_loss_value

                # Only log if there's significant improvement
                if improvement > self.min_delta:
                    print(f"\nEpoch {trainer.current_epoch}: val_loss improved by {improvement:.6f}")
                    print(f"Previous best: {prev_best:.6f} → New best: {val_loss_value:.6f}")

            self.val_loss_history.append(val_loss_value)

            # Write to log file
            csv_line = f"{trainer.current_epoch},{val_loss_value:.6f}\n"
            with open(self.log_val_path, "a") as f:
                f.write(csv_line)

    def on_train_epoch_end(self, trainer, pl_module):
        # Only process on global rank 0
        if not trainer.is_global_zero:
            return

        train_loss = trainer.callback_metrics.get('train_epoch_loss')

        if train_loss is not None:
            train_loss_value = train_loss.item() if torch.is_tensor(train_loss) else float(train_loss)
            self.train_loss_history.append(train_loss_value)

            # Write to log file
            csv_line = f"{trainer.current_epoch},{train_loss_value:.6f}\n"
            with open(self.log_train_path, "a") as f:
                f.write(csv_line)

class CustomEarlyStopping(EarlyStopping):
    """Enhanced early stopping with cleaner logging"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.best_values = []
    def _eval_improvement(self, current: torch.Tensor, best: torch.Tensor) -> bool:
        improvement = self._improvement_fn(current, best)

        # Format message with 6 decimal places instead of default 3
        if improvement > 0 and self.verbose and self.trainer.is_global_zero:
            print(
                f"Metric {self.monitor} improved by {improvement:.6f} >= min_delta = {self.min_delta:.6f}."
                f" New best score: {current:.6f}"
            )

        return improvement > self.min_delta

    def on_validation_epoch_end(self, trainer, pl_module):
        # Only process on global rank 0
        if not trainer.is_global_zero:
            return

        current = self._get_monitor_value(trainer)

        if current is not None:
            current_value = current.item() if torch.is_tensor(current) else float(current)

            # Track best values
            if not self.best_values or current_value < min(self.best_values):
                self.best_values.append(current_value)
                # Log only significant improvements
                if len(self.best_values) > 1:
                    improvement = min(self.best_values[:-1]) - current_value
                    if improvement > self.min_delta:
                        print(f"\nNew best {self.monitor}: {current_value:.6f}")

            super().on_validation_epoch_end(trainer, pl_module)

    def _get_monitor_value(self, trainer):
        """Safely get the monitored value"""
        logs = trainer.callback_metrics
        return logs.get(self.monitor) if logs else None

class CustomWandbLogger(WandbLogger):
    """Enhanced WandB logger with improved handling of hyperparameters"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric_history = {}

    def log_hyperparams(self, params):
        # Only log non-empty hyperparameters
        if params:
            sanitized_params = {k: v for k, v in params.items() if v is not None}
            super().log_hyperparams(sanitized_params)

    def log_metrics(self, metrics, step=None):
        # Track metric history for better monitoring
        for key, value in metrics.items():
            if key not in self.metric_history:
                self.metric_history[key] = []
            self.metric_history[key].append(value)

        # Call parent class log_metrics
        super().log_metrics(metrics, step)

class CustomSaveConfigCallback(SaveConfigCallback):
    def save_config(
        self, trainer: Trainer, pl_module: LightningModule, stage: str
    ) -> None:
        for logger in trainer.loggers:
            if issubclass(type(logger), WandbLogger):
                logger.experiment.config.update(self.config.as_dict())
        return super().save_config(trainer, pl_module, stage)

def _safe_eval(s: str, max_len: int = 1024) -> Union[int, float]:
    is_safe = all(ch in "e0123456789_+-*/(). " for ch in s)
    if not is_safe:
        raise ValueError(
            "Only simple arithmetic expressions involving digits, parentheses, "
            "the letter e, or the symbols '+-*/_.' are allowed"
        )
    if len(s) > max_len:
        raise ValueError(f"String length is {len(s)}, maximum allowed is {max_len}")
    return eval(s)

OmegaConf.register_new_resolver("eval", _safe_eval, use_cache=True)