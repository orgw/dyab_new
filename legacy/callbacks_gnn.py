# in models/callbacks.py
import pytorch_lightning as pl
import torch
import os
import wandb
import time
import sys
import json
import numpy as np

# Assuming these are defined elsewhere in your project
class SetupCallback(pl.Callback):
    def __init__(self, prefix, setup_time, save_dir, ckpt_dir, args, argv_content):
        self.prefix = prefix
        self.setup_time = setup_time
        self.save_dir = save_dir
        self.ckpt_dir = ckpt_dir
        self.args = args
        self.argv_content = argv_content
    
    def on_train_start(self, trainer, pl_module):
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        # Log setup information if needed
        
class BestCheckpointCallback(pl.callbacks.ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class EpochEndCallback(pl.Callback):
    def on_epoch_end(self, trainer, pl_module):
        # Placeholder for any end-of-epoch logic
        pass


class VisualizationCallback(pl.Callback):
    """
    Callback to generate and log ellipsoid visualizations during validation epochs.
    """
    def __init__(self, viz_every_n_epochs: int):
        super().__init__()
        self.viz_every_n_epochs = viz_every_n_epochs

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        """
        Called when the validation epoch ends.
        """
        epoch = trainer.current_epoch
        if (epoch + 1) % self.viz_every_n_epochs == 0 and trainer.is_global_zero:
            print(f"\n--- Generating visualizations for epoch {epoch} ---")
            
            # Get a fixed batch from the validation dataloader
            val_dataloader = trainer.val_dataloaders
            if not val_dataloader:
                print("No validation dataloader found.")
                return

            try:
                fixed_batch = next(iter(val_dataloader))
                # Move batch to the correct device
                fixed_batch = {k: v.to(pl_module.device) for k, v in fixed_batch.items() if hasattr(v, 'to')}

                # Call the model's visualization method
                pl_module.generate_and_log_visualizations(fixed_batch, epoch)
            except StopIteration:
                print("Validation dataloader is empty. Skipping visualization.")
            except Exception as e:
                print(f"Error during visualization: {e}")