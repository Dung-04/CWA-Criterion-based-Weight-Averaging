"""Early stopping and checkpoint retention.

EXPERIMENT-CRITICAL: ``get_best_checkpoint`` and ``get_top_k_checkpoints`` are
what Baseline and CWA select on. Both rank purely by validation loss.
"""
import json
import os

import torch


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class CheckpointManager:
    """Keep only the last N epochs plus the top K checkpoints by val_loss.

    The top-K set is retained unconditionally, so ``keep_last_n`` is a disk
    budget only — it never changes which checkpoints CWA can average.
    """

    def __init__(self, save_dir, model_name, keep_last_n=10, keep_top_k=5):
        self.save_dir = os.path.join(save_dir, model_name)
        os.makedirs(self.save_dir, exist_ok=True)
        self.checkpoints = []  # List of (epoch, val_loss, checkpoint_path)
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.keep_last_n = keep_last_n
        self.keep_top_k = keep_top_k

    def save_checkpoint(self, model, epoch, val_loss, is_best=False):
        """Save checkpoint and manage storage efficiently."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss
        }

        checkpoint_path = os.path.join(
            self.save_dir, f'epoch_{epoch:03d}_val_loss_{val_loss:.4f}.pth'
        )
        torch.save(checkpoint, checkpoint_path)

        self.checkpoints.append((epoch, val_loss, checkpoint_path))

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch

        if is_best:
            torch.save(checkpoint, os.path.join(self.save_dir, 'best_checkpoint.pth'))

        self._cleanup_checkpoints()

        return checkpoint_path

    def _cleanup_checkpoints(self):
        """Remove checkpoints outside the last N epochs and the top K val_loss."""
        if len(self.checkpoints) <= self.keep_last_n + self.keep_top_k:
            return  # Not enough checkpoints to cleanup

        sorted_by_epoch = sorted(self.checkpoints, key=lambda x: x[0])
        last_n_epochs = set(cp[0] for cp in sorted_by_epoch[-self.keep_last_n:])

        sorted_by_loss = sorted(self.checkpoints, key=lambda x: x[1])
        top_k_epochs = set(cp[0] for cp in sorted_by_loss[:self.keep_top_k])

        epochs_to_keep = last_n_epochs | top_k_epochs

        checkpoints_to_keep = []
        for epoch, val_loss, path in self.checkpoints:
            if epoch in epochs_to_keep:
                checkpoints_to_keep.append((epoch, val_loss, path))
            else:
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except Exception as e:
                    print(f"  Warning: Could not delete checkpoint {path}: {e}")

        self.checkpoints = checkpoints_to_keep

    def get_best_checkpoint(self):
        """Checkpoint with the lowest val_loss — the Baseline method."""
        if not self.checkpoints:
            return None
        return min(self.checkpoints, key=lambda x: x[1])

    def get_top_k_checkpoints(self, k):
        """The K lowest-val_loss checkpoints — the input to CWA."""
        if not self.checkpoints:
            return []
        return sorted(self.checkpoints, key=lambda x: x[1])[:k]

    def save_checkpoint_info(self):
        """Save checkpoint information to JSON."""
        info = {
            'checkpoints': [(epoch, val_loss, path)
                            for epoch, val_loss, path in self.checkpoints]
        }
        with open(os.path.join(self.save_dir, 'checkpoint_info.json'), 'w') as f:
            json.dump(info, f, indent=4)
