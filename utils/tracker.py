import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class BestEpochTracker:
    """Class to track the best performing epoch during training."""
    
    def __init__(self, model, patience=None):
        """
        Initialize the tracker.
        
        Args:
            model: The PyTorch model being trained
            patience: Optional early stopping patience (None to disable)
        """
        self.model = model
        self.patience = patience
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.best_model_state = None
        self.counter = 0  # For early stopping
        self.stop_training = False
    
    def update(self, epoch, test_loss):
        """
        Update the best epoch tracker.
        
        Args:
            epoch: Current epoch number
            test_loss: Current test loss value
            
        Returns:
            bool: True if this is the best epoch so far
        """
        is_best = False
        
        if test_loss < self.best_loss:
            self.best_loss = test_loss
            self.best_epoch = epoch
            self.best_model_state = {key: value.cpu().clone() for key, value in 
                                    self.model.state_dict().items()}
            is_best = True
            self.counter = 0 
        else:
            self.counter += 1
            
        if self.patience is not None and self.counter >= self.patience:
            self.stop_training = True
            
        return is_best
    
    def restore_best_model(self):
        """Restore the model to its best state."""
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"Restored model from best epoch: {self.best_epoch+1} with loss: {self.best_loss:.4f}")
    
    def get_best_info(self):
        """Return information about the best epoch."""
        return {
            'epoch': self.best_epoch + 1,
            'loss': self.best_loss,
            'model_state': self.best_model_state
        }