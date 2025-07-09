import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import joblib
import yaml
from utils.model import RegressionModel
from utils.tracker import BestEpochTracker


class RegressionTrainer:
    """Class for training and evaluating regression models."""
    
    def __init__(self, config_path):
        """
        Initialize the trainer with configuration from a YAML file.
        
        Args:
            config_path: Path to the YAML config file
        """
        self.config = self._load_config(config_path)
        
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        
        torch.manual_seed(42)
        
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.train_losses = []
        self.test_losses = []
        self.epoch_tracker = None
        
    def _load_config(self, config_path):
        """Load configuration from YAML file."""
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    
    def load_data(self):
        """Load and preprocess data according to the configuration."""
        # Get configuration parameters
        data_path = self.config['data']['path']
        target_column = self.config['data']['target_column']
        test_size = self.config['data']['test_size']
        
        data = pd.read_csv(data_path)
        
        print(f"Dataset loaded from {data_path}")
        print(f"Dataset shape: {data.shape}")
        print(data.head(6))
        
        X = data.drop(target_column, axis=1).values
        y = data[target_column].values.reshape(-1, 1)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Standardize features
        self.X_train_scaled = self.scaler_X.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler_X.transform(self.X_test)
        
        self.y_train_scaled = self.scaler_y.fit_transform(self.y_train)
        self.y_test_scaled = self.scaler_y.transform(self.y_test)
        
        # Convert to PyTorch tensors
        self.X_train_tensor = torch.FloatTensor(self.X_train_scaled).to(self.device)
        self.y_train_tensor = torch.FloatTensor(self.y_train_scaled).to(self.device)
        self.X_test_tensor = torch.FloatTensor(self.X_test_scaled).to(self.device)
        self.y_test_tensor = torch.FloatTensor(self.y_test_scaled).to(self.device)
        
        print(f"Data split: Train={self.X_train.shape[0]} samples, Test={self.X_test.shape[0]} samples")
        
    def setup_model(self):
        """Set up the model, loss function, optimizer based on configuration."""
        # Get configuration parameters
        hidden_layers = self.config['model']['hidden_layers']
        activation = self.config['model']['activation']
        learning_rate = self.config['training']['learning_rate']
        
        # Initialize model
        input_size = self.X_train_scaled.shape[1]
        self.model = RegressionModel(input_size, hidden_layers, activation).to(self.device)
        
        # Setup loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Create epoch tracker
        patience = self.config['training'].get('patience', None)
        self.epoch_tracker = BestEpochTracker(self.model, patience)
        
        print(f"Model initialized with architecture: {hidden_layers}")
        print(f"Using device: {self.device}")
        
    def train(self):
        """Train the model based on the configuration."""
        # Get training parameters
        num_epochs = self.config['training']['num_epochs']
        batch_size = self.config['training']['batch_size']
        
        # Calculate number of batches
        n_samples = len(self.X_train_tensor)
        n_batches = n_samples // batch_size
        if n_samples % batch_size != 0:
            n_batches += 1
        
        print(f"Starting training for {num_epochs} epochs with batch size {batch_size}")
        
        # Training loop
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            
            # Mini-batch training
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                
                batch_X = self.X_train_tensor[start_idx:end_idx]
                batch_y = self.y_train_tensor[start_idx:end_idx]
                
                # Forward pass
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item() * (end_idx - start_idx)
            
            # Calculate average loss for the epoch
            train_loss /= n_samples
            self.train_losses.append(train_loss)
            
            # Evaluate on test set
            test_loss = self.evaluate()
            self.test_losses.append(test_loss)
            
            # Track the best epoch
            is_best = self.epoch_tracker.update(epoch, test_loss)
            best_marker = " (Best)" if is_best else ""
            
            # Print progress
            if (epoch + 1) % 100 == 0 or is_best:
                print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, '
                      f'Test Loss: {test_loss:.6f}{best_marker}')
            
            # Check for early stopping
            if self.epoch_tracker.stop_training:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Save last model state before restoring the best one
        self.last_model_state = {key: value.cpu().clone() for key, value in 
                                self.model.state_dict().items()}
        print(f"Saved last model state from epoch {epoch+1}")
                
        # Restore the best model
        self.epoch_tracker.restore_best_model()
        best_info = self.epoch_tracker.get_best_info()
        print(f"\nTraining completed. Best epoch: {best_info['epoch']} with loss: {best_info['loss']:.6f}")
        
        return best_info
    
    def evaluate(self):
        """Evaluate the model on the test set."""
        self.model.eval()
        with torch.no_grad():
            test_outputs = self.model(self.X_test_tensor)
            test_loss = self.criterion(test_outputs, self.y_test_tensor).item()
        return test_loss
        
    def final_evaluation(self):
        """Perform final evaluation of the model and calculate metrics."""
        self.model.eval()
        with torch.no_grad():
            y_pred_scaled = self.model(self.X_test_tensor).cpu().numpy()
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
            y_true = self.scaler_y.inverse_transform(self.y_test_tensor.cpu().numpy())
            
            # Calculate metrics
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            print("\nFinal Model Evaluation:")
            print(f"Mean Squared Error (MSE): {mse:.4f}")
            print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
            print(f"Mean Absolute Error (MAE): {mae:.4f}")
            print(f"R-squared (R2): {r2:.4f}")
            
            return {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'y_pred': y_pred,
                'y_true': y_true
            }
    
    def save_model(self, results_dir=None):
        """Save the model, scalers, and results."""
        if results_dir is None:
            base_dir = "results"
            os.makedirs(base_dir, exist_ok=True)

            # Cari nomor direktori berikutnya
            existing_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("model_")]
            numbers = []
            for d in existing_dirs:
                try:
                    number = int(d.split("_")[1])
                    numbers.append(number)
                except (IndexError, ValueError):
                    continue
            next_number = max(numbers, default=0) + 1
            results_dir = os.path.join(base_dir, f"model_{next_number}")

        os.makedirs(results_dir, exist_ok=True)
        
        # Save best model
        best_model_dir = os.path.join(results_dir, "best_model")
        os.makedirs(best_model_dir, exist_ok=True)
        
        # Restore best model first
        self.epoch_tracker.restore_best_model()
        
        # Save best model
        torch.save(self.model.state_dict(), os.path.join(best_model_dir, "model_weights.pth"))
        torch.save(self.model, os.path.join(best_model_dir, "model_full.pth"))
        
        # Save traced best model for production
        example_input = torch.randn(1, self.X_train_scaled.shape[1]).to(self.device)
        traced_model = torch.jit.trace(self.model, example_input)
        traced_model.save(os.path.join(best_model_dir, "model_libtorch.pt"))
        
        # Now, if we have the last model state saved, restore it and save as last model
        if hasattr(self, 'last_model_state'):
            last_model_dir = os.path.join(results_dir, "last_model")
            os.makedirs(last_model_dir, exist_ok=True)
            
            # Restore last model state
            self.model.load_state_dict(self.last_model_state)
            
            # Save last model
            torch.save(self.model.state_dict(), os.path.join(last_model_dir, "model_weights.pth"))
            torch.save(self.model, os.path.join(last_model_dir, "model_full.pth"))
            
            # Save traced last model for production
            traced_model = torch.jit.trace(self.model, example_input)
            traced_model.save(os.path.join(last_model_dir, "model_libtorch.pt"))
        
        # Save scalers
        joblib.dump(self.scaler_X, os.path.join(results_dir, "scaler_X.pkl"))
        joblib.dump(self.scaler_y, os.path.join(results_dir, "scaler_y.pkl"))
        
        with open(os.path.join(results_dir, "scaler_X.txt"), 'w') as f:
            mean_str = ",".join([f"{x:.8f}" for x in self.scaler_X.mean_])
            scale_str = ",".join([f"{x:.8f}" for x in self.scaler_X.scale_])
            f.write(mean_str + "\n" + scale_str + "\n")

        with open(os.path.join(results_dir, "scaler_y.txt"), 'w') as f:
            mean_str = ",".join([f"{x:.8f}" for x in np.atleast_1d(self.scaler_y.mean_)])
            scale_str = ",".join([f"{x:.8f}" for x in np.atleast_1d(self.scaler_y.scale_)])
            f.write(mean_str + "\n" + scale_str + "\n")
        
        # Save configuration
        with open(os.path.join(results_dir, "config.yaml"), 'w') as f:
            yaml.dump(self.config, f)
        
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'best_epoch': self.epoch_tracker.get_best_info()
        }
        joblib.dump(history, os.path.join(results_dir, "training_history.pkl"))
        
        # Restore best model for further use
        self.epoch_tracker.restore_best_model()
        
        print(f"\nModel and results saved to: {results_dir}")
        print(f"- Best model saved in: {best_model_dir}")
        if hasattr(self, 'last_model_state'):
            print(f"- Last model saved in: {last_model_dir}")
        
        return results_dir
    
    def plot_training_curve(self, save_path=None):
        """Plot the training and validation loss curves."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.test_losses, label='Validation Loss')
        
        # Mark the best epoch
        best_epoch = self.epoch_tracker.best_epoch
        plt.axvline(x=best_epoch, color='r', linestyle='--', alpha=0.5)
        
        # Mark the best epoch in the legend label
        best_epoch = self.epoch_tracker.best_epoch
        plt.axvline(x=best_epoch, color='r', linestyle='--', alpha=0.5, 
                label=f'Best Epoch: {best_epoch+1}')
        
        plt.xlabel('Epochs')
        plt.ylabel('Loss (MSE)')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_predictions(self, save_path=None):
        """Plot predictions vs actual values."""
        # Get predictions
        eval_results = self.final_evaluation()
        y_true = eval_results['y_true']
        y_pred = eval_results['y_pred']
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs Predicted Values')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def save_metrics(self, results_dir=None):
        """Save evaluation metrics to a text file."""
        if results_dir is None:
            results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)

        metrics = self.final_evaluation()
        
        metrics_path = os.path.join(results_dir, "evaluation_metrics.txt")
        with open(metrics_path, 'w') as f:
            f.write("Final Model Evaluation Metrics:\n")
            f.write(f"Mean Squared Error (MSE): {metrics['mse']:.6f}\n")
            f.write(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.6f}\n")
            f.write(f"Mean Absolute Error (MAE): {metrics['mae']:.6f}\n")
            f.write(f"R-squared (R2): {metrics['r2']:.6f}\n")
        
        print(f"Metrics saved to {metrics_path}")
