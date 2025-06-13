import torch
import joblib
import numpy as np


class RegressionPredictor:
    """Class for making predictions with a trained regression model."""
    
    def __init__(self, model_path, scaler_x_path, scaler_y_path):
        """
        Initialize the predictor with paths to the model and scalers.
        
        Args:
            model_path: Path to the saved model file
            scaler_x_path: Path to the saved feature scaler
            scaler_y_path: Path to the saved target scaler
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(model_path, map_location=self.device,  weights_only=False)
        
        # Load scalers
        self.scaler_X = joblib.load(scaler_x_path)
        self.scaler_y = joblib.load(scaler_y_path)
        
        # Set model to evaluation mode
        if hasattr(self.model, 'eval'):
            self.model.eval()
    
    def initialize_model_from_state(self, input_size, hidden_layers=[32, 16, 8]):
        """Initialize model from state dict (used when loaded model state instead of full model)."""
        from utils.model import RegressionModel
        self.model = RegressionModel(input_size, hidden_layers).to(self.device)
        self.model.load_state_dict(self.model_state)
        self.model.eval()
    
    def predict(self, features):
        """
        Make predictions for new data.
        
        Args:
            features: Numpy array or DataFrame of features (unscaled)
            
        Returns:
            Numpy array of predictions (unscaled)
        """
        # Convert to numpy if needed
        if hasattr(features, 'values'):
            features = features.values
        
        # Scale features
        features_scaled = self.scaler_X.transform(features)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features_scaled).to(self.device)
        
        # Make predictions
        with torch.no_grad():
            predictions_scaled = self.model(features_tensor).cpu().numpy()
        
        # Inverse transform to get original scale
        predictions = self.scaler_y.inverse_transform(predictions_scaled)
        
        return predictions