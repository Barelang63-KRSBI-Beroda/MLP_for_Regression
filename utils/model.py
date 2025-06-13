import torch
import torch.nn as nn


class RegressionModel(nn.Module):
    """Neural network model for regression tasks."""
    
    def __init__(self, input_size, hidden_layers=[32, 16, 8], activation='sigmoid'):
        """
        Initialize the regression model.
        
        Args:
            input_size: Number of input features
            hidden_layers: List of hidden layer sizes
        """
        super(RegressionModel, self).__init__()
        
        # Ensure hidden_layers is a list
        if not isinstance(hidden_layers, list):
            hidden_layers = [hidden_layers]
        
        # Create layers dynamically based on the hidden_layers parameter
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_layers[0]))
        
        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_layers[-1], 1))
        
        
        # Activation function
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation function: {activation}. Please choose from 'sigmoid', 'relu', or 'tanh'. "
                              "If you want to use another activation available in PyTorch, feel free to add it manually.")       
        
    def forward(self, x):
        """Forward pass through the network."""
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation(layer(x))
        
        # Last layer without activation function
        x = self.layers[-1](x)
        return x