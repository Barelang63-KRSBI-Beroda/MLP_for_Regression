import argparse
import pandas as pd
import os
from utils.predictor import RegressionPredictor
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def main():
    """Make predictions with a trained regression model."""
    parser = argparse.ArgumentParser(description='Make predictions with a trained regression model.')
    parser.add_argument('--model', type=str, required=True, help='Path to the saved model file')
    parser.add_argument('--scaler-x', type=str, required=True, help='Path to the saved feature scaler')
    parser.add_argument('--scaler-y', type=str, required=True, help='Path to the saved target scaler')
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV file with features and expected target')
    parser.add_argument('--output', type=str, default='predictions.xlsx', help='Base path to save predictions (e.g., predictions.xlsx)')
    parser.add_argument('--input-size', type=int, default=None, help='Input size (needed only if loading state dict)')
    parser.add_argument('--hidden-layers', type=str, default=None, help='Hidden layers as comma-separated values (needed only if loading state dict)')
    parser.add_argument('--target-col', type=str, default='Expected', help='Name of the column in input CSV with expected target values')
    
    args = parser.parse_args()
    
    try:
        # Initialize predictor
        predictor = RegressionPredictor(args.model, args.scaler_x, args.scaler_y)
        
        # Initialize model from state dict if needed
        if args.input_size is not None:
            hidden_layers = [32, 16, 8] 
            if args.hidden_layers:
                hidden_layers = [int(x) for x in args.hidden_layers.split(',')]
            predictor.initialize_model_from_state(args.input_size, hidden_layers)
        
        # Load input data
        input_data = pd.read_csv(args.input)
        print(f"Loaded input data with shape: {input_data.shape}")

        # Check if target column exists
        if args.target_col not in input_data.columns:
            raise ValueError(f"Target column '{args.target_col}' not found in input CSV.")

        # Extract features and target
        features = input_data.drop(columns=[args.target_col])
        expected = input_data[args.target_col].values
        
        # Make predictions
        predictions = predictor.predict(features).flatten()

        # Calculate absolute and percentage error
        absolute_error = np.abs(predictions - expected)
        epsilon = 1e-10
        error_pct = np.abs(predictions / expected) * 100

        # Compute evaluation metrics
        mse = mean_squared_error(expected, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(expected, predictions)
        r2 = r2_score(expected, predictions)

        # Prepare results DataFrame
        results = pd.DataFrame({
            'Sample': range(1, len(predictions)+1),
            'Predicted Value': np.round(predictions, 4),
            'Actual Value': np.round(expected, 4),
            'Error': np.round(absolute_error, 4)
        })

        # Prepare metrics DataFrame
        metrics_df = pd.DataFrame({
            'Metric': ['MSE', 'RMSE', 'MAE', 'R²'],
            'Value': [mse, rmse, mae, r2]
        })

        # Output Excel file name
        excel_output_path = os.path.splitext(args.output)[0] + '.xlsx'

        # Save both sheets to Excel
        with pd.ExcelWriter(excel_output_path) as writer:
            results.to_excel(writer, index=False, sheet_name='Predictions')
            metrics_df.to_excel(writer, index=False, sheet_name='Metrics')

        print(f"\n✅ Predictions and metrics saved to {excel_output_path}\n")
        print("📊 Evaluation Metrics:")
        print(metrics_df.to_string(index=False))
        
    except Exception as e:
        print(f"❌ Error during prediction: {str(e)}")
        raise

if __name__ == "__main__":
    main()
