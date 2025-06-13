import argparse
import os
from utils.trainer import RegressionTrainer

def main():
    """Main entry point for training the regression model."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a regression model with configuration from YAML.')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration YAML file (default: config.yaml)')
    parser.add_argument('--output', type=str, default=None,
                        help='Directory to save results (default: auto-generated)')
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Config file '{args.config}' not found.")
        return
    
    results_dir = args.output
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
    
    try:
        # Initialize and train the model
        trainer = RegressionTrainer(args.config)
        trainer.load_data()
        trainer.setup_model()
        
        # Train the model
        best_info = trainer.train()
        eval_results = trainer.final_evaluation()

        save_dir = trainer.save_model(results_dir)
        
        trainer.plot_training_curve(os.path.join(save_dir, "training_loss.png"))
        trainer.plot_predictions(os.path.join(save_dir, "predictions.png"))
        trainer.save_metrics(save_dir)
        
        print(f"Training completed successfully. Results saved to {save_dir}")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()