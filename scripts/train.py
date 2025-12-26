"""
Training script for Hospital Length of Stay prediction.

This script handles the complete training pipeline including data loading,
model training, evaluation, and saving results.
"""

import os
import sys
import yaml
import logging
import argparse
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.core import set_deterministic_seed, get_device
from data.processor import LOSDataProcessor
from models.base import ModelFactory
from evaluation.metrics import LOSEvaluator
from explainability.shap_explainer import LOSExplainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_directories(config: Dict[str, Any]) -> None:
    """
    Create necessary directories.
    
    Args:
        config: Configuration dictionary
    """
    directories = [
        config['output']['model_path'],
        config['output']['results_path'],
        'assets',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def train_model(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train the model and return results.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dict[str, Any]: Training results
    """
    # Set random seed
    set_deterministic_seed(config['data']['random_state'])
    
    # Initialize components
    data_processor = LOSDataProcessor(config.get('data', {}))
    evaluator = LOSEvaluator(config.get('evaluation', {}))
    explainer = LOSExplainer(config.get('explainability', {}))
    
    # Generate or load data
    logger.info("Generating synthetic data...")
    df = data_processor.generate_synthetic_data(
        n_samples=config['data']['n_samples'],
        seed=config['data']['random_state']
    )
    
    # Get data summary
    data_summary = data_processor.get_data_summary(df)
    logger.info(f"Data summary: {data_summary['n_samples']} samples, {data_summary['n_features']} features")
    
    # Preprocess data
    logger.info("Preprocessing data...")
    X, y = data_processor.preprocess_data(df, fit=True)
    
    # Split data
    logger.info("Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = data_processor.split_data(
        X, y,
        test_size=config['data']['test_size'],
        val_size=config['data']['val_size'],
        random_state=config['data']['random_state']
    )
    
    # Create model
    logger.info(f"Creating {config['model']['type']} model...")
    model = ModelFactory.create_model(
        config['model']['type'],
        config['model']['config']
    )
    
    # Train model
    logger.info("Training model...")
    model.fit(X_train, y_train, X_val, y_val)
    
    # Make predictions
    logger.info("Making predictions...")
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)
    
    # Evaluate model
    logger.info("Evaluating model...")
    train_metrics = evaluator.evaluate_regression(y_train, y_pred_train)
    val_metrics = evaluator.evaluate_regression(y_val, y_pred_val)
    test_metrics = evaluator.evaluate_regression(y_test, y_pred_test)
    
    # Generate comprehensive evaluation report
    test_report = evaluator.generate_evaluation_report(
        y_test, y_pred_test, 
        model_name=config['model']['type']
    )
    
    # Create SHAP explainer if enabled
    explanations = None
    if config.get('explainability', {}).get('shap', {}).get('enabled', False):
        logger.info("Creating SHAP explainer...")
        feature_names = data_processor.feature_names
        explainer.create_shap_explainer(model, X_train, feature_names)
        
        if explainer.explainer is not None:
            logger.info("Generating SHAP explanations...")
            explanations = explainer.explain_predictions(
                X_test, 
                n_samples=config['explainability']['shap']['n_samples']
            )
    
    # Save model if requested
    if config['output']['save_model']:
        model_path = Path(config['output']['model_path']) / f"{config['model']['type']}_model.pkl"
        model.save_model(str(model_path))
        logger.info(f"Model saved to {model_path}")
    
    # Prepare results
    results = {
        'config': config,
        'data_summary': data_summary,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'test_report': test_report,
        'explanations': explanations,
        'feature_names': data_processor.feature_names,
        'model_type': config['model']['type']
    }
    
    # Save results if requested
    if config['output']['save_predictions']:
        results_path = Path(config['output']['results_path']) / f"{config['model']['type']}_results.yaml"
        
        # Convert numpy arrays to lists for YAML serialization
        results_for_save = results.copy()
        if explanations:
            results_for_save['explanations'] = {
                'feature_importance': explanations['feature_importance'],
                'base_value': float(explanations['base_value']),
                'n_samples': len(explanations['data'])
            }
        
        with open(results_path, 'w') as f:
            yaml.dump(results_for_save, f, default_flow_style=False)
        logger.info(f"Results saved to {results_path}")
    
    return results


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Hospital LOS prediction model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model', type=str, default=None,
                       help='Override model type')
    parser.add_argument('--samples', type=int, default=None,
                       help='Override number of samples')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.model:
        config['model']['type'] = args.model
    if args.samples:
        config['data']['n_samples'] = args.samples
    
    # Setup directories
    setup_directories(config)
    
    # Train model
    try:
        results = train_model(config)
        
        # Print summary
        logger.info("Training completed successfully!")
        logger.info(f"Test MAE: {results['test_metrics']['mae']:.2f}")
        logger.info(f"Test RMSE: {results['test_metrics']['rmse']:.2f}")
        logger.info(f"Test R²: {results['test_metrics']['r2']:.3f}")
        
        # Print clinical interpretation
        if 'clinical_interpretation' in results['test_report']:
            logger.info("Clinical Interpretation:")
            for metric, interpretation in results['test_report']['clinical_interpretation'].items():
                logger.info(f"  {metric}: {interpretation}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
