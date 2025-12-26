"""
Evaluation script for Hospital Length of Stay prediction.

This script loads a trained model and evaluates it on test data,
generating comprehensive reports and visualizations.
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
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.core import set_deterministic_seed
from data.processor import LOSDataProcessor
from models.base import BaseModel
from evaluation.metrics import LOSEvaluator
from explainability.shap_explainer import LOSExplainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(model_path: str) -> BaseModel:
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        BaseModel: Loaded model
    """
    return BaseModel.load_model(model_path)


def generate_evaluation_report(model: BaseModel, 
                             data_processor: LOSDataProcessor,
                             evaluator: LOSEvaluator,
                             explainer: LOSExplainer,
                             n_samples: int = 1000) -> Dict[str, Any]:
    """
    Generate comprehensive evaluation report.
    
    Args:
        model: Trained model
        data_processor: Data processor
        evaluator: Evaluator
        explainer: Explainer
        n_samples: Number of samples for evaluation
        
    Returns:
        Dict[str, Any]: Evaluation report
    """
    # Generate fresh test data
    df = data_processor.generate_synthetic_data(n_samples=n_samples, seed=42)
    X, y = data_processor.preprocess_data(df, fit=False)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Evaluate
    metrics = evaluator.evaluate_regression(y, y_pred)
    report = evaluator.generate_evaluation_report(y, y_pred, model_name=type(model).__name__)
    
    # Generate explanations
    explanations = None
    if explainer.explainer is not None:
        explanations = explainer.explain_predictions(X, n_samples=100)
    
    return {
        'metrics': metrics,
        'report': report,
        'explanations': explanations,
        'y_true': y,
        'y_pred': y_pred,
        'feature_names': data_processor.feature_names
    }


def create_visualizations(report: Dict[str, Any], output_dir: str) -> None:
    """
    Create evaluation visualizations.
    
    Args:
        report: Evaluation report
        output_dir: Output directory for plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Prediction vs Actual scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(report['y_true'], report['y_pred'], alpha=0.6)
    
    # Perfect prediction line
    max_val = max(max(report['y_true']), max(report['y_pred']))
    ax.plot([0, max_val], [0, max_val], 'r--', label='Perfect Prediction')
    
    ax.set_xlabel('Actual LOS (days)')
    ax.set_ylabel('Predicted LOS (days)')
    ax.set_title('Predicted vs Actual Length of Stay')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'prediction_vs_actual.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Residuals plot
    residuals = report['y_true'] - report['y_pred']
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(report['y_pred'], residuals, alpha=0.6)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_xlabel('Predicted LOS (days)')
    ax.set_ylabel('Residuals (days)')
    ax.set_title('Residuals Plot')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'residuals.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Feature importance plot
    if report['explanations'] and 'feature_importance' in report['explanations']:
        importance = report['explanations']['feature_importance']
        features = list(importance.keys())
        values = list(importance.values())
        
        # Sort by importance
        sorted_data = sorted(zip(features, values), key=lambda x: x[1], reverse=True)
        features_sorted, values_sorted = zip(*sorted_data[:10])  # Top 10
        
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.barh(range(len(features_sorted)), values_sorted)
        ax.set_yticks(range(len(features_sorted)))
        ax.set_yticklabels(features_sorted)
        ax.set_xlabel('Importance (%)')
        ax.set_title('Top 10 Feature Importance')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                   f'{width:.1f}%', ha='left', va='center')
        
        plt.tight_layout()
        plt.savefig(output_path / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info(f"Visualizations saved to {output_path}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate Hospital LOS prediction model')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--output-dir', type=str, default='assets',
                       help='Output directory for results')
    parser.add_argument('--n-samples', type=int, default=1000,
                       help='Number of samples for evaluation')
    parser.add_argument('--create-plots', action='store_true',
                       help='Create visualization plots')
    
    args = parser.parse_args()
    
    # Set random seed
    set_deterministic_seed(42)
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = load_model(args.model_path)
    
    # Initialize components
    data_processor = LOSDataProcessor()
    evaluator = LOSEvaluator()
    explainer = LOSExplainer()
    
    # Create explainer
    logger.info("Creating SHAP explainer...")
    df_temp = data_processor.generate_synthetic_data(n_samples=100, seed=42)
    X_temp, _ = data_processor.preprocess_data(df_temp, fit=True)
    explainer.create_shap_explainer(model, X_temp, data_processor.feature_names)
    
    # Generate evaluation report
    logger.info("Generating evaluation report...")
    report = generate_evaluation_report(
        model, data_processor, evaluator, explainer, args.n_samples
    )
    
    # Print results
    metrics = report['metrics']
    print(f"\n📊 Evaluation Results:")
    print(f"   MAE: {metrics['mae']:.2f} days")
    print(f"   RMSE: {metrics['rmse']:.2f} days")
    print(f"   R²: {metrics['r2']:.3f}")
    print(f"   Within 1 day: {metrics['within_1_day']:.1f}%")
    print(f"   Within 2 days: {metrics['within_2_days']:.1f}%")
    print(f"   Within 3 days: {metrics['within_3_days']:.1f}%")
    
    # Clinical interpretation
    if 'clinical_interpretation' in report['report']:
        print(f"\n🏥 Clinical Interpretation:")
        for metric, interpretation in report['report']['clinical_interpretation'].items():
            print(f"   {metric.replace('_', ' ').title()}: {interpretation}")
    
    # Feature importance
    if report['explanations'] and 'feature_importance' in report['explanations']:
        importance = report['explanations']['feature_importance']
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
        
        print(f"\n🔍 Top 5 Most Important Features:")
        for i, (feature, importance_pct) in enumerate(top_features, 1):
            print(f"   {i}. {feature}: {importance_pct:.1f}%")
    
    # Create visualizations
    if args.create_plots:
        logger.info("Creating visualizations...")
        create_visualizations(report, args.output_dir)
    
    # Save report
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    report_path = output_path / 'evaluation_report.yaml'
    with open(report_path, 'w') as f:
        # Convert numpy arrays to lists for YAML serialization
        report_for_save = report.copy()
        report_for_save['y_true'] = report['y_true'].tolist()
        report_for_save['y_pred'] = report['y_pred'].tolist()
        
        if report['explanations']:
            report_for_save['explanations'] = {
                'feature_importance': report['explanations']['feature_importance'],
                'base_value': float(report['explanations']['base_value']),
                'n_samples': len(report['explanations']['data'])
            }
        
        yaml.dump(report_for_save, f, default_flow_style=False)
    
    logger.info(f"Evaluation report saved to {report_path}")
    print(f"\n✅ Evaluation completed successfully!")
    print(f"   Report saved to: {report_path}")
    if args.create_plots:
        print(f"   Plots saved to: {output_path}")


if __name__ == "__main__":
    main()
