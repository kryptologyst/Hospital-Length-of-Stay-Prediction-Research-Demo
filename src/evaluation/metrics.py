"""
Evaluation metrics and utilities for Hospital Length of Stay prediction.

This module provides comprehensive evaluation metrics including
clinical metrics, calibration analysis, and fairness evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error
)
from sklearn.calibration import calibration_curve
import logging

logger = logging.getLogger(__name__)


class LOSEvaluator:
    """Evaluator for Hospital Length of Stay prediction models."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the evaluator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
    def evaluate_regression(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          y_pred_std: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Evaluate regression performance with comprehensive metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            y_pred_std: Standard deviation of predictions (for uncertainty)
            
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Basic regression metrics
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics['r2'] = r2_score(y_true, y_pred)
        metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred)
        
        # Clinical-specific metrics
        metrics['mae_days'] = metrics['mae']  # MAE in days
        metrics['rmse_days'] = metrics['rmse']  # RMSE in days
        
        # Percentage of predictions within clinically acceptable ranges
        metrics['within_1_day'] = np.mean(np.abs(y_true - y_pred) <= 1) * 100
        metrics['within_2_days'] = np.mean(np.abs(y_true - y_pred) <= 2) * 100
        metrics['within_3_days'] = np.mean(np.abs(y_true - y_pred) <= 3) * 100
        
        # Direction accuracy (over/under prediction)
        over_pred = np.sum(y_pred > y_true)
        under_pred = np.sum(y_pred < y_true)
        metrics['over_prediction_rate'] = over_pred / len(y_true) * 100
        metrics['under_prediction_rate'] = under_pred / len(y_true) * 100
        
        # Uncertainty metrics (if available)
        if y_pred_std is not None:
            metrics['mean_uncertainty'] = np.mean(y_pred_std)
            metrics['uncertainty_coverage'] = self._calculate_coverage(y_true, y_pred, y_pred_std)
        
        logger.info(f"Evaluation completed - MAE: {metrics['mae']:.2f}, RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2']:.3f}")
        return metrics
    
    def _calculate_coverage(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          y_pred_std: np.ndarray, confidence: float = 0.95) -> float:
        """
        Calculate prediction interval coverage.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            y_pred_std: Standard deviation of predictions
            confidence: Confidence level (default 0.95 for 95% intervals)
            
        Returns:
            float: Coverage percentage
        """
        z_score = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%
        lower_bound = y_pred - z_score * y_pred_std
        upper_bound = y_pred + z_score * y_pred_std
        
        coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))
        return coverage * 100
    
    def evaluate_calibration(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           n_bins: int = 10) -> Dict[str, Any]:
        """
        Evaluate prediction calibration.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            n_bins: Number of bins for calibration analysis
            
        Returns:
            Dict[str, Any]: Calibration metrics and data
        """
        # Create bins based on predicted values
        bin_boundaries = np.linspace(y_pred.min(), y_pred.max(), n_bins + 1)
        bin_indices = np.digitize(y_pred, bin_boundaries) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        calibration_data = []
        for i in range(n_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                bin_pred_mean = np.mean(y_pred[mask])
                bin_true_mean = np.mean(y_true[mask])
                bin_size = np.sum(mask)
                
                calibration_data.append({
                    'bin': i,
                    'predicted_mean': bin_pred_mean,
                    'actual_mean': bin_true_mean,
                    'bin_size': bin_size,
                    'calibration_error': abs(bin_pred_mean - bin_true_mean)
                })
        
        calibration_df = pd.DataFrame(calibration_data)
        
        # Calculate calibration metrics
        ece = np.average(calibration_df['calibration_error'], 
                        weights=calibration_df['bin_size'])
        mce = np.max(calibration_df['calibration_error'])
        
        return {
            'ece': ece,
            'mce': mce,
            'calibration_data': calibration_df,
            'n_bins': n_bins
        }
    
    def evaluate_fairness(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         groups: np.ndarray, group_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate fairness across different groups.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            groups: Group assignments for each sample
            group_names: Names for groups
            
        Returns:
            Dict[str, Any]: Fairness metrics
        """
        unique_groups = np.unique(groups)
        if group_names is None:
            group_names = [f"Group_{i}" for i in unique_groups]
        
        fairness_results = {}
        
        for group, group_name in zip(unique_groups, group_names):
            mask = groups == group
            if np.sum(mask) > 0:
                group_mae = mean_absolute_error(y_true[mask], y_pred[mask])
                group_rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
                group_r2 = r2_score(y_true[mask], y_pred[mask])
                
                fairness_results[group_name] = {
                    'mae': group_mae,
                    'rmse': group_rmse,
                    'r2': group_r2,
                    'n_samples': np.sum(mask),
                    'mean_prediction': np.mean(y_pred[mask]),
                    'mean_actual': np.mean(y_true[mask])
                }
        
        # Calculate fairness gaps
        if len(fairness_results) > 1:
            mae_values = [metrics['mae'] for metrics in fairness_results.values()]
            fairness_results['mae_gap'] = max(mae_values) - min(mae_values)
            fairness_results['mae_ratio'] = max(mae_values) / min(mae_values)
        
        return fairness_results
    
    def create_leaderboard(self, results: Dict[str, Dict[str, float]], 
                          primary_metric: str = 'mae') -> pd.DataFrame:
        """
        Create a model leaderboard.
        
        Args:
            results: Dictionary of model results
            primary_metric: Primary metric for ranking
            
        Returns:
            pd.DataFrame: Leaderboard
        """
        leaderboard_data = []
        
        for model_name, metrics in results.items():
            row = {'Model': model_name}
            row.update(metrics)
            leaderboard_data.append(row)
        
        leaderboard_df = pd.DataFrame(leaderboard_data)
        
        # Sort by primary metric (lower is better for MAE/RMSE)
        ascending = primary_metric in ['mae', 'rmse', 'mape']
        leaderboard_df = leaderboard_df.sort_values(primary_metric, ascending=ascending)
        
        return leaderboard_df
    
    def generate_evaluation_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 model_name: str = "Model",
                                 groups: Optional[np.ndarray] = None,
                                 group_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            groups: Group assignments for fairness analysis
            group_names: Names for groups
            
        Returns:
            Dict[str, Any]: Comprehensive evaluation report
        """
        report = {
            'model_name': model_name,
            'n_samples': len(y_true),
            'regression_metrics': self.evaluate_regression(y_true, y_pred),
            'calibration_metrics': self.evaluate_calibration(y_true, y_pred)
        }
        
        if groups is not None:
            report['fairness_metrics'] = self.evaluate_fairness(y_true, y_pred, groups, group_names)
        
        # Clinical interpretation
        report['clinical_interpretation'] = self._generate_clinical_interpretation(report['regression_metrics'])
        
        return report
    
    def _generate_clinical_interpretation(self, metrics: Dict[str, float]) -> Dict[str, str]:
        """
        Generate clinical interpretation of metrics.
        
        Args:
            metrics: Regression metrics
            
        Returns:
            Dict[str, str]: Clinical interpretations
        """
        interpretations = {}
        
        # MAE interpretation
        mae = metrics['mae']
        if mae <= 1.0:
            interpretations['mae'] = "Excellent prediction accuracy"
        elif mae <= 2.0:
            interpretations['mae'] = "Good prediction accuracy"
        elif mae <= 3.0:
            interpretations['mae'] = "Moderate prediction accuracy"
        else:
            interpretations['mae'] = "Poor prediction accuracy"
        
        # R² interpretation
        r2 = metrics['r2']
        if r2 >= 0.8:
            interpretations['r2'] = "Strong predictive power"
        elif r2 >= 0.6:
            interpretations['r2'] = "Moderate predictive power"
        elif r2 >= 0.4:
            interpretations['r2'] = "Weak predictive power"
        else:
            interpretations['r2'] = "Very weak predictive power"
        
        # Clinical accuracy
        within_2_days = metrics['within_2_days']
        if within_2_days >= 80:
            interpretations['clinical_accuracy'] = "Clinically acceptable for most use cases"
        elif within_2_days >= 60:
            interpretations['clinical_accuracy'] = "Clinically acceptable for some use cases"
        else:
            interpretations['clinical_accuracy'] = "Not clinically acceptable"
        
        return interpretations
