"""
Explainability and interpretability utilities for Hospital Length of Stay prediction.

This module provides SHAP-based explainability, uncertainty quantification,
and model interpretation tools.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import logging

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available")

logger = logging.getLogger(__name__)


class LOSExplainer:
    """Explainability utilities for LOS prediction models."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the explainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.explainer = None
        self.feature_names = None
        
    def create_shap_explainer(self, model: Any, X_train: np.ndarray, 
                            feature_names: Optional[List[str]] = None) -> None:
        """
        Create SHAP explainer for the model.
        
        Args:
            model: Trained model
            X_train: Training data for background
            feature_names: Names of features
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available, skipping explainer creation")
            return
        
        self.feature_names = feature_names
        
        try:
            # Try different SHAP explainers based on model type
            model_type = type(model).__name__.lower()
            
            if 'xgboost' in model_type or 'lightgbm' in model_type:
                self.explainer = shap.TreeExplainer(model)
            elif 'gradientboosting' in model_type:
                self.explainer = shap.TreeExplainer(model)
            else:
                # Fallback to KernelExplainer for other models
                self.explainer = shap.KernelExplainer(model.predict, X_train[:100])
            
            logger.info("SHAP explainer created successfully")
            
        except Exception as e:
            logger.warning(f"Could not create SHAP explainer: {e}")
            self.explainer = None
    
    def explain_predictions(self, X: np.ndarray, n_samples: int = 100) -> Optional[Dict[str, Any]]:
        """
        Generate SHAP explanations for predictions.
        
        Args:
            X: Input data
            n_samples: Number of samples to explain
            
        Returns:
            Dict[str, Any]: SHAP explanations
        """
        if self.explainer is None:
            logger.warning("No SHAP explainer available")
            return None
        
        try:
            # Limit samples for computational efficiency
            X_sample = X[:n_samples] if len(X) > n_samples else X
            
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(X_sample)
            
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # Take first output for regression
            
            explanations = {
                'shap_values': shap_values,
                'base_value': self.explainer.expected_value,
                'feature_names': self.feature_names,
                'data': X_sample
            }
            
            # Calculate feature importance
            explanations['feature_importance'] = self._calculate_feature_importance(shap_values)
            
            logger.info(f"Generated SHAP explanations for {len(X_sample)} samples")
            return explanations
            
        except Exception as e:
            logger.error(f"Error generating SHAP explanations: {e}")
            return None
    
    def _calculate_feature_importance(self, shap_values: np.ndarray) -> Dict[str, float]:
        """
        Calculate feature importance from SHAP values.
        
        Args:
            shap_values: SHAP values array
            
        Returns:
            Dict[str, float]: Feature importance scores
        """
        if self.feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(shap_values.shape[1])]
        else:
            feature_names = self.feature_names
        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        
        # Normalize to percentages
        total_importance = np.sum(mean_abs_shap)
        if total_importance > 0:
            importance_percentages = (mean_abs_shap / total_importance) * 100
        else:
            importance_percentages = np.zeros_like(mean_abs_shap)
        
        return dict(zip(feature_names, importance_percentages))
    
    def explain_single_prediction(self, x: np.ndarray, model: Any) -> Optional[Dict[str, Any]]:
        """
        Explain a single prediction.
        
        Args:
            x: Single input sample
            model: Trained model
            
        Returns:
            Dict[str, Any]: Single prediction explanation
        """
        if self.explainer is None:
            # Create a simple explanation without SHAP
            prediction = model.predict(x.reshape(1, -1))[0]
            return {
                'prediction': prediction,
                'feature_values': x.tolist(),
                'feature_names': self.feature_names or [f"Feature_{i}" for i in range(len(x))],
                'explanation_method': 'basic'
            }
        
        try:
            # Get SHAP values for single prediction
            shap_values = self.explainer.shap_values(x.reshape(1, -1))
            
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            prediction = model.predict(x.reshape(1, -1))[0]
            
            explanation = {
                'prediction': prediction,
                'base_value': self.explainer.expected_value,
                'shap_values': shap_values[0].tolist(),
                'feature_values': x.tolist(),
                'feature_names': self.feature_names or [f"Feature_{i}" for i in range(len(x))],
                'explanation_method': 'shap'
            }
            
            # Add feature contributions
            explanation['feature_contributions'] = dict(zip(
                explanation['feature_names'],
                shap_values[0].tolist()
            ))
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error explaining single prediction: {e}")
            return None
    
    def generate_feature_importance_plot_data(self, shap_values: np.ndarray) -> Dict[str, Any]:
        """
        Generate data for feature importance plots.
        
        Args:
            shap_values: SHAP values array
            
        Returns:
            Dict[str, Any]: Plot data
        """
        if self.feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(shap_values.shape[1])]
        else:
            feature_names = self.feature_names
        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        
        # Sort by importance
        sorted_indices = np.argsort(mean_abs_shap)[::-1]
        
        plot_data = {
            'feature_names': [feature_names[i] for i in sorted_indices],
            'importance_values': mean_abs_shap[sorted_indices].tolist(),
            'sorted_indices': sorted_indices.tolist()
        }
        
        return plot_data
    
    def generate_waterfall_plot_data(self, explanation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate data for waterfall plot.
        
        Args:
            explanation: Single prediction explanation
            
        Returns:
            Dict[str, Any]: Waterfall plot data
        """
        if explanation.get('explanation_method') != 'shap':
            return {}
        
        shap_values = explanation['shap_values']
        feature_names = explanation['feature_names']
        base_value = explanation['base_value']
        prediction = explanation['prediction']
        
        # Sort features by absolute SHAP value
        abs_shap = np.abs(shap_values)
        sorted_indices = np.argsort(abs_shap)[::-1]
        
        # Calculate cumulative values
        cumulative_values = [base_value]
        for i in sorted_indices:
            cumulative_values.append(cumulative_values[-1] + shap_values[i])
        
        waterfall_data = {
            'feature_names': ['Base'] + [feature_names[i] for i in sorted_indices],
            'shap_values': [0] + shap_values[sorted_indices].tolist(),
            'cumulative_values': cumulative_values,
            'final_prediction': prediction,
            'base_value': base_value
        }
        
        return waterfall_data
    
    def analyze_prediction_uncertainty(self, predictions: np.ndarray, 
                                     confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Analyze prediction uncertainty.
        
        Args:
            predictions: Model predictions
            confidence_level: Confidence level for intervals
            
        Returns:
            Dict[str, Any]: Uncertainty analysis
        """
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        
        # Calculate confidence intervals
        z_score = 1.96 if confidence_level == 0.95 else 2.576
        lower_bound = mean_pred - z_score * std_pred
        upper_bound = mean_pred + z_score * std_pred
        
        uncertainty_analysis = {
            'mean_prediction': mean_pred,
            'std_prediction': std_pred,
            'confidence_level': confidence_level,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'prediction_range': upper_bound - lower_bound,
            'coefficient_of_variation': std_pred / mean_pred if mean_pred != 0 else 0
        }
        
        return uncertainty_analysis
    
    def generate_explanation_summary(self, explanations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate summary of explanations.
        
        Args:
            explanations: SHAP explanations
            
        Returns:
            Dict[str, Any]: Explanation summary
        """
        if explanations is None:
            return {}
        
        summary = {
            'n_samples_explained': len(explanations['data']),
            'feature_importance': explanations['feature_importance'],
            'top_features': self._get_top_features(explanations['feature_importance'], top_k=5),
            'explanation_quality': self._assess_explanation_quality(explanations)
        }
        
        return summary
    
    def _get_top_features(self, feature_importance: Dict[str, float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Get top K most important features.
        
        Args:
            feature_importance: Feature importance dictionary
            top_k: Number of top features
            
        Returns:
            List[Dict[str, Any]]: Top features with importance scores
        """
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        top_features = []
        for i, (feature, importance) in enumerate(sorted_features[:top_k]):
            top_features.append({
                'rank': i + 1,
                'feature': feature,
                'importance': importance,
                'importance_percent': f"{importance:.1f}%"
            })
        
        return top_features
    
    def _assess_explanation_quality(self, explanations: Dict[str, Any]) -> Dict[str, str]:
        """
        Assess the quality of explanations.
        
        Args:
            explanations: SHAP explanations
            
        Returns:
            Dict[str, str]: Quality assessment
        """
        feature_importance = explanations['feature_importance']
        
        # Check if explanations are concentrated in few features
        max_importance = max(feature_importance.values())
        top_3_importance = sum(sorted(feature_importance.values(), reverse=True)[:3])
        
        quality_assessment = {}
        
        if max_importance > 50:
            quality_assessment['concentration'] = "High concentration in single feature"
        elif top_3_importance > 80:
            quality_assessment['concentration'] = "Concentrated in top 3 features"
        else:
            quality_assessment['concentration'] = "Distributed across features"
        
        # Check explanation stability
        if len(explanations['shap_values']) > 1:
            shap_std = np.std(explanations['shap_values'], axis=0)
            mean_abs_shap = np.mean(np.abs(explanations['shap_values']), axis=0)
            stability_ratio = np.mean(shap_std / (mean_abs_shap + 1e-8))
            
            if stability_ratio < 0.5:
                quality_assessment['stability'] = "High stability"
            elif stability_ratio < 1.0:
                quality_assessment['stability'] = "Moderate stability"
            else:
                quality_assessment['stability'] = "Low stability"
        
        return quality_assessment
