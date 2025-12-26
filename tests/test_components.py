"""
Unit tests for Hospital Length of Stay prediction project.

This module contains tests for all major components including
data processing, model training, evaluation, and explainability.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.core import set_deterministic_seed, get_device, EarlyStopping, deidentify_text
from data.processor import LOSDataProcessor
from models.base import ModelFactory, GradientBoostingModel
from evaluation.metrics import LOSEvaluator
from explainability.shap_explainer import LOSExplainer


class TestCoreUtils:
    """Test core utility functions."""
    
    def test_set_deterministic_seed(self):
        """Test deterministic seeding."""
        set_deterministic_seed(42)
        
        # Test numpy randomness
        np.random.seed(42)
        val1 = np.random.random()
        
        set_deterministic_seed(42)
        val2 = np.random.random()
        
        assert val1 == val2
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert device is not None
        assert hasattr(device, 'type')
    
    def test_early_stopping(self):
        """Test early stopping functionality."""
        early_stopping = EarlyStopping(patience=3, min_delta=0.1)
        
        # Test improvement
        assert not early_stopping(0.8)  # First call
        assert not early_stopping(0.9)  # Improvement
        
        # Test no improvement
        assert not early_stopping(0.85)  # No improvement
        assert not early_stopping(0.85)  # Still no improvement
        assert not early_stopping(0.85)  # Still no improvement
        assert early_stopping(0.85)  # Should stop now
    
    def test_deidentify_text(self):
        """Test text de-identification."""
        text = "Patient John Doe, SSN 123-45-6789, phone 555-123-4567"
        deidentified = deidentify_text(text)
        
        assert "John Doe" not in deidentified
        assert "123-45-6789" not in deidentified
        assert "555-123-4567" not in deidentified
        assert "[SSN]" in deidentified
        assert "[PHONE]" in deidentified


class TestDataProcessor:
    """Test data processing functionality."""
    
    def test_generate_synthetic_data(self):
        """Test synthetic data generation."""
        processor = LOSDataProcessor()
        df = processor.generate_synthetic_data(n_samples=100, seed=42)
        
        assert len(df) == 100
        assert 'length_of_stay' in df.columns
        assert df['length_of_stay'].min() >= 1
        assert df['length_of_stay'].max() <= 30
    
    def test_preprocess_data(self):
        """Test data preprocessing."""
        processor = LOSDataProcessor()
        df = processor.generate_synthetic_data(n_samples=50, seed=42)
        
        X, y = processor.preprocess_data(df, fit=True)
        
        assert X.shape[0] == 50
        assert X.shape[1] > 0
        assert len(y) == 50
        assert processor.scaler is not None
        assert processor.feature_names is not None
    
    def test_split_data(self):
        """Test data splitting."""
        processor = LOSDataProcessor()
        df = processor.generate_synthetic_data(n_samples=100, seed=42)
        X, y = processor.preprocess_data(df, fit=True)
        
        splits = processor.split_data(X, y, test_size=0.2, val_size=0.1)
        X_train, X_val, X_test, y_train, y_val, y_test = splits
        
        assert len(X_train) + len(X_val) + len(X_test) == 100
        assert len(y_train) + len(y_val) + len(y_test) == 100
        assert len(X_test) == 20  # 20% of 100
        assert len(X_val) == 8   # 10% of 80
    
    def test_get_data_summary(self):
        """Test data summary generation."""
        processor = LOSDataProcessor()
        df = processor.generate_synthetic_data(n_samples=50, seed=42)
        
        summary = processor.get_data_summary(df)
        
        assert summary['n_samples'] == 50
        assert summary['n_features'] == len(df.columns) - 1
        assert 'target_stats' in summary
        assert 'missing_values' in summary


class TestModels:
    """Test model functionality."""
    
    def test_model_factory(self):
        """Test model factory."""
        available_models = ModelFactory.get_available_models()
        assert 'gradient_boosting' in available_models
        
        # Test creating a model
        model = ModelFactory.create_model('gradient_boosting')
        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
    
    def test_gradient_boosting_model(self):
        """Test gradient boosting model."""
        model = GradientBoostingModel()
        
        # Generate test data
        X = np.random.random((50, 5))
        y = np.random.random(50) * 10 + 1
        
        # Train model
        model.fit(X, y)
        
        # Make predictions
        predictions = model.predict(X)
        
        assert len(predictions) == 50
        assert all(pred >= 0 for pred in predictions)
        
        # Test feature importance
        importance = model.get_feature_importance()
        assert importance is not None
        assert len(importance) == 5


class TestEvaluation:
    """Test evaluation functionality."""
    
    def test_evaluate_regression(self):
        """Test regression evaluation."""
        evaluator = LOSEvaluator()
        
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        
        metrics = evaluator.evaluate_regression(y_true, y_pred)
        
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'r2' in metrics
        assert 'within_1_day' in metrics
        assert 'within_2_days' in metrics
    
    def test_evaluate_calibration(self):
        """Test calibration evaluation."""
        evaluator = LOSEvaluator()
        
        y_true = np.random.random(100) * 10 + 1
        y_pred = y_true + np.random.normal(0, 0.5, 100)
        
        calibration = evaluator.evaluate_calibration(y_true, y_pred)
        
        assert 'ece' in calibration
        assert 'mce' in calibration
        assert 'calibration_data' in calibration
    
    def test_evaluate_fairness(self):
        """Test fairness evaluation."""
        evaluator = LOSEvaluator()
        
        y_true = np.random.random(100) * 10 + 1
        y_pred = y_true + np.random.normal(0, 0.5, 100)
        groups = np.random.choice([0, 1], 100)
        
        fairness = evaluator.evaluate_fairness(y_true, y_pred, groups)
        
        assert 'Group_0' in fairness
        assert 'Group_1' in fairness
        assert 'mae_gap' in fairness
    
    def test_create_leaderboard(self):
        """Test leaderboard creation."""
        evaluator = LOSEvaluator()
        
        results = {
            'Model1': {'mae': 2.1, 'rmse': 2.8, 'r2': 0.73},
            'Model2': {'mae': 2.3, 'rmse': 3.0, 'r2': 0.71},
            'Model3': {'mae': 2.0, 'rmse': 2.7, 'r2': 0.75}
        }
        
        leaderboard = evaluator.create_leaderboard(results, 'mae')
        
        assert len(leaderboard) == 3
        assert leaderboard.iloc[0]['Model'] == 'Model3'  # Lowest MAE


class TestExplainability:
    """Test explainability functionality."""
    
    def test_los_explainer_init(self):
        """Test explainer initialization."""
        explainer = LOSExplainer()
        assert explainer.explainer is None
        assert explainer.feature_names is None
    
    def test_explain_single_prediction_basic(self):
        """Test basic single prediction explanation."""
        explainer = LOSExplainer()
        
        # Mock model
        model = MagicMock()
        model.predict.return_value = np.array([5.2])
        
        x = np.array([1, 2, 3, 4, 5])
        explanation = explainer.explain_single_prediction(x, model)
        
        assert explanation is not None
        assert explanation['prediction'] == 5.2
        assert explanation['explanation_method'] == 'basic'
        assert len(explanation['feature_values']) == 5
    
    def test_analyze_prediction_uncertainty(self):
        """Test uncertainty analysis."""
        explainer = LOSExplainer()
        
        predictions = np.array([5.1, 5.2, 5.0, 5.3, 4.9])
        uncertainty = explainer.analyze_prediction_uncertainty(predictions)
        
        assert 'mean_prediction' in uncertainty
        assert 'std_prediction' in uncertainty
        assert 'confidence_level' in uncertainty
        assert 'lower_bound' in uncertainty
        assert 'upper_bound' in uncertainty


class TestIntegration:
    """Test integration between components."""
    
    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline."""
        # Set seed
        set_deterministic_seed(42)
        
        # Initialize components
        processor = LOSDataProcessor()
        evaluator = LOSEvaluator()
        explainer = LOSExplainer()
        
        # Generate data
        df = processor.generate_synthetic_data(n_samples=100, seed=42)
        
        # Preprocess
        X, y = processor.preprocess_data(df, fit=True)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(X, y)
        
        # Train model
        model = ModelFactory.create_model('gradient_boosting')
        model.fit(X_train, y_train, X_val, y_val)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate
        metrics = evaluator.evaluate_regression(y_test, y_pred)
        
        # Test basic functionality
        assert metrics['mae'] > 0
        assert metrics['r2'] >= 0
        assert len(y_pred) == len(y_test)
        
        # Test explainer
        explainer.create_shap_explainer(model, X_train, processor.feature_names)
        
        # Test single prediction explanation
        explanation = explainer.explain_single_prediction(X_test[0], model)
        assert explanation is not None
        assert 'prediction' in explanation


if __name__ == "__main__":
    pytest.main([__file__])
