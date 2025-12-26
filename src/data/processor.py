"""
Data processing utilities for Hospital Length of Stay prediction.

This module handles data loading, preprocessing, feature engineering,
and synthetic data generation for the LOS prediction project.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional, Union
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


class LOSDataProcessor:
    """Data processor for Hospital Length of Stay prediction."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the data processor.
        
        Args:
            config: Configuration dictionary for data processing
        """
        self.config = config or {}
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = None
        
    def generate_synthetic_data(self, n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
        """
        Generate synthetic hospital admission data for demonstration.
        
        Args:
            n_samples: Number of samples to generate
            seed: Random seed for reproducibility
            
        Returns:
            pd.DataFrame: Synthetic hospital data
        """
        np.random.seed(seed)
        
        # Define realistic distributions based on clinical knowledge
        data = {
            'age': np.random.normal(65, 15, n_samples).astype(int).clip(18, 100),
            'gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.52, 0.48]),
            'diagnosis': np.random.choice([
                'Heart Failure', 'Pneumonia', 'Infection', 'Stroke', 'COPD',
                'Diabetes', 'Hypertension', 'Sepsis', 'Trauma', 'Surgery'
            ], n_samples, p=[0.15, 0.12, 0.10, 0.08, 0.08, 0.10, 0.12, 0.08, 0.09, 0.08]),
            'num_medications': np.random.poisson(8, n_samples).clip(1, 30),
            'num_lab_procedures': np.random.poisson(15, n_samples).clip(5, 50),
            'severity_score': np.random.choice([1, 2, 3, 4], n_samples, p=[0.2, 0.3, 0.3, 0.2]),
            'admission_type': np.random.choice(['Emergency', 'Elective', 'Urgent'], n_samples, p=[0.6, 0.2, 0.2]),
            'insurance_type': np.random.choice(['Medicare', 'Medicaid', 'Private', 'Self-pay'], n_samples, p=[0.4, 0.2, 0.35, 0.05]),
            'comorbidities': np.random.poisson(2, n_samples).clip(0, 8),
            'vital_signs_stable': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        }
        
        df = pd.DataFrame(data)
        
        # Create realistic LOS based on clinical factors
        base_los = np.random.exponential(5, n_samples)
        
        # Adjust LOS based on clinical factors
        los_adjustments = {
            'Heart Failure': 1.5,
            'Pneumonia': 1.3,
            'Infection': 1.4,
            'Stroke': 2.0,
            'COPD': 1.2,
            'Diabetes': 1.1,
            'Hypertension': 1.0,
            'Sepsis': 2.5,
            'Trauma': 1.8,
            'Surgery': 1.6
        }
        
        for diagnosis, multiplier in los_adjustments.items():
            mask = df['diagnosis'] == diagnosis
            base_los[mask] *= multiplier
        
        # Additional adjustments
        base_los *= (1 + df['severity_score'] * 0.3)  # Severity impact
        base_los *= (1 + df['comorbidities'] * 0.1)   # Comorbidity impact
        base_los *= (1 + df['age'] / 100 * 0.2)       # Age impact
        base_los *= (2 - df['vital_signs_stable'])    # Stability impact
        
        df['length_of_stay'] = np.round(base_los).astype(int).clip(1, 30)
        
        logger.info(f"Generated {n_samples} synthetic hospital records")
        return df
    
    def preprocess_data(self, df: pd.DataFrame, fit: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the data for machine learning.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit transformers (True for training, False for inference)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Features and target arrays
        """
        df_processed = df.copy()
        
        # Encode categorical variables
        categorical_cols = ['gender', 'diagnosis', 'admission_type', 'insurance_type']
        
        for col in categorical_cols:
            if col in df_processed.columns:
                if fit:
                    le = LabelEncoder()
                    df_processed[col] = le.fit_transform(df_processed[col])
                    self.label_encoders[col] = le
                else:
                    if col in self.label_encoders:
                        # Handle unseen categories
                        le = self.label_encoders[col]
                        df_processed[col] = df_processed[col].apply(
                            lambda x: le.transform([x])[0] if x in le.classes_ else -1
                        )
        
        # Separate features and target
        feature_cols = [col for col in df_processed.columns if col != 'length_of_stay']
        X = df_processed[feature_cols].values
        y = df_processed['length_of_stay'].values
        
        # Store feature names
        if fit:
            self.feature_names = feature_cols
        
        # Scale features
        if fit:
            self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
            X = self.scaler.fit_transform(X)
        else:
            if self.scaler is not None:
                X = self.scaler.transform(X)
        
        logger.info(f"Preprocessed data: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    def create_feature_importance_data(self) -> pd.DataFrame:
        """
        Create a DataFrame with feature importance information.
        
        Returns:
            pd.DataFrame: Feature importance data
        """
        if self.feature_names is None:
            logger.warning("Feature names not available. Run preprocessing first.")
            return pd.DataFrame()
        
        # Clinical knowledge-based feature importance (for demonstration)
        importance_data = {
            'feature': self.feature_names,
            'clinical_importance': [
                'High' if 'severity' in name.lower() or 'diagnosis' in name.lower() else
                'Medium' if 'age' in name.lower() or 'comorbidities' in name.lower() else
                'Low'
                for name in self.feature_names
            ],
            'description': [
                'Patient age in years',
                'Patient gender (encoded)',
                'Primary diagnosis (encoded)',
                'Number of medications prescribed',
                'Number of laboratory procedures',
                'Clinical severity score (1-4)',
                'Type of admission (encoded)',
                'Insurance type (encoded)',
                'Number of comorbidities',
                'Vital signs stability (0/1)'
            ][:len(self.feature_names)]
        }
        
        return pd.DataFrame(importance_data)
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                   test_size: float = 0.2, val_size: float = 0.1, 
                   random_state: int = 42) -> Tuple[np.ndarray, ...]:
        """
        Split data into train/validation/test sets.
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation
            random_state: Random seed
            
        Returns:
            Tuple: X_train, X_val, X_test, y_train, y_val, y_test
        """
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
        )
        
        logger.info(f"Data split - Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get summary statistics for the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dict: Summary statistics
        """
        summary = {
            'n_samples': len(df),
            'n_features': len(df.columns) - 1,  # Excluding target
            'target_stats': {
                'mean': df['length_of_stay'].mean(),
                'std': df['length_of_stay'].std(),
                'min': df['length_of_stay'].min(),
                'max': df['length_of_stay'].max(),
                'median': df['length_of_stay'].median()
            },
            'missing_values': df.isnull().sum().to_dict(),
            'categorical_features': df.select_dtypes(include=['object']).columns.tolist(),
            'numerical_features': df.select_dtypes(include=[np.number]).columns.tolist()
        }
        
        return summary
