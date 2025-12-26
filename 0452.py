"""
Modernized Hospital Length of Stay Prediction Script

This script demonstrates the modernized LOS prediction system with:
- Multiple model types (XGBoost, LightGBM, TabNet)
- Comprehensive evaluation metrics
- SHAP explainability
- Clinical interpretation
- Proper error handling and logging

DISCLAIMER: This is a research demonstration project and is NOT intended for clinical use.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from utils.core import set_deterministic_seed
from data.processor import LOSDataProcessor
from models.base import ModelFactory
from evaluation.metrics import LOSEvaluator
from explainability.shap_explainer import LOSExplainer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main function demonstrating the modernized LOS prediction system."""
    
    print("🏥 Hospital Length of Stay Prediction - Modernized Demo")
    print("=" * 60)
    print("DISCLAIMER: This is a research demonstration project and is NOT intended for clinical use.")
    print("=" * 60)
    
    # Set deterministic seed
    set_deterministic_seed(42)
    
    # Initialize components
    data_processor = LOSDataProcessor()
    evaluator = LOSEvaluator()
    explainer = LOSExplainer()
    
    # Generate synthetic data
    logger.info("Generating synthetic hospital data...")
    df = data_processor.generate_synthetic_data(n_samples=1000, seed=42)
    
    # Display data summary
    data_summary = data_processor.get_data_summary(df)
    print(f"\n📊 Dataset Summary:")
    print(f"   Samples: {data_summary['n_samples']}")
    print(f"   Features: {data_summary['n_features']}")
    print(f"   Mean LOS: {data_summary['target_stats']['mean']:.1f} days")
    print(f"   LOS Range: {data_summary['target_stats']['min']}-{data_summary['target_stats']['max']} days")
    
    # Preprocess data
    logger.info("Preprocessing data...")
    X, y = data_processor.preprocess_data(df, fit=True)
    
    # Split data
    logger.info("Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = data_processor.split_data(
        X, y, test_size=0.2, val_size=0.1, random_state=42
    )
    
    print(f"\n📈 Data Split:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Validation: {len(X_val)} samples")
    print(f"   Test: {len(X_test)} samples")
    
    # Test multiple models
    available_models = ModelFactory.get_available_models()
    print(f"\n🤖 Available Models: {', '.join(available_models)}")
    
    results = {}
    
    for model_type in available_models[:3]:  # Test first 3 available models
        print(f"\n🔄 Training {model_type} model...")
        
        try:
            # Create model
            model = ModelFactory.create_model(model_type, {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'random_state': 42
            })
            
            # Train model
            model.fit(X_train, y_train, X_val, y_val)
            
            # Make predictions
            y_pred_test = model.predict(X_test)
            
            # Evaluate
            test_metrics = evaluator.evaluate_regression(y_test, y_pred_test)
            results[model_type] = test_metrics
            
            # Display results
            print(f"   ✅ MAE: {test_metrics['mae']:.2f} days")
            print(f"   ✅ RMSE: {test_metrics['rmse']:.2f} days")
            print(f"   ✅ R²: {test_metrics['r2']:.3f}")
            print(f"   ✅ Within 2 days: {test_metrics['within_2_days']:.1f}%")
            
        except Exception as e:
            logger.warning(f"Could not train {model_type}: {e}")
            continue
    
    # Create leaderboard
    if results:
        print(f"\n🏆 Model Leaderboard (sorted by MAE):")
        leaderboard = evaluator.create_leaderboard(results, 'mae')
        
        for i, (_, row) in enumerate(leaderboard.iterrows(), 1):
            print(f"   {i}. {row['Model']}: MAE={row['mae']:.2f}, RMSE={row['rmse']:.2f}, R²={row['r2']:.3f}")
        
        # Get best model for explainability
        best_model_name = leaderboard.iloc[0]['Model']
        best_model = ModelFactory.create_model(best_model_name, {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'random_state': 42
        })
        best_model.fit(X_train, y_train, X_val, y_val)
        
        # Generate explanations
        print(f"\n🧠 Generating explanations for best model ({best_model_name})...")
        explainer.create_shap_explainer(best_model, X_train, data_processor.feature_names)
        
        if explainer.explainer is not None:
            explanations = explainer.explain_predictions(X_test, n_samples=50)
            
            if explanations:
                print(f"   ✅ Generated SHAP explanations for {len(explanations['data'])} samples")
                
                # Display top features
                importance = explanations['feature_importance']
                top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
                
                print(f"\n🔍 Top 5 Most Important Features:")
                for i, (feature, importance_pct) in enumerate(top_features, 1):
                    print(f"   {i}. {feature}: {importance_pct:.1f}%")
        
        # Clinical interpretation
        best_metrics = results[best_model_name]
        print(f"\n🏥 Clinical Interpretation:")
        
        if best_metrics['mae'] <= 2.0:
            print("   ✅ Good prediction accuracy for clinical use")
        elif best_metrics['mae'] <= 3.0:
            print("   ⚠️  Moderate prediction accuracy")
        else:
            print("   ❌ Poor prediction accuracy")
        
        if best_metrics['within_2_days'] >= 80:
            print("   ✅ Clinically acceptable for most use cases")
        elif best_metrics['within_2_days'] >= 60:
            print("   ⚠️  Clinically acceptable for some use cases")
        else:
            print("   ❌ Not clinically acceptable")
    
    print(f"\n🎯 Next Steps:")
    print("   1. Run the interactive demo: streamlit run demo/app.py")
    print("   2. Train with custom config: python scripts/train.py --config configs/default.yaml")
    print("   3. Run tests: pytest tests/")
    print("   4. View detailed documentation in README.md")
    
    print(f"\n⚠️  Remember: This is a research demonstration project.")
    print("   For clinical use, external validation and regulatory approval are required.")


if __name__ == "__main__":
    main()
