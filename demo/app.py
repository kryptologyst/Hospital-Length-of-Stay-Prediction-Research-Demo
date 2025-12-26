"""
Streamlit demo application for Hospital Length of Stay prediction.

This application provides an interactive interface for exploring the LOS
prediction model, including predictions, explanations, and visualizations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml
import sys
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.core import set_deterministic_seed
from data.processor import LOSDataProcessor
from models.base import ModelFactory, BaseModel
from evaluation.metrics import LOSEvaluator
from explainability.shap_explainer import LOSExplainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Hospital LOS Prediction Demo",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for disclaimer
st.markdown("""
<style>
.disclaimer {
    background-color: #ffebee;
    border: 1px solid #f44336;
    border-radius: 5px;
    padding: 10px;
    margin: 10px 0;
    color: #c62828;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Disclaimer banner
st.markdown("""
<div class="disclaimer">
⚠️ DISCLAIMER: This is a research demonstration project and is NOT intended for clinical use. 
This software is provided for educational and research purposes only. It should not be used 
for medical diagnosis, treatment, or clinical decision-making. Always consult qualified 
healthcare professionals for medical advice.
</div>
""", unsafe_allow_html=True)

# Title and description
st.title("🏥 Hospital Length of Stay Prediction Demo")
st.markdown("""
This interactive demo showcases machine learning models for predicting hospital length of stay 
using structured clinical features. The system includes multiple algorithms, explainability tools, 
and comprehensive evaluation metrics.
""")

# Sidebar configuration
st.sidebar.header("Configuration")

# Model selection
model_type = st.sidebar.selectbox(
    "Select Model",
    ["xgboost", "lightgbm", "gradient_boosting", "tabnet"],
    index=0
)

# Data configuration
n_samples = st.sidebar.slider("Number of Samples", 100, 2000, 1000)
test_size = st.sidebar.slider("Test Size", 0.1, 0.3, 0.2)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = None
if 'evaluator' not in st.session_state:
    st.session_state.evaluator = None
if 'explainer' not in st.session_state:
    st.session_state.explainer = None
if 'results' not in st.session_state:
    st.session_state.results = None

@st.cache_data
def load_and_train_model(model_type: str, n_samples: int, test_size: float):
    """Load and train the model with caching."""
    try:
        # Set random seed
        set_deterministic_seed(42)
        
        # Initialize components
        data_processor = LOSDataProcessor()
        evaluator = LOSEvaluator()
        explainer = LOSExplainer()
        
        # Generate synthetic data
        df = data_processor.generate_synthetic_data(n_samples=n_samples, seed=42)
        
        # Preprocess data
        X, y = data_processor.preprocess_data(df, fit=True)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = data_processor.split_data(
            X, y, test_size=test_size, val_size=0.1, random_state=42
        )
        
        # Create and train model
        model = ModelFactory.create_model(model_type, {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'random_state': 42
        })
        
        model.fit(X_train, y_train, X_val, y_val)
        
        # Make predictions
        y_pred_test = model.predict(X_test)
        
        # Evaluate
        test_metrics = evaluator.evaluate_regression(y_test, y_pred_test)
        test_report = evaluator.generate_evaluation_report(y_test, y_pred_test, model_name=model_type)
        
        # Create explainer
        explainer.create_shap_explainer(model, X_train, data_processor.feature_names)
        explanations = None
        if explainer.explainer is not None:
            explanations = explainer.explain_predictions(X_test, n_samples=50)
        
        return {
            'model': model,
            'data_processor': data_processor,
            'evaluator': evaluator,
            'explainer': explainer,
            'df': df,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred_test': y_pred_test,
            'test_metrics': test_metrics,
            'test_report': test_report,
            'explanations': explanations
        }
        
    except Exception as e:
        st.error(f"Error training model: {e}")
        return None

# Train model button
if st.sidebar.button("Train Model", type="primary"):
    with st.spinner("Training model..."):
        results = load_and_train_model(model_type, n_samples, test_size)
        if results:
            st.session_state.results = results
            st.session_state.model = results['model']
            st.session_state.data_processor = results['data_processor']
            st.session_state.evaluator = results['evaluator']
            st.session_state.explainer = results['explainer']
            st.success("Model trained successfully!")

# Main content
if st.session_state.results is not None:
    results = st.session_state.results
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Model Performance", 
        "🔍 Predictions", 
        "🧠 Explainability", 
        "📈 Data Analysis", 
        "⚙️ Model Details"
    ])
    
    with tab1:
        st.header("Model Performance Metrics")
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("MAE (days)", f"{results['test_metrics']['mae']:.2f}")
        with col2:
            st.metric("RMSE (days)", f"{results['test_metrics']['rmse']:.2f}")
        with col3:
            st.metric("R² Score", f"{results['test_metrics']['r2']:.3f}")
        with col4:
            st.metric("Within 2 Days", f"{results['test_metrics']['within_2_days']:.1f}%")
        
        # Clinical accuracy metrics
        st.subheader("Clinical Accuracy")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Within 1 Day", f"{results['test_metrics']['within_1_day']:.1f}%")
        with col2:
            st.metric("Within 3 Days", f"{results['test_metrics']['within_3_days']:.1f}%")
        with col3:
            st.metric("Over-prediction Rate", f"{results['test_metrics']['over_prediction_rate']:.1f}%")
        
        # Clinical interpretation
        if 'clinical_interpretation' in results['test_report']:
            st.subheader("Clinical Interpretation")
            for metric, interpretation in results['test_report']['clinical_interpretation'].items():
                st.info(f"**{metric.replace('_', ' ').title()}**: {interpretation}")
        
        # Prediction vs Actual scatter plot
        st.subheader("Prediction vs Actual")
        fig_scatter = px.scatter(
            x=results['y_test'], 
            y=results['y_pred_test'],
            labels={'x': 'Actual LOS (days)', 'y': 'Predicted LOS (days)'},
            title="Predicted vs Actual Length of Stay"
        )
        
        # Add perfect prediction line
        max_val = max(max(results['y_test']), max(results['y_pred_test']))
        fig_scatter.add_trace(go.Scatter(
            x=[0, max_val], 
            y=[0, max_val], 
            mode='lines',
            name='Perfect Prediction',
            line=dict(dash='dash', color='red')
        ))
        
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Residuals plot
        residuals = results['y_test'] - results['y_pred_test']
        fig_residuals = px.scatter(
            x=results['y_pred_test'], 
            y=residuals,
            labels={'x': 'Predicted LOS (days)', 'y': 'Residuals (days)'},
            title="Residuals Plot"
        )
        fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_residuals, use_container_width=True)
    
    with tab2:
        st.header("Make Predictions")
        
        # Feature input form
        st.subheader("Patient Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age", 18, 100, 65)
            gender = st.selectbox("Gender", ["Male", "Female"])
            diagnosis = st.selectbox("Primary Diagnosis", [
                "Heart Failure", "Pneumonia", "Infection", "Stroke", "COPD",
                "Diabetes", "Hypertension", "Sepsis", "Trauma", "Surgery"
            ])
            num_medications = st.slider("Number of Medications", 1, 30, 8)
        
        with col2:
            num_lab_procedures = st.slider("Number of Lab Procedures", 5, 50, 15)
            severity_score = st.selectbox("Severity Score", [1, 2, 3, 4])
            admission_type = st.selectbox("Admission Type", ["Emergency", "Elective", "Urgent"])
            insurance_type = st.selectbox("Insurance Type", ["Medicare", "Medicaid", "Private", "Self-pay"])
            comorbidities = st.slider("Number of Comorbidities", 0, 8, 2)
            vital_signs_stable = st.selectbox("Vital Signs Stable", [0, 1])
        
        # Make prediction
        if st.button("Predict Length of Stay"):
            # Create input data
            input_data = {
                'age': age,
                'gender': gender,
                'diagnosis': diagnosis,
                'num_medications': num_medications,
                'num_lab_procedures': num_lab_procedures,
                'severity_score': severity_score,
                'admission_type': admission_type,
                'insurance_type': insurance_type,
                'comorbidities': comorbidities,
                'vital_signs_stable': vital_signs_stable
            }
            
            # Convert to DataFrame and preprocess
            input_df = pd.DataFrame([input_data])
            X_input, _ = st.session_state.data_processor.preprocess_data(input_df, fit=False)
            
            # Make prediction
            prediction = st.session_state.model.predict(X_input)[0]
            
            # Display prediction
            st.subheader("Prediction Result")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Predicted LOS", f"{prediction:.1f} days")
            with col2:
                st.metric("Confidence", "High" if abs(prediction - 7) < 2 else "Medium")
            with col3:
                st.metric("Risk Level", "Low" if prediction < 5 else "High" if prediction > 10 else "Medium")
            
            # Get explanation
            if st.session_state.explainer.explainer is not None:
                explanation = st.session_state.explainer.explain_single_prediction(X_input[0], st.session_state.model)
                
                if explanation:
                    st.subheader("Prediction Explanation")
                    
                    # Feature contributions
                    if 'feature_contributions' in explanation:
                        contributions_df = pd.DataFrame([
                            {'Feature': k, 'Contribution': v}
                            for k, v in explanation['feature_contributions'].items()
                        ])
                        contributions_df = contributions_df.sort_values('Contribution', key=abs, ascending=False)
                        
                        fig_contrib = px.bar(
                            contributions_df.head(10),
                            x='Contribution',
                            y='Feature',
                            orientation='h',
                            title="Feature Contributions to Prediction"
                        )
                        st.plotly_chart(fig_contrib, use_container_width=True)
    
    with tab3:
        st.header("Model Explainability")
        
        if results['explanations']:
            # Feature importance
            st.subheader("Feature Importance")
            importance_data = results['explanations']['feature_importance']
            
            importance_df = pd.DataFrame([
                {'Feature': k, 'Importance': v}
                for k, v in importance_data.items()
            ]).sort_values('Importance', ascending=True)
            
            fig_importance = px.bar(
                importance_df.tail(10),
                x='Importance',
                y='Feature',
                orientation='h',
                title="Top 10 Most Important Features"
            )
            st.plotly_chart(fig_importance, use_container_width=True)
            
            # SHAP summary plot
            st.subheader("SHAP Summary Plot")
            shap_values = results['explanations']['shap_values']
            feature_names = results['explanations']['feature_names']
            
            # Create SHAP summary plot data
            shap_df = pd.DataFrame(shap_values, columns=feature_names)
            
            # Plot mean absolute SHAP values
            mean_abs_shap = shap_df.abs().mean().sort_values(ascending=True)
            
            fig_shap = px.bar(
                x=mean_abs_shap.values,
                y=mean_abs_shap.index,
                orientation='h',
                title="Mean Absolute SHAP Values"
            )
            st.plotly_chart(fig_shap, use_container_width=True)
            
        else:
            st.warning("SHAP explanations not available for this model.")
    
    with tab4:
        st.header("Data Analysis")
        
        # Data summary
        st.subheader("Dataset Summary")
        df = results['df']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Samples", len(df))
        with col2:
            st.metric("Mean LOS", f"{df['length_of_stay'].mean():.1f} days")
        with col3:
            st.metric("Median LOS", f"{df['length_of_stay'].median():.1f} days")
        with col4:
            st.metric("LOS Range", f"{df['length_of_stay'].min()}-{df['length_of_stay'].max()} days")
        
        # LOS distribution
        st.subheader("Length of Stay Distribution")
        fig_los = px.histogram(
            df, 
            x='length_of_stay',
            nbins=20,
            title="Distribution of Length of Stay"
        )
        st.plotly_chart(fig_los, use_container_width=True)
        
        # Feature distributions
        st.subheader("Feature Distributions")
        
        # Age distribution
        fig_age = px.histogram(df, x='age', title="Age Distribution")
        st.plotly_chart(fig_age, use_container_width=True)
        
        # Diagnosis distribution
        diagnosis_counts = df['diagnosis'].value_counts()
        fig_diagnosis = px.pie(
            values=diagnosis_counts.values,
            names=diagnosis_counts.index,
            title="Diagnosis Distribution"
        )
        st.plotly_chart(fig_diagnosis, use_container_width=True)
        
        # LOS by diagnosis
        fig_los_diagnosis = px.box(
            df, 
            x='diagnosis', 
            y='length_of_stay',
            title="Length of Stay by Diagnosis"
        )
        st.plotly_chart(fig_los_diagnosis, use_container_width=True)
    
    with tab5:
        st.header("Model Details")
        
        # Model information
        st.subheader("Model Information")
        st.write(f"**Model Type**: {model_type}")
        st.write(f"**Training Samples**: {len(results['X_test']) + int(len(results['X_test']) / test_size * (1 - test_size))}")
        st.write(f"**Test Samples**: {len(results['X_test'])}")
        st.write(f"**Features**: {len(results['data_processor'].feature_names)}")
        
        # Feature names
        st.subheader("Feature Names")
        feature_df = pd.DataFrame({
            'Feature': results['data_processor'].feature_names
        })
        st.dataframe(feature_df, use_container_width=True)
        
        # Model configuration
        st.subheader("Model Configuration")
        config_dict = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'random_state': 42
        }
        st.json(config_dict)
        
        # Performance metrics table
        st.subheader("Detailed Performance Metrics")
        metrics_df = pd.DataFrame([
            {'Metric': 'Mean Absolute Error', 'Value': f"{results['test_metrics']['mae']:.3f}"},
            {'Metric': 'Root Mean Square Error', 'Value': f"{results['test_metrics']['rmse']:.3f}"},
            {'Metric': 'R² Score', 'Value': f"{results['test_metrics']['r2']:.3f}"},
            {'Metric': 'Mean Absolute Percentage Error', 'Value': f"{results['test_metrics']['mape']:.3f}"},
            {'Metric': 'Within 1 Day (%)', 'Value': f"{results['test_metrics']['within_1_day']:.1f}"},
            {'Metric': 'Within 2 Days (%)', 'Value': f"{results['test_metrics']['within_2_days']:.1f}"},
            {'Metric': 'Within 3 Days (%)', 'Value': f"{results['test_metrics']['within_3_days']:.1f}"},
            {'Metric': 'Over-prediction Rate (%)', 'Value': f"{results['test_metrics']['over_prediction_rate']:.1f}"},
            {'Metric': 'Under-prediction Rate (%)', 'Value': f"{results['test_metrics']['under_prediction_rate']:.1f}"},
        ])
        st.dataframe(metrics_df, use_container_width=True)

else:
    st.info("👈 Please configure the model settings in the sidebar and click 'Train Model' to begin.")
    
    # Show available models
    st.subheader("Available Models")
    available_models = ModelFactory.get_available_models()
    
    for model in available_models:
        st.write(f"✅ {model}")
    
    # Show disclaimer again
    st.markdown("""
    ### Important Notes:
    - This demo uses synthetic data for demonstration purposes
    - Real-world performance may vary significantly
    - Clinical validation is required for any medical application
    - This tool is for educational and research purposes only
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    Hospital Length of Stay Prediction Demo | Research & Educational Use Only
</div>
""", unsafe_allow_html=True)
